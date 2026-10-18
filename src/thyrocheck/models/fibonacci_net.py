"""FibonacciNet - lightweight CNN with Fibonacci filter counts and Avg-2Max parallel concat blocks.

Re-implementation of: S. Roy et al., "Fibonacci-Net: A Lightweight CNN model for
Automatic Brain Tumor Classification", arXiv:2503.13928 (2025), adapted here to
thyroid ultrasound.

Architecture (default, 224x224 input, 7 blocks):

  block k : Conv3x3(f_k) -> BN -> ReLU [-> MaxPool2 for k <= 5]   f = 21, 34, 55, 89, 144, 233, 377
            (the last `dwsc_last_n` blocks use SeparableConv2D; final map is 7x7x377 for 224 input)

  pcb 2->4: out(block2) --Avg2MaxPool(3,2)--> concat with out(block3) --> input of block4
  pcb 3->5: out(block3) --Conv3x3(24)->BN->ReLU->Avg2MaxPool(3,2)--> concat with out(block4) --> input of block5

  head    : last conv activation ("last_conv", used by Grad-CAM) -> GAP -> Dropout -> Dense(softmax)

Avg-2Max pooling = AvgPool(3x3, s2) - 2 * MaxPool(3x3, s2): an edge-emphasising,
"negative-image" view of earlier features that acts as an in-network feature augmenter.

Where the paper's Fig. 1 leaves details open (pooling placement, dropout), the
choices above are documented assumptions - see docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import tensorflow as tf
import keras
from keras import layers


@keras.saving.register_keras_serializable(package="thyrocheck")
class Avg2MaxPooling2D(layers.Layer):
    """AvgPool(x) - 2 * MaxPool(x) with the same pool size / stride."""

    def __init__(self, pool_size: int = 3, strides: int = 2, padding: str = "same", **kwargs):
        super().__init__(**kwargs)
        self.pool_size, self.strides, self.padding = pool_size, strides, padding
        self.avg = layers.AveragePooling2D(pool_size, strides, padding)
        self.max = layers.MaxPooling2D(pool_size, strides, padding)

    def call(self, x):
        return self.avg(x) - 2.0 * self.max(x)

    def compute_output_shape(self, input_shape):
        return self.avg.compute_output_shape(input_shape)

    def get_config(self):
        return {**super().get_config(), "pool_size": self.pool_size,
                "strides": self.strides, "padding": self.padding}


def _conv_block(x, filters: int, idx: int, separable: bool, is_last: bool, pool: bool, n_convs: int = 1,
                bn_momentum: float = 0.9):
    conv = layers.SeparableConv2D if separable else layers.Conv2D
    kind = "dwsc" if separable else "conv"
    for j in range(n_convs):
        suffix = "" if n_convs == 1 else f"_{j + 1}"
        x = conv(filters, 3, padding="same", use_bias=False, name=f"block{idx}_{kind}{suffix}")(x)
        x = layers.BatchNormalization(momentum=bn_momentum, name=f"block{idx}_bn{suffix}")(x)
        last_act = is_last and j == n_convs - 1
        x = layers.Activation("relu", name="last_conv" if last_act else f"block{idx}_relu{suffix}")(x)
    if pool:
        x = layers.MaxPooling2D(2, name=f"block{idx}_pool")(x)
    return x


def build_fibonacci_net(
    num_classes: int,
    input_size: int = 224,
    channels: int = 3,
    filters: list[int] | tuple[int, ...] = (21, 34, 55, 89, 144, 233, 377),
    dwsc_last_n: int = 2,
    use_pcb_2_4: bool = True,
    use_pcb_3_5: bool = True,
    pcb_3_5_filters: int = 24,
    dropout: float = 0.3,
    pool_blocks: int = 5,
    convs_per_block: int = 1,
    bn_momentum: float = 0.9,
) -> keras.Model:
    """pool_blocks: MaxPool after the first N blocks (5 -> 224/32 = 7x7 final map, as in the paper).
    convs_per_block: stack this many convs in each block to scale capacity up.
    bn_momentum: 0.9 (vs Keras' 0.99) so BatchNorm statistics settle on small medical datasets."""
    filters = list(filters)
    n = len(filters)
    if n < 5 and (use_pcb_2_4 or use_pcb_3_5):
        raise ValueError("Parallel concat blocks need at least 5 conv blocks")

    inputs = keras.Input((input_size, input_size, channels), name="image")
    x = inputs
    outs = {}
    for i, f in enumerate(filters, start=1):
        # merge parallel branches into the input of blocks 4 and 5
        if i == 4 and use_pcb_2_4:
            branch = Avg2MaxPooling2D(name="pcb24_avg2max")(outs[2])
            x = layers.Concatenate(name="pcb24_concat")([x, branch])
        if i == 5 and use_pcb_3_5:
            b = layers.Conv2D(pcb_3_5_filters, 3, padding="same", use_bias=False, name="pcb35_conv")(outs[3])
            b = layers.BatchNormalization(momentum=bn_momentum, name="pcb35_bn")(b)
            b = layers.Activation("relu", name="pcb35_relu")(b)
            b = Avg2MaxPooling2D(name="pcb35_avg2max")(b)
            x = layers.Concatenate(name="pcb35_concat")([x, b])
        x = _conv_block(x, f, i, separable=i > n - dwsc_last_n, is_last=i == n,
                        pool=i <= pool_blocks, n_convs=convs_per_block, bn_momentum=bn_momentum)
        outs[i] = x

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="FibonacciNet")


def build_from_config(cfg, num_classes: int) -> keras.Model:
    m = cfg.model
    return build_fibonacci_net(
        num_classes=num_classes, input_size=m.input_size, channels=m.channels,
        filters=m.filters, dwsc_last_n=m.dwsc_last_n, use_pcb_2_4=m.use_pcb_2_4,
        use_pcb_3_5=m.use_pcb_3_5, pcb_3_5_filters=m.pcb_3_5_filters, dropout=m.dropout,
        pool_blocks=m.get("pool_blocks", 5), convs_per_block=m.get("convs_per_block", 1),
        bn_momentum=m.get("bn_momentum", 0.9),
    )


def load_model(path: str) -> keras.Model:
    return keras.models.load_model(path, custom_objects={"Avg2MaxPooling2D": Avg2MaxPooling2D})


if __name__ == "__main__":
    model = build_fibonacci_net(num_classes=3)
    model.summary(line_length=110)
