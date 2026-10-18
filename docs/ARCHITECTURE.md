# FibonacciNet — implementation notes

Source: S. Roy et al., *Fibonacci-Net: A Lightweight CNN model for Automatic Brain Tumor Classification*, arXiv:2503.13928. The paper targets brain MRI; this project applies the same architecture to thyroid ultrasound.

## What the paper specifies (implemented as described)

| Element | Paper | Here |
|---|---|---|
| Filter counts | Fibonacci series 21, 34, 55, 89, 144, 233, 377 | `model.filters` |
| Kernels | 3×3, stride 1, `same` padding | ✓ |
| BatchNorm | after every conv layer | ✓ |
| Depthwise separable convs | in the last two blocks | `model.dwsc_last_n: 2` |
| pcb 2→4 | output of block 2 → Avg-2Max pooling → concatenated into the input of block 4 | `pcb24_*` layers |
| pcb 3→5 | output of block 3 → 24 conv 3×3 filters → Avg-2Max pooling → concatenated into the input of block 5 | `pcb35_*` layers |
| Avg-2Max pooling | AvgPool(3×3, s2) − 2·MaxPool(3×3, s2) | `Avg2MaxPooling2D` |
| Head | Global Average Pooling → single-layer perceptron (softmax) | `gap` → `predictions` |
| Final map | 7×7×377 (mentioned when comparing GAP to flatten) | `pool_blocks: 5` gives 7×7×377 |
| Training | Adam, batch 8, 224×224, stratified 80/10/10, lr 1e-4 for 13 epochs then ×0.9 decay up to 25 epochs, no augmentation, no early stopping | `train.*` defaults |
| Blocks | 6 for small datasets, 7 for larger ones | drop `377` from `filters` for 6 |

## Assumptions (Fig. 1 detail not stated in the text)

- **One conv per block** by default, MaxPool(2) after blocks 1–5. This gives ~0.38 M parameters; the paper reports ~1.2–1.4 M, so its blocks likely contain more layers. `model.convs_per_block: 2` gives ~0.89 M.
- **Dropout 0.3** before the classifier (not mentioned in the paper; set `model.dropout: 0` to remove).
- **BatchNorm momentum 0.9** instead of Keras' 0.99, so running statistics settle on datasets of a few hundred images. With 0.99 the validation predictions can stay flat for many epochs.
- **Grad-CAM layer** is the ReLU after the last block (`last_conv`, 7×7 for 224 input).

## Avg-2Max pooling intuition

Average pooling is a blur; max pooling keeps the strongest response. `avg − 2·max` subtracts the "sharp" image from its blurred version twice, producing an edge-emphasised, negative-like copy of earlier features. Concatenating it back into the main path mixes features with different statistics, which the authors argue acts as in-network augmentation and helps minority classes.

## Shape trace (224×224×3 input, defaults)

```
block1 conv21  224 → pool 112
block2 conv34  112 → pool 56 ───────────────┐ Avg2Max → 28×28×34
block3 conv55   56 → pool 28 ──┐            │
                               │ conv24     │
block4 input  = concat(28×28×55, 28×28×34) = 28×28×89
block4 conv89   28 → pool 14   │ Avg2Max → 14×14×24
block5 input  = concat(14×14×89, 14×14×24) = 14×14×113
block5 conv144  14 → pool 7
block6 dwsc233   7
block7 dwsc377   7  → last_conv (Grad-CAM) → GAP 377 → Dropout → Dense(n_classes)
```
