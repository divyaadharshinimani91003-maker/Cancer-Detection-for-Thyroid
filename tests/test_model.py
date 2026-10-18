import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
import keras  # noqa: E402

from thyrocheck.models.fibonacci_net import Avg2MaxPooling2D, build_fibonacci_net, load_model  # noqa: E402
from thyrocheck.xai.gradcam import GradCAM  # noqa: E402


def test_avg2max_matches_definition():
    x = np.random.rand(1, 8, 8, 2).astype("float32")
    out = Avg2MaxPooling2D()(x).numpy()
    avg = keras.layers.AveragePooling2D(3, 2, "same")(x).numpy()
    mx = keras.layers.MaxPooling2D(3, 2, "same")(x).numpy()
    np.testing.assert_allclose(out, avg - 2 * mx, rtol=1e-5)
    assert out.shape == (1, 4, 4, 2)


def test_model_shapes_and_size():
    m = build_fibonacci_net(num_classes=3)
    assert m.output_shape == (None, 3)
    assert tuple(m.get_layer("last_conv").output.shape[1:]) == (7, 7, 377)
    assert m.count_params() < 2_000_000          # lightweight, like the paper (~1.3-1.4 M)
    assert m.get_layer("pcb24_concat").output.shape[-1] == 55 + 34
    assert m.get_layer("pcb35_concat").output.shape[-1] == 89 + 24


def test_save_load_and_gradcam(tmp_path):
    m = build_fibonacci_net(num_classes=2, input_size=96, filters=[8, 13, 21, 34, 55])
    p = tmp_path / "m.keras"
    m.save(p)
    m2 = load_model(str(p))
    x = np.random.rand(1, 96, 96, 3).astype("float32")
    np.testing.assert_allclose(m.predict(x, verbose=0), m2.predict(x, verbose=0), rtol=1e-5)
    heat = GradCAM(m2, "last_conv").heatmap(x)
    assert heat.shape == (96, 96) and heat.min() >= 0 and heat.max() <= 1
