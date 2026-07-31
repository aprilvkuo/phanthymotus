"""预处理纯 numpy 测试（无需 torch）。"""
import numpy as np
from preprocess import letterbox_resize, normalize, to_tensor, preprocess


def test_letterbox_shape_and_pad():
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    out = letterbox_resize(img, 224, [0.485, 0.456, 0.406])
    assert out.shape == (224, 224, 3)
    # pad 区域应等于 ImageNet 均值(uint8 ≈ 123,116,104)
    assert out[0, 0].tolist() == [123, 116, 104]


def test_letterbox_keeps_content_centered():
    img = np.full((224, 224, 3), 200, dtype=np.uint8)
    out = letterbox_resize(img, 224, [0, 0, 0])
    assert int(out[112, 112, 0]) == 200


def test_normalize_range():
    img = np.full((10, 10, 3), 128, dtype=np.uint8)
    out = normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    assert out.dtype == np.float32
    expected = (128 / 255.0 - 0.485) / 0.229
    assert abs(out[0, 0, 0] - expected) < 1e-4


def test_to_tensor_shape():
    arr = np.zeros((5, 5, 3), dtype=np.float32)
    t = to_tensor(arr)
    # numpy 路径返回 [1,3,5,5]；torch 路径返回同形状张量
    assert tuple(t.shape) == (1, 3, 5, 5)


def test_preprocess_output_shape():
    img = np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8)
    out = preprocess(
        img, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], [0.485, 0.456, 0.406]
    )
    assert tuple(out.shape) == (1, 3, 224, 224)
