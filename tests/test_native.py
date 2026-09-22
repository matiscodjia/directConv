"""The Rust core, through PyO3, must agree with numpy."""

import numpy as np
import pytest

from directConv import _native
from directConv.backends import conv2d
from directConv.filters import CONV_FILTER

RNG = np.random.default_rng(0)

# (C, H, W, K, KH, KW) shapes small enough to run in a test.
SMALL_SHAPES = [s for s in map(tuple, _native.registered_shapes()) if s[1] * s[2] <= 256 * 256]


@pytest.mark.parametrize("shape", SMALL_SHAPES, ids=str)
@pytest.mark.parametrize("n", [1, 3])
def test_matches_numpy(shape, n):
    c, h, w, k, kh, kw = shape
    video = RNG.random((n, c, h, w), dtype=np.float32)
    filters = RNG.standard_normal((k, c, kh, kw)).astype(np.float32)
    ref, out = conv2d(video, filters, "numpy"), conv2d(video, filters, "local")
    assert out.shape == ref.shape == (n, h - kh + 1, w - kw + 1, k)
    assert out.dtype == np.float32
    assert np.abs(ref - out).max() / np.abs(ref).max() < 1e-5


def test_sobel_gaussian_bank_on_uint8_video():
    video = RNG.integers(0, 256, (2, 3, 360, 640), dtype=np.uint8)
    ref, out = conv2d(video, CONV_FILTER, "numpy"), conv2d(video, CONV_FILTER, "local")
    assert np.abs(ref - out).max() / np.abs(ref).max() < 1e-5


def test_non_contiguous_input_is_accepted_by_the_python_layer():
    video = RNG.random((1, 16, 16, 3), dtype=np.float32).transpose(0, 3, 1, 2)  # NHWC memory
    filters = RNG.random((2, 3, 3, 3), dtype=np.float32)
    assert conv2d(video, filters, "local").shape == (1, 14, 14, 2)


def test_unregistered_shape_says_what_to_add():
    with pytest.raises(ValueError, match=r"add `\(3, 20, 20, 2, 3, 3\)`.*register_shapes!"):
        _native.cross_correlate2d(np.zeros((1, 3, 20, 20), np.float32), np.zeros((2, 3, 3, 3), np.float32))


def test_channel_mismatch():
    with pytest.raises(ValueError, match="channels"):
        _native.cross_correlate2d(np.zeros((1, 3, 16, 16), np.float32), np.zeros((2, 1, 3, 3), np.float32))


def test_device_selection():
    video, filters = np.zeros((1, 3, 16, 16), np.float32), np.zeros((2, 3, 3, 3), np.float32)
    with pytest.raises(NotImplementedError):
        conv2d(video, filters, "ssh:pi")
    with pytest.raises(ValueError, match="unknown device"):
        conv2d(video, filters, "gpu")
