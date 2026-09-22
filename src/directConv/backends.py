"""Where a convolution runs.

`conv2d(video, filters, device=...)` is the one entry point: the caller picks
the device, the layout stays the same everywhere (video N x C x H x W, filters
K x C x KH x KW, result N x H_OUT x W_OUT x K, float32).

- ``"local"``: the frugal_ml Rust core, called in-process through PyO3.
- ``"numpy"``: the reference, `directConv.convolver.compute`.
- ``"ssh:<host>"``: the edge device, reserved for a later step.
"""

import numpy as np

from directConv import _native
from directConv.convolver import compute as numpy_compute


def _as_f32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def conv2d(video: np.ndarray, filters: np.ndarray, device: str = "local") -> np.ndarray:
    if device == "local":
        return _native.cross_correlate2d(_as_f32(video), _as_f32(filters))
    if device == "numpy":
        return numpy_compute(video, conv_filter=filters)
    if device.startswith("ssh:"):
        raise NotImplementedError(
            f"device {device!r}: the SSH backend is not implemented yet, only 'local' and 'numpy' are"
        )
    raise ValueError(f"unknown device {device!r}, expected 'local', 'numpy' or 'ssh:<host>'")
