"""Rust (frugal_ml, via PyO3) vs numpy on a real video.

    uv run python benchmarks/compare_rust_numpy.py                    # cockatoo, 8 frames
    uv run python benchmarks/compare_rust_numpy.py URL --frames 16    # downloads the video first

Both sides get the same float32 (N, C, H, W) input and must agree on the
(N, H_OUT, W_OUT, K) output before any timing is reported.
"""

import argparse
import statistics
from time import perf_counter

import numpy as np

from directConv import _native
from directConv.backends import conv2d
from directConv.filters import CONV_FILTER
from directConv.loader import load_video

TOLERANCE = 1e-5  # same relative criterion as src/compare_outputs.py


def timed(fn, repeat: int) -> list[float]:
    fn()  # warm-up: page faults, BLAS thread pool, allocator
    runs = []
    for _ in range(repeat):
        start = perf_counter()
        fn()
        runs.append(perf_counter() - start)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default="imageio:cockatoo.mp4", help="path, imageio: resource or http(s) URL")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    # load_video returns a transposed view (memory is NHWC): lay it out as NCHW once, outside
    # the timings, so neither side pays for the layout change.
    video = np.ascontiguousarray(load_video(args.source, args.frames), dtype=np.float32)
    filters = CONV_FILTER.astype(np.float32)
    n, c, h, w = video.shape
    print(f"video {video.shape} float32 ({video.nbytes / 1e6:.0f} MB), filters {filters.shape}")

    ref = conv2d(video, filters, device="numpy")
    out = conv2d(video, filters, device="local")
    assert ref.shape == out.shape, f"shape mismatch: numpy {ref.shape}, rust {out.shape}"
    rel = np.abs(ref - out).max() / np.abs(ref).max()
    print(f"max relative error rust vs numpy: {rel:.2e} ({'PASS' if rel < TOLERANCE else 'FAIL'})\n")

    numpy_s = statistics.median(timed(lambda: conv2d(video, filters, "numpy"), args.repeat))
    rust_s = statistics.median(timed(lambda: conv2d(video, filters, "local"), args.repeat))
    kernels = [_native.cross_correlate2d_timed(video, filters)[1] for _ in range(args.repeat)]
    kernel_s = statistics.median(t["kernel_s"] for t in kernels)

    print(f"{'':28}{'median (s)':>12}{'per frame (ms)':>17}{'vs numpy':>11}")
    for label, s in [
        ("numpy (Accelerate BLAS)", numpy_s),
        ("rust, end to end", rust_s),
        ("rust, kernel only", kernel_s),
    ]:
        print(f"{label:28}{s:12.4f}{1000 * s / n:17.2f}{s / numpy_s:10.2f}x")
    print("\n'end to end' includes the numpy<->Tensor4D copies; 'kernel only' is cross_correlate2d alone.")


if __name__ == "__main__":
    main()
