"""Peak extra memory of one convolution, measured in a fresh process.

    python -m directConv.demo.memprobe <device> <video.npy> <filters.npy>

A process's peak RSS never goes down, so two devices cannot be compared inside
one process: the runner starts one of these per device. The input is loaded
first (its size is the baseline), then the convolution runs once, and the report
is how far the peak rose above that baseline: the buffers the convolution itself
allocates, output included.
"""

import json
import os
import resource
import subprocess
import sys

import numpy as np


def _rss_bytes() -> int:
    """Current resident set size of this process."""
    try:
        with open("/proc/self/statm") as f:  # Linux
            return int(f.read().split()[1]) * resource.getpagesize()
    except OSError:  # macOS
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True)
        return int(out.stdout.strip()) * 1024


def _peak_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # bytes on macOS, KiB on Linux


def main() -> None:

    from directConv.backends import conv2d

    device, video_path, filters_path = sys.argv[1:4]
    video, filters = np.load(video_path), np.load(filters_path)
    baseline = _rss_bytes()
    out = conv2d(video, filters, device)
    peak = _peak_bytes()
    print(json.dumps({"baseline_mb": baseline / 1e6, "peak_extra_mb": max(0, peak - baseline) / 1e6, "out_mb": out.nbytes / 1e6}))


if __name__ == "__main__":
    main()
