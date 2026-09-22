"""Where numpy's tensordot beats the frugal_ml kernel, and where it does not.

    uv run python benchmarks/sweep_tradeoffs.py

Sweeps the number of input channels C and of filters K on 128x128 frames, batches of
4, and writes the medians to `src/directConv/demo/static/tradeoffs.json`, which the demo
page shows. The demo's own workload (C=1, K=2) is the first row: it is the favourable
end of the sweep, and the page says so.
"""

import json
import platform
import statistics
import subprocess
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np

from directConv import _native
from directConv.convolver import compute

OUT = Path(__file__).parent.parent / "src/directConv/demo/static/tradeoffs.json"
HW, N, REPEAT = 128, 4, 9
CASES = [(1, 2), (3, 2), (8, 2), (16, 2), (3, 1), (3, 4), (3, 8), (3, 16)]  # (C, K)


def median_s(fn) -> float:
    fn()
    runs = []
    for _ in range(REPEAT):
        start = perf_counter()
        fn()
        runs.append(perf_counter() - start)
    return statistics.median(runs)


def cpu() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return platform.processor() or platform.machine()


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []
    print(f"{'C':>3}{'K':>4} | {'numpy ms':>9}{'rust ms':>9}{'numpy / rust':>14}   (per frame, {HW}x{HW})")
    for c, k in CASES:
        x = rng.random((N, c, HW, HW), dtype=np.float32)
        f = rng.random((k, c, 3, 3), dtype=np.float32)
        numpy_ms = 1000 * median_s(lambda: compute(x, conv_filter=f)) / N
        rust_ms = 1000 * median_s(lambda: _native.cross_correlate2d(x, f)) / N
        rows.append({"c": c, "k": k, "numpy_ms": numpy_ms, "rust_ms": rust_ms, "ratio": numpy_ms / rust_ms})
        print(f"{c:>3}{k:>4} | {numpy_ms:9.3f}{rust_ms:9.3f}{numpy_ms / rust_ms:13.1f}x")
    OUT.write_text(json.dumps({"cpu": cpu(), "date": date.today().isoformat(), "hw": HW, "batch": N, "rows": rows}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
