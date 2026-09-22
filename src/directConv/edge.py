"""What a target device can hold and how fast it would go.

Two kinds of numbers live here and the page keeps them apart:

- **Exact**: bytes. Tensor sizes are known at compile time, so `direct_bytes` and
  `stream_bytes` are formulas that the tests check against `size_of` in Rust.
- **Projected**: the time a frame takes on the STM32F446RE. It is *not* run
  here; it is the cycle counts that `ferrite-embedded` measured on the real chip
  (DWT cycle counter, read back over ST-Link) applied to the shapes at hand.
  Once the board is reachable over SSH these projections get replaced by
  live measurements.

Sources, all in `~/Dev/projects/ferrite-embedded/` (and `frugal_ml/docs/BENCHMARKS.md`):
  scripts/csv/regime.csv            direct full-frame convolution, gray, K=1
  scripts/logs/scaling_bench-*.log  streaming convolution, gray, K=1
  scripts/logs/regime_bench-final.log  the stack margin at the entry of each regime

What the chip's stack allows, and how sure we are:

- The stack is 126.4 kB (`_stack_start - _stack_end`, BENCHMARKS.md).
- The compiler reserves the tensors plus a small constant per regime: from the SP margin at the
  entry of each executed regime, 376 to 1 732 B on top of 27 to 72 kB of tensors (x1.006 to x1.038).
- So tensors up to 126.4 kB / 1.04 = 121.5 kB are *predicted* to fit.
- The largest convolution that actually ran on the chip is 72 244 B (gray 96x96, K=1). The
  75 000 B `RAM_SAFETY_BUDGET` of `regime_bench.rs` is a safety value fixed before the
  `black_box(out)` bug was found (it inflated the stack to 2x); after the fix, the 78 032 B regime
  is SKIPped, never run. Nothing between 72 kB and 121 kB has been run since.
"""

from dataclasses import dataclass

KH = KW = 3
K = 2  # filters: Sobel-x and Gaussian
F32 = 4

# 16:9 frames, sensor-sized up to full HD. The Rust side is compiled for these.
LADDER: list[tuple[str, int, int]] = [
    ("64×36", 36, 64),
    ("96×54", 54, 96),
    ("128×72", 72, 128),
    ("160×90", 90, 160),
    ("192×108", 108, 192),
    ("256×144", 144, 256),
    ("320×180", 180, 320),
    ("426×240", 240, 426),
    ("640×360", 360, 640),
    ("854×480", 480, 854),
    ("1280×720", 720, 1280),
    ("1920×1080", 1080, 1920),
]


def direct_bytes(h: int, w: int) -> int:
    """Frame + filter bank + output of a full-frame run (stack or heap)."""
    return F32 * (h * w + K * KH * KW + (h - KH + 1) * (w - KW + 1) * K)


def stream_bytes(w: int, word: int = 4) -> int:
    """K ring buffers of KH rows (plus their two index fields, `word` bytes each: 4 on a
    Cortex-M, 8 on a 64-bit host), one input row, K output rows. Independent of H."""
    return F32 * (K * KH * w + w + K * (w - KW + 1)) + K * 2 * word


@dataclass(frozen=True)
class Device:
    key: str
    name: str
    budget: int  # bytes of tensors the device can hold
    note: str  # where the budget comes from, shown as is
    clock_hz: float | None = None  # None: the device is this machine, times are measured
    ram: int | None = None
    verified: int | None = None  # largest tensor bytes actually run on the device, if `budget` is a prediction


STACK_BYTES = 126_400  # linker symbols, BENCHMARKS.md
STACK_OVERHEAD = 1.04  # worst measured ratio of reserved stack to tensor bytes
STM32F446RE = Device(
    key="stm32",
    name="STM32F446RE (Cortex-M4F, 168 MHz, 128 KB RAM)",
    budget=int(STACK_BYTES / STACK_OVERHEAD),
    note="The stack is 126.4 kB. The compiler reserves the tensors plus 0.4 to 1.7 kB (measured: x1.006 to x1.038), "
    "so tensors up to 121.5 kB are predicted to fit. The largest convolution actually run on the chip is 72.2 kB: "
    "beyond that, a fit is a prediction, not yet confirmed on the board.",
    clock_hz=168e6,
    ram=128 * 1024,
    verified=72_244,
)
HOST = Device(
    key="host",
    name="This machine (8 MiB thread stack)",
    budget=8 * 1024 * 1024,
    note="The default stack of a thread. The same rule as on the chip (tensor bytes must fit) applied to "
    "8 MiB; the multiplier measured on the chip (1.0 to 1.04) is assumed to hold here, it was not measured.",
)
DEVICES = {d.key: d for d in (STM32F446RE, HOST)}


# --- measured on the STM32F446RE ------------------------------------------------------------
# streaming, K=1, gray: (width, cycles per row). scaling_bench-20260820-123703.log
_STREAM_CYCLES_PER_ROW = [
    (80, 6369), (160, 13088), (240, 18689), (320, 26819), (400, 34599),
    (480, 41001), (640, 54521), (800, 64543), (960, 80238), (1280, 97943),
]  # fmt: skip
# direct, K=1, gray: (MACs, cycles per MAC). csv/regime.csv, grid_c1_*
_DIRECT_CYCLES_PER_MAC = [
    (1764, 9.104), (4356, 8.947), (8100, 9.019), (12996, 8.985), (19044, 8.589), (26244, 7.977),
    (34596, 7.952), (44100, 7.911), (54756, 7.677), (66564, 7.808), (79524, 7.824),
]  # fmt: skip


def _interp(points: list[tuple[float, float]], x: float) -> tuple[float, bool]:
    """Piecewise-linear y(x), extended along the end segments. Returns (y, extrapolated)."""
    if x <= points[0][0]:
        (x0, y0), (x1, y1) = points[0], points[1]
    elif x >= points[-1][0]:
        (x0, y0), (x1, y1) = points[-2], points[-1]
    else:
        x0, y0 = max((p for p in points if p[0] <= x), key=lambda p: p[0])
        x1, y1 = min((p for p in points if p[0] >= x), key=lambda p: p[0])
        if x0 == x1:
            return y0, False
    y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return max(y, 0.0), not (points[0][0] <= x <= points[-1][0])


def chip_seconds_per_frame(strategy: str, h: int, w: int) -> tuple[float, bool]:
    """Projected time of one h x w frame on the STM32F446RE, and whether it extrapolates
    beyond the measured range. `strategy` is "direct" (stack or heap) or "stream"."""
    hz = STM32F446RE.clock_hz
    if strategy == "stream":
        per_row, extrapolated = _interp([(float(a), float(b)) for a, b in _STREAM_CYCLES_PER_ROW], float(w))
        return K * h * per_row / hz, extrapolated
    macs = (h - KH + 1) * (w - KW + 1) * KH * KW
    cycles_per_mac, extrapolated = _interp(_DIRECT_CYCLES_PER_MAC, float(macs))
    return K * macs * cycles_per_mac / hz, extrapolated


def fits(bytes_needed: int, device: Device) -> bool:
    return bytes_needed <= device.budget


def largest_fitting(device: Device, max_h: int | None = None) -> tuple[str, int, int] | None:
    """The biggest 16:9 frame whose full-frame run fits `device`, or None."""
    ok = [(n, h, w) for n, h, w in LADDER if fits(direct_bytes(h, w), device) and (max_h is None or h <= max_h)]
    return ok[-1] if ok else None


def drop_schedule(n_frames: int, camera_fps: float, seconds_per_frame: float) -> list[int]:
    """Indices of the frames a device actually processes.

    The camera delivers frame i at t = i / camera_fps. A device that is busy when a
    frame arrives lets it go (nothing buffers a full frame on a chip without RAM);
    the next frame it handles is the first one to arrive after it finishes.
    """
    processed, free_at = [], 0.0
    for i in range(n_frames):
        if i / camera_fps >= free_at - 1e-12:
            processed.append(i)
            free_at = i / camera_fps + seconds_per_frame
    return processed
