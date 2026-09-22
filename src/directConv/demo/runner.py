"""The demo's two jobs.

`run` measures the same gray-scale convolution (Sobel-x and Gaussian, 3x3) on numpy and on the
`frugal_ml` strategies (stack, heap, streaming) over a long clip, batch by batch, and checks that
every output equals numpy's. `play` then encodes what one strategy gives on one device, dropping
the frames that device could not have processed in time.

What is measured live is measured on this machine. What is said about the STM32F446RE is either
exact (bytes) or projected from cycle counts measured on the chip earlier (see `directConv.edge`);
the two are never mixed in one figure.

Long clips: the source is played forward then backward as often as needed (no jump cut), frames are
stored once as 8-bit luminance, and everything is processed in batches, so neither memory nor the
number of frames limits the length. Timing is taken per batch, which gives a distribution
(median and 5th to 95th percentile) instead of a single number.
"""

import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import uuid
from time import perf_counter

import imageio.v2 as iio2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from directConv import _native, edge
from directConv.convolver import compute as numpy_compute
from directConv.filters import GAUSSIAN_FILTER, SOBEL_FILTER
from directConv.loader import _fit, iter_frames, resolve_source, video_fps

PRESETS = {
    "Cockatoo (720p, bundled with imageio)": "imageio:cockatoo.mp4",
    "Big Buck Bunny (small sample, downloaded)": "https://www.w3schools.com/html/mov_bbb.mp4",
}
MAX_FRAMES = 3600
DEFAULT_FRAMES = 600
LENGTH_STEPS = [60, 120, 300, 600, 1200, 1800, 3600]
MAX_CAMERA_FPS = 240
# numpy is by far the slowest side (about 12 ns per pixel): the frame count is bounded so that its
# pass stays around this long, whatever the frame size.
NUMPY_BUDGET_S = 8.0
NUMPY_S_PER_PIXEL = 12e-9
NUMPY_BATCH_BYTES = 300_000_000  # windows numpy may build at once
EQUALITY_FRAMES = 8
PLAY_BATCH = 32

# (2, 1, 3, 3): Sobel-x, then Gaussian, one channel each.
BANK = np.stack([SOBEL_FILTER[0], GAUSSIAN_FILTER[0]])[:, None].astype(np.float32)
NATIVE = {"stack": "stack", "reduced": "stack", "heap": "heap", "stream": "stream"}
LADDER = {name: (h, w) for name, h, w in edge.LADDER}


def _noop(step: str, text: str = "", pct: float | None = None) -> None:
    return None


def patch_bytes(h: int, w: int, n_frames: int = 1) -> int:
    """Size of the im2col buffer numpy builds inside tensordot (one channel), in bytes."""
    return n_frames * (h - 2) * (w - 2) * 9 * 4


def max_camera_fps(h: int, w: int) -> int:
    """Highest frame rate H.264 level 5.1 allows for h x w frames (983 040 macroblocks per second),
    which is what browsers decode in hardware; past it a video would simply not play."""
    macroblocks = ((w + 15) // 16) * ((h + 15) // 16)
    return max(1, min(MAX_CAMERA_FPS, 983_040 // macroblocks))


def max_frames(resolution: str) -> int:
    h, w = LADDER[resolution]
    return int(max(30, min(MAX_FRAMES, NUMPY_BUDGET_S / (h * w * NUMPY_S_PER_PIXEL))))


def validate_run(n_frames: int, resolution: str) -> None:
    if resolution not in LADDER:
        raise ValueError(f"resolution must be one of {list(LADDER)}")
    if not 1 <= n_frames <= max_frames(resolution):
        raise ValueError(f"frames must be between 1 and {max_frames(resolution)} at {resolution}")


def sequence(m: int, n: int) -> np.ndarray:
    """Indices of n frames drawn from m source frames, played forward then backward (no jump cut)."""
    if m <= 1:
        return np.zeros(n, dtype=np.int64)
    period = 2 * (m - 1)
    i = np.arange(n) % period
    return np.where(i < m, i, period - i)


# --- what each strategy is, on each device -----------------------------------------------------


def _size(h: int, w: int) -> dict:
    return {"name": f"{w}×{h}", "h": h, "w": w}


def cards_for(resolution: str, device_key: str, measured: dict | None = None) -> dict:
    """The four ways to run `resolution` on a device, described by exact bytes, whether they fit,
    what was measured on this machine (if `measured`) and what the chip would do."""
    device = edge.DEVICES[device_key]
    h, w = LADDER[resolution]
    measured = measured or {}
    on_chip = device.clock_hz is not None
    full_bytes = edge.direct_bytes(h, w)
    full_fits = edge.fits(full_bytes, device)

    def card(sid: str, size_hw: tuple[int, int] | None, nbytes: int, fits: bool, reason: str | None, kind: str) -> dict:
        out = {"id": sid, "bytes": nbytes, "budget": device.budget, "fits": fits, "reason": reason, "unneeded": False}
        if size_hw is None:
            return out | {"size": None, "playable": False, "host_ms": None, "chip_ms": None}
        sh, sw = size_hw
        compiled = size_hw in {(a, b) for a, b in _native.strategy_shapes()["stack"]} if kind == "stack" else True
        host_ms = measured.get(f"{NATIVE[sid]}@{sh}x{sw}", {}).get("ms_per_frame")
        chip_ms = None
        if on_chip and fits:
            chip_ms = 1000 * edge.chip_seconds_per_frame("stream" if sid == "stream" else "direct", sh, sw)[0]
        return out | {
            "size": _size(sh, sw),
            "pixels_kept": (sh * sw) / (h * w),
            # A fit beyond the largest run seen on the chip is a prediction.
            "predicted": bool(fits and device.verified and nbytes > device.verified),
            "playable": fits and compiled,
            "host_ms": host_ms,
            "chip_ms": chip_ms,
        }

    stack_reason = None if full_fits else f"needs {full_bytes:,} B of tensors, the budget is {device.budget:,} B"
    if on_chip:
        heap_reason = (
            "the bare-metal build has no allocator: heap storage does not exist on this chip, "
            f"whatever the size ({full_bytes:,} B here, {device.ram:,} B of RAM in total)"
        )
        heap_fits = False
    else:
        heap_reason, heap_fits = None, True
    stream_bytes = edge.stream_bytes(w)
    stream_fits = edge.fits(stream_bytes, device)

    cards = {
        "stack": card("stack", (h, w) if _has(h, w) else None, full_bytes, full_fits, stack_reason, "stack"),
        "heap": card("heap", (h, w), full_bytes, heap_fits, heap_reason, "heap"),
        "stream": card(
            "stream", (h, w), stream_bytes, stream_fits,
            None if stream_fits else f"needs {stream_bytes:,} B, the budget is {device.budget:,} B", "stream",
        ),
    }  # fmt: skip
    if full_fits:
        cards["reduced"] = {**card("reduced", None, 0, True, None, "stack"), "unneeded": True}
    else:
        best = edge.largest_fitting(device)
        cards["reduced"] = (
            card("reduced", (best[1], best[2]), edge.direct_bytes(best[1], best[2]), True, None, "stack")
            if best
            else card("reduced", None, 0, False, "no frame of the ladder fits this budget", "stack")
        )
    if not _has(h, w):  # the stack variant is not compiled for this size, and would not fit a thread
        cards["stack"] |= {"playable": False, "size": _size(h, w)}
    return cards


def _has(h: int, w: int) -> bool:
    return (h, w) in {(a, b) for a, b in _native.strategy_shapes()["stack"]}


# --- run ----------------------------------------------------------------------------------------


def _luminance_u8(frame: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (H, W) uint8, the 0.299 R + 0.587 G + 0.114 B `frugal_ml` uses."""
    lum = frame[..., 0] * np.float32(0.299) + frame[..., 1] * np.float32(0.587) + frame[..., 2] * np.float32(0.114)
    return np.rint(lum).astype(np.uint8)


def _batch(store: np.ndarray, seq: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Frames lo..hi of the sequence as a contiguous (k, 1, H, W) float32 array."""
    return np.ascontiguousarray(store[seq[lo:hi]][:, None], dtype=np.float32)


def _stats(per_frame_s: list[float], frames: int) -> dict:
    ms = 1000 * np.asarray(per_frame_s)
    return {
        "ms_per_frame": float(np.median(ms)),
        "p5_ms": float(np.percentile(ms, 5)),
        "p95_ms": float(np.percentile(ms, 95)),
        "frames": frames,
        "batches": len(per_frame_s),
    }


def _rust_batch(n: int) -> int:
    return max(1, min(32, n // 24))


def _numpy_batch(n: int, h: int, w: int) -> int:
    return max(1, min(_rust_batch(n), NUMPY_BATCH_BYTES // patch_bytes(h, w)))


def _numpy_breakdown(gray: np.ndarray, total_ms: float) -> dict:
    """Where numpy's time goes: `sliding_window_view` is a zero-copy view, but `tensordot` must
    flatten it into a matrix for BLAS, and flattening a strided view copies."""
    n = gray.shape[0]
    view = sliding_window_view(gray, (3, 3), axis=(2, 3)).transpose(0, 2, 3, 1, 4, 5)
    cols = view.reshape(-1, 9)
    matrix = BANK.reshape(2, 9).T.copy()

    def timed(fn):
        fn()
        runs = []
        for _ in range(5):
            start = perf_counter()
            fn()
            runs.append(perf_counter() - start)
        return statistics.median(runs)

    copy_ms = 1000 * timed(lambda: view.reshape(-1, 9)) / n
    gemm_ms = 1000 * timed(lambda: cols @ matrix) / n
    return {
        "total_ms": total_ms,
        "copy_ms": copy_ms,
        "gemm_ms": gemm_ms,
        "copy_share": copy_ms / total_ms,
        "copy_mb_per_frame": cols.nbytes / n / 1e6,
        "view_is_zero_copy": bool(np.shares_memory(view, gray)),
        "flattened_is_a_copy": not np.shares_memory(cols, gray),
    }


def _numpy_memory(batch: np.ndarray, scratch: str) -> dict:
    """Peak extra memory of one numpy call on a batch, in a fresh process (see memprobe)."""
    video_path, filters_path = os.path.join(scratch, "video.npy"), os.path.join(scratch, "filters.npy")
    np.save(video_path, batch)
    np.save(filters_path, BANK)
    out = subprocess.run(
        [sys.executable, "-m", "directConv.demo.memprobe", "numpy", video_path, filters_path],
        capture_output=True,
        text=True,
        check=True,
    )
    probe = json.loads(out.stdout.strip().splitlines()[-1])
    working = probe["peak_extra_mb"] - probe["out_mb"]
    return {"working_mb": working, "per_frame_mb": working / batch.shape[0], "batch_frames": batch.shape[0]}


class Context:
    """What `play` needs from the last `run`: the luminance frames at each size it measured."""

    def __init__(self, run_id, resolution, fps, store, seq):
        self.run_id, self.resolution, self.fps, self.store, self.seq = run_id, resolution, fps, store, seq
        self.measured: dict = {}
        self.result: dict | None = None


_last: Context | None = None
_lock = threading.Lock()


def last_result() -> dict | None:
    with _lock:
        return _last.result if _last else None


def run(source: str, n_frames: int, resolution: str, out_dir: str, progress=_noop) -> dict:
    global _last
    validate_run(n_frames, resolution)
    h, w = LADDER[resolution]
    n = n_frames

    # Sizes to measure: the target, and for each device the biggest frame that fits its stack.
    sizes = {(h, w)}
    for dev in edge.DEVICES.values():
        if not edge.fits(edge.direct_bytes(h, w), dev) and (best := edge.largest_fitting(dev)):
            sizes.add((best[1], best[2]))
    sizes = sorted(sizes)

    progress("decode", "Decoding the clip", 0.0)
    start = perf_counter()
    source = resolve_source(source)
    fps = video_fps(source)
    planes: dict[tuple[int, int], list[np.ndarray]] = {size: [] for size in sizes}
    for rgb in iter_frames(source, n):  # at most n source frames; the sequence loops beyond them
        for size in sizes:
            planes[size].append(_luminance_u8(_fit(rgb, size)))
    store = {size: np.stack(frames) for size, frames in planes.items()}
    del planes
    m = len(store[sizes[0]])
    seq = sequence(m, n)
    load_s = perf_counter() - start

    # --- numpy, batch by batch
    batch = _numpy_batch(n, h, w)
    numpy_compute(_batch(store[(h, w)], seq, 0, batch), conv_filter=BANK)  # warm-up
    samples = []
    for lo in range(0, n, batch):
        hi = min(n, lo + batch)
        x = _batch(store[(h, w)], seq, lo, hi)
        t = perf_counter()
        numpy_compute(x, conv_filter=BANK)
        samples.append((perf_counter() - t) / (hi - lo))
        progress("numpy", f"numpy at {w}×{h}: {hi} of {n} frames", hi / n)
    numpy_stats = _stats(samples, n)
    breakdown = _numpy_breakdown(_batch(store[(h, w)], seq, 0, min(n, 4)), numpy_stats["ms_per_frame"])

    # --- frugal_ml strategies, batch by batch
    todo = [("stream", h, w), ("heap", h, w)]
    if edge.fits(edge.direct_bytes(h, w), edge.HOST) and _has(h, w):
        todo.append(("stack", h, w))
    for size in [s for s in sizes if s != (h, w)] + ([(h, w)] if edge.fits(edge.direct_bytes(h, w), edge.STM32F446RE) else []):
        todo += [("stack", *size), ("heap", *size)]
    todo = list(dict.fromkeys(todo))

    measured, worst = {}, 0.0
    rust_batch = _rust_batch(n)
    for i, (strategy, sh, sw) in enumerate(todo):
        label = f"frugal_ml, {strategy} strategy, {sw}×{sh}"
        first = _batch(store[(sh, sw)], seq, 0, min(n, rust_batch))
        _native.run_gray(strategy, first, BANK)  # warm-up
        samples, info = [], None
        for lo in range(0, n, rust_batch):
            hi = min(n, lo + rust_batch)
            _, info = _native.run_gray(strategy, _batch(store[(sh, sw)], seq, lo, hi), BANK)
            samples.append(info["kernel_s"] / (hi - lo))
        progress("rust", f"{label}", (i + 1) / len(todo))
        # equality with numpy on the first and the last frames of the sequence
        diff = 0.0
        for lo, hi in ((0, min(n, EQUALITY_FRAMES)), (max(0, n - EQUALITY_FRAMES), n)):
            x = _batch(store[(sh, sw)], seq, lo, hi)
            diff = max(diff, float(np.abs(_native.run_gray(strategy, x, BANK)[0] - numpy_compute(x, conv_filter=BANK)).max()))
        worst = max(worst, diff)
        measured[f"{strategy}@{sh}x{sw}"] = _stats(samples, n) | {"working_bytes": info["working_bytes"], "max_abs_diff": diff}

    progress("memory", "Measuring numpy's memory in a fresh process", None)
    with tempfile.TemporaryDirectory(prefix="probe-", dir=out_dir) as scratch:
        numpy_memory = _numpy_memory(_batch(store[(h, w)], seq, 0, batch), scratch)

    cards = {key: cards_for(resolution, key, measured) for key in edge.DEVICES}
    heap_vs_stack = {}
    for key, per_device in cards.items():
        size = per_device["reduced"]["size"] if not per_device["reduced"]["unneeded"] else per_device["stack"]["size"]
        if size and f"stack@{size['h']}x{size['w']}" in measured and f"heap@{size['h']}x{size['w']}" in measured:
            s, hp = measured[f"stack@{size['h']}x{size['w']}"]["ms_per_frame"], measured[f"heap@{size['h']}x{size['w']}"]["ms_per_frame"]
            heap_vs_stack[key] = {"size": size["name"], "stack_ms": s, "heap_ms": hp, "ratio": hp / s}

    run_id = uuid.uuid4().hex
    result = {
        "run_id": run_id,
        "resolution": _size(h, w) | {"key": resolution},
        "frames": n,
        "source_frames": m,
        "looped": n > m,
        "clip_fps": fps,
        "load_s": load_s,
        "numpy": numpy_stats | {"memory": numpy_memory, "breakdown": breakdown, "patch_mb": patch_bytes(h, w, batch) / 1e6, "batch_frames": batch},
        "measured": measured,
        "max_abs_diff": worst,
        "cards": cards,
        "heap_vs_stack": heap_vs_stack,
    }
    with _lock:
        _last = Context(run_id, resolution, fps, store, seq)
        _last.measured, _last.result = measured, result
    return result


# --- play ---------------------------------------------------------------------------------------


def _rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[..., None], 3, axis=-1)


def play(run_id: str, strategy: str, device_key: str, camera_fps: float, out_dir: str, progress=_noop) -> dict:
    with _lock:
        ctx = _last
    if ctx is None or ctx.run_id != run_id:
        raise ValueError("that run is gone, run again")
    if strategy not in NATIVE or device_key not in edge.DEVICES:
        raise ValueError("unknown strategy or device")
    if not 1 <= camera_fps <= MAX_CAMERA_FPS:
        raise ValueError(f"camera fps must be between 1 and {MAX_CAMERA_FPS}")
    card = cards_for(ctx.resolution, device_key, ctx.measured)[strategy]
    if not card["playable"]:
        raise ValueError(card["reason"] or "this strategy is not available here")
    size = card["size"]
    h, w = size["h"], size["w"]
    store, seq = ctx.store[(h, w)], ctx.seq
    n = len(seq)

    requested_fps = camera_fps
    h_out, w_out = h - 2, w - 2
    he, we = h_out - h_out % 2, w_out - w_out % 2
    camera_fps = min(camera_fps, max_camera_fps(he, we))  # what the encoded video can carry
    ms = card["chip_ms"] if edge.DEVICES[device_key].clock_hz else card["host_ms"]
    processed = edge.drop_schedule(n, camera_fps, ms / 1000)
    proc = np.asarray(processed)

    names = ("original", "edges", "blur")
    files = {name: f"{uuid.uuid4().hex}.mp4" for name in names}
    writers = {
        name: iio2.get_writer(
            os.path.join(out_dir, files[name]), fps=camera_fps, codec="libx264", pixelformat="yuv420p",
            macro_block_size=1, output_params=["-preset", "veryfast", "-crf", "20", "-movflags", "+faststart"],
        )
        for name in names
    }  # fmt: skip
    scale = None
    try:
        for c in range(0, len(proc), PLAY_BATCH):
            idx = proc[c : c + PLAY_BATCH]
            out, _ = _native.run_gray(NATIVE[strategy], np.ascontiguousarray(store[seq[idx]][:, None], dtype=np.float32), BANK)
            edges = np.abs(out[..., 0])
            if scale is None:  # one scale for the whole clip: a per-frame scale would flicker
                scale = 255.0 / max(float(np.percentile(edges[:, ::4, ::4], 99.5)), 1e-6)
            edges_rgb = _rgb(edges * scale)[:, :he, :we]
            blur_rgb = _rgb(out[..., 1])[:, :he, :we]
            nxt = int(proc[c + PLAY_BATCH]) if c + PLAY_BATCH < len(proc) else n
            for i in range(int(idx[0]), nxt):
                j = int(np.searchsorted(idx, i, side="right")) - 1
                # Output pixel (i, j) is centred on input pixel (i + 1, j + 1): trim the border.
                writers["original"].append_data(_rgb(store[seq[i], 1 : 1 + h_out, 1 : 1 + w_out])[:he, :we])
                writers["edges"].append_data(edges_rgb[j])
                writers["blur"].append_data(blur_rgb[j])
            progress("encode", f"Encoding {n} frames", min(1.0, (int(nxt)) / n))
    finally:
        for wr in writers.values():
            wr.close()
    return {
        "strategy": strategy,
        "device": device_key,
        "size": size,
        "camera_fps": camera_fps,
        "clamped": camera_fps < requested_fps,
        "frames": n,
        "duration_s": n / camera_fps,
        "processed": len(processed),
        "processed_indices": processed,
        "dropped": n - len(processed),
        "dropped_pct": 100 * (n - len(processed)) / n,
        "ms_per_frame": ms,
        "effective_fps": min(camera_fps, 1000 / ms),
        "videos": {name: f"/media/{file}" for name, file in files.items()},
    }
