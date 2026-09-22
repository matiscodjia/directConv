# directConv

One convolution, four ways to hold it in memory, on a chip that has 128 KB.

[`frugal_ml`](https://github.com/matiscodjia/frugal_ml) is a `no_std` Rust core whose
tensor shapes are fixed at compile time. This project drives it from Python (PyO3) and
answers, with measurements, the question that core exists for: **what do you do when the
frame no longer fits on the stack?**

```
uv sync                 # builds the Rust extension (needs a Rust toolchain)
uv run directconv-demo  # opens http://localhost:8000
uv run pytest           # 101 tests
```

## The demo

A four-page site, no extra dependency (standard-library server, plain HTML/CSS/JS).

| Page | What it is |
|---|---|
| **Home** (`/`) | The idea in one screen: an animated memory wall, and the four ways to fit a frame. |
| **Lab** (`/lab`) | The whole comparison on one screen, no scrolling. Sliders for frame size, camera rate and video length; a chart of the memory each option needs against the device limit (it moves as you drag); four cards with three meters each (**resolution, memory, speed**); and the result: three synchronised videos, a strip of the frames the device kept or dropped, and the numbers. One-click scenarios (*it fits, the wall, shrink it, stream it, on a computer*) run everything. |
| **Why not numpy?** (`/why`) | Where one frame's time goes, with your own last run's numbers, and a table of what it costs you. |
| **Method** (`/method`) | Which numbers are exact, measured or projected, and where each comes from. |

The four options: a full frame **on the stack** (what runs on the microcontroller; memory grows with
the frame area, so it hits a wall), a full frame **on the heap** (any size the machine has RAM for,
but it needs an allocator the chip does not have), a **smaller frame** (stay on the stack, keep the
speed, give up pixels), and **streaming** (`ConvStreaming`, one row at a time: memory depends on the
width only, the price is speed).

**Long videos, steadier statistics.** A run takes up to 3 600 frames (723 at 720p, bounded by how long numpy takes),
played forward then backward so the source clip loops without a cut. Everything is processed batch by batch
(about 24 batches per strategy) and timed per batch, so the lab shows a **median and a 5th to 95th percentile**,
not a single average. Every strategy's output is compared with numpy's and must be identical.

## What the measurements say

Measured / derived facts, on an Apple M4 Pro for the host figures, on the STM32F446RE
(Cortex-M4F, 168 MHz, 128 KB RAM) for the chip figures:

| | |
|---|---|
| Stack on the chip | 126.4 kB. The compiler reserves the tensors plus 0.4 to 1.7 kB (measured), so tensors up to about **121.5 kB** are predicted to fit |
| Largest convolution actually run on the chip | 72.2 kB. Nothing above it has been run since a stack bug was fixed (see below) |
| Largest 16:9 frame that fits the stack | **128×72** (107.5 kB, a prediction), 1.0 % of 1280×720. 160×90 needs 168.9 kB, more than the whole stack |
| 1280×720 full frame | 11.0 MB, 91× the predicted limit |
| 1280×720 streaming | 46 kB, fits, **1.19 fps** on the chip (projected; the chip measured 1.22) |
| Heap vs stack, on the host | no measurable time difference (0.8 to 1.0×) |
| numpy vs `frugal_ml` heap, 720p, this machine | 10.7 ms vs 0.55 ms per frame, about 19× |
| Where numpy's time goes | 86 % is one copy of the 3×3 windows (33 MB per frame) |

**Why it is faster, precisely.** `sliding_window_view` is a zero-copy view in numpy too.
The copy happens next: `np.tensordot` needs a flat 2-D matrix for BLAS, and flattening a
strided view copies it. `tensordot_3` in `frugal_ml` never flattens: it reads the 6-D view
through its strides. So the gain is "never materialise the windows", not faster arithmetic,
and it shrinks as the filter bank grows: at K=16 filters numpy is level (0.9×), because
its one copy is shared by all filters. `benchmarks/sweep_tradeoffs.py` regenerates that table.

## Measured vs projected

- **Measured live**: everything labelled *This machine*.
- **Exact**: all byte counts.
- **Projected**: STM32F446RE times. They apply the cycle counts measured on the chip
  (`ferrite-embedded`, DWT counter over ST-Link) to these shapes, see `src/directConv/edge.py`
  for the sources. Nothing runs on the board yet; once it is reachable over SSH these
  become live measurements.
- **The chip's stack limit is a prediction beyond 72.2 kB.** `regime_bench` keeps a 75 kB
  safety budget from before its `black_box(out)` bug was found (it doubled the stack use);
  the 78 kB regime it once crashed on is now skipped, never run. The wall (128×72) holds for any
  threshold between 107 kB and 126 kB, but confirming it means running that regime on the board.
- The 8 MiB "This machine" budget applies the chip's rule (tensor bytes must fit) to a
  thread stack; the chip's 1.0 to 1.04 stack multiplier is assumed to hold there, not measured.

## Layout

| | |
|---|---|
| `rust/src/strategies.rs` | The three strategies (stack, heap, streaming) and the ladder of compiled sizes. |
| `rust/src/shapes.rs` | The general RGB path (`cross_correlate2d`), used by `benchmarks/`. |
| `src/directConv/edge.py` | Device budgets, exact byte formulas, chip cycle model, frame-dropping schedule. |
| `src/directConv/demo/` | The site: a standard-library server (`server.py`), the measuring job (`runner.py`) and `static/` (four pages, one stylesheet). |
| `benchmarks/` | `compare_rust_numpy.py` (terminal), `sweep_tradeoffs.py` (numpy vs Rust over C and K). |

## Limits

- Shapes are compile-time: the demo ships a fixed ladder of 16:9 sizes. Anything else
  raises an error naming the line to add to `register!` in `rust/src/strategies.rs`. The
  number of frames is not part of a shape; streaming accepts any height.
- Luminance only: `ConvStreaming` is one channel, and RGB would triple its RAM.
- The Rust core runs on one thread with no BLAS.
- f32, stride 1, no padding.
- The Rust crate expects `frugal_ml` at `../../projects/frugal_ml` (see `rust/Cargo.toml`).
