"""The three frugal_ml strategies, the sizes they claim and the edge model behind the page."""

from dataclasses import replace

import numpy as np
import pytest

from directConv import _native, edge
from directConv.convolver import compute
from directConv.demo import runner

RNG = np.random.default_rng(0)
BANK = runner.BANK
SHAPES = _native.strategy_shapes()


def gray(n, h, w):
    return (RNG.random((n, 1, h, w), dtype=np.float32) * 255).astype(np.float32)


@pytest.mark.parametrize("strategy", ["stack", "heap"])
@pytest.mark.parametrize("h,w", [tuple(s) for s in SHAPES["stack"][:6]], ids=str)
def test_direct_strategies_equal_numpy_exactly(strategy, h, w):
    x = gray(2, h, w)
    out, _ = _native.run_gray(strategy, x, BANK)
    assert np.array_equal(out, compute(x, conv_filter=BANK))


@pytest.mark.parametrize("w", SHAPES["stream"][:8])
@pytest.mark.parametrize("h", [3, 4, 37])
def test_streaming_equals_numpy_exactly_at_any_height(w, h):
    x = gray(2, h, w)
    out, _ = _native.run_gray("stream", x, BANK)
    assert np.array_equal(out, compute(x, conv_filter=BANK))


def test_the_three_strategies_agree_with_each_other():
    x = gray(3, 54, 96)
    outs = [_native.run_gray(s, x, BANK)[0] for s in ("stack", "heap", "stream")]
    assert np.array_equal(outs[0], outs[1]) and np.array_equal(outs[0], outs[2])


def test_byte_formulas_match_rust_size_of():
    for _, h, w in edge.LADDER:
        assert edge.direct_bytes(h, w) == _native.direct_bytes(h, w)
        x = gray(1, h, w) if h * w < 300_000 else None
        if x is not None:
            info = _native.run_gray("stream", x, BANK)[1]
            assert info["working_bytes"] == edge.stream_bytes(w, word=8)  # usize is 8 bytes on the host
    x = gray(1, 54, 96)
    assert _native.run_gray("stack", x, BANK)[1]["working_bytes"] == edge.direct_bytes(54, 96)
    assert _native.run_gray("heap", x, BANK)[1]["working_bytes"] == edge.direct_bytes(54, 96)


def test_streaming_ram_does_not_depend_on_height_but_direct_does():
    small, tall = gray(1, 8, 128), gray(1, 200, 128)
    assert _native.run_gray("stream", small, BANK)[1]["working_bytes"] == _native.run_gray("stream", tall, BANK)[1]["working_bytes"]
    assert edge.direct_bytes(200, 128) > 20 * edge.direct_bytes(8, 128)


def test_uncompiled_sizes_say_what_to_add():
    with pytest.raises(ValueError, match="stack strategy is not compiled for 37x100"):
        _native.run_gray("stack", gray(1, 37, 100), BANK)
    with pytest.raises(ValueError, match="unknown strategy"):
        _native.run_gray("gpu", gray(1, 54, 96), BANK)


def test_every_ladder_frame_is_compiled_for_heap_and_streaming():
    assert {(h, w) for _, h, w in edge.LADDER} <= {tuple(s) for s in SHAPES["heap"]}
    assert {w for _, _, w in edge.LADDER} <= set(SHAPES["stream"])


# --- the edge model ----------------------------------------------------------------------------


def test_stm32_budget_comes_from_the_126_kb_stack_not_from_the_bench_safety_value():
    d = edge.STM32F446RE
    assert d.budget == int(126_400 / 1.04) == 121_538  # BENCHMARKS.md: 126.4 kB of stack; worst measured overhead x1.04
    assert d.verified == 72_244  # the largest convolution actually run on the chip (gray 96x96, K=1)
    assert edge.fits(72_244, d) and edge.fits(78_032, d)  # 78 032 B was SKIPped by a stale 75 kB budget, never run
    assert edge.largest_fitting(d)[0] == "128×72"


def test_the_wall_does_not_depend_on_the_exact_threshold():
    # 128x72 needs 107 496 B and 160x90 needs 168 904 B, more than the whole stack: anything from
    # the first to the second gives the same answer, so the multiplier's exact value does not matter.
    for budget in (107_496, 115_000, 121_538, 126_400):
        assert edge.largest_fitting(replace(edge.STM32F446RE, budget=budget))[0] == "128×72"
    assert edge.direct_bytes(90, 160) > edge.STACK_BYTES


def test_measured_stack_overhead_stays_under_the_assumed_multiplier():
    # regime_bench-final.log: (tensor bytes, SP margin at the entry of the regime); the margin at
    # the start of main was 129 872 B, so the frame reserved by the prologue is the difference.
    start = 129_872
    for tensors, margin in [(27_120, 102_376), (31_796, 96_864), (61_936, 67_560), (72_244, 55_904)]:
        reserved = start - margin
        assert tensors <= reserved <= tensors * edge.STACK_OVERHEAD


def test_streaming_fits_the_chip_at_every_resolution_of_the_ladder():
    assert all(edge.fits(edge.stream_bytes(w), edge.STM32F446RE) for _, _, w in edge.LADDER)


def test_chip_projection_reproduces_the_measured_points():
    # scaling_bench, c2_w1280: 191 381 cycles per row for two filters -> 1.22 fps at 720 rows.
    seconds, extrapolated = edge.chip_seconds_per_frame("stream", 720, 1280)
    assert not extrapolated and 1 / seconds == pytest.approx(1.22, rel=0.05)
    # regime_bench, gray96_k1: 622 164 cycles per iteration at 168 MHz for one filter.
    seconds, _ = edge.chip_seconds_per_frame("direct", 96, 96)
    assert seconds / edge.K == pytest.approx(622_164 / 168e6, rel=0.02)


def test_dropping_frames():
    assert edge.drop_schedule(40, 20, 0.0) == list(range(40))  # instant device: nothing dropped
    assert edge.drop_schedule(40, 20, 0.84) == [0, 17, 34]  # the chip streaming 720p, camera at 20 fps
    assert edge.drop_schedule(10, 10, 0.25) == [0, 3, 6, 9]  # busy for 2.5 frame periods -> takes every 3rd


# --- what the page is told ----------------------------------------------------------------------


def cards(resolution, device):
    return runner.cards_for(resolution, device)


def test_at_720p_on_the_chip_only_reduced_and_streaming_are_possible():
    c = cards("1280×720", "stm32")
    assert not c["stack"]["fits"] and not c["stack"]["playable"]
    assert not c["heap"]["playable"] and "no allocator" in c["heap"]["reason"]
    assert c["reduced"]["playable"] and c["reduced"]["size"]["name"] == "128×72"
    assert c["reduced"]["predicted"]  # 107 kB is beyond the 72 kB run seen on the chip
    assert c["stream"]["playable"] and c["stream"]["bytes"] < edge.STM32F446RE.budget and not c["stream"]["predicted"]
    assert c["reduced"]["pixels_kept"] == pytest.approx(0.01)


def test_at_720p_on_this_machine_heap_and_streaming_work_and_reduced_is_480p():
    c = cards("1280×720", "host")
    assert not c["stack"]["playable"] and c["heap"]["playable"] and c["stream"]["playable"]
    assert c["reduced"]["size"]["name"] == "854×480"


def test_when_the_frame_fits_the_reduced_card_is_not_needed():
    c = cards("96×54", "stm32")
    assert c["stack"]["fits"] and c["stack"]["playable"] and c["reduced"]["unneeded"]
    assert not c["stack"]["predicted"]  # 59.9 kB is under the 72.2 kB that ran on the chip
    assert not c["heap"]["playable"]  # still no allocator on the chip
    assert cards("128×72", "stm32")["stack"]["predicted"]
