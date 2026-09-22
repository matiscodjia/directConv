//! Runtime shapes -> compile-time shapes.
//!
//! `frugal_ml::sp::cross_correlate2d` fixes every dimension as a const generic,
//! because that is what lets it run without allocation on a microcontroller. A
//! Python caller only knows its shapes at runtime, so the bridge is a table of
//! shapes instantiated ahead of time and a `match` that picks one.
//!
//! To support a new video resolution or filter bank, add one line to the
//! `register_shapes!` call at the bottom of this file and rebuild. `N` (the
//! number of frames) is deliberately *not* part of the key: every frame goes
//! through its own `Tensor4D<1, C, H, W>`, the same per-frame kernel the
//! embedded target runs, so any sequence length works with one instantiation.

use frugal_ml::linalg::Tensor4DBoxed;
use frugal_ml::sp::cross_correlate2d;
use std::time::Instant;

/// Where the wall-clock time of one call went, in seconds. `kernel` is the
/// `cross_correlate2d` call alone; the two copies are the price of crossing
/// the Python boundary (numpy owns its buffer, `Tensor4D` owns its own).
#[derive(Default, Clone, Copy)]
pub struct Timings {
    pub copy_in: f64,
    pub kernel: f64,
    pub copy_out: f64,
}

/// (C, H, W, K, KH, KW)
pub type Shape = [usize; 6];

fn conv_frames<
    const C: usize,
    const H: usize,
    const W: usize,
    const K: usize,
    const KH: usize,
    const KW: usize,
    const H_OUT: usize,
    const W_OUT: usize,
>(
    x: &[f32],
    filters: &[f32],
    n: usize,
    out: &mut [f32],
) -> Timings {
    let frame_in = C * H * W;
    let frame_out = H_OUT * W_OUT * K;
    let mut t = Timings::default();

    let start = Instant::now();
    let bank = Tensor4DBoxed::<K, C, KH, KW>::from_vec(filters.to_vec())
        .ok()
        .expect("filter length validated by the caller");
    t.copy_in += start.elapsed().as_secs_f64();

    for i in 0..n {
        let start = Instant::now();
        let frame = Tensor4DBoxed::<1, C, H, W>::from_vec(x[i * frame_in..(i + 1) * frame_in].to_vec())
            .ok()
            .expect("frame length validated by the caller");
        t.copy_in += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let y: Tensor4DBoxed<1, H_OUT, W_OUT, K> = cross_correlate2d(&frame, &bank, 1);
        t.kernel += start.elapsed().as_secs_f64();

        let start = Instant::now();
        out[i * frame_out..(i + 1) * frame_out].copy_from_slice(y.get_data());
        t.copy_out += start.elapsed().as_secs_f64();
    }
    t
}

macro_rules! register_shapes {
    ($(($c:literal, $h:literal, $w:literal, $k:literal, $kh:literal, $kw:literal)),+ $(,)?) => {
        /// Every shape this build can run.
        pub const REGISTERED: &[Shape] = &[$([$c, $h, $w, $k, $kh, $kw]),+];

        /// Runs `n` frames through the instantiation matching `shape`, or
        /// returns `None` if this build has none.
        pub fn dispatch(
            shape: Shape,
            x: &[f32],
            filters: &[f32],
            n: usize,
            out: &mut [f32],
        ) -> Option<Timings> {
            match shape {
                $(
                    [$c, $h, $w, $k, $kh, $kw] => Some(conv_frames::<
                        $c, $h, $w, $k, $kh, $kw, { $h - $kh + 1 }, { $w - $kw + 1 },
                    >(x, filters, n, out)),
                )+
                _ => None,
            }
        }
    };
}

// (C, H, W, K, KH, KW), stride 1, no padding.
register_shapes!(
    // Small shapes, for tests.
    (1, 8, 8, 1, 3, 3),
    (3, 16, 16, 2, 3, 3),
    // Video: the (2, 3, 3, 3) Sobel + Gaussian bank of `directConv.filters`.
    (3, 360, 640, 2, 3, 3),
    (3, 720, 1280, 2, 3, 3),
    (3, 1080, 1920, 2, 3, 3),
    // Sweeps of `benchmarks/`: baseline C=3, K=2, HW=128, one axis at a time.
    // Channels.
    (1, 128, 128, 2, 3, 3),
    (3, 128, 128, 2, 3, 3),
    (8, 128, 128, 2, 3, 3),
    (16, 128, 128, 2, 3, 3),
    // Filters.
    (3, 128, 128, 1, 3, 3),
    (3, 128, 128, 4, 3, 3),
    (3, 128, 128, 8, 3, 3),
    (3, 128, 128, 16, 3, 3),
    // Resolution.
    (3, 32, 32, 2, 3, 3),
    (3, 64, 64, 2, 3, 3),
    (3, 256, 256, 2, 3, 3),
    (3, 720, 720, 2, 3, 3),
);
