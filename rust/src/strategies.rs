//! Three ways to run the same gray-scale convolution with `frugal_ml`.
//!
//! All three compute the same thing, a bank of `K = 2` 3x3 filters (Sobel-x and
//! Gaussian in the demo) over one-channel frames, and return (N, H-2, W-2, K).
//! What differs is where the memory lives and how much of it there is:
//!
//! - **stack**: `Tensor4D` on the stack, the whole frame and the whole output
//!   at once. This is exactly what runs on the microcontroller. Memory grows
//!   with H*W, and a 128 KB chip runs out at a handful of tens of pixels.
//! - **heap**: the same code with `HeapStorage`. It runs at any resolution the
//!   host has RAM for, but a chip without an allocator cannot.
//! - **stream**: `ConvStreaming`, one row at a time. RAM is `O(KH * W)`, it
//!   does not depend on the frame height at all.
//!
//! The direct variants are const-generic over (H, W), so they exist only for
//! the resolutions listed in `register!` at the bottom. Streaming is generic
//! over W alone: any height works.

use crate::shapes::Timings;
use frugal_ml::linalg::tensor::Tensor4DBuffer;
use frugal_ml::linalg::{HeapStorage, OwnedStorage, StackStorage, Storage, Tensor4D};
use frugal_ml::sp::{cross_correlate2d, ConvStreaming};
use std::mem::size_of;
use std::time::Instant;

pub const K: usize = 2;
const KH: usize = 3;
const KW: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Strategy {
    Stack,
    Heap,
    Stream,
}

impl Strategy {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "stack" => Some(Self::Stack),
            "heap" => Some(Self::Heap),
            "stream" => Some(Self::Stream),
            _ => None,
        }
    }
}

/// What one call did: time split, the bytes the algorithm itself needs (exact,
/// from `size_of`, harness copies not included) and, for the stack variant,
/// the size of the thread it ran in.
pub struct Report {
    pub timings: Timings,
    pub working_bytes: usize,
    pub thread_stack_bytes: usize,
}

/// Bytes of the three tensors of a direct run: frame, filter bank, output.
pub const fn direct_bytes(h: usize, w: usize) -> usize {
    4 * (h * w + K * KH * KW + (h - KH + 1) * (w - KW + 1) * K)
}

fn direct_frames<
    const H: usize,
    const W: usize,
    const H_OUT: usize,
    const W_OUT: usize,
    SX,
    SF,
    SY,
>(
    x: &[f32],
    bank: &[f32],
    n: usize,
    out: &mut [f32],
    load_frame: impl Fn(&[f32]) -> Tensor4D<1, 1, H, W, SX>,
    load_bank: impl FnOnce(&[f32]) -> Tensor4D<K, 1, KH, KW, SF>,
) -> Timings
where
    SX: Storage<Tensor4DBuffer<1, 1, H, W>>,
    SF: Storage<Tensor4DBuffer<K, 1, KH, KW>>,
    SY: OwnedStorage<Tensor4DBuffer<1, H_OUT, W_OUT, K>>,
{
    let mut t = Timings::default();
    let start = Instant::now();
    let bank = load_bank(bank);
    t.copy_in += start.elapsed().as_secs_f64();

    for i in 0..n {
        let start = Instant::now();
        let frame = load_frame(&x[i * H * W..(i + 1) * H * W]);
        t.copy_in += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let y: Tensor4D<1, H_OUT, W_OUT, K, SY> = cross_correlate2d(&frame, &bank, 1);
        t.kernel += start.elapsed().as_secs_f64();

        let start = Instant::now();
        out[i * H_OUT * W_OUT * K..(i + 1) * H_OUT * W_OUT * K].copy_from_slice(y.get_data());
        t.copy_out += start.elapsed().as_secs_f64();
    }
    t
}

fn stack_frame<const H: usize, const W: usize>(s: &[f32]) -> Tensor4D<1, 1, H, W> {
    let mut t = Tensor4D::<1, 1, H, W>::new([[[[0.0; W]; H]; 1]; 1]);
    t.load_slice(s).ok().expect("frame length validated by the caller");
    t
}

fn stack_bank(s: &[f32]) -> Tensor4D<K, 1, KH, KW> {
    let mut t = Tensor4D::<K, 1, KH, KW>::new([[[[0.0; KW]; KH]; 1]; K]);
    t.load_slice(s).ok().expect("bank length validated by the caller");
    t
}

fn heap_frame<const H: usize, const W: usize>(s: &[f32]) -> Tensor4D<1, 1, H, W, HeapStorage<Tensor4DBuffer<1, 1, H, W>>> {
    Tensor4D::from_vec(s.to_vec()).ok().expect("frame length validated by the caller")
}

fn heap_bank(s: &[f32]) -> Tensor4D<K, 1, KH, KW, HeapStorage<Tensor4DBuffer<K, 1, KH, KW>>> {
    Tensor4D::from_vec(s.to_vec()).ok().expect("bank length validated by the caller")
}

fn stream_frames<const W: usize, const W_OUT: usize>(
    x: &[f32],
    h: usize,
    bank: &[f32],
    n: usize,
    out: &mut [f32],
) -> (Timings, usize) {
    let kernels: [[[f32; KW]; KH]; K] =
        core::array::from_fn(|k| core::array::from_fn(|i| core::array::from_fn(|j| bank[k * KH * KW + i * KW + j])));
    let h_out = h - KH + 1;
    let start = Instant::now();
    for f in 0..n {
        // One ring buffer per filter, fed the same rows.
        let mut rings = [ConvStreaming::<W, KH, KW>::new(), ConvStreaming::<W, KH, KW>::new()];
        let frame = &x[f * h * W..(f + 1) * h * W];
        let dst = &mut out[f * h_out * W_OUT * K..(f + 1) * h_out * W_OUT * K];
        for r in 0..h {
            let row: [f32; W] = frame[r * W..(r + 1) * W].try_into().expect("row width is W");
            for ring in rings.iter_mut() {
                ring.push_row(row);
            }
            if rings[0].ready_to_compute() {
                let out_row = r + 1 - KH;
                for (k, ring) in rings.iter().enumerate() {
                    let o: [f32; W_OUT] = ring.conv2d(&kernels[k]);
                    for c in 0..W_OUT {
                        dst[(out_row * W_OUT + c) * K + k] = o[c];
                    }
                }
            }
        }
    }
    let t = Timings { kernel: start.elapsed().as_secs_f64(), ..Timings::default() };
    // Ring buffers, one input row, one output row per filter.
    (t, size_of::<[ConvStreaming<W, KH, KW>; K]>() + 4 * W + K * 4 * W_OUT)
}

/// The stack variant runs in a thread of its own, sized for the frame: the
/// caller's thread (Python's) has a fixed 8 MiB, which a large budget can exceed.
fn on_big_stack<T: Send>(bytes: usize, f: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|s| {
        std::thread::Builder::new()
            .stack_size(3 * bytes + (1 << 20))
            .spawn_scoped(s, f)
            .expect("spawning the stack thread")
            .join()
            .expect("the stack thread panicked")
    })
}

macro_rules! register {
    (
        stack: [$(($sh:literal, $sw:literal)),+ $(,)?],
        heap: [$(($hh:literal, $hw:literal)),+ $(,)?],
        stream: [$($tw:literal),+ $(,)?] $(,)?
    ) => {
        /// (H, W) frames the stack variant is compiled for.
        pub const STACK: &[(usize, usize)] = &[$(($sh, $sw)),+];
        /// (H, W) frames the heap variant is compiled for.
        pub const HEAP: &[(usize, usize)] = &[$(($hh, $hw)),+];
        /// Widths the streaming variant is compiled for (any height).
        pub const STREAM: &[usize] = &[$($tw),+];

        /// Runs `n` (h x w) frames. `None` if this build has no instantiation.
        pub fn run(strategy: Strategy, h: usize, w: usize, x: &[f32], bank: &[f32], n: usize, out: &mut [f32]) -> Option<Report> {
            match strategy {
                Strategy::Stack => match (h, w) {
                    $(
                        ($sh, $sw) => {
                            let bytes = direct_bytes($sh, $sw);
                            let thread_stack_bytes = 3 * bytes + (1 << 20);
                            let timings = on_big_stack(bytes, || {
                                direct_frames::<
                                    $sh, $sw, { $sh - KH + 1 }, { $sw - KW + 1}, _, _,
                                    StackStorage<Tensor4DBuffer<1, { $sh - KH + 1 }, { $sw - KW + 1 }, K>>,
                                >(x, bank, n, out, stack_frame::<$sh, $sw>, stack_bank)
                            });
                            Some(Report { timings, working_bytes: bytes, thread_stack_bytes })
                        }
                    )+
                    _ => None,
                },
                Strategy::Heap => match (h, w) {
                    $(
                        ($hh, $hw) => {
                            let timings = direct_frames::<
                                $hh, $hw, { $hh - KH + 1 }, { $hw - KW + 1 }, _, _,
                                HeapStorage<Tensor4DBuffer<1, { $hh - KH + 1 }, { $hw - KW + 1 }, K>>,
                            >(x, bank, n, out, heap_frame::<$hh, $hw>, heap_bank);
                            Some(Report { timings, working_bytes: direct_bytes($hh, $hw), thread_stack_bytes: 0 })
                        }
                    )+
                    _ => None,
                },
                Strategy::Stream => match w {
                    $(
                        $tw => {
                            let (timings, working_bytes) = stream_frames::<$tw, { $tw - KW + 1 }>(x, h, bank, n, out);
                            Some(Report { timings, working_bytes, thread_stack_bytes: 0 })
                        }
                    )+
                    _ => None,
                },
            }
        }
    };
}

// 16:9 frames, from a sensor that fits a microcontroller to full HD.
register!(
    // The stack variant stops at 480p: past that a frame no longer fits any thread.
    stack: [(36, 64), (54, 96), (72, 128), (90, 160), (108, 192), (144, 256), (180, 320), (240, 426), (360, 640), (480, 854)],
    heap: [(36, 64), (54, 96), (72, 128), (90, 160), (108, 192), (144, 256), (180, 320), (240, 426), (360, 640), (480, 854), (720, 1280), (1080, 1920)],
    stream: [64, 96, 128, 160, 192, 256, 320, 426, 640, 854, 1280, 1920],
);
