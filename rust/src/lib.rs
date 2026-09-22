//! Python bindings for `frugal_ml::sp::cross_correlate2d`, run on the host.
//!
//! The Rust core is built for a microcontroller (`no_std`, shapes fixed at
//! compile time); this crate is the desktop harness that lets numpy and the
//! same core be compared on the same input. See [`shapes`] for how runtime
//! shapes reach the const-generic kernel.

mod shapes;
mod strategies;

use numpy::{PyArray1, PyArray4, PyArrayMethods, PyReadonlyArray4, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use shapes::{Shape, Timings};

const CONTIGUOUS_HINT: &str = "expected a C-contiguous float32 array, use np.ascontiguousarray(a, dtype=np.float32)";

/// Validates shapes, runs the kernel with the GIL released, wraps the result.
fn run<'py>(
    py: Python<'py>,
    video: &PyReadonlyArray4<'py, f32>,
    filters: &PyReadonlyArray4<'py, f32>,
) -> PyResult<(Bound<'py, PyArray4<f32>>, Timings)> {
    let x = video.as_slice().map_err(|_| PyValueError::new_err(CONTIGUOUS_HINT))?;
    let f = filters.as_slice().map_err(|_| PyValueError::new_err(CONTIGUOUS_HINT))?;
    let &[n, c, h, w] = video.shape() else { unreachable!("PyReadonlyArray4 is 4-dimensional") };
    let &[k, fc, kh, kw] = filters.shape() else { unreachable!("PyReadonlyArray4 is 4-dimensional") };

    if n == 0 {
        return Err(PyValueError::new_err("video has no frames"));
    }
    if fc != c {
        return Err(PyValueError::new_err(format!(
            "filters span {fc} channels but the video has {c} (video is N x C x H x W, filters K x C x KH x KW)"
        )));
    }
    if kh > h || kw > w {
        return Err(PyValueError::new_err(format!("{kh}x{kw} window does not fit in {h}x{w} frames")));
    }
    let shape: Shape = [c, h, w, k, kh, kw];
    let (h_out, w_out) = (h - kh + 1, w - kw + 1);

    let mut out = vec![0f32; n * h_out * w_out * k];
    let timings = py
        .detach(|| shapes::dispatch(shape, x, f, n, &mut out))
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "shape (C, H, W, K, KH, KW) = {shape:?} is not compiled into this build, \
                 add `({c}, {h}, {w}, {k}, {kh}, {kw})` to register_shapes! in rust/src/shapes.rs \
                 and rebuild. Available: {:?}",
                shapes::REGISTERED
            ))
        })?;

    let out = PyArray1::from_vec(py, out).reshape([n, h_out, w_out, k])?;
    Ok((out, timings))
}

/// Cross-correlates an (N, C, H, W) float32 video with a (K, C, KH, KW) bank,
/// stride 1, no padding. Returns (N, H - KH + 1, W - KW + 1, K), the layout
/// of `directConv.convolver.compute`.
#[pyfunction]
fn cross_correlate2d<'py>(
    py: Python<'py>,
    video: PyReadonlyArray4<'py, f32>,
    filters: PyReadonlyArray4<'py, f32>,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    Ok(run(py, &video, &filters)?.0)
}

/// Same as `cross_correlate2d`, plus a dict splitting the time spent in Rust
/// into `copy_in_s`, `kernel_s` and `copy_out_s`.
#[pyfunction]
fn cross_correlate2d_timed<'py>(
    py: Python<'py>,
    video: PyReadonlyArray4<'py, f32>,
    filters: PyReadonlyArray4<'py, f32>,
) -> PyResult<(Bound<'py, PyArray4<f32>>, Bound<'py, PyDict>)> {
    let (out, t) = run(py, &video, &filters)?;
    let timings = PyDict::new(py);
    timings.set_item("copy_in_s", t.copy_in)?;
    timings.set_item("kernel_s", t.kernel)?;
    timings.set_item("copy_out_s", t.copy_out)?;
    Ok((out, timings))
}

/// Runs the gray-scale, two-filter convolution with one of the three `frugal_ml`
/// strategies: "stack", "heap" or "stream". `video` is (N, 1, H, W), `filters`
/// (2, 1, 3, 3), the result (N, H - 2, W - 2, 2). The dict splits the time
/// (`copy_in_s`, `kernel_s`, `copy_out_s`) and gives `working_bytes`, what the
/// algorithm itself needs (exact, from `size_of`), and `thread_stack_bytes`.
#[pyfunction]
fn run_gray<'py>(
    py: Python<'py>,
    strategy: &str,
    video: PyReadonlyArray4<'py, f32>,
    filters: PyReadonlyArray4<'py, f32>,
) -> PyResult<(Bound<'py, PyArray4<f32>>, Bound<'py, PyDict>)> {
    let kind = strategies::Strategy::parse(strategy)
        .ok_or_else(|| PyValueError::new_err(format!("unknown strategy {strategy:?}, expected 'stack', 'heap' or 'stream'")))?;
    let x = video.as_slice().map_err(|_| PyValueError::new_err(CONTIGUOUS_HINT))?;
    let f = filters.as_slice().map_err(|_| PyValueError::new_err(CONTIGUOUS_HINT))?;
    let &[n, c, h, w] = video.shape() else { unreachable!("PyReadonlyArray4 is 4-dimensional") };
    if filters.shape() != [strategies::K, 1, 3, 3] {
        return Err(PyValueError::new_err(format!("filters must be (2, 1, 3, 3), got {:?}", filters.shape())));
    }
    if c != 1 || n == 0 || h < 3 || w < 3 {
        return Err(PyValueError::new_err(format!("video must be (N >= 1, 1, H >= 3, W >= 3), got {:?}", video.shape())));
    }

    let (h_out, w_out) = (h - 2, w - 2);
    let mut out = vec![0f32; n * h_out * w_out * strategies::K];
    let report = py
        .detach(|| strategies::run(kind, h, w, x, f, n, &mut out))
        .ok_or_else(|| {
            let available = match kind {
                strategies::Strategy::Stack => format!("{:?}", strategies::STACK),
                strategies::Strategy::Heap => format!("{:?}", strategies::HEAP),
                strategies::Strategy::Stream => format!("widths {:?}", strategies::STREAM),
            };
            PyValueError::new_err(format!(
                "the {strategy} strategy is not compiled for {h}x{w} frames, add it to register! in \
                 rust/src/strategies.rs and rebuild. Available: {available}"
            ))
        })?;

    let out = PyArray1::from_vec(py, out).reshape([n, h_out, w_out, strategies::K])?;
    let info = PyDict::new(py);
    info.set_item("copy_in_s", report.timings.copy_in)?;
    info.set_item("kernel_s", report.timings.kernel)?;
    info.set_item("copy_out_s", report.timings.copy_out)?;
    info.set_item("working_bytes", report.working_bytes)?;
    info.set_item("thread_stack_bytes", report.thread_stack_bytes)?;
    Ok((out, info))
}

/// The frames each strategy is compiled for: {"stack": [(H, W)], "heap": [(H, W)], "stream": [W]}.
#[pyfunction]
fn strategy_shapes<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("stack", strategies::STACK.to_vec())?;
    d.set_item("heap", strategies::HEAP.to_vec())?;
    d.set_item("stream", strategies::STREAM.to_vec())?;
    Ok(d)
}

/// Bytes of the three tensors of a direct (stack or heap) run of an H x W frame.
#[pyfunction]
fn direct_bytes(h: usize, w: usize) -> usize {
    strategies::direct_bytes(h, w)
}

/// The (C, H, W, K, KH, KW) shapes this build can run.
#[pyfunction]
fn registered_shapes() -> Vec<Shape> {
    shapes::REGISTERED.to_vec()
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cross_correlate2d, m)?)?;
    m.add_function(wrap_pyfunction!(cross_correlate2d_timed, m)?)?;
    m.add_function(wrap_pyfunction!(registered_shapes, m)?)?;
    m.add_function(wrap_pyfunction!(run_gray, m)?)?;
    m.add_function(wrap_pyfunction!(strategy_shapes, m)?)?;
    m.add_function(wrap_pyfunction!(direct_bytes, m)?)?;
    Ok(())
}
