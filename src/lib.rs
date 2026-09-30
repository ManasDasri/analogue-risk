mod analog;
mod backtest;
mod features;
mod garch;
mod hawkes;
mod legacy;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

type A1<'py> = Bound<'py, PyArray1<f64>>;

fn to2d<'py>(py: Python<'py>, flat: Vec<f64>, cols: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows = flat.len() / cols;
    flat.into_pyarray(py).reshape([rows, cols])
}

/// (log returns, EWMA sigma, Wilder ATR, regime state 0/1)
#[pyfunction]
#[pyo3(name = "features")]
fn py_features<'py>(
    py: Python<'py>, h: PyReadonlyArray1<f64>, l: PyReadonlyArray1<f64>, c: PyReadonlyArray1<f64>,
    halflife: f64, atr_len: usize, regime_len: usize,
) -> PyResult<(A1<'py>, A1<'py>, A1<'py>, Bound<'py, PyArray1<u8>>)> {
    let (h, l, c) = (h.as_slice()?, l.as_slice()?, c.as_slice()?);
    let r = features::log_returns(c);
    let sig = features::ewma_vol(&r, halflife);
    let a = features::atr(h, l, c, atr_len);
    let st = features::regime(&a, regime_len);
    Ok((r.into_pyarray(py), sig.into_pyarray(py), a.into_pyarray(py), st.into_pyarray(py)))
}

#[pyfunction]
fn embed<'py>(
    py: Python<'py>, r: PyReadonlyArray1<f64>, sigma: PyReadonlyArray1<f64>, w: usize, m: usize, normalize: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    to2d(py, analog::embed(r.as_slice()?, sigma.as_slice()?, w, m, normalize), m)
}

/// (R long, R short, up) per anchor for the trade rule (stop = sl_mult*ATR, target = rr*stop).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn labels<'py>(
    py: Python<'py>, o: PyReadonlyArray1<f64>, h: PyReadonlyArray1<f64>, l: PyReadonlyArray1<f64>,
    c: PyReadonlyArray1<f64>, atr: PyReadonlyArray1<f64>, sl_mult: f64, rr: f64, max_hold: usize, cost: f64,
) -> PyResult<(A1<'py>, A1<'py>, A1<'py>)> {
    let (rl, rs, up) = analog::labels(
        o.as_slice()?, h.as_slice()?, l.as_slice()?, c.as_slice()?, atr.as_slice()?, sl_mult, rr, max_hold, cost,
    );
    Ok((rl.into_pyarray(py), rs.into_pyarray(py), up.into_pyarray(py)))
}

/// Analogue forecast of each column of `y` (n x p). Output n x (2p+2): means, standard errors,
/// analogues used, P(high-vol regime over the horizon).
#[pyfunction]
#[pyo3(signature = (emb, state, y, w, delay, h, lookback, k, k_min, excl, regime, warmup, bandwidth=0.0))]
#[allow(clippy::too_many_arguments)]
fn forecast<'py>(
    py: Python<'py>, emb: PyReadonlyArray2<f64>, state: PyReadonlyArray1<u8>, y: PyReadonlyArray2<f64>,
    w: usize, delay: usize, h: usize, lookback: usize, k: usize, k_min: usize, excl: usize, regime: bool,
    warmup: usize, bandwidth: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (m, p) = (emb.as_array().ncols(), y.as_array().ncols());
    let (e, st, y) = (emb.as_slice()?, state.as_slice()?, y.as_slice()?);
    let prm = analog::Params { w, delay, h, lookback, k, k_min, excl, regime, warmup, bandwidth };
    let flat = py.detach(|| analog::forecast(e, m, st, y, p, &prm));
    to2d(py, flat, 2 * p + 2)
}

/// Zero-mean GARCH(1,1): (one-step variance forecasts, log-likelihood from `start`).
#[pyfunction]
fn garch_filter<'py>(
    py: Python<'py>, r: PyReadonlyArray1<f64>, omega: f64, alpha: f64, beta: f64, s2_0: f64, start: usize,
) -> PyResult<(A1<'py>, f64)> {
    let (s2, ll) = garch::filter(r.as_slice()?, omega, alpha, beta, s2_0, start);
    Ok((s2.into_pyarray(py), ll))
}

#[pyfunction]
fn hawkes_loglik(times: PyReadonlyArray1<f64>, t_end: f64, mu: f64, alpha: f64, beta: f64) -> PyResult<f64> {
    Ok(hawkes::loglik(times.as_slice()?, t_end, mu, alpha, beta))
}

#[pyfunction]
fn hawkes_prob<'py>(
    py: Python<'py>, events: PyReadonlyArray1<bool>, mu: f64, alpha: f64, beta: f64, h: f64,
) -> PyResult<A1<'py>> {
    Ok(hawkes::prob(events.as_slice()?, mu, alpha, beta, h).into_pyarray(py))
}

/// Returns (equity per close, dict of trade-log arrays).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(name = "backtest")]
fn py_backtest<'py>(
    py: Python<'py>, o: PyReadonlyArray1<f64>, h: PyReadonlyArray1<f64>, l: PyReadonlyArray1<f64>,
    c: PyReadonlyArray1<f64>, dir: PyReadonlyArray1<i8>, stop_dist: PyReadonlyArray1<f64>,
    size: PyReadonlyArray1<f64>, rr: f64, max_hold: usize, cost: f64, protect_entry: bool,
    close_on_opposite: bool, risk_sizing: bool, lev_cap: f64,
) -> PyResult<(A1<'py>, Bound<'py, PyDict>)> {
    let p = backtest::Params { rr, max_hold, cost, protect_entry, close_on_opposite, risk_sizing, lev_cap };
    let (o, h, l, c) = (o.as_slice()?, h.as_slice()?, l.as_slice()?, c.as_slice()?);
    let (d, s, z) = (dir.as_slice()?, stop_dist.as_slice()?, size.as_slice()?);
    let (eq, t) = py.detach(|| backtest::run(o, h, l, c, d, s, z, &p));
    let out = PyDict::new(py);
    out.set_item("entry_bar", t.entry_bar.into_pyarray(py))?;
    out.set_item("exit_bar", t.exit_bar.into_pyarray(py))?;
    out.set_item("dir", t.dir.into_pyarray(py))?;
    out.set_item("entry_px", t.entry_px.into_pyarray(py))?;
    out.set_item("exit_px", t.exit_px.into_pyarray(py))?;
    out.set_item("r", t.r.into_pyarray(py))?;
    out.set_item("ret", t.ret.into_pyarray(py))?;
    out.set_item("reason", t.reason.into_pyarray(py))?;
    Ok((eq.into_pyarray(py), out))
}

/// Original Pine strategy: (dir, stop_dist, notional size fraction, adjusted bull prob, jump prob).
#[pyfunction]
fn legacy_v1<'py>(
    py: Python<'py>, h: PyReadonlyArray1<f64>, l: PyReadonlyArray1<f64>, c: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, PyArray1<i8>>, A1<'py>, A1<'py>, A1<'py>, A1<'py>)> {
    let (h, l, c) = (h.as_slice()?, l.as_slice()?, c.as_slice()?);
    let o = py.detach(|| legacy::signals(h, l, c));
    Ok((
        o.dir.into_pyarray(py), o.stop_dist.into_pyarray(py), o.size.into_pyarray(py),
        o.adj_prob_bull.into_pyarray(py), o.jump_prob.into_pyarray(py),
    ))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_features, m)?)?;
    m.add_function(wrap_pyfunction!(embed, m)?)?;
    m.add_function(wrap_pyfunction!(labels, m)?)?;
    m.add_function(wrap_pyfunction!(forecast, m)?)?;
    m.add_function(wrap_pyfunction!(garch_filter, m)?)?;
    m.add_function(wrap_pyfunction!(hawkes_loglik, m)?)?;
    m.add_function(wrap_pyfunction!(hawkes_prob, m)?)?;
    m.add_function(wrap_pyfunction!(py_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(legacy_v1, m)?)?;
    Ok(())
}
