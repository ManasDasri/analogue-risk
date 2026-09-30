//! Regime-conditioned analogue (k-nearest-neighbour) forecaster.
//!
//! At each bar t the embedding row x_t (any causal feature vector) is compared with past anchors j.
//! The `k` nearest anchors per volatility regime are selected greedily under an exclusion zone
//! (no two analogues closer than `excl` bars, so their outcome windows do not overlap). Each anchor
//! carries a row of targets y_j that is only used once fully observed (j <= t - delay). Regime
//! groups are mixed with weights from the h-step Markov forecast of the regime:
//!   E[y | now] = sum_s P(regime s over the horizon | now) * mean(y over analogues in regime s).

use crate::backtest::{r_multiple, simulate};
use rayon::prelude::*;

pub struct Params {
    pub w: usize,
    pub delay: usize,
    pub h: usize,
    pub lookback: usize,
    pub k: usize,
    pub k_min: usize,
    pub excl: usize,
    pub regime: bool,
    pub warmup: usize,
    /// > 0: Gaussian-kernel weights over all candidates (Nadaraya-Watson) instead of k nearest.
    pub bandwidth: f64,
}

/// Row-major n x m embedding; NaN rows where the window is incomplete.
pub fn embed(r: &[f64], sigma: &[f64], w: usize, m: usize, normalize: bool) -> Vec<f64> {
    assert!(m >= 1 && w % m == 0, "window must be divisible by m");
    let n = r.len();
    let g = w / m;
    let mut cs = vec![0.0; n + 1];
    for t in 0..n {
        cs[t + 1] = cs[t] + r[t];
    }
    let mut out = vec![f64::NAN; n * m];
    for j in w..n {
        let scale = if normalize { sigma[j] * (g as f64).sqrt() } else { 1.0 };
        for q in 0..m {
            let a = j + 1 - w + q * g; // first return in segment
            out[j * m + q] = (cs[a + g] - cs[a]) / scale;
        }
    }
    out
}

/// Outcome of the strategy's own trade rule taken at each anchor j (entry at open j+1),
/// long and short, plus whether close rose over the next `max_hold` bars.
/// NaN where the outcome is not fully observed by bar j + max_hold + 1.
#[allow(clippy::too_many_arguments)]
pub fn labels(
    o: &[f64], h: &[f64], l: &[f64], c: &[f64], atr: &[f64],
    sl_mult: f64, rr: f64, max_hold: usize, cost: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = c.len();
    let mut rl = vec![f64::NAN; n];
    let mut rs = vec![f64::NAN; n];
    let mut up = vec![f64::NAN; n];
    for j in 0..n {
        let e = j + 1;
        if e + max_hold >= n || !(atr[j] > 0.0) {
            continue;
        }
        let sd = sl_mult * atr[j];
        for (d, out) in [(1.0, &mut rl), (-1.0, &mut rs)] {
            let (_, px, _) = simulate(o, h, l, c, e, d, sd, rr, max_hold, true);
            out[j] = r_multiple(d, o[e], px, sd, cost);
        }
        up[j] = (c[j + max_hold] > c[j]) as u8 as f64;
    }
    (rl, rs, up)
}

/// Cumulative regime-transition counts: c[u][2*prev + cur] over transitions ending at bars 1..=u.
pub fn transition_prefix(state: &[u8]) -> Vec<[u32; 4]> {
    let mut acc = [0u32; 4];
    let mut out = vec![acc; state.len()];
    for u in 1..state.len() {
        acc[2 * state[u - 1] as usize + state[u] as usize] += 1;
        out[u] = acc;
    }
    out
}

/// Average over the next h steps of P(regime = high | regime now), from a 2-state chain
/// estimated on the last `lookback` transitions.
pub fn markov_pi_high(state: &[u8], pre: &[[u32; 4]], t: usize, lookback: usize, h: usize) -> f64 {
    let lo = t.saturating_sub(lookback).max(1) - 1; // transitions ending in (lo, t]
    let n: Vec<f64> = (0..4).map(|i| (pre[t][i] - pre[lo][i]) as f64).collect();
    let a = if n[0] + n[1] > 0.0 { n[1] / (n[0] + n[1]) } else { 0.0 };
    let b = if n[2] + n[3] > 0.0 { n[2] / (n[2] + n[3]) } else { 0.0 };
    let s0 = state[t] as f64;
    if a + b == 0.0 {
        return s0;
    }
    let q = a / (a + b);
    let lam = 1.0 - a - b;
    (1..=h).map(|k| q + (s0 - q) * lam.powi(k as i32)).sum::<f64>() / h as f64
}

/// Per-bar forecast over `p` target columns.
/// Output row: [mean_0..mean_p, se_0..se_p, n_used, pi_high]; NaN rows where no forecast exists.
pub fn forecast(emb: &[f64], m: usize, state: &[u8], y: &[f64], p: usize, prm: &Params) -> Vec<f64> {
    let n = state.len();
    let pre = transition_prefix(state);
    (0..n)
        .into_par_iter()
        .flat_map_iter(|t| one(t, emb, m, state, &pre, y, p, prm))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn one(t: usize, emb: &[f64], m: usize, state: &[u8], pre: &[[u32; 4]], y: &[f64], p: usize, prm: &Params) -> Vec<f64> {
    let width = 2 * p + 2;
    let nan = vec![f64::NAN; width];
    let x = &emb[t * m..t * m + m];
    if t < prm.warmup || t < prm.delay || x.iter().any(|v| !v.is_finite()) {
        return nan;
    }
    let hi = t - prm.delay; // newest anchor whose targets are fully observed at t
    let lo = t.saturating_sub(prm.lookback).max(prm.w);
    if hi < lo {
        return nan;
    }
    let mut cand: Vec<(f64, usize)> = (lo..=hi)
        .filter(|&j| {
            y[j * p..j * p + p].iter().all(|v| v.is_finite())
                && emb[j * m..j * m + m].iter().all(|v| v.is_finite())
        })
        .map(|j| {
            let e = &emb[j * m..j * m + m];
            (x.iter().zip(e).map(|(a, b)| (a - b) * (a - b)).sum::<f64>(), j)
        })
        .collect();
    if prm.bandwidth > 0.0 {
        return kernel(&cand, state, pre, y, p, prm, t);
    }
    let groups = if prm.regime { 2 } else { 1 };
    let mut acc: Vec<usize> = Vec::with_capacity(groups * prm.k);
    let mut grp: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    let cmp = |a: &(f64, usize), b: &(f64, usize)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
    // Each accepted analogue can block up to 2*excl neighbours; take a generous head first.
    let cap = (4 * groups * prm.k * (2 * prm.excl + 1)).max(256);
    let head = if cand.len() > cap {
        cand.select_nth_unstable_by(cap, cmp);
        cap
    } else {
        cand.len()
    };
    cand[..head].sort_unstable_by(cmp);
    let mut full = false;
    for pass in 0..2 {
        if pass == 1 {
            if full || head == cand.len() {
                break;
            }
            cand[head..].sort_unstable_by(cmp);
        }
        let range = if pass == 0 { 0..head } else { head..cand.len() };
        for &(_, j) in &cand[range] {
            let s = if prm.regime { state[j] as usize } else { 0 };
            if grp[s].len() >= prm.k || acc.iter().any(|&a| a.abs_diff(j) < prm.excl) {
                continue;
            }
            acc.push(j);
            grp[s].push(j);
            if grp[..groups].iter().all(|g| g.len() >= prm.k) {
                full = true;
                break;
            }
        }
    }

    let pi1 = if prm.regime { markov_pi_high(state, pre, t, prm.lookback, prm.h) } else { 0.0 };
    let mut wts = if prm.regime { [1.0 - pi1, pi1] } else { [1.0, 0.0] };
    for s in 0..2 {
        if grp[s].len() < prm.k_min.max(2) {
            wts[s] = 0.0;
        }
    }
    let wsum = wts[0] + wts[1];
    if wsum <= 0.0 {
        return nan;
    }
    let mut out = vec![0.0; width];
    for s in 0..2 {
        let w = wts[s] / wsum;
        if w == 0.0 {
            continue;
        }
        let g = &grp[s];
        let k = g.len() as f64;
        for c in 0..p {
            let mean = g.iter().map(|&j| y[j * p + c]).sum::<f64>() / k;
            let ss = g.iter().map(|&j| (y[j * p + c] - mean).powi(2)).sum::<f64>();
            out[c] += w * mean;
            out[p + c] += w * w * ss / (k - 1.0) / k; // variance of the mixture mean
        }
    }
    for c in 0..p {
        out[p + c] = out[p + c].sqrt();
    }
    out[2 * p] = acc.len() as f64;
    out[2 * p + 1] = pi1;
    out
}

/// Kernel mode: every candidate j gets weight exp(-d_j^2 / (2 bw^2)); regime groups are mixed as in
/// kNN mode. Standard errors use the Kish effective sample size (sum w)^2 / sum w^2.
fn kernel(cand: &[(f64, usize)], state: &[u8], pre: &[[u32; 4]], y: &[f64], p: usize, prm: &Params, t: usize) -> Vec<f64> {
    let width = 2 * p + 2;
    let inv = 1.0 / (2.0 * prm.bandwidth * prm.bandwidth);
    let dmin = cand.iter().map(|c| c.0).fold(f64::INFINITY, f64::min); // stabilise exp()
    let mut sw = [0.0f64; 2];
    let mut sw2 = [0.0f64; 2];
    let mut sy = vec![[0.0f64; 2]; p];
    let mut syy = vec![[0.0f64; 2]; p];
    for &(d, j) in cand {
        let z = (d - dmin) * inv;
        if z > 30.0 {
            continue;
        }
        let s = if prm.regime { state[j] as usize } else { 0 };
        let w = (-z).exp();
        sw[s] += w;
        sw2[s] += w * w;
        for c in 0..p {
            let v = y[j * p + c];
            sy[c][s] += w * v;
            syy[c][s] += w * v * v;
        }
    }
    let neff = [sw[0] * sw[0] / sw2[0].max(1e-300), sw[1] * sw[1] / sw2[1].max(1e-300)];
    let pi1 = if prm.regime { markov_pi_high(state, pre, t, prm.lookback, prm.h) } else { 0.0 };
    let mut wts = if prm.regime { [1.0 - pi1, pi1] } else { [1.0, 0.0] };
    for s in 0..2 {
        if !(neff[s] >= prm.k_min.max(2) as f64) {
            wts[s] = 0.0;
        }
    }
    let wsum = wts[0] + wts[1];
    let mut out = vec![f64::NAN; width];
    if wsum <= 0.0 {
        return out;
    }
    out.iter_mut().for_each(|v| *v = 0.0);
    for s in 0..2 {
        let w = wts[s] / wsum;
        if w == 0.0 {
            continue;
        }
        for c in 0..p {
            let mean = sy[c][s] / sw[s];
            let var = (syy[c][s] / sw[s] - mean * mean).max(0.0);
            out[c] += w * mean;
            out[p + c] += w * w * var / neff[s];
        }
    }
    for c in 0..p {
        out[p + c] = out[p + c].sqrt();
    }
    out[2 * p] = neff[0] + neff[1];
    out[2 * p + 1] = pi1;
    out
}
