//! Bar-for-bar port of the original TradingView strategy (`codes/script.pine`, defaults),
//! used as the prior-work baseline. Execution semantics are reproduced in backtest::run with
//! max_hold = 21 (close when bars-in-trade >= 20, filled next open), protect_entry = false
//! (exit orders are placed at the entry bar's close), close_on_opposite = true and notional sizing.

use crate::features::{atr, log_returns, rolling_sum, sma, stdev};

pub struct Out {
    pub dir: Vec<i8>,
    pub stop_dist: Vec<f64>,
    pub size: Vec<f64>,
    pub adj_prob_bull: Vec<f64>,
    pub jump_prob: Vec<f64>,
}

pub fn signals(h: &[f64], l: &[f64], c: &[f64]) -> Out {
    const LOOKBACK: usize = 1000;
    const K: usize = 10;
    const W: usize = 20;
    const SCAN: usize = 300;
    const CONF: f64 = 65.0;
    const MIN_MATCHES: usize = 8;
    const GATE: f64 = 50.0;
    const WIDEN: f64 = 10.0;
    const MAX_POS: f64 = 10.0;
    const STOP_MULT: f64 = 2.5;
    const RR: f64 = 1.5;

    let n = c.len();
    let a = atr(h, l, c, 14);
    let a_ma = sma(&a, LOOKBACK);
    let lr = log_returns(c);
    let hv = stdev(&lr, LOOKBACK);
    let jumps: Vec<f64> = lr.iter().zip(&hv).map(|(r, s)| (r.abs() > 2.5 * s) as u8 as f64).collect();
    let total = rolling_sum(&jumps, LOOKBACK);

    let mut out = Out {
        dir: vec![0; n],
        stop_dist: vec![f64::NAN; n],
        size: vec![0.0; n],
        adj_prob_bull: vec![f64::NAN; n],
        jump_prob: vec![f64::NAN; n],
    };
    let st = |i: usize| (a[i] > a_ma[i]) as usize; // NaN compares false, as in Pine

    for t in (LOOKBACK + W + 11)..n {
        // Markov chain over the last SCAN transitions.
        let mut cnt = [[0f64; 2]; 2];
        for i in 1..=SCAN {
            cnt[st(t - i - 1)][st(t - i)] += 1.0;
        }
        let p0 = (cnt[0][0] + cnt[0][1]).max(1.0);
        let p1 = (cnt[1][0] + cnt[1][1]).max(1.0);
        let (p00, p11) = (cnt[0][0] / p0, cnt[1][1] / p1);

        let pj = (1.0 - (-(total[t] / LOOKBACK as f64) * W as f64).exp()) * 100.0;
        out.jump_prob[t] = pj;
        let suppressed = pj >= GATE;
        let wide = pj >= WIDEN && !suppressed;

        // 1D KNN on the 20-bar simple return.
        let cur = (c[t] - c[t - W]) / c[t - W];
        let max_scan = SCAN.min(t - W);
        let mut d: Vec<(f64, usize)> = ((W + 1)..=max_scan)
            .map(|i| (((c[t - i] - c[t - i - W]) / c[t - i - W] - cur).abs(), i))
            .collect();
        d.sort_by(|x, y| x.0.total_cmp(&y.0)); // stable, like array.sort_indices
        let k = K.min(d.len());
        let bull = d[..k].iter().filter(|&&(_, i)| c[t - i + W] > c[t - i]).count() as f64;
        let prior = bull / (k.max(1) as f64) * 100.0;

        let (mut lo, mut hi) = (0.0, 100.0);
        if k > 0 {
            let (z, nn) = (1.96f64, k as f64);
            let ph = bull / nn;
            let den = 1.0 + z * z / nn;
            let ctr = ph + z * z / (2.0 * nn);
            let mg = z * ((ph * (1.0 - ph) / nn) + z * z / (4.0 * nn * nn)).sqrt();
            lo = (((ctr - mg) / den) * 100.0).max(0.0);
            hi = (((ctr + mg) / den) * 100.0).min(100.0);
        }

        let sp = if st(t) == 1 { p11 } else { p00 };
        let adj = (prior * sp + 50.0 * (1.0 - sp)).clamp(0.0, 100.0);
        let adj_bear = 100.0 - adj;
        out.adj_prob_bull[t] = adj;

        let mult = if wide { STOP_MULT * 1.5 } else { STOP_MULT };
        let sd = a[t] * mult;
        let rr = (sd * RR) / sd.max(0.0001);
        let lk = adj / 100.0 - (1.0 - adj / 100.0) / rr;
        let sk = adj_bear / 100.0 - (1.0 - adj_bear / 100.0) / rr;
        let kelly = if adj > adj_bear { lk } else { sk };
        let size = MAX_POS.min((kelly * 0.5 * 100.0).max(0.0));

        let enough = k >= MIN_MATCHES.min(K);
        let bull_sig = enough && lo >= CONF && adj >= CONF && size > 0.0 && !suppressed;
        let bear_sig = enough && (100.0 - hi) >= CONF && adj_bear >= CONF && size > 0.0 && !suppressed;
        out.dir[t] = if bull_sig { 1 } else if bear_sig { -1 } else { 0 };
        out.stop_dist[t] = sd;
        out.size[t] = size.max(0.01) / 100.0;
    }
    out
}
