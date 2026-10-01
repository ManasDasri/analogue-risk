//! Trade simulation. The same `simulate` scores historical analogues (labels) and live trades,
//! so what the forecaster learns from is exactly what the strategy executes.

pub const STOP: u8 = 1;
pub const TARGET: u8 = 2;
pub const TIME: u8 = 3;
pub const REVERSE: u8 = 4;
pub const END: u8 = 5;

/// Enter at the open of bar `e` in direction `d` (+1/-1) with stop distance `sd` (price units).
/// Target = rr * sd. Held for at most `max_hold` bars, then exited at the next open.
/// If stop and target are both touched inside one bar, the stop is assumed first (conservative).
/// Returns (exit bar, exit price, reason).
pub fn simulate(
    o: &[f64], h: &[f64], l: &[f64], c: &[f64],
    e: usize, d: f64, sd: f64, rr: f64, max_hold: usize, protect_entry: bool,
) -> (usize, f64, u8) {
    let n = o.len();
    let entry = o[e];
    let stop = entry - d * sd;
    let tp = entry + d * rr * sd;
    let last = (e + max_hold).min(n);
    for b in e..last {
        if b > e {
            // Gap through a level fills at the open.
            if d * (o[b] - stop) <= 0.0 { return (b, o[b], STOP); }
            if d * (o[b] - tp) >= 0.0 { return (b, o[b], TARGET); }
        }
        if b > e || protect_entry {
            let (adv, fav) = if d > 0.0 { (l[b], h[b]) } else { (h[b], l[b]) };
            if d * (adv - stop) <= 0.0 { return (b, stop, STOP); }
            if d * (fav - tp) >= 0.0 { return (b, tp, TARGET); }
        }
    }
    if last < n { (last, o[last], TIME) } else { (n - 1, c[n - 1], END) }
}

/// Net R-multiple of a trade; `cost` is the fractional cost per side on notional.
pub fn r_multiple(d: f64, entry: f64, exit: f64, sd: f64, cost: f64) -> f64 {
    (d * (exit - entry) - cost * (entry + exit)) / sd
}

pub struct Params {
    pub rr: f64,
    pub max_hold: usize,
    pub cost: f64,
    pub protect_entry: bool,
    pub close_on_opposite: bool,
    /// true: `size` is the fraction of equity risked (qty = size*eq/sd); false: fraction of equity as notional.
    pub risk_sizing: bool,
    pub lev_cap: f64,
}

#[derive(Default)]
pub struct Trades {
    pub entry_bar: Vec<i64>,
    pub exit_bar: Vec<i64>,
    pub dir: Vec<i64>,
    pub entry_px: Vec<f64>,
    pub exit_px: Vec<f64>,
    pub r: Vec<f64>,
    pub ret: Vec<f64>,
    pub reason: Vec<i64>,
}

/// Signals decided at the close of bar t are filled at the open of t+1, one position at a time.
/// Returns the mark-to-market equity at every close (starting at 1.0) and the trade log.
pub fn run(
    o: &[f64], h: &[f64], l: &[f64], c: &[f64],
    dir: &[i8], sd: &[f64], size: &[f64], p: &Params,
) -> (Vec<f64>, Trades) {
    let n = c.len();
    let mut eq = vec![1.0; n];
    let mut tr = Trades::default();
    let mut cash = 1.0;
    let mut t = 0;
    while t + 1 < n {
        eq[t] = cash;
        let (s, z) = (sd[t], size[t]);
        if dir[t] == 0 || !(s > 0.0) || !(z > 0.0) || !s.is_finite() || cash <= 0.0 {
            t += 1;
            continue;
        }
        let d = dir[t] as f64;
        let e = t + 1;
        let entry = o[e];
        let mut qty = if p.risk_sizing { z * cash / s } else { z * cash / c[t] };
        qty = qty.min(p.lev_cap * cash / c[t]);
        let (mut x, mut px, mut why) = simulate(o, h, l, c, e, d, s, p.rr, p.max_hold, p.protect_entry);
        if p.close_on_opposite {
            if let Some(b) = (e..x).find(|&b| dir[b] == -dir[t]) {
                (x, px, why) = (b + 1, o[b + 1], REVERSE);
            }
        }
        for b in e..x {
            eq[b] = cash + d * qty * (c[b] - entry) - p.cost * qty * entry;
        }
        let pnl = d * qty * (px - entry) - p.cost * qty * (entry + px);
        tr.entry_bar.push(e as i64);
        tr.exit_bar.push(x as i64);
        tr.dir.push(dir[t] as i64);
        tr.entry_px.push(entry);
        tr.exit_px.push(px);
        tr.r.push(r_multiple(d, entry, px, s, p.cost));
        tr.ret.push(pnl / cash);
        tr.reason.push(why as i64);
        cash += pnl;
        eq[x] = cash;
        t = x;
    }
    eq[n - 1] = cash;
    (eq, tr)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Five bars; long entry at the open of bar 1 (price 100), stop distance 2, target 3.
    fn bars(h: [f64; 5], l: [f64; 5]) -> ([f64; 5], [f64; 5], [f64; 5], [f64; 5]) {
        ([100.0; 5], h, l, [100.0; 5])
    }

    #[test]
    fn stop_wins_when_both_levels_hit_in_one_bar() {
        let (o, h, l, c) = bars([101.0, 104.0, 101.0, 101.0, 101.0], [99.0, 97.0, 99.0, 99.0, 99.0]);
        assert_eq!(simulate(&o, &h, &l, &c, 1, 1.0, 2.0, 1.5, 3, true), (1, 98.0, STOP));
    }

    #[test]
    fn target_and_time_exits() {
        let (o, h, l, c) = bars([101.0, 101.0, 103.5, 101.0, 101.0], [99.0; 5]);
        assert_eq!(simulate(&o, &h, &l, &c, 1, 1.0, 2.0, 1.5, 3, true), (2, 103.0, TARGET));
        let (o, h, l, c) = bars([101.0; 5], [99.0; 5]);
        assert_eq!(simulate(&o, &h, &l, &c, 1, 1.0, 2.0, 1.5, 3, true), (4, 100.0, TIME));
    }

    #[test]
    fn gap_through_stop_fills_at_open_and_unprotected_entry_bar_is_skipped() {
        let o = [100.0, 100.0, 95.0, 95.0, 95.0];
        let (h, l, c) = ([101.0; 5], [94.0; 5], [95.0; 5]);
        assert_eq!(simulate(&o, &h, &l, &c, 1, 1.0, 2.0, 1.5, 3, false), (2, 95.0, STOP));
    }

    #[test]
    fn r_multiple_includes_costs() {
        assert!((r_multiple(1.0, 100.0, 103.0, 2.0, 0.0) - 1.5).abs() < 1e-12);
        assert!((r_multiple(-1.0, 100.0, 98.0, 2.0, 0.001) - (2.0 - 0.198) / 2.0).abs() < 1e-12);
    }
}
