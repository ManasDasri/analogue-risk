//! Expanding and rolling empirical quantiles and expected shortfall, maintained on a sorted buffer
//! (one insertion per bar instead of a full sort). Quantiles use numpy's default "linear" method,
//! including its two-sided interpolation, so results match `np.quantile` to the last bit.

fn quantile_sorted(x: &[f64], alpha: f64) -> f64 {
    let pos = (x.len() - 1) as f64 * alpha;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(x.len() - 1);
    let t = pos - lo as f64;
    let (a, b) = (x[lo], x[hi]);
    let d = b - a;
    if t >= 0.5 {
        b - d * (1.0 - t)
    } else {
        a + d * t
    }
}

fn tail(x: &[f64], alpha: f64) -> (f64, f64) {
    let q = quantile_sorted(x, alpha);
    let m = x.partition_point(|&v| v <= q);
    (q, x[..m].iter().sum::<f64>() / m as f64)
}

fn insert(buf: &mut Vec<f64>, v: f64) {
    let i = buf.partition_point(|&u| u < v);
    buf.insert(i, v);
}

/// For every t >= start: alpha-quantile and mean of values <= it, over finite z[start..=t].
pub fn expanding_tail(z: &[f64], start: usize, alpha: f64) -> (Vec<f64>, Vec<f64>) {
    let n = z.len();
    let (mut q, mut es) = (vec![f64::NAN; n], vec![f64::NAN; n]);
    let mut buf = Vec::with_capacity(n);
    for t in start..n {
        if z[t].is_finite() {
            insert(&mut buf, z[t]);
        }
        if !buf.is_empty() {
            (q[t], es[t]) = tail(&buf, alpha);
        }
    }
    (q, es)
}

/// For every t >= window - 1: alpha-quantile and mean of values <= it over r[t-window+1..=t].
pub fn rolling_tail(r: &[f64], window: usize, alpha: f64) -> (Vec<f64>, Vec<f64>) {
    let n = r.len();
    let (mut q, mut es) = (vec![f64::NAN; n], vec![f64::NAN; n]);
    if window == 0 || n < window {
        return (q, es);
    }
    let mut buf: Vec<f64> = r[..window].to_vec();
    buf.sort_by(f64::total_cmp);
    (q[window - 1], es[window - 1]) = tail(&buf, alpha);
    for t in window..n {
        let old = r[t - window];
        let i = buf.partition_point(|&u| u < old);
        buf.remove(i);
        insert(&mut buf, r[t]);
        (q[t], es[t]) = tail(&buf, alpha);
    }
    (q, es)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_matches_numpy_linear() {
        let x = [1.0, 2.0, 4.0, 8.0];
        assert_eq!(quantile_sorted(&x, 0.5), 3.0); // pos 1.5 -> between 2 and 4
        assert_eq!(quantile_sorted(&x, 0.25), 1.75); // pos 0.75 -> b - (b - a) * 0.25
        assert_eq!(quantile_sorted(&x, 0.0), 1.0);
        assert_eq!(quantile_sorted(&x, 1.0), 8.0);
    }

    #[test]
    fn rolling_equals_recomputation() {
        let r: Vec<f64> = (0..50).map(|i| ((i * 37) % 23) as f64 - 11.0).collect();
        let (q, es) = rolling_tail(&r, 10, 0.2);
        for t in 9..50 {
            let mut w = r[t - 9..=t].to_vec();
            w.sort_by(f64::total_cmp);
            let (q2, e2) = tail(&w, 0.2);
            assert_eq!((q[t], es[t]), (q2, e2));
        }
    }
}
