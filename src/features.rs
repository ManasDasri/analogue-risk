//! Causal per-bar features. Every value at index t uses bars 0..=t only.

pub fn log_returns(c: &[f64]) -> Vec<f64> {
    let mut r = vec![0.0; c.len()];
    for t in 1..c.len() {
        r[t] = (c[t] / c[t - 1]).ln();
    }
    r
}

/// EWMA volatility of log returns, including bar t.
/// Seeded with the mean square of the first 50 returns (burn-in bars are never traded).
pub fn ewma_vol(r: &[f64], halflife: f64) -> Vec<f64> {
    let n = r.len();
    if n < 2 {
        return vec![f64::NAN; n];
    }
    let lam = 0.5f64.powf(1.0 / halflife);
    let seed = n.min(50);
    let mut v = r[1..seed].iter().map(|x| x * x).sum::<f64>() / (seed - 1).max(1) as f64;
    r.iter()
        .map(|x| {
            v = lam * v + (1.0 - lam) * x * x;
            v.sqrt()
        })
        .collect()
}

/// Wilder ATR, identical to Pine's `ta.atr` (RMA seeded with an SMA; NaN before `len` bars).
pub fn atr(h: &[f64], l: &[f64], c: &[f64], len: usize) -> Vec<f64> {
    let n = c.len();
    let mut out = vec![f64::NAN; n];
    let tr: Vec<f64> = (0..n)
        .map(|t| {
            if t == 0 {
                h[0] - l[0]
            } else {
                (h[t] - l[t]).max((h[t] - c[t - 1]).abs()).max((l[t] - c[t - 1]).abs())
            }
        })
        .collect();
    if n < len {
        return out;
    }
    out[len - 1] = tr[..len].iter().sum::<f64>() / len as f64;
    for t in len..n {
        out[t] = (out[t - 1] * (len - 1) as f64 + tr[t]) / len as f64;
    }
    out
}

/// Rolling mean with Pine semantics: NaN if the window is incomplete or holds a NaN.
pub fn sma(x: &[f64], len: usize) -> Vec<f64> {
    rolling(x, len, |s, _| s / len as f64)
}

/// Rolling population standard deviation (Pine `ta.stdev`, biased).
pub fn stdev(x: &[f64], len: usize) -> Vec<f64> {
    let mean = sma(x, len);
    let sq: Vec<f64> = x.iter().map(|v| v * v).collect();
    let msq = sma(&sq, len);
    mean.iter().zip(&msq).map(|(m, q)| (q - m * m).max(0.0).sqrt()).collect()
}

/// Rolling sum (Pine `math.sum`).
pub fn rolling_sum(x: &[f64], len: usize) -> Vec<f64> {
    rolling(x, len, |s, _| s)
}

fn rolling(x: &[f64], len: usize, f: impl Fn(f64, usize) -> f64) -> Vec<f64> {
    let mut out = vec![f64::NAN; x.len()];
    let (mut s, mut nans) = (0.0, 0usize);
    for t in 0..x.len() {
        if x[t].is_nan() { nans += 1 } else { s += x[t] }
        if t >= len {
            if x[t - len].is_nan() { nans -= 1 } else { s -= x[t - len] }
        }
        if t + 1 >= len && nans == 0 {
            out[t] = f(s, len);
        }
    }
    out
}

/// Volatility regime: 1 when ATR is above its long moving average (the v1 definition), else 0.
pub fn regime(atr: &[f64], len: usize) -> Vec<u8> {
    let ma = sma(atr, len);
    atr.iter().zip(&ma).map(|(a, m)| (a > m) as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atr_matches_wilder_definition() {
        let h = [10.0, 11.0, 12.0, 11.5];
        let l = [9.0, 9.5, 10.0, 10.0];
        let c = [9.5, 10.5, 11.0, 10.5];
        let a = atr(&h, &l, &c, 2);
        // TR = [1.0, 1.5, 2.0, 1.5]; seed = mean(1.0, 1.5); then RMA with alpha = 1/2
        assert!(a[0].is_nan());
        assert!((a[1] - 1.25).abs() < 1e-12 && (a[2] - 1.625).abs() < 1e-12 && (a[3] - 1.5625).abs() < 1e-12);
    }

    #[test]
    fn rolling_statistics_and_nan_semantics() {
        let x = [1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0];
        let m = sma(&x, 2);
        assert!(m[0].is_nan() && (m[1] - 1.5).abs() < 1e-12 && (m[2] - 2.5).abs() < 1e-12);
        assert!(m[3].is_nan() && m[4].is_nan() && (m[5] - 5.5).abs() < 1e-12);
        let s = stdev(&[1.0, 3.0, 5.0], 3);
        assert!((s[2] - (8.0f64 / 3.0).sqrt()).abs() < 1e-12); // population standard deviation
    }
}
