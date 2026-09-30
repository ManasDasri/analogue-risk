//! GARCH(1,1) with zero mean: s2[t+1] = omega + alpha * r[t]^2 + beta * s2[t].

/// Returns (s2_next, loglik): s2_next[t] is the one-step variance forecast for bar t+1 made at t;
/// the Gaussian log-likelihood is summed over bars start..n.
pub fn filter(r: &[f64], omega: f64, alpha: f64, beta: f64, s2_0: f64, start: usize) -> (Vec<f64>, f64) {
    let mut s2 = s2_0;
    let mut ll = 0.0;
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    let out = r
        .iter()
        .enumerate()
        .map(|(t, &x)| {
            if t >= start {
                ll -= 0.5 * (ln2pi + s2.ln() + x * x / s2);
            }
            s2 = omega + alpha * x * x + beta * s2;
            s2
        })
        .collect();
    (out, ll)
}
