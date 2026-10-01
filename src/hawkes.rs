//! Univariate Hawkes process with exponential kernel phi(u) = alpha * beta * exp(-beta u),
//! alpha = branching ratio (< 1 for stationarity), time measured in bars.

/// Exact log-likelihood of event times on [0, t_end], O(n) via the standard recursion.
pub fn loglik(times: &[f64], t_end: f64, mu: f64, alpha: f64, beta: f64) -> f64 {
    let mut a = 0.0;
    let mut ll = 0.0;
    for i in 0..times.len() {
        if i > 0 {
            a = (-beta * (times[i] - times[i - 1])).exp() * (a + 1.0);
        }
        ll += (mu + alpha * beta * a).ln();
    }
    let comp: f64 = times.iter().map(|&ti| 1.0 - (-beta * (t_end - ti)).exp()).sum();
    ll - mu * t_end - alpha * comp
}

/// P(at least one event in bars t+1..=t+h | events up to and including bar t).
/// Uses the expected count from the baseline plus the decaying excitation of past events.
pub fn prob(events: &[bool], mu: f64, alpha: f64, beta: f64, h: f64) -> Vec<f64> {
    let decay = (-beta).exp();
    let tail = 1.0 - (-beta * h).exp();
    let mut s = 0.0;
    events
        .iter()
        .map(|&ev| {
            s = s * decay + ev as u8 as f64;
            1.0 - (-(mu * h + alpha * s * tail)).exp()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_limit_and_excitation() {
        // alpha = 0: a homogeneous Poisson process, P = 1 - exp(-mu h) whatever the history
        let p = prob(&[true, false, true], 0.01, 0.0, 0.5, 10.0);
        assert!(p.iter().all(|&x| (x - (1.0 - (-0.1f64).exp())).abs() < 1e-12));
        // an event raises the probability, which then decays back toward the baseline
        let q = prob(&[false, true, false, false], 0.01, 0.5, 0.5, 10.0);
        assert!(q[1] > q[2] && q[2] > q[3] && q[3] > q[0]);
    }

    #[test]
    fn loglik_of_no_events_is_minus_compensator() {
        assert!((loglik(&[], 100.0, 0.02, 0.3, 0.1) + 2.0).abs() < 1e-12);
    }
}
