# Deviations from the pre-registration (`prereg-v1`)

None. The confirmatory evaluation ran on the registered code without modification:

- All 15 registered series were downloaded; none met the exclusion rule
  (every hourly series had at least 20,000 bars and every daily series at least 3,000).
- `python -m sq.experiments --universe confirmatory --only e1 e2 e3 e4 e5 e8` and
  `python -m sq.hypotheses results/confirmatory` were run at the registered commit.
- No series, model, parameter or test was added, removed or changed.
