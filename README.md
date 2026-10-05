# Did the Promo Actually Work? Causal Impact of a Discount Rollout

**Business question:** Sales rose after a discount was rolled out to some stores. How much of that lift was caused by the promo, and did it pay for itself?

## Headline result (simulated data, known true lift = 10%)

| Method | Est. lift | Note |
|---|---|---|
| Naive before/after | 1.3% | Badly wrong: ignores seasonality and trends |
| Diff-in-Diff (TWFE) | 12.0% | Biased up: treated stores were already growing faster |
| DiD + store-specific trends | 10.5% | Fixes the pre-trend problem |
| Propensity matching (ATT) | 10.3% | Matches on pre-period features |
| Synthetic control | 10.1% | Placebo-in-space p = 0.024 |

Promo adds roughly Rs 9.9 lakh incremental revenue vs an assumed Rs 5.2 lakh discount cost -> **keep the promo**.
(Discount cost of 5% of revenue is an assumption, so replace it with real numbers.)

## Why this is interesting
- Treatment was **not random**: bigger, faster-growing stores got the promo (confounding).
- Plain DiD **fails** here because parallel trends is violated. The placebo test and event study expose it, and we fix it.
- Three independent methods agree, which gives confidence.

## Run it
```bash
pip install -r requirements.txt
python run_all.py
```
Outputs (tables + 5 charts) land in `outputs/`.

## Structure
- `src/simulate_data.py`: store-week panel with confounded assignment and a known effect
- `src/did.py`: naive, TWFE DiD, trend-adjusted DiD, event study, placebo
- `src/psm.py`: propensity score, 1:1 matching, SMD balance, bootstrap CI
- `src/synthetic_control.py`: convex-weight synthetic control + placebo-in-space
- `run_all.py`: runs everything and prints the comparison + business verdict

## Next steps (to make it portfolio-grade)
1. Swap in real data (Rossmann Store Sales or Dunnhumby) in place of the simulator.
2. Add heterogeneous effects (by region/store size) and a Streamlit dashboard.
3. Write the discussion: when does each method fail, and what assumptions did you test?
