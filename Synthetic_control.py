"""
Method 3: Synthetic Control (a CausalImpact-style counterfactual)

Idea: build a weighted mix of untreated "donor" stores that tracks the treated
group's PRE-promo path. Project that mix forward as the counterfactual
("what would have happened without the promo"). Gap = causal effect.

Weights are >= 0 and sum to 1 (convex combination), fit by constrained least squares.
Series are demeaned by their pre-period mean, so we match *shape*, not level
(a "demeaned" synthetic control, more robust when treated stores are larger).

Inference: placebo-in-space. Run the same procedure on every donor as if it were
treated, then compare the real post/pre RMSPE ratio against that distribution.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def wide_panel(df):
    return df.pivot(index="week", columns="store_id", values="log_sales")


def fit_weights(y_pre, X_pre):
    """y_pre: (T,) treated series; X_pre: (T, J) donor series. Returns (J,) weights."""
    J = X_pre.shape[1]
    w0 = np.full(J, 1 / J)
    res = minimize(
        lambda w: np.sum((y_pre - X_pre @ w) ** 2),
        w0,
        method="SLSQP",
        bounds=[(0, 1)] * J,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"maxiter": 500},
    )
    return res.x


def synth(y, donors, promo_week):
    """y: Series (weeks,), donors: DataFrame (weeks x J). Returns dict of results."""
    pre = y.index < promo_week
    y_c = y - y[pre].mean()
    d_c = donors - donors[pre].mean()
    w = fit_weights(y_c[pre].values, d_c[pre].values)
    synth_path = d_c.values @ w
    gap = y_c.values - synth_path
    rmspe_pre = np.sqrt(np.mean(gap[pre] ** 2))
    rmspe_post = np.sqrt(np.mean(gap[~pre] ** 2))
    return {
        "weights": pd.Series(w, index=donors.columns),
        "gap": pd.Series(gap, index=y.index),
        "synth": pd.Series(synth_path, index=y.index),
        "actual": y_c,
        "att": gap[~pre].mean(),
        "rmspe_pre": rmspe_pre,
        "rmspe_ratio": rmspe_post / max(rmspe_pre, 1e-8),
    }


def run(df, promo_week, n_placebos=40, seed=0):
    wide = wide_panel(df)
    treated_ids = df[df.treated == 1].store_id.unique()
    control_ids = df[df.treated == 0].store_id.unique()

    y = wide[treated_ids].mean(axis=1)        # average treated store
    donors = wide[control_ids]
    real = synth(y, donors, promo_week)

    # placebo-in-space: pretend a random donor is "treated"
    rng = np.random.default_rng(seed)
    picks = rng.choice(control_ids, size=min(n_placebos, len(control_ids)), replace=False)
    placebo_gaps, placebo_ratios = {}, []
    for p in picks:
        others = donors.drop(columns=p)
        r = synth(donors[p], others, promo_week)
        placebo_gaps[p] = r["gap"]
        placebo_ratios.append(r["rmspe_ratio"])
    p_value = (np.sum(np.array(placebo_ratios) >= real["rmspe_ratio"]) + 1) / (len(placebo_ratios) + 1)
    return real, pd.DataFrame(placebo_gaps), p_value


def plot_synth(real, promo_week, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(real["actual"], label="Treated (actual)")
    ax[0].plot(real["synth"], "--", label="Synthetic (counterfactual)")
    ax[0].axvline(promo_week, color="red", ls="--", lw=1)
    ax[0].set_title("Actual vs synthetic control (demeaned log sales)")
    ax[0].legend()
    ax[1].plot(real["gap"], color="green")
    ax[1].axhline(0, color="grey", lw=1)
    ax[1].axvline(promo_week, color="red", ls="--", lw=1)
    ax[1].set_title("Gap = estimated promo effect over time")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_placebos(real, placebo_gaps, promo_week, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in placebo_gaps.columns:
        ax.plot(placebo_gaps[c], color="lightgrey", lw=0.8)
    ax.plot(real["gap"], color="green", lw=2, label="Treated group")
    ax.axvline(promo_week, color="red", ls="--", lw=1)
    ax.axhline(0, color="grey", lw=1)
    ax.set_title("Placebo-in-space test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
  
