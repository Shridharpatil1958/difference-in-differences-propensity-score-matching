"""
Method 1: Difference-in-Differences (two-way fixed effects)

 log(sales_it) = store FE + week FE + beta * (treated_i * post_t) + e_it
 beta ~ log-lift caused by the promo. Lift % = exp(beta) - 1.

Also includes:
 - naive before/after estimate (biased)
 - event-study (lead/lag) to check the parallel-trends assumption
 - placebo test using a fake promo date in the pre-period
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def naive_before_after(df):
    """Treated stores only: mean log sales after minus before. Ignores trend/seasonality."""
    t = df[df.treated == 1]
    return t[t.post == 1].log_sales.mean() - t[t.post == 0].log_sales.mean()


def did_twfe(df, outcome="log_sales", treat_col="treat_post"):
    model = smf.ols(f"{outcome} ~ {treat_col} + C(store_id) + C(week)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["store_id"]}
    )
    beta = model.params[treat_col]
    lo, hi = model.conf_int().loc[treat_col]
    return beta, (lo, hi), model.pvalues[treat_col]


def did_with_trends(df, outcome="log_sales", treat_col="treat_post"):
    """DiD + store-specific linear trends. Fixes bias when treated stores were already growing faster."""
    model = smf.ols(f"{outcome} ~ {treat_col} + C(store_id) + C(week) + C(store_id):week", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["store_id"]}
    )
    beta = model.params[treat_col]
    lo, hi = model.conf_int().loc[treat_col]
    return beta, (lo, hi), model.pvalues[treat_col]


def event_study(df, promo_week, window=(-12, 18), ref=-1):
    """Interact treated with relative-week dummies. Pre-period coefficients ~ 0 supports parallel trends."""
    d = df.copy()
    d["rel"] = d["week"] - promo_week
    d = d[(d.rel >= window[0]) & (d.rel <= window[1])]
    rels = [r for r in range(window[0], window[1] + 1) if r != ref]
    for r in rels:
        d[f"r_{r}".replace("-", "m")] = ((d.rel == r) & (d.treated == 1)).astype(int)
    cols = [f"r_{r}".replace("-", "m") for r in rels]
    m = smf.ols(f"log_sales ~ {' + '.join(cols)} + C(store_id) + C(week)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["store_id"]}
    )
    est = pd.DataFrame({
        "rel_week": rels,
        "coef": [m.params[c] for c in cols],
        "lo": [m.conf_int().loc[c, 0] for c in cols],
        "hi": [m.conf_int().loc[c, 1] for c in cols],
    })
    return est


def plot_event_study(est, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(est.rel_week, est.coef, yerr=[est.coef - est.lo, est.hi - est.coef], fmt="o", capsize=3)
    ax.axhline(0, color="grey", lw=1)
    ax.axvline(-0.5, color="red", ls="--", lw=1, label="Promo start")
    ax.set_xlabel("Weeks relative to promo")
    ax.set_ylabel("Effect on log sales (vs week -1)")
    ax.set_title("Event study: pre-trend check + dynamic effect")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_trends(df, promo_week, path):
    g = df.groupby(["week", "treated"]).log_sales.mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(g.index, g[0], label="Control stores")
    ax.plot(g.index, g[1], label="Treated stores")
    ax.axvline(promo_week, color="red", ls="--", label="Promo start")
    ax.set_xlabel("Week")
    ax.set_ylabel("Mean log sales")
    ax.set_title("Raw trends: treated vs control")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def placebo_did(df, promo_week, fake_week):
    """Pretend the promo started at fake_week using PRE-period data only. True effect should be ~0."""
    d = df[df.week < promo_week].copy()
    d["fake_post"] = (d.week >= fake_week).astype(int)
    d["fake_tp"] = d["treated"] * d["fake_post"]
    return did_twfe(d, treat_col="fake_tp")
  
