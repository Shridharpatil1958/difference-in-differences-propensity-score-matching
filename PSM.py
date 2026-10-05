"""
Method 2: Propensity Score Matching (store level)

Unit = one store. Covariates are PRE-promo only (never use post-period info).
Outcome = change in mean log sales (post minus pre).
Estimand = ATT (effect on stores that actually got the promo).

Steps: logistic propensity model -> 1:1 nearest-neighbour matching with caliper
       -> balance check (standardized mean difference) -> ATT + bootstrap CI.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def build_store_table(df, promo_week, pre_window=20):
    pre = df[df.week < promo_week]
    post = df[df.week >= promo_week]
    recent = pre[pre.week >= promo_week - pre_window]

    def slope(g):
        return np.polyfit(g.week, g.log_sales, 1)[0]

    s = pd.DataFrame({
        "treated": df.groupby("store_id").treated.first(),
        "pre_mean": pre.groupby("store_id").log_sales.mean(),
        "pre_trend": recent.groupby("store_id").apply(slope, include_groups=False),
        "competitor": df.groupby("store_id").competitor.first(),
        "region": df.groupby("store_id").region.first(),
    })
    for r in range(4):
        s[f"region_{r}"] = (s.region == r).astype(int)
    s["delta"] = post.groupby("store_id").log_sales.mean() - pre.groupby("store_id").log_sales.mean()
    return s


FEATURES = ["pre_mean", "pre_trend", "competitor", "region_0", "region_1", "region_2"]


def fit_propensity(s):
    X = StandardScaler().fit_transform(s[FEATURES])
    lr = LogisticRegression(max_iter=1000).fit(X, s.treated)
    s = s.copy()
    s["ps"] = lr.predict_proba(X)[:, 1]
    return s


def match(s, caliper_sd=0.2):
    """1:1 nearest neighbour on logit(ps), with replacement, within caliper."""
    s = s.copy()
    s["logit_ps"] = np.log(s.ps / (1 - s.ps))
    cal = caliper_sd * s.logit_ps.std()
    treated = s[s.treated == 1]
    control = s[s.treated == 0]
    pairs = []
    for i, row in treated.iterrows():
        d = (control.logit_ps - row.logit_ps).abs()
        j = d.idxmin()
        if d[j] <= cal:
            pairs.append((i, j))
    return pairs


def smd(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


def balance_table(s, pairs):
    t_all, c_all = s[s.treated == 1], s[s.treated == 0]
    t_m = s.loc[[p[0] for p in pairs]]
    c_m = s.loc[[p[1] for p in pairs]]
    return pd.DataFrame({
        "SMD_before": [smd(t_all[f], c_all[f]) for f in FEATURES],
        "SMD_after": [smd(t_m[f], c_m[f]) for f in FEATURES],
    }, index=FEATURES)


def att(s, pairs):
    t = s.loc[[p[0] for p in pairs], "delta"].values
    c = s.loc[[p[1] for p in pairs], "delta"].values
    return (t - c).mean()


def bootstrap_ci(s, pairs, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = s.loc[[p[0] for p in pairs], "delta"].values
    c = s.loc[[p[1] for p in pairs], "delta"].values
    diffs = t - c
    boots = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n)]
    return np.percentile(boots, [2.5, 97.5])


def plot_balance(bal, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(bal))
    ax.scatter(bal.SMD_before, y, label="Before matching", marker="o")
    ax.scatter(bal.SMD_after, y, label="After matching", marker="s")
    ax.axvline(0, color="grey", lw=1)
    ax.axvline(0.1, color="red", ls="--", lw=0.8)
    ax.axvline(-0.1, color="red", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(bal.index)
    ax.set_xlabel("Standardized mean difference (|SMD| < 0.1 is good)")
    ax.set_title("Covariate balance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
  
