#!/usr/bin/env python3


import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = ""
IN = os.path.join(BASE, "stage2", "stage2_estimates.csv")
OUT = os.path.join(BASE, "stage3")

SQI_COLS = ["snr_db", "bpr", "ipr", "periodicity", "hjorth_m", "hjorth_c",
            "zcr", "kurt", "skew", "spec_ent", "peak_prom", "p95p5_std"]

# for each SQI, does a HIGHER value mean better quality?
SQI_HIGHER_BETTER = {
    "snr_db": True, "bpr": True, "ipr": False, "periodicity": True,
    "hjorth_m": False, "hjorth_c": False, "zcr": False,
    "kurt": None, "skew": None,          # closer to zero is better
    "spec_ent": False, "peak_prom": True, "p95p5_std": False,
}

COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def mae(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    return float(e.mean()) if len(e) else np.nan


def main():
    os.makedirs(OUT, exist_ok=True)
    print("loading table ...")
    df = pd.read_csv(IN)
    print(f"{len(df):,} rows, {df.video.nunique()} videos, "
          f"{df.groupby(['video','win_start_s']).ngroups:,} windows")

    df["combo"] = (df["signal"] + "|" + df["detrend"] + "|" +
                   df["band"] + "|" + df["estimator"])

    # ---------------------------------------------------------------- validity
    print()
    print("=" * 74)
    print("VALIDITY - did the pipeline produce a usable estimate at all?")
    print("=" * 74)
    by_sig = df.groupby("signal").agg(
        valid_rate=("valid", "mean"),
        degenerate_rate=("degenerate", "mean"),
        n=("valid", "size"))
    for s, r in by_sig.iterrows():
        print(f"  {s:16s} valid {r.valid_rate*100:5.1f}%   "
              f"degenerate {r.degenerate_rate*100:5.1f}%   n={int(r.n):,}")

    v = df[df.valid == 1].copy()
    print(f"\n  usable rows: {len(v):,} of {len(df):,}")

    # ---------------------------------------------------------------- dummy
    win = v.groupby(["video", "win_start_s"])["gt_hr"].first()
    gt = win.to_numpy()
    dummy_const = float(np.mean(gt))
    dummy_mae = mae(np.abs(gt - dummy_const))

    print()
    print("=" * 74)
    print("LEVELS")
    print("=" * 74)
    print(f"  DUMMY (always {dummy_const:.1f} bpm, ignores the video)"
          f"          MAE {dummy_mae:7.2f} bpm")

    # ---------------------------------------------------------------- baseline
    combo_mae = v.groupby("combo")["abs_err"].mean().sort_values()
    best_combo = combo_mae.index[0]
    baseline_mae = float(combo_mae.iloc[0])
    print(f"  BASELINE best fixed combination                    "
          f"MAE {baseline_mae:7.2f} bpm")
    print(f"           -> {best_combo}")

    # ---------------------------------------------------------------- oracle A
    idx = v.groupby(["video", "win_start_s"])["abs_err"].idxmin()
    oa = v.loc[idx]
    oracle_a_mae = mae(oa["abs_err"])
    print(f"  ORACLE-A per-window best of {v.combo.nunique()} combinations   "
          f"MAE {oracle_a_mae:7.2f} bpm")

    # ---------------------------------------------------------------- SQI
    base = v[v.combo == best_combo].copy()
    sqi_results = {}
    for col in SQI_COLS:
        if col not in v.columns:
            continue
        sub = v[["video", "win_start_s", "combo", "abs_err", col]].dropna()
        if sub.empty:
            continue
        hb = SQI_HIGHER_BETTER.get(col)
        if hb is None:
            sub["_score"] = -sub[col].abs()      # closer to zero is better
        else:
            sub["_score"] = sub[col] if hb else -sub[col]
        pick = sub.loc[sub.groupby(["video", "win_start_s"])["_score"].idxmax()]
        sqi_results[col] = mae(pick["abs_err"])

    if sqi_results:
        best_sqi = min(sqi_results, key=sqi_results.get)
        print(f"  SQI      per-window choice by quality index only   "
              f"MAE {sqi_results[best_sqi]:7.2f} bpm   (best index: {best_sqi})")

    # ---------------------------------------------------------------- verdict
    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    gap = dummy_mae - oracle_a_mae
    if oracle_a_mae >= dummy_mae:
        print(f"  The ORACLE ceiling ({oracle_a_mae:.2f} bpm) is NO BETTER than a")
        print(f"  constant predictor ({dummy_mae:.2f} bpm).")
        print("  Perfect hindsight selection across every filter and estimator")
        print("  cannot extract heart rate from this signal.")
    else:
        print(f"  Oracle beats dummy by {gap:.2f} bpm.")
        print(f"  Achievable headroom exists; SQI captures "
              f"{(dummy_mae - sqi_results[best_sqi]) / gap * 100:.0f}% of it."
              if sqi_results else "")

    # ---------------------------------------------------------------- oracle B
    print()
    print("=" * 74)
    print("ORACLE-B - discard the worst windows (best fixed combination)")
    print("=" * 74)
    print(f"{'coverage':>9s} {'oracle reject':>14s} {'SQI reject':>12s} "
          f"{'random':>9s}")

    err = base["abs_err"].to_numpy()
    rng = np.random.default_rng(0)
    curve = {}
    sqi_for_reject = best_sqi if sqi_results else None
    for cov in COVERAGES:
        k = max(int(len(err) * cov), 1)
        o = mae(np.sort(err)[:k])                      # oracle: keep smallest
        r = mae(rng.permutation(err)[:k])              # random
        s = np.nan
        if sqi_for_reject and sqi_for_reject in base.columns:
            b = base.dropna(subset=[sqi_for_reject])
            hb = SQI_HIGHER_BETTER.get(sqi_for_reject)
            sc = (b[sqi_for_reject] if hb else
                  -b[sqi_for_reject] if hb is False else -b[sqi_for_reject].abs())
            keep = b.loc[sc.sort_values(ascending=False).index[:max(int(len(b)*cov), 1)]]
            s = mae(keep["abs_err"])
        curve[cov] = (o, s, r)
        print(f"{cov*100:8.0f}% {o:14.2f} {s:12.2f} {r:9.2f}")

    # ---------------------------------------------------------------- by signal
    print()
    print("=" * 74)
    print("PER-SIGNAL ORACLE (best of 24 filter/estimator combos, per window)")
    print("=" * 74)
    print(f"{'signal':16s} {'oracle MAE':>11s} {'baseline MAE':>13s} "
          f"{'valid%':>8s}")
    per_sig = []
    for s, g in v.groupby("signal"):
        i = g.groupby(["video", "win_start_s"])["abs_err"].idxmin()
        om = mae(g.loc[i, "abs_err"])
        bm = float(g.groupby("combo")["abs_err"].mean().min())
        vr = float(df[df.signal == s]["valid"].mean() * 100)
        per_sig.append((s, om, bm, vr))
        print(f"{s:16s} {om:11.2f} {bm:13.2f} {vr:7.1f}%")

    # ---------------------------------------------------------------- estimators
    print()
    print("=" * 74)
    print("BY SPECTRAL ESTIMATOR (pooled over signals and filters)")
    print("=" * 74)
    est = v.groupby("estimator").agg(mae_=("abs_err", "mean"),
                                     n=("abs_err", "size"))
    for e, r in est.sort_values("mae_").iterrows():
        print(f"  {e:8s} MAE {r.mae_:7.2f}   n={int(r.n):,}")

    # ---------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    levels = ["dummy\n(no video)", "baseline\n(fixed)", "SQI\n(deployable)",
              "oracle\n(ceiling)"]
    vals = [dummy_mae, baseline_mae,
            sqi_results[best_sqi] if sqi_results else np.nan, oracle_a_mae]
    cols = ["#999999", "#4C72B0", "#DD8452", "#55A868"]
    ax.bar(levels, vals, color=cols)
    ax.axhline(dummy_mae, color="k", linestyle="--", linewidth=1)
    for i, val in enumerate(vals):
        if np.isfinite(val):
            ax.text(i, val + 0.3, f"{val:.1f}", ha="center", fontsize=10)
    ax.set_ylabel("HR MAE (bpm)")
    ax.set_title("Best achievable vs a predictor that ignores the video")

    ax = axes[1]
    covs = [c * 100 for c in COVERAGES]
    ax.plot(covs, [curve[c][0] for c in COVERAGES], "o-", label="oracle rejection")
    if sqi_for_reject:
        ax.plot(covs, [curve[c][1] for c in COVERAGES], "s-", label="SQI rejection")
    ax.plot(covs, [curve[c][2] for c in COVERAGES], "^--", label="random")
    ax.axhline(dummy_mae, color="k", linestyle=":", label="dummy")
    ax.invert_xaxis()
    ax.set_xlabel("coverage (% of windows kept)")
    ax.set_ylabel("HR MAE (bpm)")
    ax.set_title("Error vs coverage")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "oracle_summary.png"), dpi=400)
    plt.close(fig)

    # ---------------------------------------------------------------- save
    pd.DataFrame(per_sig, columns=["signal", "oracle_mae", "baseline_mae",
                                   "valid_pct"]).to_csv(
        os.path.join(OUT, "per_signal.csv"), index=False)
    combo_mae.to_csv(os.path.join(OUT, "combo_ranking.csv"))
    pd.Series(sqi_results).sort_values().to_csv(
        os.path.join(OUT, "sqi_ranking.csv"))
    pd.DataFrame([{"coverage": c, "oracle": curve[c][0],
                   "sqi": curve[c][1], "random": curve[c][2]}
                  for c in COVERAGES]).to_csv(
        os.path.join(OUT, "coverage_curve.csv"), index=False)

    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(f"dummy    {dummy_mae:.3f}\n")
        f.write(f"baseline {baseline_mae:.3f}  ({best_combo})\n")
        if sqi_results:
            f.write(f"sqi      {sqi_results[best_sqi]:.3f}  ({best_sqi})\n")
        f.write(f"oracle   {oracle_a_mae:.3f}\n")

    print(f"\nfigures and tables written to {OUT}")


if __name__ == "__main__":
    main()
