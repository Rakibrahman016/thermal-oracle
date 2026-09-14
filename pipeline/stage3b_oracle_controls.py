#!/usr/bin/env python3


import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = ""
REAL = os.path.join(BASE, "stage2", "stage2_estimates.csv")
CTRL = os.path.join(BASE, "stage2", "stage2_controls.csv")
OUT = os.path.join(BASE, "stage3b")

SQI_COLS = ["snr_db", "bpr", "ipr", "periodicity", "hjorth_m", "hjorth_c",
            "zcr", "kurt", "skew", "spec_ent", "peak_prom"]
SQI_HIGHER_BETTER = {
    "snr_db": True, "bpr": True, "ipr": False, "periodicity": True,
    "hjorth_m": False, "hjorth_c": False, "zcr": False,
    "kurt": None, "skew": None, "spec_ent": False, "peak_prom": True,
}
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def mae(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    return float(e.mean()) if len(e) else np.nan


def main():
    os.makedirs(OUT, exist_ok=True)

    print("loading ...")
    df = pd.read_csv(REAL)
    df["kind"] = "real"
    if os.path.isfile(CTRL):
        dc = pd.read_csv(CTRL)
        dc["kind"] = "control"
        df = pd.concat([df, dc], ignore_index=True)
        print(f"  real {len(df[df.kind=='real']):,} + "
              f"control {len(dc):,} rows")
    else:
        print("  WARNING: no control file, run stage2b_controls.py first")

    df["combo"] = (df["signal"] + "|" + df["detrend"] + "|" +
                   df["band"] + "|" + df["estimator"])
    v = df[df.valid == 1].copy()

    # ---------------------------------------------------------------- dummy
    gt = v.groupby(["video", "win_start_s"])["gt_hr"].first().to_numpy()
    dummy_const = float(np.mean(gt))
    dummy_mae = mae(np.abs(gt - dummy_const))

    # ---------------------------------------------------------------- table
    print()
    print("=" * 84)
    print("ORACLE vs BASELINE, REAL SIGNALS AND CONTROLS")
    print("=" * 84)
    print(f"{'signal':18s} {'kind':8s} {'oracle MAE':>11s} {'baseline MAE':>13s} "
          f"{'valid%':>8s} {'n win':>8s}")

    recs = []
    for (sig, kind), g in v.groupby(["signal", "kind"]):
        idx = g.groupby(["video", "win_start_s"])["abs_err"].idxmin()
        om = mae(g.loc[idx, "abs_err"])
        bm = float(g.groupby("combo")["abs_err"].mean().min())
        vr = float(df[(df.signal == sig)]["valid"].mean() * 100)
        nw = g.groupby(["video", "win_start_s"]).ngroups
        recs.append(dict(signal=sig, kind=kind, oracle_mae=om,
                         baseline_mae=bm, valid_pct=vr, n_windows=nw))

    recs.sort(key=lambda r: (r["kind"], r["oracle_mae"]))
    for r in recs:
        print(f"{r['signal']:18s} {r['kind']:8s} {r['oracle_mae']:11.2f} "
              f"{r['baseline_mae']:13.2f} {r['valid_pct']:7.1f}% "
              f"{r['n_windows']:8,d}")

    print(f"\n{'DUMMY (no video)':18s} {'--':8s} {'--':>11s} "
          f"{dummy_mae:13.2f}")

    # ---------------------------------------------------------------- verdict
    real_o = [r["oracle_mae"] for r in recs if r["kind"] == "real"]
    ctrl_o = [r["oracle_mae"] for r in recs if r["kind"] == "control"]
    real_b = [r["baseline_mae"] for r in recs if r["kind"] == "real"]
    ctrl_b = [r["baseline_mae"] for r in recs if r["kind"] == "control"]

    print()
    print("=" * 84)
    print("VERDICT")
    print("=" * 84)
    if ctrl_o:
        print(f"  best oracle,   real signals : {min(real_o):6.2f} bpm")
        print(f"  best oracle,   CONTROLS     : {min(ctrl_o):6.2f} bpm")
        print(f"  best baseline, real signals : {min(real_b):6.2f} bpm")
        print(f"  best baseline, CONTROLS     : {min(ctrl_b):6.2f} bpm")
        print(f"  dummy (ignores video)       : {dummy_mae:6.2f} bpm")
        print()
        if min(ctrl_o) <= min(real_o) * 1.5:
            print("  Controls containing NO cardiac information reach an oracle")
            print("  score comparable to the real signals. Oracle selection is")
            print("  therefore measuring candidate-set density, not signal content.")
            print("  The only meaningful ceiling is the fixed-configuration")
            print(f"  baseline, and that ({min(real_b):.2f} bpm) is WORSE than a")
            print(f"  constant predictor ({dummy_mae:.2f} bpm).")
        else:
            print("  Real signals beat the controls at oracle selection, so some")
            print("  genuine cardiac information is present.")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    order = sorted(recs, key=lambda r: r["oracle_mae"])
    names = [r["signal"] for r in order]
    vals = [r["oracle_mae"] for r in order]
    cols = ["#C44E52" if r["kind"] == "control" else "#4C72B0" for r in order]
    ax.barh(names, vals, color=cols)
    ax.axvline(dummy_mae, color="k", linestyle="--", linewidth=1.2,
               label=f"dummy {dummy_mae:.1f}")
    ax.set_xlabel("oracle MAE (bpm)  -  lower looks better")
    ax.set_title("Oracle selection\nred = control with no cardiac content")
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    ax = axes[1]
    order = sorted(recs, key=lambda r: r["baseline_mae"])
    names = [r["signal"] for r in order]
    vals = [r["baseline_mae"] for r in order]
    cols = ["#C44E52" if r["kind"] == "control" else "#4C72B0" for r in order]
    ax.barh(names, vals, color=cols)
    ax.axvline(dummy_mae, color="k", linestyle="--", linewidth=1.2,
               label=f"dummy {dummy_mae:.1f}")
    ax.set_xlabel("baseline MAE (bpm)  -  best fixed configuration")
    ax.set_title("Deployable performance\n(no hindsight)")
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "oracle_vs_controls.png"), dpi=120)
    plt.close(fig)

    pd.DataFrame(recs).to_csv(os.path.join(OUT, "oracle_vs_controls.csv"),
                              index=False)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
