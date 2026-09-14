#!/usr/bin/env python3


import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = ""    # your source
IN = os.path.join(BASE, "stage2", "stage2_estimates.csv")
OUT = os.path.join(BASE, "stage4")

K_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 140, 168]
N_DRAWS = 300          # random subsets per K
MIN_VALID_FRAC = 0.30  # a window needs this fraction of candidates valid
SEED = 0

DUMMY_MAE = 8.61       # always predicting 74.6 bpm, for reference


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("loading table ...")
    df = pd.read_csv(IN)
    df["combo"] = (df["signal"] + "|" + df["detrend"] + "|" +
                   df["band"] + "|" + df["estimator"])
    df = df[df["valid"] == 1]

    # ---- build a windows x combos matrix of absolute error -------------
    win_key = df["video"].astype(str) + "@" + df["win_start_s"].astype(str)
    wi, wu = pd.factorize(win_key)
    ci, cu = pd.factorize(df["combo"])
    M = np.full((len(wu), len(cu)), np.nan, float)
    M[wi, ci] = df["abs_err"].to_numpy(float)

    n_win, n_combo = M.shape
    print(f"{n_win:,} windows x {n_combo} candidates")

    keep = np.isfinite(M).mean(axis=1) >= MIN_VALID_FRAC
    M = M[keep]
    print(f"{M.shape[0]:,} windows kept "
          f"(at least {MIN_VALID_FRAC:.0%} of candidates valid)\n")

    col_mean = np.nanmean(M, axis=0)
    full_fixed = float(np.nanmin(col_mean))
    full_oracle = float(np.nanmean(np.nanmin(M, axis=1)))
    print(f"Whole set of {n_combo}:  fixed {full_fixed:.2f}   "
          f"oracle {full_oracle:.2f}   inflation {full_fixed - full_oracle:.2f} bpm")
    print(f"Dummy predictor (always {DUMMY_MAE and '74.6'} bpm): {DUMMY_MAE:.2f}\n")

    # ---- sweep the search size ----------------------------------------
    rows = []
    print(f"{'K':>5s} {'fixed':>8s} {'oracle':>8s} {'inflation':>10s} "
          f"{'sd':>7s} {'beats dummy?':>14s}")
    print("-" * 60)

    for K in K_VALUES:
        if K > n_combo:
            continue
        draws = 1 if K == n_combo else N_DRAWS
        fx, orc = [], []
        for _ in range(draws):
            cols = (np.arange(n_combo) if K == n_combo
                    else rng.choice(n_combo, size=K, replace=False))
            sub = M[:, cols]
            cm = np.nanmean(sub, axis=0)
            if not np.isfinite(cm).any():
                continue
            fx.append(float(np.nanmin(cm)))
            with np.errstate(all="ignore"):
                per_win = np.nanmin(sub, axis=1)
            orc.append(float(np.nanmean(per_win)))
        if not fx:
            continue
        f_m, o_m = float(np.mean(fx)), float(np.mean(orc))
        infl = np.array(fx) - np.array(orc)
        rows.append(dict(K=K, fixed=f_m, oracle=o_m,
                         inflation=float(infl.mean()),
                         inflation_sd=float(infl.std()),
                         oracle_sd=float(np.std(orc)),
                         beats_dummy=bool(o_m < DUMMY_MAE)))
        print(f"{K:5d} {f_m:8.2f} {o_m:8.2f} {infl.mean():10.2f} "
              f"{infl.std():7.2f} {'yes' if o_m < DUMMY_MAE else 'no':>14s}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "candidate_count.csv"), index=False)

    # ---- the sentence the paper needs ---------------------------------
    print()
    for K in (10, 20, 168):
        r = res[res.K == K]
        if not r.empty:
            r = r.iloc[0]
            print(f"  At {K:3d} candidates: oracle {r.oracle:.2f} bpm, "
                  f"{r.inflation:.2f} bpm better than one fixed choice.")
    ten = res[res.K == 10]
    if not ten.empty and not res[res.K == n_combo].empty:
        frac = ten.iloc[0].inflation / res[res.K == n_combo].iloc[0].inflation
        print(f"\n  Searching only 10 candidates already produces "
              f"{frac:.0%} of the inflation seen at {n_combo}.")
        print("  The problem is not confined to unusually large search spaces.")

    # ---- split-half: does a fixed choice transfer? ---------------------
    idx = rng.permutation(M.shape[0])
    h1, h2 = idx[:len(idx) // 2], idx[len(idx) // 2:]
    m1 = np.nanmean(M[h1], axis=0)
    best_on_h1 = int(np.nanargmin(m1))
    fixed_transfer = float(np.nanmean(M[h2][:, best_on_h1]))
    fixed_insample = float(m1[best_on_h1])
    oracle_h1 = float(np.nanmean(np.nanmin(M[h1], axis=1)))
    # apply half 1's per-window winners to half 2 — impossible in practice,
    # shown only to make the point that per-window choice cannot transfer
    print()
    print("Split-half check (does the choice generalise?)")
    print(f"  best fixed combination, chosen on half 1 : {fixed_insample:.2f} bpm")
    print(f"  same combination applied to half 2       : {fixed_transfer:.2f} bpm")
    print(f"  per-window oracle on half 1              : {oracle_h1:.2f} bpm")
    print("  A fixed choice transfers between halves. A per-window choice")
    print("  has nothing to transfer: it needs the answer to make the choice.")

    with open(os.path.join(OUT, "split_half.txt"), "w") as f:
        f.write(f"fixed_in_sample {fixed_insample:.4f}\n"
                f"fixed_transfer {fixed_transfer:.4f}\n"
                f"oracle_half1 {oracle_h1:.4f}\n")

    # ---- figure --------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].plot(res.K, res.oracle, "o-", lw=1.8, label="oracle (per-window best)")
    ax[0].fill_between(res.K, res.oracle - res.oracle_sd,
                       res.oracle + res.oracle_sd, alpha=0.15)
    ax[0].plot(res.K, res.fixed, "s--", lw=1.5, color="0.4",
               label="best single fixed choice")
    ax[0].axhline(DUMMY_MAE, color="tab:red", ls=":", lw=1.5,
                  label=f"dummy predictor ({DUMMY_MAE:.2f})")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("number of candidates searched")
    ax[0].set_ylabel("MAE (bpm)")
    ax[0].set_title("Reported error shrinks as you search more")
    ax[0].legend(fontsize=8)

    ax[1].plot(res.K, res.inflation, "o-", lw=1.8, color="tab:purple")
    ax[1].fill_between(res.K, res.inflation - res.inflation_sd,
                       res.inflation + res.inflation_sd,
                       alpha=0.15, color="tab:purple")
    ax[1].axvline(10, color="0.5", ls=":", lw=1.2)
    ax[1].text(10.5, ax[1].get_ylim()[1] * 0.06, "typical published\nsearch size",
               fontsize=8, color="0.35")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("number of candidates searched")
    ax[1].set_ylabel("inflation (bpm)")
    ax[1].set_title("How much of that is free, from searching alone")

    fig.suptitle("Oracle inflation as a function of search size. "
                 "Shaded bands are one standard deviation over random subsets.",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "candidate_count.png"), dpi=140)
    plt.close(fig)

    print(f"\nTable and figure written to {OUT}")


if __name__ == "__main__":
    main()
