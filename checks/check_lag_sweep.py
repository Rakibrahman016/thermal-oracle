#!/usr/bin/env python3


import os
import re
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------- config

# Set these to your local paths before running.
# CACHE:   rPPG-Toolbox preprocessed cache for the fold you are analysing
#          (contains <video>_input<N>.npy and <video>_label<N>.npy)
# DATASET: root of iBVP_Dataset_ready (one folder per session, e.g. p02_a/)


CACHE = ("")
OUTDIR = ""

FS = 30.0                 # Hz
WIN_SEC = 10.0            # HR window length
HOP_SEC = 1.0             # hop between windows
MAX_LAG_SEC = 30          # sweep -30 .. +30 s
HR_LO, HR_HI = 0.75, 3.0  # cardiac band
DETREND_ORDER = 3
MIN_WINDOWS = 40          # skip videos with too little usable overlap

WIN = int(WIN_SEC * FS)
HOP = int(HOP_SEC * FS)
LAGS = np.arange(-MAX_LAG_SEC, MAX_LAG_SEC + 1)   # in window steps (hop = 1 s)

# Same ROI set as check_roi_pulse.py, inside the 72x72 face-aligned crop.
ROIS = {
    "forehead":    (8, 22, 24, 48),
    "periorbital": (24, 32, 16, 56),
    "perinasal":   (34, 46, 28, 44),
    "cheek_left":  (36, 52, 12, 28),
    "cheek_right": (36, 52, 44, 60),
    "full_face":   (10, 62, 10, 62),
}


# --------------------------------------------------------------- helpers
def detrend(x, order=DETREND_ORDER):
    x = np.asarray(x, float).reshape(-1)
    t = np.arange(len(x))
    return x - np.polyval(np.polyfit(t, x, order), t)


def band_hr(x, fs=FS):
    """HR in bpm from the strongest cardiac-band spectral peak."""
    x = np.asarray(x, float).reshape(-1)
    if x.size < 32 or not np.isfinite(x).all() or x.std() == 0:
        return np.nan
    x = detrend(x)
    if x.std() == 0:
        return np.nan
    n = len(x)
    sp = np.abs(np.fft.rfft(x * np.hanning(n), n=8192)) ** 2
    f = np.fft.rfftfreq(8192, d=1.0 / fs)
    band = (f >= HR_LO) & (f <= HR_HI)
    if not band.any() or sp[band].sum() <= 0:
        return np.nan
    return float(f[band][np.argmax(sp[band])] * 60.0)


def hr_series(sig):
    """Sliding-window HR series: 10 s window, 1 s hop."""
    sig = np.asarray(sig, float).reshape(-1)
    out = []
    for s in range(0, len(sig) - WIN + 1, HOP):
        out.append(band_hr(sig[s:s + WIN]))
    return np.asarray(out, float)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    if a[m].std() == 0 or b[m].std() == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    return float(np.mean(np.abs(a[m] - b[m])))


def shift_pair(est, ref, lag):
    """
    Align est against ref at the given lag (in window steps = seconds).

    lag > 0 : thermal lags behind, so est[t] is compared with ref[t - lag].
    lag < 0 : thermal leads.
    Returns the overlapping slices.
    """
    if lag > 0:
        return est[lag:], ref[:len(ref) - lag]
    if lag < 0:
        k = -lag
        return est[:len(est) - k], ref[k:]
    return est, ref


def fisher_mean(rs):
    """Average correlations properly, via Fisher z."""
    rs = np.asarray([r for r in rs if np.isfinite(r)], float)
    if rs.size == 0:
        return np.nan
    z = np.arctanh(np.clip(rs, -0.999, 0.999))
    return float(np.tanh(z.mean()))


def load_video(vid):
    """Concatenate all cached chunks for one video -> (frames, 72, 72), labels."""
    ins = sorted(
        glob.glob(os.path.join(CACHE, f"{vid}_input*.npy")),
        key=lambda p: int(re.search(r"_input(\d+)\.npy$", p).group(1)),
    )
    lbs = sorted(
        glob.glob(os.path.join(CACHE, f"{vid}_label*.npy")),
        key=lambda p: int(re.search(r"_label(\d+)\.npy$", p).group(1)),
    )
    if not ins or not lbs:
        return None, None
    k = min(len(ins), len(lbs))
    x = np.concatenate([np.squeeze(np.load(p)) for p in ins[:k]], axis=0)
    y = np.concatenate([np.asarray(np.load(p), float).reshape(-1) for p in lbs[:k]])
    n = min(len(x), len(y))
    return x[:n], y[:n]


# --------------------------------------------------------------- main
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    inputs = sorted(glob.glob(os.path.join(CACHE, "*_input0.npy")))
    vids = sorted({os.path.basename(p).split("_input")[0] for p in inputs})
    print(f"{len(vids)} videos in cache")
    print(f"window {WIN_SEC:.0f} s, hop {HOP_SEC:.0f} s, "
          f"lags {-MAX_LAG_SEC}..{MAX_LAG_SEC} s\n")

    # ---- stage 1: build per-video HR series -----------------------------
    est_series = {r: {} for r in ROIS}   # roi -> vid -> array
    ref_series = {}                      # vid -> array

    for i, vid in enumerate(vids, 1):
        x, y = load_video(vid)
        if x is None:
            print(f"  [{i:3d}/{len(vids)}] {vid:10s} skipped (missing chunks)")
            continue

        ref = hr_series(y)
        if np.isfinite(ref).sum() < MIN_WINDOWS:
            print(f"  [{i:3d}/{len(vids)}] {vid:10s} skipped "
                  f"(only {int(np.isfinite(ref).sum())} valid reference windows)")
            continue
        ref_series[vid] = ref

        n = x.shape[0]
        for name, (r0, r1, c0, c1) in ROIS.items():
            s = x[:, r0:r1, c0:c1].reshape(n, -1).mean(axis=1)
            est_series[name][vid] = hr_series(s)

        print(f"  [{i:3d}/{len(vids)}] {vid:10s} "
              f"{n:5d} frames, {len(ref):4d} windows, "
              f"ref HR {np.nanmean(ref):5.1f} bpm "
              f"(sd {np.nanstd(ref):4.1f})")

        del x

    good = sorted(ref_series)
    print(f"\n{len(good)} videos usable\n")
    if len(good) < 5:
        print("Too few usable videos. Stopping.")
        return

    # ---- stage 2: lag sweep, real pairs and mismatched controls ---------
    # Mismatched control: thermal from video i against reference from i+1.
    partner = {v: good[(k + 1) % len(good)] for k, v in enumerate(good)}

    rows = []
    curves = {}   # (roi, kind) -> array over LAGS

    print("=" * 78)
    print("LAG SWEEP")
    print("=" * 78)

    for name in ROIS:
        for kind in ("real", "control"):
            r_by_lag, mae_by_lag = [], []
            for lag in LAGS:
                rs, ms = [], []
                for vid in good:
                    est = est_series[name].get(vid)
                    ref = ref_series[partner[vid]] if kind == "control" \
                        else ref_series[vid]
                    if est is None:
                        continue
                    m = min(len(est), len(ref))
                    a, b = shift_pair(est[:m], ref[:m], int(lag))
                    if len(a) < MIN_WINDOWS:
                        continue
                    rs.append(pearson(a, b))
                    ms.append(mae(a, b))
                r_by_lag.append(fisher_mean(rs))
                mae_by_lag.append(np.nanmean(ms) if ms else np.nan)

            r_by_lag = np.asarray(r_by_lag, float)
            mae_by_lag = np.asarray(mae_by_lag, float)
            curves[(name, kind)] = r_by_lag

            if np.isfinite(r_by_lag).any():
                bi = int(np.nanargmax(r_by_lag))
                best_lag, best_r = int(LAGS[bi]), float(r_by_lag[bi])
                r0 = float(r_by_lag[LAGS == 0][0])
                best_mae = float(mae_by_lag[bi])
            else:
                best_lag, best_r, r0, best_mae = 0, np.nan, np.nan, np.nan

            rows.append({
                "roi": name, "kind": kind,
                "r_at_lag0": round(r0, 4),
                "best_lag_s": best_lag,
                "r_at_best_lag": round(best_r, 4),
                "mae_at_best_lag": round(best_mae, 2),
            })

            for lag, r, m in zip(LAGS, r_by_lag, mae_by_lag):
                rows.append({
                    "roi": name, "kind": f"{kind}_curve",
                    "r_at_lag0": "", "best_lag_s": int(lag),
                    "r_at_best_lag": round(r, 4) if np.isfinite(r) else "",
                    "mae_at_best_lag": round(m, 2) if np.isfinite(m) else "",
                })

    # ---- stage 3: verdict table ----------------------------------------
    print(f"{'ROI':14s} {'r @ lag 0':>10s} {'best lag':>9s} {'r @ best':>9s} "
          f"{'ctrl best':>10s} {'beats ctrl?':>12s}")
    print("-" * 78)

    verdict_any = False
    for name in ROIS:
        real = next(r for r in rows if r["roi"] == name and r["kind"] == "real")
        ctrl = next(r for r in rows if r["roi"] == name and r["kind"] == "control")
        beats = (np.isfinite(real["r_at_best_lag"])
                 and np.isfinite(ctrl["r_at_best_lag"])
                 and real["r_at_best_lag"] > ctrl["r_at_best_lag"] + 0.05)
        verdict_any |= bool(beats)
        print(f"{name:14s} {real['r_at_lag0']:10.3f} "
              f"{real['best_lag_s']:8d}s {real['r_at_best_lag']:9.3f} "
              f"{ctrl['r_at_best_lag']:10.3f} {'YES' if beats else 'no':>12s}")

    

    # ---- stage 4: outputs ----------------------------------------------
    with open(os.path.join(OUTDIR, "lag_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ncol = 3
    nrow = int(np.ceil(len(ROIS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                             sharey=True)
    axes = np.array(axes).reshape(-1)
    for ax, name in zip(axes, ROIS):
        ax.plot(LAGS, curves[(name, "real")], lw=1.8, label="real pairs")
        ax.plot(LAGS, curves[(name, "control")], lw=1.4, ls="--",
                color="0.45", label="mismatched subject")
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.axvspan(7, 23, color="tab:orange", alpha=0.12,
                   label="slow thermal response (7-23 s)")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("lag (s): positive = thermal late")
        ax.set_ylabel("mean within-video r")
    for ax in axes[len(ROIS):]:
        ax.axis("off")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Thermal HR vs reference HR across time lags. "
                 "A real delayed signal would lift the solid line "
                 "clearly above the dashed control.", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "lag_sweep.png"), dpi=130)
    plt.close(fig)

    print(f"\nTable and figure written to {OUTDIR}")


if __name__ == "__main__":
    main()
