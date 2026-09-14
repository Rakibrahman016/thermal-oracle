#!/usr/bin/env python3

import os
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set these to your local paths before running.
# CACHE:   rPPG-Toolbox preprocessed cache for the fold you are analysing
#          (contains <video>_input<N>.npy and <video>_label<N>.npy)
# DATASET: root of iBVP_Dataset_ready (one folder per session, e.g. p02_a/)

CACHE = ("")
DATASET = ""
OUTDIR = ""

FS = 30.0
HR_LO, HR_HI = 0.75, 3.0
DETREND_ORDER = 3

# Regions inside the 72x72 face-aligned crop, as (r0, r1, c0, c1).
# The crop is a face box, so these follow the thesis ROI set.
ROIS = {
    "forehead":    (8, 22, 24, 48),
    "periorbital": (24, 32, 16, 56),
    "perinasal":   (34, 46, 28, 44),
    "cheek_left":  (36, 52, 12, 28),
    "cheek_right": (36, 52, 44, 60),
    "full_face":   (10, 62, 10, 62),
}


def detrend(x, order=DETREND_ORDER):
    x = np.asarray(x, float).reshape(-1)
    t = np.arange(len(x))
    return x - np.polyval(np.polyfit(t, x, order), t)


def band_peak(x, fs=FS):
    """HR in bpm from the strongest cardiac-band peak, plus prominence."""
    x = detrend(x)
    if x.std() == 0:
        return np.nan, 0.0, 0.0
    n = len(x)
    sp = np.abs(np.fft.rfft(x * np.hanning(n), n=8192)) ** 2
    f = np.fft.rfftfreq(8192, d=1.0 / fs)
    band = (f >= HR_LO) & (f <= HR_HI)
    tot = sp[1:].sum()
    if not band.any() or tot <= 0:
        return np.nan, 0.0, 0.0
    i = np.argmax(sp[band])
    return (float(f[band][i] * 60),
            float(sp[band][i] / sp[band].mean()),
            float(sp[band].sum() / tot))


def ref_hr(video_id, n):
    """True HR from the reference BVP over the same window."""
    sess = f"{video_id[:-1]}_{video_id[-1]}"
    p = os.path.join(DATASET, sess, f"{sess}_bvp.csv")
    if not os.path.isfile(p):
        return np.nan
    a = np.genfromtxt(p, delimiter=",", names=True)
    if a.dtype.names is None or "BVP" not in a.dtype.names:
        return np.nan
    bvp = np.asarray(a["BVP"], float)[:n]
    hr, _, _ = band_peak(bvp)
    return hr


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or a[m].std() == 0 or b[m].std() == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    inputs = sorted(glob.glob(os.path.join(CACHE, "*_input0.npy")))
    vids = [os.path.basename(p).split("_input")[0] for p in inputs]
    print(f"{len(vids)} videos available in fold00 cache\n")

    rows = []
    for vid in vids:
        x = np.squeeze(np.load(os.path.join(CACHE, f"{vid}_input0.npy")))
        n = x.shape[0]
        true_hr = ref_hr(vid, n)

        row = {"video": vid, "true_hr": round(true_hr, 2) if np.isfinite(true_hr) else ""}
        for name, (r0, r1, c0, c1) in ROIS.items():
            s = x[:, r0:r1, c0:c1].reshape(n, -1).mean(axis=1)
            hr, prom, share = band_peak(s)
            row[f"{name}_hr"] = round(hr, 2) if np.isfinite(hr) else ""
            row[f"{name}_prom"] = round(prom, 2)
            row[f"{name}_share"] = round(share, 4)
            row[f"{name}_err"] = (round(abs(hr - true_hr), 2)
                                  if np.isfinite(hr) and np.isfinite(true_hr) else "")
        rows.append(row)

        print(f"{vid}  true={true_hr:6.1f}  " +
              "  ".join(f"{k[:4]}={row[k+'_hr'] if False else row[f'{k}_hr']}"
                        for k in ROIS))

    # ------------------------------------------------------------ the test
    print()
    print("=" * 78)
    print("DOES THERMAL HR TRACK TRUE HR ACROSS VIDEOS?")
    print("=" * 78)
    print(f"{'ROI':14s} {'Pearson r':>10s} {'MAE bpm':>9s} {'median prom':>12s} "
          f"{'n':>4s}")

    true = [r["true_hr"] if r["true_hr"] != "" else np.nan for r in rows]
    true = np.array([float(t) if t == t else np.nan for t in true], dtype=float)

    summary = {}
    for name in ROIS:
        est = np.array([float(r[f"{name}_hr"]) if r[f"{name}_hr"] != "" else np.nan
                        for r in rows])
        prom = np.array([r[f"{name}_prom"] for r in rows], dtype=float)
        r = pearson(true, est)
        m = np.isfinite(true) & np.isfinite(est)
        mae = float(np.mean(np.abs(true[m] - est[m]))) if m.sum() else np.nan
        summary[name] = (r, mae)
        print(f"{name:14s} {r:10.3f} {mae:9.2f} {np.median(prom):12.2f} {m.sum():4d}")

    # a dummy that always guesses the mean, for comparison
    if np.isfinite(true).sum():
        const = np.nanmean(true)
        dummy_mae = float(np.nanmean(np.abs(true - const)))
        print(f"{'(dummy: always ' + f'{const:.0f}' + ')':14s} "
              f"{'--':>10s} {dummy_mae:9.2f}")

    # ------------------------------------------------------------ save
    with open(os.path.join(OUTDIR, "roi_pulse.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------ figure
    ncol = 3
    nrow = int(np.ceil(len(ROIS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow))
    axes = np.array(axes).reshape(-1)
    lo, hi = 40, 120
    for ax, name in zip(axes, ROIS):
        est = np.array([float(r[f"{name}_hr"]) if r[f"{name}_hr"] != "" else np.nan
                        for r in rows])
        ax.scatter(true, est, s=26, alpha=0.75)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        rr, mae = summary[name]
        ax.set_title(f"{name}   r={rr:.2f}  MAE={mae:.1f}", fontsize=10)
        ax.set_xlabel("true HR (bpm)")
        ax.set_ylabel("thermal HR (bpm)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    for ax in axes[len(ROIS):]:
        ax.axis("off")
    fig.suptitle("Thermal ROI heart rate vs reference. Points on the diagonal "
                 "mean a real pulse.", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "roi_hr_scatter.png"), dpi=400)
    plt.close(fig)




if __name__ == "__main__":
    main()
