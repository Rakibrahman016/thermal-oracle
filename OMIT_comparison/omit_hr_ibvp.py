#!/usr/bin/env python3
"""
OMIT + Welch heart-rate pipeline, reimplemented and run on iBVP.

"""

import os
import re
import csv
import glob
import numpy as np
from scipy.signal import butter, filtfilt, get_window, medfilt
from scipy.interpolate import CubicSpline

# --------------------------------------------------------------- config

# Set these to your local paths before running.
# CACHE:   rPPG-Toolbox preprocessed cache for the fold you are analysing
#          (contains <video>_input<N>.npy and <video>_label<N>.npy)
# DATASET: root of iBVP_Dataset_ready (one folder per session, e.g. p02_a/)


CACHE = ("")
OUTDIR = ""

FS = 30.0                    # iBVP native rate
SIM1_FS = 7.5                # rate the paper was limited to
WIN_SEC = 15.0               # Welch window, per the paper
STEP_SEC = 1.0               # Welch step, per the paper
PRE_LO, PRE_HI = 0.3, 4.0    # first bandpass
CARD_LO, CARD_HI = 1.0, 3.5  # cardiac bandpass
VALID_LO, VALID_HI = 60, 180 # the paper's valid range
MAX_GAP = 10                 # interpolate runs of invalid samples up to this
MEDFILT_N = 7                # final smoothing
MIN_WINDOWS = 20

# The four ROIs the paper combines for HR, mapped onto the 72x72 crop.
# Proportions follow Sec. III.A: forehead 0.45w x 0.18h, nose 0.30 x 0.15,
# cheeks 0.20 x 0.20.
ROIS = {
    "forehead": (8, 22, 20, 52),
    "nose":     (34, 46, 28, 44),
    "cheek_l":  (36, 52, 12, 26),
    "cheek_r":  (36, 52, 46, 60),
}


# --------------------------------------------------------------- filters
def bandpass(x, lo, hi, fs, order=4):
    """Zero-phase Butterworth bandpass. Returns x unchanged if degenerate."""
    x = np.asarray(x, float)
    ny = fs / 2.0
    lo_n, hi_n = lo / ny, min(hi / ny, 0.99)
    if lo_n <= 0 or hi_n <= lo_n or x.size < 4 * order:
        return x
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, x, method="gust")


def band_peak_power(x, fs, lo=CARD_LO, hi=CARD_HI):
    """Strongest spectral power inside the cardiac band. Used to rank channels."""
    x = np.asarray(x, float)
    if x.size < 16 or x.std() == 0:
        return 0.0
    n = x.size
    sp = np.abs(np.fft.rfft(x * np.hanning(n), n=8192)) ** 2
    f = np.fft.rfftfreq(8192, d=1.0 / fs)
    m = (f >= lo) & (f <= hi)
    return float(sp[m].max()) if m.any() else 0.0


# --------------------------------------------------------------- OMIT
def omit(channels, fs):
    """
    Orthogonal matrix image transformation.

    channels : (C, N) array, one row per ROI trace.

    QR-factorise the channel matrix, project out the dominant component
    (assumed motion-correlated), then return whichever residual channel
    has the strongest cardiac-band peak.
    """
    X = np.asarray(channels, float)
    X = X - X.mean(axis=1, keepdims=True)

    sd = X.std(axis=1)
    keep = sd > 0
    if keep.sum() == 0:
        return np.zeros(X.shape[1]), -1
    X = X[keep] / sd[keep][:, None]
    if X.shape[0] == 1:
        return X[0], 0

    # QR of the channel matrix; Q's first column spans the dominant direction
    Q, _ = np.linalg.qr(X)
    q1 = Q[:, [0]]
    P = np.eye(X.shape[0]) - q1 @ q1.T
    Y = P @ X

    powers = [band_peak_power(Y[i], fs) for i in range(Y.shape[0])]
    best = int(np.argmax(powers))
    return Y[best], best


# --------------------------------------------------------------- Welch
def welch_peak_bpm(x, fs, lo=CARD_LO, hi=CARD_HI):
    """
    Welch spectral estimate over one window, peak picked in the cardiac
    band and refined by parabolic interpolation across the two neighbours.
    """
    x = np.asarray(x, float)
    n = x.size
    if n < 32 or not np.isfinite(x).all() or x.std() == 0:
        return np.nan

    seg = max(64, n // 2)
    nfft = 4096
    win = get_window("hann", seg)
    step = seg // 2
    psd = np.zeros(nfft // 2 + 1)
    count = 0
    for s in range(0, n - seg + 1, step):
        d = x[s:s + seg]
        d = d - d.mean()
        psd += np.abs(np.fft.rfft(d * win, n=nfft)) ** 2
        count += 1
    if count == 0:
        return np.nan
    psd /= count

    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    band = np.where((f >= lo) & (f <= hi))[0]
    if band.size < 3:
        return np.nan
    i = band[int(np.argmax(psd[band]))]

    # parabolic refinement around the discrete peak
    if 0 < i < len(psd) - 1:
        a, b, c = psd[i - 1], psd[i], psd[i + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    df = f[1] - f[0]
    return float((f[i] + delta * df) * 60.0)


def clean_series(bpm, lo=VALID_LO, hi=VALID_HI, max_gap=MAX_GAP, med=MEDFILT_N):
    """Invalid-range rejection, short-gap interpolation, median smoothing."""
    y = np.asarray(bpm, float).copy()
    bad = ~np.isfinite(y) | (y < lo) | (y > hi)
    y[bad] = np.nan

    idx = np.arange(y.size)
    good = np.isfinite(y)
    if good.sum() >= 2:
        # interpolate only across runs no longer than max_gap
        runs, start = [], None
        for k in range(y.size):
            if not good[k] and start is None:
                start = k
            elif good[k] and start is not None:
                runs.append((start, k - 1))
                start = None
        if start is not None:
            runs.append((start, y.size - 1))
        for a, b in runs:
            if (b - a + 1) <= max_gap and a > 0 and b < y.size - 1:
                y[a:b + 1] = np.interp(idx[a:b + 1], idx[good], y[good])

    if np.isfinite(y).sum() >= med:
        filled = np.where(np.isfinite(y), y, np.nanmedian(y))
        sm = medfilt(filled, kernel_size=med)
        y = np.where(np.isfinite(y), sm, np.nan)
    return y


# --------------------------------------------------------------- rates
def to_sim1_rate(x, fs=FS, target=SIM1_FS):
    """
    Decimate to SIM1's 7.5 Hz, then cubic-spline back to 30 Hz — the
    paper's own resampling step, which adds no information.
    """
    x = np.asarray(x, float)
    factor = int(round(fs / target))
    xd = x[::factor]
    if xd.size < 4:
        return x
    t_d = np.arange(xd.size) / target
    t_f = np.arange(x.size) / fs
    t_f = t_f[t_f <= t_d[-1]]
    return CubicSpline(t_d, xd)(t_f)


# --------------------------------------------------------------- data
def load_video(vid):
    """Concatenate all cached chunks for one video -> frames, labels."""
    ins = sorted(glob.glob(os.path.join(CACHE, f"{vid}_input*.npy")),
                 key=lambda p: int(re.search(r"_input(\d+)\.npy$", p).group(1)))
    lbs = sorted(glob.glob(os.path.join(CACHE, f"{vid}_label*.npy")),
                 key=lambda p: int(re.search(r"_label(\d+)\.npy$", p).group(1)))
    if not ins or not lbs:
        return None, None
    k = min(len(ins), len(lbs))
    x = np.concatenate([np.squeeze(np.load(p)) for p in ins[:k]], axis=0)
    y = np.concatenate([np.asarray(np.load(p), float).reshape(-1) for p in lbs[:k]])
    n = min(len(x), len(y))
    return x[:n], y[:n]


def hr_series_from_signal(sig, fs, win_sec=WIN_SEC, step_sec=STEP_SEC):
    win, step = int(win_sec * fs), int(step_sec * fs)
    out = [welch_peak_bpm(sig[s:s + win], fs)
           for s in range(0, len(sig) - win + 1, step)]
    return np.asarray(out, float)


def reference_series(bvp, fs):
    """Reference HR, windowed identically, from the contact BVP."""
    b = bandpass(bvp, CARD_LO, CARD_HI, fs)
    return hr_series_from_signal(b, fs)


# --------------------------------------------------------------- metrics
def metrics(est, ref):
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    n = min(est.size, ref.size)
    est, ref = est[:n], ref[:n]
    m = np.isfinite(est) & np.isfinite(ref)
    if m.sum() < MIN_WINDOWS:
        return dict(mae=np.nan, rmse=np.nan, pcc=np.nan, bias=np.nan, n=int(m.sum()))
    e, r = est[m], ref[m]
    pcc = (float(np.corrcoef(e, r)[0, 1])
           if e.std() > 0 and r.std() > 0 else np.nan)
    return dict(mae=float(np.mean(np.abs(e - r))),
                rmse=float(np.sqrt(np.mean((e - r) ** 2))),
                pcc=pcc,
                bias=float(np.mean(e - r)),
                n=int(m.sum()))


def pool(rows, key):
    v = np.array([r[key] for r in rows if np.isfinite(r[key])], float)
    if v.size == 0:
        return np.nan, np.nan
    if key == "pcc":
        z = np.arctanh(np.clip(v, -0.999, 0.999))
        return float(np.tanh(z.mean())), float(np.tanh(z.std()))
    return float(v.mean()), float(v.std())


# --------------------------------------------------------------- main
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    vids = sorted({os.path.basename(p).split("_input")[0]
                   for p in glob.glob(os.path.join(CACHE, "*_input0.npy"))})
    print(f"{len(vids)} videos in cache")
    print(f"Welch {WIN_SEC:.0f} s window, {STEP_SEC:.0f} s step, "
          f"valid range {VALID_LO}-{VALID_HI} bpm\n")

    est30, est75, refs, chosen = {}, {}, {}, []

    for i, vid in enumerate(vids, 1):
        x, y = load_video(vid)
        if x is None:
            continue
        n = x.shape[0]

        traces = []
        for r0, r1, c0, c1 in ROIS.values():
            traces.append(x[:, r0:r1, c0:c1].reshape(n, -1).mean(axis=1))
        traces = np.vstack(traces)
        del x

        pre = np.vstack([bandpass(t, PRE_LO, PRE_HI, FS) for t in traces])

        pulse30, which = omit(pre, FS)
        pulse30 = bandpass(pulse30, CARD_LO, CARD_HI, FS)
        chosen.append(which)

        pre75 = np.vstack([to_sim1_rate(t) for t in pre])
        pulse75, _ = omit(pre75, FS)
        pulse75 = bandpass(pulse75, CARD_LO, CARD_HI, FS)

        ref = reference_series(y, FS)
        if np.isfinite(ref).sum() < MIN_WINDOWS:
            print(f"  [{i:3d}/{len(vids)}] {vid:10s} skipped (reference too sparse)")
            continue

        refs[vid] = ref
        est30[vid] = clean_series(hr_series_from_signal(pulse30, FS))
        est75[vid] = clean_series(hr_series_from_signal(pulse75, FS))

        m30 = metrics(est30[vid], ref)
        print(f"  [{i:3d}/{len(vids)}] {vid:10s} "
              f"ref {np.nanmean(ref):5.1f} bpm   "
              f"30Hz MAE {m30['mae']:6.2f}   PCC {m30['pcc']:6.3f}   "
              f"ch {list(ROIS)[which] if which >= 0 else 'none'}")

    good = sorted(refs)
    if len(good) < 5:
        print("\nToo few usable videos. Stopping.")
        return

    partner = {v: good[(k + 1) % len(good)] for k, v in enumerate(good)}

    runs = {
        "30 Hz (native)":      [metrics(est30[v], refs[v]) for v in good],
        "7.5 Hz (SIM1 rate)":  [metrics(est75[v], refs[v]) for v in good],
        "30 Hz, wrong subject": [metrics(est30[v], refs[partner[v]]) for v in good],
        "7.5 Hz, wrong subject": [metrics(est75[v], refs[partner[v]]) for v in good],
    }

    print()
    print("=" * 78)
    print("OMIT + WELCH ON iBVP")
    print("=" * 78)
    print(f"{'run':24s} {'MAE':>8s} {'RMSE':>8s} {'PCC':>8s} {'bias':>8s} {'n':>7s}")
    print("-" * 78)

    summary = {}
    for name, rows in runs.items():
        mae, mae_sd = pool(rows, "mae")
        rmse, _ = pool(rows, "rmse")
        pcc, _ = pool(rows, "pcc")
        bias, _ = pool(rows, "bias")
        nw = int(sum(r["n"] for r in rows))
        summary[name] = (mae, mae_sd, rmse, pcc, bias, nw)
        print(f"{name:24s} {mae:8.2f} {rmse:8.2f} {pcc:8.3f} {bias:8.2f} {nw:7d}")

    print()
    print("Paper's SIM1 result for reference:  MAE 13.8 +/- 7.5   "
          "PCC ~ 0   bias +10.3")
    print()

    m30 = summary["30 Hz (native)"]
    m75 = summary["7.5 Hz (SIM1 rate)"]
    c30 = summary["30 Hz, wrong subject"]

    print("Interpretation:")
    if np.isfinite(m30[0]) and np.isfinite(m75[0]):
        d = m75[0] - m30[0]
        if d > 2.0:
            print(f"  Native 30 Hz beats the 7.5 Hz version by {d:.2f} bpm.")
            print("  Frame rate does contribute. Report the size of the effect.")
        else:
            print(f"  30 Hz and 7.5 Hz differ by only {abs(d):.2f} bpm.")
            print("  Frame rate is not the limiting factor.")
    if np.isfinite(m30[3]) and np.isfinite(c30[3]):
        if abs(m30[3]) <= abs(c30[3]) + 0.05:
            print("  Correct-subject pairing is no better than wrong-subject "
                  "pairing.")
            print("  The estimator is not reading cardiac activity.")
        else:
            print("  Correct-subject pairing beats the control. Inspect further.")

    with open(os.path.join(OUTDIR, "omit_hr_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "mae", "mae_sd", "rmse", "pcc", "bias", "windows"])
        for name, s in summary.items():
            w.writerow([name] + [round(v, 4) if isinstance(v, float) else v
                                 for v in s])

    with open(os.path.join(OUTDIR, "omit_hr_per_video.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "run", "mae", "rmse", "pcc", "bias", "n"])
        for name, rows in runs.items():
            for v, r in zip(good, rows):
                w.writerow([v, name, round(r["mae"], 3), round(r["rmse"], 3),
                            round(r["pcc"], 4), round(r["bias"], 3), r["n"]])

    if chosen:
        names = list(ROIS)
        counts = {names[c]: chosen.count(c) for c in set(chosen) if c >= 0}
        print()
        print("OMIT channel selection across videos:")
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {k:10s} {v:3d}")
        print("  A real pulse would favour one region consistently.")

    print(f"\nTables written to {OUTDIR}")


if __name__ == "__main__":
    main()
