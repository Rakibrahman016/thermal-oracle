#!/usr/bin/env python3


import os
import glob
import csv
import numpy as np
from scipy import signal as sps
from scipy.linalg import toeplitz, eigh

# ---------------------------------------------------------------- config
BASE = ""
SIGDIR = os.path.join(BASE, "stage1", "signals")
CACHE = ("")
OUTDIR = os.path.join(BASE, "stage2")

FS = 30.0
WIN_SEC = 10.0                 # window length
STEP_SEC = 1.0                 # hop
N_CHUNKS = 4                   # 4 x 600 frames = 80 s of ROI signal per video
HR_MIN, HR_MAX = 45.0, 180.0   # plausible HR range, bpm

# bandpass variants (Hz) - the "filter bank"
BANDS = {
    "bp_075_300": (0.75, 3.00),
    "bp_070_400": (0.70, 4.00),
    "bp_060_250": (0.60, 2.50),
}
DETRENDS = {"dt1": 1, "dt3": 3}

# ROI boxes inside the 72x72 face-aligned crop
ROIS = {
    "forehead":    (8, 22, 24, 48),
    "periorbital": (24, 32, 16, 56),
    "perinasal":   (34, 46, 28, 44),
    "cheek_left":  (36, 52, 12, 28),
    "cheek_right": (36, 52, 44, 60),
    "full_face":   (10, 62, 10, 62),
}


# ---------------------------------------------------------------- signal utils
def detrend_poly(x, order):
    t = np.arange(len(x))
    return x - np.polyval(np.polyfit(t, x, order), t)


def bandpass(x, lo, hi, fs=FS, order=2):
    ny = fs / 2.0
    lo_n, hi_n = max(lo / ny, 1e-4), min(hi / ny, 0.999)
    if lo_n >= hi_n:
        return x
    b, a = sps.butter(order, [lo_n, hi_n], btype="band")
    try:
        return sps.filtfilt(b, a, x)
    except Exception:
        return x


# ---------------------------------------------------------------- estimators
def hr_welch(x, fs=FS, lo=0.75, hi=3.0):
    nper = min(len(x), int(8 * fs))
    f, p = sps.welch(x, fs=fs, nperseg=nper, nfft=4096)
    return _peak_bpm(f, p, lo, hi)


def hr_fft(x, fs=FS, lo=0.75, hi=3.0):
    n = len(x)
    p = np.abs(np.fft.rfft(x * np.hanning(n), n=4096)) ** 2
    f = np.fft.rfftfreq(4096, d=1.0 / fs)
    return _peak_bpm(f, p, lo, hi)


def hr_music(x, fs=FS, lo=0.75, hi=3.0, p_order=2):
    """Single-channel MUSIC via a lagged autocorrelation matrix."""
    n = len(x)
    M = 2 * p_order
    if n <= M + 2:
        return np.nan
    r = np.array([np.dot(x[:n - k], x[k:]) / n for k in range(M)])
    R = toeplitz(r)
    try:
        w, v = eigh(R)
    except Exception:
        return np.nan
    noise = v[:, :M - p_order]              # smallest eigenvectors
    f = np.linspace(lo, hi, 2048)
    m = np.arange(M)[:, None]
    A = np.exp(-2j * np.pi * m * f[None, :] / fs)
    proj = np.abs(noise.conj().T @ A) ** 2
    denom = proj.sum(axis=0)
    denom[denom <= 0] = np.nan
    ps = 1.0 / denom
    if not np.isfinite(ps).any():
        return np.nan
    return float(f[np.nanargmax(ps)] * 60)


def hr_peak(x, fs=FS, lo=0.75, hi=3.0):
    """Peak counting - median inter-peak interval."""
    min_dist = int(fs / hi)
    pk, _ = sps.find_peaks(x, distance=max(min_dist, 1))
    if len(pk) < 3:
        return np.nan
    ibi = np.diff(pk) / fs
    ibi = ibi[(ibi > 1.0 / hi) & (ibi < 1.0 / lo)]
    if len(ibi) < 2:
        return np.nan
    return float(60.0 / np.median(ibi))


def _peak_bpm(f, p, lo, hi):
    band = (f >= lo) & (f <= hi)
    if not band.any() or not np.isfinite(p[band]).any():
        return np.nan
    return float(f[band][np.argmax(p[band])] * 60)


ESTIMATORS = {"welch": hr_welch, "fft": hr_fft, "music": hr_music, "peak": hr_peak}


# ---------------------------------------------------------------- quality indices
def quality_indices(x, fs=FS, lo=0.75, hi=3.0):
    """Per-window SQIs. Higher is not always better - direction handled later."""
    out = {}
    n = len(x)
    if n < 10 or np.std(x) == 0:
        keys = ["snr_db", "bpr", "ipr", "periodicity", "hjorth_m", "hjorth_c",
                "zcr", "kurt", "skew", "spec_ent", "peak_prom"]
        return {k: np.nan for k in keys}

    xz = (x - x.mean()) / x.std()

    # spectrum
    p = np.abs(np.fft.rfft(xz * np.hanning(n), n=4096)) ** 2
    f = np.fft.rfftfreq(4096, d=1.0 / fs)
    band = (f >= lo) & (f <= hi)
    total = p[1:].sum()

    band_power = p[band].sum()
    out["bpr"] = float(band_power / total) if total > 0 else np.nan
    out["ipr"] = float(1.0 - out["bpr"]) if np.isfinite(out["bpr"]) else np.nan

    if band.any() and p[band].mean() > 0:
        pk = p[band].max()
        out["peak_prom"] = float(pk / p[band].mean())
        # SNR: power within +/-0.2 Hz of the peak vs rest of band
        fp = f[band][np.argmax(p[band])]
        near = (f >= fp - 0.2) & (f <= fp + 0.2)
        sig = p[near].sum()
        noise = total - sig
        out["snr_db"] = float(10 * np.log10(sig / noise)) if noise > 0 else np.nan
    else:
        out["peak_prom"] = np.nan
        out["snr_db"] = np.nan

    # spectral entropy - low means one dominant rhythm
    pn = p[1:] / p[1:].sum() if p[1:].sum() > 0 else None
    out["spec_ent"] = (float(-(pn * np.log(pn + 1e-12)).sum() / np.log(len(pn)))
                       if pn is not None else np.nan)

    # periodicity from autocorrelation
    ac = np.correlate(xz, xz, mode="full")[n - 1:]
    ac = ac / (ac[0] if ac[0] != 0 else 1)
    lo_lag, hi_lag = int(fs / hi), int(fs / lo)
    out["periodicity"] = (float(ac[lo_lag:hi_lag].max())
                          if hi_lag < len(ac) and hi_lag > lo_lag else np.nan)

    # Hjorth
    d1 = np.diff(xz)
    d2 = np.diff(d1)
    v0, v1, v2 = xz.var(), d1.var(), d2.var()
    out["hjorth_m"] = float(np.sqrt(v1 / v0)) if v0 > 0 else np.nan
    out["hjorth_c"] = (float(np.sqrt(v2 / v1) / np.sqrt(v1 / v0))
                       if v0 > 0 and v1 > 0 else np.nan)

    out["zcr"] = float(np.mean(np.diff(np.sign(xz)) != 0))
    out["kurt"] = float(((xz ** 4).mean()) - 3.0)
    out["skew"] = float((xz ** 3).mean())
    return out


# ---------------------------------------------------------------- data loading
def load_roi_signals(video_id):
    """Mean value per ROI per frame, from the face-aligned cache crops."""
    parts = []
    for c in range(N_CHUNKS):
        p = os.path.join(CACHE, f"{video_id}_input{c}.npy")
        if not os.path.isfile(p):
            break
        parts.append(np.squeeze(np.load(p)))
    if not parts:
        return {}, 0
    x = np.concatenate(parts, axis=0)          # (T, 72, 72)
    n = x.shape[0]
    sigs = {}
    for name, (r0, r1, c0, c1) in ROIS.items():
        sigs[f"roi_{name}"] = x[:, r0:r1, c0:c1].reshape(n, -1).mean(axis=1)
    # spatial p95-p5 range per frame, from the thesis feature set
    face = x[:, 10:62, 10:62].reshape(n, -1)
    sigs["_p95p5"] = np.percentile(face, 95, axis=1) - np.percentile(face, 5, axis=1)
    return sigs, n


def load_model_signal(video_id):
    p = os.path.join(SIGDIR, f"{video_id}.npz")
    if not os.path.isfile(p):
        return None, None
    d = np.load(p, allow_pickle=True)
    return (np.asarray(d["pred"], float),
            np.asarray(d["label"], float))


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    vids = sorted(os.path.basename(p).split("_input")[0]
                  for p in glob.glob(os.path.join(CACHE, "*_input0.npy")))
    print(f"{len(vids)} videos\n")

    win = int(WIN_SEC * FS)
    step = int(STEP_SEC * FS)

    rows = []
    for vi, vid in enumerate(vids, 1):
        roi_sigs, n_roi = load_roi_signals(vid)
        pred, label = load_model_signal(vid)
        if pred is None or not roi_sigs:
            print(f"[{vi:3d}/{len(vids)}] {vid}: missing data, skipped")
            continue

        n = min(n_roi, len(pred), len(label))
        bank = {"model_pred": pred[:n]}
        p95 = roi_sigs.pop("_p95p5")[:n]
        for k, v in roi_sigs.items():
            bank[k] = v[:n]
        ref = label[:n]

        n_win = 0
        for s0 in range(0, n - win + 1, step):
            s1 = s0 + win
            seg_ref = ref[s0:s1]

            # reference HR for this window, from the label
            ref_bp = bandpass(detrend_poly(seg_ref, 1), 0.75, 3.0)
            gt_hr = hr_welch(ref_bp)
            if not np.isfinite(gt_hr) or not (HR_MIN <= gt_hr <= HR_MAX):
                continue
            n_win += 1

            p95_win = float(np.mean(p95[s0:s1]))
            p95_var = float(np.std(p95[s0:s1]))

            for sig_name, sig in bank.items():
                seg = sig[s0:s1]
                degenerate = int(np.std(seg) == 0)

                for dt_name, dt_order in DETRENDS.items():
                    seg_dt = detrend_poly(seg, dt_order) if not degenerate else seg

                    for band_name, (lo, hi) in BANDS.items():
                        seg_bp = bandpass(seg_dt, lo, hi) if not degenerate else seg_dt
                        sqi = quality_indices(seg_bp, lo=lo, hi=hi)

                        for est_name, est in ESTIMATORS.items():
                            hr = est(seg_bp, lo=lo, hi=hi) if not degenerate else np.nan
                            valid = int(np.isfinite(hr) and HR_MIN <= hr <= HR_MAX)

                            rows.append({
                                "video": vid,
                                "subject": vid[:-1],
                                "condition": vid[-1],
                                "win_start_s": round(s0 / FS, 2),
                                "signal": sig_name,
                                "detrend": dt_name,
                                "band": band_name,
                                "estimator": est_name,
                                "gt_hr": round(gt_hr, 3),
                                "pred_hr": round(hr, 3) if np.isfinite(hr) else "",
                                "abs_err": (round(abs(hr - gt_hr), 3)
                                            if valid else ""),
                                "valid": valid,
                                "degenerate": degenerate,
                                "p95p5_mean": round(p95_win, 3),
                                "p95p5_std": round(p95_var, 3),
                                **{k: (round(v, 5) if np.isfinite(v) else "")
                                   for k, v in sqi.items()},
                            })

        print(f"[{vi:3d}/{len(vids)}] {vid}: {n_win} windows, "
              f"{len(bank)} signals -> {n_win * len(bank) * 24} rows")

    # ------------------------------------------------------------ write
    out = os.path.join(OUTDIR, "stage2_estimates.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows):,} rows written to {out}")
    print(f"combinations per window: {len(BANDS)} bands x {len(DETRENDS)} detrends "
          f"x {len(ESTIMATORS)} estimators = "
          f"{len(BANDS)*len(DETRENDS)*len(ESTIMATORS)} per signal")


if __name__ == "__main__":
    main()
