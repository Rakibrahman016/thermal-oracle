#!/usr/bin/env python3


import os
import glob
import numpy as np

# Set these to your local paths before running.
# CACHE:   rPPG-Toolbox preprocessed cache for the fold you are analysing
#          (contains <video>_input<N>.npy and <video>_label<N>.npy)
# DATASET: root of iBVP_Dataset_ready (one folder per session, e.g. p02_a/)

CACHE = ("")

DATASET = ""

FS = 30.0
HR_LO, HR_HI = 0.75, 3.0      # cardiac band, Hz  (45-180 bpm)
N_VIDEOS = 6                   # how many videos to inspect


def band_peak(x, fs=FS, lo=HR_LO, hi=HR_HI):
    """Dominant frequency in the cardiac band and its share of total power."""
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x - x.mean()
    if x.std() == 0:
        return np.nan, 0.0, 0.0
    n = len(x)
    win = np.hanning(n)
    sp = np.abs(np.fft.rfft(x * win, n=4096)) ** 2
    f = np.fft.rfftfreq(4096, d=1.0 / fs)
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return np.nan, 0.0, 0.0
    total = sp[1:].sum()
    peak_i = np.argmax(sp[band])
    peak_f = f[band][peak_i]
    peak_p = sp[band][peak_i]
    ratio = float(sp[band].sum() / total) if total > 0 else 0.0
    prominence = float(peak_p / sp[band].mean()) if sp[band].mean() > 0 else 0.0
    return float(peak_f), ratio, prominence


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return np.nan
    a, b = a[m], b[m]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def raw_bvp(video_id):
    """Raw BVP straight from the dataset csv."""
    sess = f"{video_id[:-1]}_{video_id[-1]}"
    p = os.path.join(DATASET, sess, f"{sess}_bvp.csv")
    if not os.path.isfile(p):
        return None
    a = np.genfromtxt(p, delimiter=",", names=True)
    return np.asarray(a["BVP"], dtype=float) if "BVP" in (a.dtype.names or ()) else None


def main():
    labels = sorted(glob.glob(os.path.join(CACHE, "*_label0.npy")))
    vids = [os.path.basename(p).split("_label")[0] for p in labels][:N_VIDEOS]
    if not vids:
        raise SystemExit(f"No label chunks found in {CACHE}")

    print("=" * 74)
    print("1. LABELS - do the training targets look like a heartbeat?")
    print("=" * 74)
    print(f"{'video':7s} {'chunk':>5s} {'std':>9s} {'peak Hz':>8s} {'bpm':>7s} "
          f"{'band%':>7s} {'promin':>7s}")
    for v in vids:
        for c in (0, 3):
            p = os.path.join(CACHE, f"{v}_label{c}.npy")
            if not os.path.isfile(p):
                continue
            y = np.load(p).reshape(-1)
            f0, ratio, prom = band_peak(y)
            bpm = f0 * 60 if np.isfinite(f0) else np.nan
            print(f"{v:7s} {c:5d} {y.std():9.4f} {f0:8.3f} {bpm:7.1f} "
                  f"{ratio*100:6.1f}% {prom:7.1f}")

    print()
    print("  A real BVP label has a sharp peak at 45-180 bpm and high prominence.")
    print("  Flat std, or prominence near 1, means the labels are not cardiac.")

    print()
    print("=" * 74)
    print("2. INPUTS - does the thermal video carry variation?")
    print("=" * 74)
    for v in vids[:3]:
        p = os.path.join(CACHE, f"{v}_input0.npy")
        if not os.path.isfile(p):
            continue
        x = np.load(p)
        print(f"{v}: shape={x.shape} dtype={x.dtype}")
        print(f"    overall  min={x.min():+.4f}  max={x.max():+.4f}  "
              f"mean={x.mean():+.4f}  std={x.std():.4f}")
        # per-frame spatial mean over time - should wobble, not be constant
        flat = x.reshape(x.shape[0], -1)
        series = flat.mean(axis=1)
        print(f"    frame-mean over time: std={series.std():.6f}  "
              f"range={series.max()-series.min():+.6f}")
        # how many channels, and are any dead
        if x.ndim == 4:
            for ch in range(x.shape[-1]):
                s = x[..., ch].std()
                print(f"    channel {ch}: std={s:.6f}" + ("   <-- DEAD" if s < 1e-6 else ""))

    print()
    print("  A dead channel or near-zero std means the model saw nothing.")

    print()
    print("=" * 74)
    print("3. ALIGNMENT - does the cached label match the raw BVP csv?")
    print("=" * 74)
    print(f"{'video':7s} {'r at lag 0':>11s} {'best r':>8s} {'best lag':>9s}")
    for v in vids:
        chunks = sorted(glob.glob(os.path.join(CACHE, f"{v}_label*.npy")),
                        key=lambda p: int(p.split("label")[-1].split(".")[0]))
        if not chunks:
            continue
        y = np.concatenate([np.load(p).reshape(-1) for p in chunks])
        raw = raw_bvp(v)
        if raw is None:
            print(f"{v:7s}  no raw bvp")
            continue
        n = min(len(y), len(raw))
        r0 = corr(y[:n], raw[:n])

        # search a shift of +/- 5 seconds
        best_r, best_lag = r0, 0
        for lag in range(-150, 151, 5):
            if lag >= 0:
                a, b = y[:n - lag], raw[lag:n]
            else:
                a, b = y[-lag:n], raw[:n + lag]
            if len(a) < 100:
                continue
            r = corr(a, b)
            if np.isfinite(r) and abs(r) > abs(best_r):
                best_r, best_lag = r, lag
        print(f"{v:7s} {r0:+11.3f} {best_r:+8.3f} {best_lag:9d}")

   


if __name__ == "__main__":
    main()
