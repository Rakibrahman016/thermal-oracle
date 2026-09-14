#!/usr/bin/env python3


import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set these to your local paths before running.
# CACHE:   rPPG-Toolbox preprocessed cache for the fold you are analysing
#          (contains <video>_input<N>.npy and <video>_label<N>.npy)
# DATASET: root of iBVP_Dataset_ready (one folder per session, e.g. p02_a/)

DATASET = ""
CACHE = ("")
OUTDIR = ""

W, H = 640, 512
FS = 30.0
HR_LO, HR_HI = 0.75, 3.0        # 45-180 bpm
N_FRAMES = 1800                 # 60 s is plenty for a spectrum
VIDEOS = ["p02a", "p02b", "p03a", "p03b"]


def read_raw(path):
    a = np.fromfile(path, dtype=np.uint16)
    if a.size != W * H:
        return None
    return a.reshape(H, W).astype(np.float64)


def spectrum(x, fs=FS):
    x = np.asarray(x, float).reshape(-1)
    x = x - x.mean()
    if x.std() == 0:
        return None, None
    # remove slow drift - thermal has strong low-frequency trends
    n = len(x)
    t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 3), t)
    sp = np.abs(np.fft.rfft(x * np.hanning(n), n=8192)) ** 2
    f = np.fft.rfftfreq(8192, d=1.0 / fs)
    return f, sp


def band_report(x, fs=FS):
    """Peak frequency in the cardiac band, its prominence, and band share."""
    f, sp = spectrum(x, fs)
    if f is None:
        return dict(bpm=np.nan, prom=0.0, share=0.0)
    band = (f >= HR_LO) & (f <= HR_HI)
    tot = sp[1:].sum()
    if not band.any() or tot <= 0:
        return dict(bpm=np.nan, prom=0.0, share=0.0)
    i = np.argmax(sp[band])
    peak = sp[band][i]
    return dict(bpm=float(f[band][i] * 60),
                prom=float(peak / sp[band].mean()),
                share=float(sp[band].sum() / tot))


def ref_bvp(video_id, n):
    sess = f"{video_id[:-1]}_{video_id[-1]}"
    p = os.path.join(DATASET, sess, f"{sess}_bvp.csv")
    if not os.path.isfile(p):
        return None
    a = np.genfromtxt(p, delimiter=",", names=True)
    if a.dtype.names is None or "BVP" not in a.dtype.names:
        return None
    return np.asarray(a["BVP"], float)[:n]


def roi_series(frames, box):
    """Mean value inside a box, per frame. box = (r0, r1, c0, c1)."""
    r0, r1, c0, c1 = box
    return np.array([f[r0:r1, c0:c1].mean() for f in frames])


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("=" * 86)
    print("CARDIAC BAND CHECK  (prominence >> 1 and a plausible bpm means a real peak)")
    print("=" * 86)
    print(f"{'video':6s} {'source':22s} {'bpm':>7s} {'promin':>8s} {'band share':>11s}")

    for vid in VIDEOS:
        sess = f"{vid[:-1]}_{vid[-1]}"
        tdir = os.path.join(DATASET, sess, f"{sess}_t")
        files = sorted(glob.glob(os.path.join(tdir, "*.raw")),
                       key=lambda p: int(os.path.basename(p)[:-4]))[:N_FRAMES]
        if not files:
            print(f"{vid}: no raw frames")
            continue

        frames = []
        for p in files:
            f = read_raw(p)
            if f is not None:
                frames.append(f)
        if len(frames) < 300:
            print(f"{vid}: too few readable frames")
            continue
        frames = np.stack(frames)
        n = len(frames)

        # --- reference
        bvp = ref_bvp(vid, n)
        if bvp is not None:
            r = band_report(bvp)
            print(f"{vid:6s} {'reference BVP':22s} {r['bpm']:7.1f} "
                  f"{r['prom']:8.1f} {r['share']*100:10.1f}%")

        # --- 16-bit raw, several regions.
        # boxes are fractions of the frame, so they roughly follow the face
        # without needing landmarks: centre block, upper block, whole frame.
        boxes = {
            "16bit whole frame": (0, H, 0, W),
            "16bit centre": (H // 3, 2 * H // 3, W // 3, 2 * W // 3),
            "16bit upper centre": (H // 4, H // 2, 2 * W // 5, 3 * W // 5),
        }
        series = {}
        for name, box in boxes.items():
            s = roi_series(frames, box)
            series[name] = s
            r = band_report(s)
            print(f"{vid:6s} {name:22s} {r['bpm']:7.1f} "
                  f"{r['prom']:8.1f} {r['share']*100:10.1f}%")

        # --- the 8-bit crop the model actually got
        cpath = os.path.join(CACHE, f"{vid}_input0.npy")
        if os.path.isfile(cpath):
            x = np.squeeze(np.load(cpath))
            s8 = x.reshape(x.shape[0], -1).mean(axis=1)
            series["8bit crop (model input)"] = s8
            r = band_report(s8)
            print(f"{vid:6s} {'8bit crop (model in)':22s} {r['bpm']:7.1f} "
                  f"{r['prom']:8.1f} {r['share']*100:10.1f}%")

        # --- quantisation check: how many distinct values are there?
        centre = frames[:, H // 3:2 * H // 3, W // 3:2 * W // 3]
        step = np.median(np.abs(np.diff(np.unique(centre.ravel()))))
        cs = series["16bit centre"]
        drift = np.abs(np.polyval(np.polyfit(np.arange(n), cs, 1),
                                  [0, n - 1]))
        print(f"{vid:6s} {'--- 16bit levels in centre: ':22s}"
              f"{len(np.unique(centre.ravel())):d}, "
              f"step {step:.1f}, "
              f"per-frame change std {np.diff(cs).std():.3f}")

        # --- plot spectra together
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name, s in series.items():
            f, sp = spectrum(s)
            if f is None:
                continue
            m = (f > 0.3) & (f < 4.0)
            y = sp[m] / sp[m].max()
            ax.plot(f[m] * 60, y, label=name, linewidth=1.2)
        if bvp is not None:
            f, sp = spectrum(bvp)
            m = (f > 0.3) & (f < 4.0)
            ax.plot(f[m] * 60, sp[m] / sp[m].max(), "k--",
                    label="reference BVP", linewidth=1.6)
        ax.axvspan(HR_LO * 60, HR_HI * 60, color="grey", alpha=0.12)
        ax.set_xlabel("bpm")
        ax.set_ylabel("normalised power")
        ax.set_title(f"{vid} - is there a cardiac peak in the thermal signal?")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"{vid}_spectra.png"), dpi=110)
        plt.close(fig)
        print()



if __name__ == "__main__":
    main()
