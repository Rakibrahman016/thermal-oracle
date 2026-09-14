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

CACHE = ("")

OUTDIR = ""

N_VIDEOS = 4          # how many videos to inspect
FRAMES = [0, 100, 200, 300, 400, 500]   # which frames to show


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    inputs = sorted(glob.glob(os.path.join(CACHE, "*_input0.npy")))
    if not inputs:
        raise SystemExit(f"No input chunks in {CACHE}")
    vids = [os.path.basename(p).split("_input")[0] for p in inputs][:N_VIDEOS]

    print("=" * 78)
    print("WHAT THE MODEL SEES  (EfficientPhys takes torch.diff internally)")
    print("=" * 78)
    print(f"{'video':7s} {'raw std':>9s} {'diff std':>9s} {'diff/raw':>9s} "
          f"{'sat lo%':>8s} {'sat hi%':>8s} {'centre-edge':>12s}")

    for v in vids:
        x = np.load(os.path.join(CACHE, f"{v}_input0.npy"))
        x = np.squeeze(x)                       # (600, 72, 72)

        raw_std = float(x.std())

        # the actual model input: consecutive frame differences
        d = np.diff(x, axis=0)
        diff_std = float(d.std())
        ratio = diff_std / raw_std if raw_std > 0 else 0.0

        # saturation - clipped pixels carry no information
        sat_lo = float((x <= 1).mean() * 100)
        sat_hi = float((x >= 254).mean() * 100)

        # is there a face? a centred face should be warmer than the border
        mean_img = x.mean(axis=0)
        c = mean_img[18:54, 18:54].mean()          # central half
        border = np.concatenate([mean_img[:10].ravel(), mean_img[-10:].ravel(),
                                 mean_img[:, :10].ravel(), mean_img[:, -10:].ravel()])
        contrast = float(c - border.mean())

        print(f"{v:7s} {raw_std:9.2f} {diff_std:9.3f} {ratio:9.4f} "
              f"{sat_lo:7.2f}% {sat_hi:7.2f}% {contrast:+12.2f}")

        # ---- picture: raw frames on top, differences below
        fig, axes = plt.subplots(2, len(FRAMES), figsize=(2.2 * len(FRAMES), 5))
        for j, fr in enumerate(FRAMES):
            if fr >= len(x):
                continue
            axes[0, j].imshow(x[fr], cmap="inferno")
            axes[0, j].set_title(f"f{fr}", fontsize=8)
            axes[0, j].axis("off")

            if fr < len(d):
                dd = d[fr]
                lim = np.percentile(np.abs(dd), 99) or 1.0
                axes[1, j].imshow(dd, cmap="coolwarm", vmin=-lim, vmax=lim)
                axes[1, j].axis("off")

        axes[0, 0].set_ylabel("raw")
        axes[1, 0].set_ylabel("diff")
        fig.suptitle(f"{v}   top: raw thermal crop   bottom: frame difference "
                     f"(what the model actually sees)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"{v}_crops.png"), dpi=110)
        plt.close(fig)




if __name__ == "__main__":
    main()
