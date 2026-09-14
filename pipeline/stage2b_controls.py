#!/usr/bin/env python3


import os
import glob
import csv
import numpy as np

# reuse everything from Stage 2 so the processing is provably identical
from stage2_estimates import (
    FS, WIN_SEC, STEP_SEC, N_CHUNKS, HR_MIN, HR_MAX,
    BANDS, DETRENDS, ESTIMATORS, ROIS,
    detrend_poly, bandpass, quality_indices,
    load_roi_signals, load_model_signal,
    CACHE, OUTDIR,
)

SEED = 12345


def pink_noise(n, rng):
    """1/f noise - the spectral shape thermal drift actually has."""
    f = np.fft.rfftfreq(n, d=1.0 / FS)
    amp = np.ones_like(f)
    amp[1:] = 1.0 / np.sqrt(f[1:])
    amp[0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, len(f))
    spec = amp * np.exp(1j * phase)
    x = np.fft.irfft(spec, n=n)
    return x / (x.std() or 1.0)


def main():
    rng = np.random.default_rng(SEED)

    vids = sorted(os.path.basename(p).split("_input")[0]
                  for p in glob.glob(os.path.join(CACHE, "*_input0.npy")))
    print(f"{len(vids)} videos\n")

    win = int(WIN_SEC * FS)
    step = int(STEP_SEC * FS)

    # cache one ROI signal per video, so ctrl_permuted can borrow across videos
    print("loading donor signals for the permuted control ...")
    donors = {}
    for vid in vids:
        sigs, n = load_roi_signals(vid)
        if sigs:
            donors[vid] = sigs["roi_forehead"]
    donor_ids = sorted(donors.keys())
    print(f"  {len(donor_ids)} donors\n")

    rows = []
    for vi, vid in enumerate(vids, 1):
        roi_sigs, n_roi = load_roi_signals(vid)
        pred, label = load_model_signal(vid)
        if pred is None or not roi_sigs:
            print(f"[{vi:3d}/{len(vids)}] {vid}: missing data, skipped")
            continue

        n = min(n_roi, len(pred), len(label))
        ref = label[:n]
        real = roi_sigs["roi_forehead"][:n]

        # a different video's forehead signal, same length
        other = donor_ids[(donor_ids.index(vid) + 1) % len(donor_ids)] \
            if vid in donor_ids else donor_ids[0]
        borrowed = donors[other]
        if len(borrowed) < n:
            borrowed = np.pad(borrowed, (0, n - len(borrowed)), mode="wrap")
        borrowed = borrowed[:n]

        bank = {
            "ctrl_white": rng.standard_normal(n),
            "ctrl_pink": pink_noise(n, rng),
            "ctrl_shuffled": rng.permutation(real.copy()),
            "ctrl_permuted": borrowed,
        }

        n_win = 0
        for s0 in range(0, n - win + 1, step):
            s1 = s0 + win
            seg_ref = ref[s0:s1]

            ref_bp = bandpass(detrend_poly(seg_ref, 1), 0.75, 3.0)
            from stage2_estimates import hr_welch
            gt_hr = hr_welch(ref_bp)
            if not np.isfinite(gt_hr) or not (HR_MIN <= gt_hr <= HR_MAX):
                continue
            n_win += 1

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
                                "p95p5_mean": "",
                                "p95p5_std": "",
                                **{k: (round(x, 5) if np.isfinite(x) else "")
                                   for k, x in sqi.items()},
                            })

        print(f"[{vi:3d}/{len(vids)}] {vid}: {n_win} windows "
              f"x 4 controls x 24 combos")

    out = os.path.join(OUTDIR, "stage2_controls.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows):,} control rows written to {out}")
    print("\nNow rerun stage3_oracle.py - it picks the control file up")
    print("automatically and reports control oracles alongside the real ones.")


if __name__ == "__main__":
    main()
