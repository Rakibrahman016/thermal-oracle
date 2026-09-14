#!/usr/bin/env python3


import os
import csv
import glob
import pickle
import numpy as np

# ---------------------------------------------------------------- config

# Set these before running.
# Runs:   working directory holding the stage outputs from earlier scripts
# OUTDIR: where this script writes its table and figure (created if missing)
# DATASET : your dataset


RUNS = ""
DATASET = ""
OUTDIR = ""

FS = 30.0                # sampling rate, Hz (confirmed in Stage 0)
CHUNK_LEN = 600          # samples per chunk (CL600 config)

# The dataset BVP csv also carries the quality columns we need in Stage 4.
SQ_COLS = ["SQPhysMD", "SQ1", "SQ2", "Perfusion"]


# ---------------------------------------------------------------- helpers
def find_fold_pickles(runs_dir):
    """One pickle per fold, sorted by fold number."""
    pat = os.path.join(runs_dir, "*", "saved_test_outputs", "*_outputs.pickle")
    paths = sorted(glob.glob(pat))
    out = []
    for p in paths:
        run = p.split(os.sep)[-3]
        fold = None
        for token in run.split("_"):
            if token.startswith("fold"):
                fold = token
        out.append((fold or run, p))
    return out


def stitch(chunk_dict):
    """Concatenate chunks in index order into one 1-D signal."""
    keys = sorted(chunk_dict.keys())
    parts = []
    for k in keys:
        t = chunk_dict[k]
        a = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        parts.append(np.asarray(a, dtype=np.float64).reshape(-1))
    return np.concatenate(parts), len(keys)


def session_dir_name(video_id):
    """pickle uses 'p02a'; the dataset folder is 'p02_a'."""
    return f"{video_id[:-1]}_{video_id[-1]}"


def load_reference(video_id, n_needed):
    """
    Load BVP + quality columns from the dataset csv, trimmed to n_needed.

    Returns dict of column -> array, or None if the file is missing.
    """
    sess = session_dir_name(video_id)
    path = os.path.join(DATASET, sess, f"{sess}_bvp.csv")
    if not os.path.isfile(path):
        return None
    try:
        arr = np.genfromtxt(path, delimiter=",", names=True)
    except Exception:
        return None
    if arr.dtype.names is None:
        return None

    out = {}
    for name in arr.dtype.names:
        v = np.asarray(arr[name], dtype=np.float64).reshape(-1)
        out[name] = v[:n_needed] if len(v) >= n_needed else np.pad(
            v, (0, n_needed - len(v)), constant_values=np.nan)
    return out


def zscore(x):
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 0 else x - np.nanmean(x)


def corr(a, b):
    """Pearson r, NaN-safe."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    a, b = a[m], b[m]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    sig_dir = os.path.join(OUTDIR, "signals")
    os.makedirs(sig_dir, exist_ok=True)

    folds = find_fold_pickles(RUNS)
    if not folds:
        raise SystemExit(f"No fold pickles found under {RUNS}")
    print(f"Found {len(folds)} fold pickles\n")

    rows = []
    seen = {}

    for fold, path in folds:
        with open(path, "rb") as f:
            d = pickle.load(f)

        pred_all = d["predictions"]
        label_all = d["labels"]
        fs = float(d.get("fs", FS))
        label_type = d.get("label_type", "")

        print(f"--- {fold}  ({len(pred_all)} videos, fs={fs}, {label_type})")

        for vid in sorted(pred_all.keys()):
            if vid in seen:
                print(f"  ! {vid} already seen in {seen[vid]}, skipping dup")
                continue
            seen[vid] = fold

            pred, n_chunks = stitch(pred_all[vid])
            label, _ = stitch(label_all[vid])
            n = min(len(pred), len(label))
            pred, label = pred[:n], label[:n]

            ref = load_reference(vid, n)

            # sanity: the pickle's label should track the dataset BVP.
            # if it doesn't, the chunk-to-frame alignment assumption is wrong.
            r_align = np.nan
            if ref is not None and "BVP" in ref:
                r_align = corr(zscore(label), zscore(ref["BVP"]))

            npz = {
                "pred": pred.astype(np.float32),
                "label": label.astype(np.float32),
                "fs": np.float64(fs),
                "n_chunks": np.int32(n_chunks),
                "fold": fold,
                "video_id": vid,
            }
            if ref is not None:
                for c in ["BVP"] + SQ_COLS:
                    if c in ref:
                        npz[f"ref_{c}"] = ref[c].astype(np.float32)

            np.savez_compressed(os.path.join(sig_dir, f"{vid}.npz"), **npz)

            subject = vid[:-1]
            condition = vid[-1]
            row = {
                "video_id": vid,
                "subject": subject,
                "condition": condition,
                "fold": fold,
                "n_samples": n,
                "duration_s": round(n / fs, 2),
                "n_chunks": n_chunks,
                "fs": fs,
                "pred_std": round(float(np.std(pred)), 6),
                "pred_flat": int(np.std(pred) < 1e-8),
                "pred_nan": int(np.isnan(pred).sum()),
                "label_align_r": round(r_align, 4) if np.isfinite(r_align) else "",
                "has_ref": int(ref is not None),
            }
            for c in SQ_COLS:
                key = f"ref_{c}"
                if ref is not None and c in ref:
                    v = ref[c]
                    v = v[np.isfinite(v)]
                    row[f"{c}_mean"] = round(float(v.mean()), 4) if len(v) else ""
                    if c.startswith("SQ"):
                        row[f"{c}_frac_low"] = round(float((v < 1.0).mean()), 4) if len(v) else ""
                else:
                    row[f"{c}_mean"] = ""
                    if c.startswith("SQ"):
                        row[f"{c}_frac_low"] = ""

            rows.append(row)
            print(f"  {vid}  n={n:5d} ({n/fs:6.1f}s)  chunks={n_chunks}  "
                  f"align_r={r_align:+.3f}" if np.isfinite(r_align)
                  else f"  {vid}  n={n:5d} ({n/fs:6.1f}s)  chunks={n_chunks}  align_r=n/a")

    # ------------------------------------------------------------ index
    idx_path = os.path.join(OUTDIR, "stage1_index.csv")
    with open(idx_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------ report
    subs = sorted({r["subject"] for r in rows})
    aligns = [r["label_align_r"] for r in rows if isinstance(r["label_align_r"], float)]
    durs = [r["duration_s"] for r in rows]

    lines = []
    add = lines.append
    add("STAGE 1 - SIGNAL STORE")
    add("=" * 60)
    add(f"videos written : {len(rows)}")
    add(f"subjects       : {len(subs)}")
    add(f"folds          : {len(folds)}")
    add(f"signals dir    : {sig_dir}")
    add(f"index          : {idx_path}")
    add("")
    add("Per-video length:")
    add(f"  samples  : min {min(r['n_samples'] for r in rows)}  "
        f"max {max(r['n_samples'] for r in rows)}")
    add(f"  seconds  : min {min(durs):.1f}  median {np.median(durs):.1f}  max {max(durs):.1f}")
    add("")
    add("Alignment check (pickle label vs dataset BVP, Pearson r):")
    if aligns:
        add(f"  median {np.median(aligns):+.3f}   min {min(aligns):+.3f}   max {max(aligns):+.3f}")
        weak = [r["video_id"] for r in rows
                if isinstance(r["label_align_r"], float) and abs(r["label_align_r"]) < 0.5]
        add(f"  videos with |r| < 0.5 : {len(weak)}  {weak[:10]}")
        add("  (high r means chunk->frame alignment is correct;")
        add("   widespread low r means the reference offset needs fixing)")
    else:
        add("  no reference available")
    add("")
    add("Degenerate predictions:")
    flat = [r["video_id"] for r in rows if r["pred_flat"]]
    nan = [r["video_id"] for r in rows if r["pred_nan"]]
    add(f"  flat (zero variance) : {flat if flat else 'none'}")
    add(f"  containing NaN       : {nan if nan else 'none'}")
    add("")
    add("Quality labels across all videos (fraction of samples < 1.0):")
    for c in ("SQPhysMD", "SQ1", "SQ2"):
        vals = [r.get(f"{c}_frac_low") for r in rows]
        vals = [v for v in vals if isinstance(v, float)]
        if vals:
            add(f"  {c}: median {np.median(vals):.4f}  max {max(vals):.4f}  "
                f"videos with any low: {sum(1 for v in vals if v > 0)}/{len(vals)}")

    report = "\n".join(lines)
    with open(os.path.join(OUTDIR, "stage1_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)


if __name__ == "__main__":
    main()
