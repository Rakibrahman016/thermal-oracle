#!/usr/bin/env python3


import os
import sys
import csv
import numpy as np

# ---------------------------------------------------------------- config

# Set these before running.
# ROOT:   working directory holding the stage outputs from earlier scripts
# OUTDIR: where this script writes its table and figure (created if missing)

ROOT = ""
OUTDIR = ""

# AnalysisOutputs belongs to a different experiment - never touch it.
SKIP_DIRS = {"AnalysisOutputs"}

THERMAL_W, THERMAL_H = 640, 512
THERMAL_DTYPE = np.uint16
THERMAL_BYTES = THERMAL_W * THERMAL_H * 2


# ---------------------------------------------------------------- helpers
def list_sessions(root):
    """Session folders like p02_a, sorted. AnalysisOutputs excluded."""
    out = []
    for name in sorted(os.listdir(root)):
        if name in SKIP_DIRS:
            continue
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if not os.access(path, os.R_OK):
            print(f"  ! unreadable, skipping: {name}")
            continue
        out.append(name)
    return out


def frame_timestamps(folder):
    """Millisecond timestamps parsed from .raw filenames, sorted."""
    if not os.path.isdir(folder):
        return np.array([])
    ts = []
    for f in os.listdir(folder):
        if f.endswith(".raw"):
            stem = f[:-4]
            if stem.isdigit():
                ts.append(int(stem))
    return np.array(sorted(ts))


def fps_stats(ts):
    """Effective fps and jitter from timestamps."""
    if len(ts) < 2:
        return None, None, None
    d = np.diff(ts).astype(float)          # ms between frames
    d = d[d > 0]
    if len(d) == 0:
        return None, None, None
    return 1000.0 / np.median(d), float(np.median(d)), float(d.std())


def read_bvp(path):
    """Load the BVP csv. Returns dict of column -> array, or None."""
    if not os.path.isfile(path):
        return None
    try:
        arr = np.genfromtxt(path, delimiter=",", names=True)
    except Exception as e:
        print(f"  ! could not read {os.path.basename(path)}: {e}")
        return None
    if arr.dtype.names is None:
        return None
    return {n: np.asarray(arr[n], dtype=float) for n in arr.dtype.names}


def check_frame_size(folder, ts):
    """Confirm frames are the expected byte size. Returns (n_checked, n_bad)."""
    if len(ts) == 0:
        return 0, 0
    probe = [ts[0], ts[len(ts) // 2], ts[-1]]
    bad = 0
    for t in probe:
        p = os.path.join(folder, f"{t}.raw")
        if os.path.isfile(p) and os.path.getsize(p) != THERMAL_BYTES:
            bad += 1
    return len(probe), bad


# ---------------------------------------------------------------- main
def main():
    if not os.path.isdir(ROOT):
        sys.exit(f"Dataset root not found: {ROOT}")

    os.makedirs(OUTDIR, exist_ok=True)
    sessions = list_sessions(ROOT)
    print(f"Found {len(sessions)} session folders\n")

    rows = []
    for i, sess in enumerate(sessions, 1):
        sdir = os.path.join(ROOT, sess)
        subject, condition = sess.rsplit("_", 1)

        t_dir = os.path.join(sdir, f"{sess}_t")
        rgb_dir = os.path.join(sdir, f"{sess}_rgb")
        bvp_path = os.path.join(sdir, f"{sess}_bvp.csv")

        t_ts = frame_timestamps(t_dir)
        rgb_ts = frame_timestamps(rgb_dir)
        t_fps, t_dt, t_jit = fps_stats(t_ts)
        rgb_fps, _, _ = fps_stats(rgb_ts)

        n_probe, n_bad = check_frame_size(t_dir, t_ts)

        bvp = read_bvp(bvp_path)
        n_bvp = len(bvp["BVP"]) if bvp and "BVP" in bvp else 0

        # duration from thermal timestamps, seconds
        dur = (t_ts[-1] - t_ts[0]) / 1000.0 if len(t_ts) > 1 else 0.0
        bvp_fs = n_bvp / dur if dur > 0 and n_bvp else 0.0

        row = {
            "session": sess,
            "subject": subject,
            "condition": condition,
            "n_thermal": len(t_ts),
            "n_rgb": len(rgb_ts),
            "n_bvp": n_bvp,
            "duration_s": round(dur, 2),
            "thermal_fps": round(t_fps, 3) if t_fps else 0,
            "thermal_dt_ms": round(t_dt, 2) if t_dt else 0,
            "thermal_jitter_ms": round(t_jit, 2) if t_jit else 0,
            "rgb_fps": round(rgb_fps, 3) if rgb_fps else 0,
            "bvp_fs_est": round(bvp_fs, 2),
            "bad_frame_size": n_bad,
        }

        # quality-label columns, if present
        for col in ("SQPhysMD", "SQ1", "SQ2", "Perfusion"):
            if bvp and col in bvp:
                v = bvp[col]
                v = v[np.isfinite(v)]
                row[f"{col}_mean"] = round(float(v.mean()), 4) if len(v) else ""
                # fraction of samples flagged as not-good (below 1.0)
                if col.startswith("SQ"):
                    row[f"{col}_frac_low"] = round(float((v < 1.0).mean()), 4) if len(v) else ""
            else:
                row[f"{col}_mean"] = ""
                if col.startswith("SQ"):
                    row[f"{col}_frac_low"] = ""

        rows.append(row)
        print(f"[{i:3d}/{len(sessions)}] {sess}  "
              f"thermal={len(t_ts):5d}  rgb={len(rgb_ts):5d}  "
              f"bvp={n_bvp:6d}  {dur:6.1f}s  {row['thermal_fps']} fps")

    # ------------------------------------------------------------ write csv
    csv_path = os.path.join(OUTDIR, "stage0_sessions.csv")
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------ summary
    subjects = sorted({r["subject"] for r in rows})
    conditions = sorted({r["condition"] for r in rows})
    t_fps_all = [r["thermal_fps"] for r in rows if r["thermal_fps"]]
    bvp_fs_all = [r["bvp_fs_est"] for r in rows if r["bvp_fs_est"]]
    durs = [r["duration_s"] for r in rows if r["duration_s"]]

    lines = []
    add = lines.append
    add("STAGE 0 - iBVP INVENTORY")
    add("=" * 60)
    add(f"root            : {ROOT}")
    add(f"sessions        : {len(rows)}")
    add(f"subjects        : {len(subjects)}")
    add(f"conditions      : {', '.join(conditions)}")
    add("")
    add("Sessions per condition:")
    for c in conditions:
        add(f"  {c}: {sum(1 for r in rows if r['condition'] == c)}")
    add("")
    add("Sessions per subject (expect 4 each):")
    odd = [s for s in subjects
           if sum(1 for r in rows if r["subject"] == s) != 4]
    add(f"  subjects without exactly 4 sessions: {odd if odd else 'none'}")
    add("")
    add("Timing:")
    if t_fps_all:
        add(f"  thermal fps  : min {min(t_fps_all):.2f}  "
            f"median {np.median(t_fps_all):.2f}  max {max(t_fps_all):.2f}")
    if bvp_fs_all:
        add(f"  bvp fs (est) : min {min(bvp_fs_all):.2f}  "
            f"median {np.median(bvp_fs_all):.2f}  max {max(bvp_fs_all):.2f}")
    if durs:
        add(f"  duration (s) : min {min(durs):.1f}  "
            f"median {np.median(durs):.1f}  max {max(durs):.1f}")
    add("")
    add("Integrity flags:")
    bad_size = [r["session"] for r in rows if r["bad_frame_size"]]
    no_bvp = [r["session"] for r in rows if r["n_bvp"] == 0]
    no_therm = [r["session"] for r in rows if r["n_thermal"] == 0]
    mismatch = [r["session"] for r in rows
                if r["n_thermal"] and r["n_rgb"]
                and abs(r["n_thermal"] - r["n_rgb"]) > 0.02 * r["n_thermal"]]
    add(f"  wrong thermal frame size : {bad_size if bad_size else 'none'}")
    add(f"  missing bvp              : {no_bvp if no_bvp else 'none'}")
    add(f"  missing thermal          : {no_therm if no_therm else 'none'}")
    add(f"  thermal/rgb count differs by >2% : {mismatch if mismatch else 'none'}")
    add("")
    add("Quality labels (fraction of samples below 1.0):")
    for col in ("SQPhysMD", "SQ1", "SQ2"):
        vals = [r.get(f"{col}_frac_low") for r in rows]
        vals = [v for v in vals if isinstance(v, float)]
        if vals:
            add(f"  {col}: median {np.median(vals):.4f}  "
                f"max {max(vals):.4f}  "
                f"sessions with any low: {sum(1 for v in vals if v > 0)}/{len(vals)}")
        else:
            add(f"  {col}: not present")
    add("")
    add(f"Per-session detail written to: {csv_path}")

    report = "\n".join(lines)
    with open(os.path.join(OUTDIR, "stage0_report.txt"), "w") as f:
        f.write(report + "\n")

    print("\n" + report)


if __name__ == "__main__":
    main()
