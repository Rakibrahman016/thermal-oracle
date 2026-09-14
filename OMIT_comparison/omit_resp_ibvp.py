#!/usr/bin/env python3


import os
import re
import csv
import glob
import numpy as np
from scipy.signal import butter, filtfilt, get_window, medfilt, find_peaks

# Set these to your local paths before running.
# CACHE:   rPPG-Toolbox preprocessed cache for the fold you are analysing
#          (contains <video>_input<N>.npy and <video>_label<N>.npy)
# DATASET: root of iBVP_Dataset_ready (one folder per session, e.g. p02_a/)

# --------------------------------------------------------------- config
CACHE = ("")
DATASET = ""
OUTDIR = ""

FS = 30.0
WIN_SEC = 25.0               # Welch window for respiration, per the paper
STEP_SEC = 1.0
PRE_LO, PRE_HI = 0.12, 2.0   # pre-filter
RESP_LO, RESP_HI = 0.12, 0.55  # respiratory band, 7-33 rpm
VALID_LO, VALID_HI = 7, 45   # valid range in rpm
MEDFILT_N = 7
MAX_LAG_S = 10               # small alignment search, applied to controls too
AGREE_TOL = 3.0              # rpm; AM and RSA must agree within this
MIN_WINDOWS = 20

# nose and both cheeks, inside the 72x72 face-aligned crop
ROIS = {
    "nose":    (34, 46, 28, 44),
    "cheek_l": (36, 52, 12, 26),
    "cheek_r": (36, 52, 46, 60),
}

CONDITIONS = {"a": "paced breathing", "b": "easy math",
              "c": "hard math", "d": "head movement"}


# --------------------------------------------------------------- signal
def bandpass(x, lo, hi, fs, order=4):
    x = np.asarray(x, float)
    ny = fs / 2.0
    lo_n, hi_n = lo / ny, min(hi / ny, 0.99)
    if lo_n <= 0 or hi_n <= lo_n or x.size < 10 * order:
        return x
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, x, method="gust")


def welch_peak_rpm(x, fs, lo=RESP_LO, hi=RESP_HI):
    x = np.asarray(x, float)
    n = x.size
    if n < 64 or not np.isfinite(x).all() or x.std() == 0:
        return np.nan
    seg = max(128, n // 2)
    nfft = 8192
    win = get_window("hann", seg)
    step = seg // 2
    psd, count = np.zeros(nfft // 2 + 1), 0
    for s in range(0, n - seg + 1, step):
        d = x[s:s + seg]
        psd += np.abs(np.fft.rfft((d - d.mean()) * win, n=nfft)) ** 2
        count += 1
    if count == 0:
        return np.nan
    psd /= count
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    band = np.where((f >= lo) & (f <= hi))[0]
    if band.size < 3:
        return np.nan
    i = band[int(np.argmax(psd[band]))]
    if 0 < i < len(psd) - 1:
        a, b, c = psd[i - 1], psd[i], psd[i + 1]
        den = a - 2 * b + c
        delta = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den else 0.0
    else:
        delta = 0.0
    return float((f[i] + delta * (f[1] - f[0])) * 60.0)


def rate_series(sig, fs, win_sec=WIN_SEC, step_sec=STEP_SEC):
    win, step = int(win_sec * fs), int(step_sec * fs)
    return np.asarray([welch_peak_rpm(sig[s:s + win], fs)
                       for s in range(0, len(sig) - win + 1, step)], float)


def clean(rpm, lo=VALID_LO, hi=VALID_HI, med=MEDFILT_N):
    y = np.asarray(rpm, float).copy()
    y[~np.isfinite(y) | (y < lo) | (y > hi)] = np.nan
    if np.isfinite(y).sum() >= med:
        filled = np.where(np.isfinite(y), y, np.nanmedian(y))
        y = np.where(np.isfinite(y), medfilt(filled, kernel_size=med), np.nan)
    return y


def omit(channels, fs, lo=RESP_LO, hi=RESP_HI):
    """QR, project out the dominant component, keep the best residual channel."""
    X = np.asarray(channels, float)
    X = X - X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1)
    keep = sd > 0
    if keep.sum() == 0:
        return np.zeros(X.shape[1]), -1
    X = X[keep] / sd[keep][:, None]
    if X.shape[0] == 1:
        return X[0], 0
    Q, _ = np.linalg.qr(X)
    q1 = Q[:, [0]]
    Y = (np.eye(X.shape[0]) - q1 @ q1.T) @ X

    def bp(v):
        n = v.size
        sp = np.abs(np.fft.rfft(v * np.hanning(n), n=8192)) ** 2
        f = np.fft.rfftfreq(8192, d=1.0 / fs)
        m = (f >= lo) & (f <= hi)
        return float(sp[m].max()) if m.any() else 0.0

    best = int(np.argmax([bp(Y[i]) for i in range(Y.shape[0])]))
    return Y[best], best


# --------------------------------------------------------- reference
def resp_from_rsa(bvp, fs):
    """
    Respiratory sinus arrhythmia: breathing speeds the heart on inhale
    and slows it on exhale. Recovered from beat-to-beat intervals, so it
    survives the high-pass filter applied to the iBVP reference.
    """
    b = bandpass(bvp, 0.7, 3.5, fs)
    if b.std() == 0:
        return np.zeros_like(b)
    pk, _ = find_peaks(b, distance=int(0.35 * fs), prominence=0.2 * b.std())
    if pk.size < 12:
        return np.zeros_like(b)

    # interval between consecutive beats, placed at the later beat
    ibi = np.diff(pk) / fs
    t_ibi = pk[1:] / fs

    # discard physiologically impossible intervals (0.3-1.5 s = 40-200 bpm)
    ok = (ibi > 0.3) & (ibi < 1.5)
    if ok.sum() < 10:
        return np.zeros_like(b)
    ibi, t_ibi = ibi[ok], t_ibi[ok]

    # resample the irregular interval series onto the uniform grid
    t_uniform = np.arange(b.size) / fs
    tach = np.interp(t_uniform, t_ibi, ibi)
    return bandpass(tach, RESP_LO, RESP_HI, fs)


def resp_from_am(bvp, fs):
    """Amplitude modulation: interpolated envelope of the pulse peaks."""
    b = bandpass(bvp, 0.7, 3.5, fs)
    if b.std() == 0:
        return np.zeros_like(b)
    pk, _ = find_peaks(b, distance=int(0.35 * fs))
    if pk.size < 8:
        return np.zeros_like(b)
    env = np.interp(np.arange(b.size), pk, b[pk])
    return bandpass(env, RESP_LO, RESP_HI, fs)


def load_bvp(video_id, n):
    sess = f"{video_id[:-1]}_{video_id[-1]}"
    p = os.path.join(DATASET, sess, f"{sess}_bvp.csv")
    if not os.path.isfile(p):
        return None
    a = np.genfromtxt(p, delimiter=",", names=True)
    if a.dtype.names is None or "BVP" not in a.dtype.names:
        return None
    return np.asarray(a["BVP"], float)[:n]


def load_thermal(vid):
    ins = sorted(glob.glob(os.path.join(CACHE, f"{vid}_input*.npy")),
                 key=lambda p: int(re.search(r"_input(\d+)\.npy$", p).group(1)))
    if not ins:
        return None
    return np.concatenate([np.squeeze(np.load(p)) for p in ins], axis=0)


# --------------------------------------------------------- controls
def subject_of(vid):
    """p02a -> p02. The condition letter is the last character."""
    return vid[:-1]


def assign_partners(vids):
    """
    Pair each video with one from a DIFFERENT subject.

    Sorted order puts a subject's four conditions next to each other, so
    taking the next video in the list pairs p02a with p02b — the same
    person. That is not a wrong-subject control. Here the list is rotated
    by a whole subject block instead, and each pair is checked.
    """
    subjects = sorted({subject_of(v) for v in vids})
    if len(subjects) < 2:
        return {v: v for v in vids}
    order = {s: i for i, s in enumerate(subjects)}
    by_subject = {s: [v for v in vids if subject_of(v) == s] for s in subjects}

    partner = {}
    for v in vids:
        # walk forward through subjects until one differs
        for step in range(1, len(subjects)):
            cand_subj = subjects[(order[subject_of(v)] + step) % len(subjects)]
            pool_ = by_subject[cand_subj]
            if pool_:
                # keep the same condition where possible, so the control
                # differs in person and not in task
                same_cond = [u for u in pool_ if u[-1] == v[-1]]
                partner[v] = (same_cond or pool_)[0]
                break
        else:
            partner[v] = v
    return partner


def low_quality_fraction(video_id, n):
    """
    Fraction of reference samples flagged low quality by SQPhysMD.
    Returns nan when the column is absent.
    """
    sess = f"{video_id[:-1]}_{video_id[-1]}"
    path = os.path.join(DATASET, sess, f"{sess}_bvp.csv")
    if not os.path.isfile(path):
        return float("nan")
    try:
        a = np.genfromtxt(path, delimiter=",", names=True)
    except Exception:
        return float("nan")
    if a.dtype.names is None or "SQPhysMD" not in a.dtype.names:
        return float("nan")
    q = np.asarray(a["SQPhysMD"], float)[:n]
    q = q[np.isfinite(q)]
    if q.size == 0:
        return float("nan")
    return float(np.mean(q < 0.5))


# --------------------------------------------------------------- metrics
def align_metrics(est, ref, fs_step=1.0, max_lag=MAX_LAG_S):
    """
    Metrics at zero lag and at the best lag within a small window. The
    same lag freedom is given to the controls, so the comparison stays
    fair.
    """
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    n = min(est.size, ref.size)
    est, ref = est[:n], ref[:n]

    def score(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < MIN_WINDOWS:
            return None
        e, r = a[m], b[m]
        pcc = (float(np.corrcoef(e, r)[0, 1])
               if e.std() > 0 and r.std() > 0 else np.nan)
        return dict(mae=float(np.mean(np.abs(e - r))),
                    rmse=float(np.sqrt(np.mean((e - r) ** 2))),
                    pcc=pcc, bias=float(np.mean(e - r)), n=int(m.sum()))

    at0 = score(est, ref)
    best, best_lag = at0, 0
    for lag in range(1, int(max_lag / fs_step) + 1):
        for a, b, sgn in ((est[lag:], ref[:n - lag], lag),
                          (est[:n - lag], ref[lag:], -lag)):
            s = score(a, b)
            if s and (best is None or s["mae"] < best["mae"]):
                best, best_lag = s, sgn
    nan = dict(mae=np.nan, rmse=np.nan, pcc=np.nan, bias=np.nan, n=0)
    return (at0 or nan), (best or nan), best_lag


def pool(rows, key):
    v = np.array([r[key] for r in rows if np.isfinite(r[key])], float)
    if v.size == 0:
        return np.nan
    if key == "pcc":
        return float(np.tanh(np.arctanh(np.clip(v, -0.999, 0.999)).mean()))
    return float(v.mean())


# --------------------------------------------------------------- main
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    vids = sorted({os.path.basename(p).split("_input")[0]
                   for p in glob.glob(os.path.join(CACHE, "*_input0.npy"))})
    print(f"{len(vids)} videos in cache")
    print(f"Welch {WIN_SEC:.0f} s window, band {RESP_LO}-{RESP_HI} Hz "
          f"({VALID_LO}-{VALID_HI} rpm)\n")

    est, ref, agree, cond, lowq = {}, {}, {}, {}, {}

    for i, vid in enumerate(vids, 1):
        x = load_thermal(vid)
        if x is None:
            continue
        n = x.shape[0]
        bvp = load_bvp(vid, n)
        if bvp is None or bvp.size < int(WIN_SEC * FS) * 2:
            print(f"  [{i:3d}/{len(vids)}] {vid:10s} skipped (no BVP)")
            del x
            continue
        n = min(n, bvp.size)

        traces = np.vstack([x[:n, r0:r1, c0:c1].reshape(n, -1).mean(axis=1)
                            for r0, r1, c0, c1 in ROIS.values()])
        del x

        pre = np.vstack([bandpass(t, PRE_LO, PRE_HI, FS) for t in traces])
        # the paper averages nose and cheeks for BR; OMIT is kept as a
        # second route so the two can be compared
        avg = bandpass(pre.mean(axis=0), RESP_LO, RESP_HI, FS)
        est[vid] = clean(rate_series(avg, FS))

        r_am = clean(rate_series(resp_from_am(bvp[:n], FS), FS))
        r_rsa = clean(rate_series(resp_from_rsa(bvp[:n], FS), FS))
        m = np.isfinite(r_am) & np.isfinite(r_rsa)
        gap = float(np.mean(np.abs(r_am[m] - r_rsa[m]))) if m.sum() else np.nan
        agree[vid] = gap
        # where the two agree, average them; otherwise fall back to AM,
        # which is the more robust of the pair
        both = np.isfinite(r_am) & np.isfinite(r_rsa)
        merged = np.where(np.isfinite(r_am), r_am, r_rsa)
        if np.isfinite(gap) and gap <= AGREE_TOL:
            merged[both] = 0.5 * (r_am[both] + r_rsa[both])
        ref[vid] = merged
        cond[vid] = vid[-1].lower()
        lowq[vid] = low_quality_fraction(vid, n)

        if np.isfinite(ref[vid]).sum() < MIN_WINDOWS:
            print(f"  [{i:3d}/{len(vids)}] {vid:10s} skipped (reference sparse)")
            del est[vid], ref[vid]
            continue

        flag = "ok " if np.isfinite(gap) and gap <= AGREE_TOL else "CHK"
        print(f"  [{i:3d}/{len(vids)}] {vid:10s} [{CONDITIONS.get(cond[vid],'?'):15s}] "
              f"ref {np.nanmean(ref[vid]):5.1f} rpm   "
              f"thermal {np.nanmean(est[vid]):5.1f} rpm   "
              f"AM-RSA gap {gap:5.2f} {flag}")

    good = sorted(ref)
    if len(good) < 5:
        print("\nToo few usable videos. Stopping.")
        return

    trusted = [v for v in good
               if np.isfinite(agree[v]) and agree[v] <= AGREE_TOL]
    print(f"\n{len(good)} usable, {len(trusted)} with a trustworthy reference "
          f"(AM and RSA within {AGREE_TOL:.0f} rpm)\n")

    # Control pairing.
    #
    # A control must be a DIFFERENT person whose reference we also trust.
    # Pairing a trusted recording against an untrusted one inflates the
    # control error and flatters the result, so trusted recordings are
    # paired only with other trusted recordings.
    partner = assign_partners(good)                       # all-videos control
    partner_t = assign_partners(trusted) if len(trusted) >= 2 else {}

    same = sum(1 for v in good if subject_of(v) == subject_of(partner[v]))
    print(f"Control pairing (all): {len(good) - same} of {len(good)} partners "
          f"are a different subject")
    if partner_t:
        same_t = sum(1 for v in trusted
                     if subject_of(v) == subject_of(partner_t[v]))
        print(f"Control pairing (trusted): {len(trusted) - same_t} of "
              f"{len(trusted)} partners are a different subject AND trusted")
    else:
        print("Control pairing (trusted): too few trusted recordings to pair")


    real0, realB, ctrl0, ctrlB, lags, rows = [], [], [], [], [], []
    for v in good:
        a0, ab, lag = align_metrics(est[v], ref[v])
        c0, cb, _ = align_metrics(est[v], ref[partner[v]])
        real0.append(a0); realB.append(ab)
        ctrl0.append(c0); ctrlB.append(cb)
        lags.append(lag)
        rows.append([v, cond[v], partner[v],
                     partner_t.get(v, ""),
                     round(lowq[v], 4) if np.isfinite(lowq[v]) else "",
                     round(agree[v], 2) if np.isfinite(agree[v]) else "",
                     round(a0["mae"], 2), round(a0["pcc"], 3),
                     round(ab["mae"], 2), round(ab["pcc"], 3), lag,
                     round(c0["mae"], 2), round(c0["pcc"], 3)])

    print("=" * 78)
    print("RESPIRATION FROM iBVP THERMAL")
    print("=" * 78)
    print(f"{'pairing':28s} {'MAE rpm':>9s} {'RMSE':>8s} {'PCC':>8s} {'bias':>8s}")
    print("-" * 78)
    for name, rs in (("real, lag 0", real0), ("real, best lag", realB),
                     ("wrong subject, lag 0", ctrl0),
                     ("wrong subject, best lag", ctrlB)):
        print(f"{name:28s} {pool(rs,'mae'):9.2f} {pool(rs,'rmse'):8.2f} "
              f"{pool(rs,'pcc'):8.3f} {pool(rs,'bias'):8.2f}")

    if trusted:
        sub = [good.index(v) for v in trusted]
        print()
        print(f"{'trusted references only':28s} "
              f"{pool([real0[i] for i in sub],'mae'):9.2f} "
              f"{pool([real0[i] for i in sub],'rmse'):8.2f} "
              f"{pool([real0[i] for i in sub],'pcc'):8.3f} "
              f"{pool([real0[i] for i in sub],'bias'):8.2f}")
        print(f"{'  its wrong-subject control':28s} "
              f"{pool([ctrl0[i] for i in sub],'mae'):9.2f} "
              f"{pool([ctrl0[i] for i in sub],'rmse'):8.2f} "
              f"{pool([ctrl0[i] for i in sub],'pcc'):8.3f} "
              f"{pool([ctrl0[i] for i in sub],'bias'):8.2f}")

    print()
    print("By condition (real pairs, lag 0):")
    print(f"  {'condition':18s} {'n':>4s} {'MAE rpm':>9s} {'PCC':>8s}")
    for c, label in CONDITIONS.items():
        idx = [k for k, v in enumerate(good) if cond[v] == c]
        if not idx:
            continue
        print(f"  {label:18s} {len(idx):4d} "
              f"{pool([real0[i] for i in idx],'mae'):9.2f} "
              f"{pool([real0[i] for i in idx],'pcc'):8.3f}")
    print("  Paced breathing should be the strongest row if this works.")

    # ---- headline: trusted recordings against trusted controls ----
    if partner_t:
        tr_real = [align_metrics(est[v], ref[v])[0] for v in trusted]
        tr_ctrl = [align_metrics(est[v], ref[partner_t[v]])[0] for v in trusted]
        rm_ = np.array([r["mae"] for r in tr_real], float)
        cm_ = np.array([r["mae"] for r in tr_ctrl], float)
        m_ = np.isfinite(rm_) & np.isfinite(cm_)
        wins = int((rm_[m_] < cm_[m_]).sum())
        print()
        print("=" * 78)
        print("HEADLINE: trusted recordings, controls also trusted")
        print("=" * 78)
        print(f"{'pairing':28s} {'MAE rpm':>9s} {'RMSE':>8s} {'PCC':>8s} {'bias':>8s}")
        print("-" * 78)
        print(f"{'real (right person)':28s} {pool(tr_real,'mae'):9.2f} "
              f"{pool(tr_real,'rmse'):8.2f} {pool(tr_real,'pcc'):8.3f} "
              f"{pool(tr_real,'bias'):8.2f}")
        print(f"{'control (wrong person)':28s} {pool(tr_ctrl,'mae'):9.2f} "
              f"{pool(tr_ctrl,'rmse'):8.2f} {pool(tr_ctrl,'pcc'):8.3f} "
              f"{pool(tr_ctrl,'bias'):8.2f}")
        print()
        print(f"  advantage      {pool(tr_ctrl,'mae') - pool(tr_real,'mae'):+.2f} rpm")
        print(f"  beats control  {wins} of {int(m_.sum())}")
        try:
            from scipy.stats import wilcoxon
            if m_.sum() >= 6:
                pv = float(wilcoxon(rm_[m_], cm_[m_])[1])
                print(f"  Wilcoxon       p = {pv:.5g}")
        except Exception:
            pass
        print()
        print("  This is the number to report. Both sides of the comparison")
        print("  now have a reference we trust, so the control is not")
        print("  penalised by a noisy reference of its own.")

    q = np.array([lowq[v] for v in good], float)
    a = np.array([agree[v] for v in good], float)
    m = np.isfinite(q) & np.isfinite(a)
    if m.sum() > 10 and q[m].std() > 0 and a[m].std() > 0:
        r_qa = float(np.corrcoef(q[m], a[m])[0, 1])
        print()
        print("Reference quality vs AM-RSA agreement:")
        print(f"  correlation between low-quality fraction and gap: {r_qa:+.3f}")
        tset = [v for v in good if np.isfinite(agree[v]) and agree[v] <= AGREE_TOL]
        fset = [v for v in good if v not in tset]
        for nm, ss in (("trusted", tset), ("flagged", fset)):
            qq = np.array([lowq[v] for v in ss], float)
            qq = qq[np.isfinite(qq)]
            if qq.size:
                print(f"  {nm:8s} n={len(ss):3d}  mean low-quality fraction "
                      f"{qq.mean():.3f}")
        if r_qa > 0.25:
            print("  The videos that fail the agreement check are the ones with")
            print("  a poor ear-PPG reference. The limit is reference quality,")
            print("  not the thermal signal.")
        else:
            print("  Agreement failures do not track reference quality; some")
            print("  other factor is driving them.")

    rm, cm = pool(real0, "mae"), pool(ctrl0, "mae")
    rp, cp = pool(real0, "pcc"), pool(ctrl0, "pcc")
    print()
    print("Interpretation:")
    if np.isfinite(rm) and np.isfinite(cm) and (cm - rm) > 2.0 and rp > cp + 0.15:
        print(f"  Real pairs beat the wrong-subject control by "
              f"{cm - rm:.2f} rpm and {rp - cp:.3f} in correlation.")
        print("  Respiration IS recovered from thermal. The pipeline works,")
        print("  so the heart-rate null is specific to the cardiac band —")
        print("  this is the positive control the paper needs.")
    else:
        print("  Real pairs do not clearly beat the control.")
        print("  Do not claim a positive control from this run. The likely")
        print("  cause is the PPG-derived reference rather than the thermal")
        print("  signal — check the paced-breathing row and the AM-RSA gap")
        print("  before concluding anything.")

    with open(os.path.join(OUTDIR, "resp_per_video.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "condition", "control_partner",
                    "trusted_control_partner", "low_quality_frac",
                    "am_rsa_gap", "mae_lag0", "pcc_lag0",
                    "mae_best", "pcc_best", "best_lag_s", "mae_ctrl", "pcc_ctrl"])
        w.writerows(rows)
    print(f"\nTable written to {OUTDIR}")


if __name__ == "__main__":
    main()
