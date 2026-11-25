"""Task 0 - Stimulus-brain alignment in primary auditory cortex (A1).

1. A1 BOLD time course per subject (mean of the raw signal over the voxels of A1_L + A1_R, then
   z-scored, as in the assignment text) and group average.  The convention of the course example
   notebook (z-score every voxel before averaging) is run as a check.
2. Audio amplitude envelope: |Hilbert|, 8-Hz low-pass, averaged within 1.5-s TR bins.
3. Normalised cross-correlation (xcorr 'coeff') envelope vs. BOLD at lags -10..+10 TRs.
4. Significance: phase-randomised surrogate envelopes (N_SURR random draws) and ALL admissible
   circular shifts of the envelope (shift of at least 2*MAX_LAG TRs from zero in either direction;
   240 distinct surrogates), each with the maximum over lags as the test statistic.
5. Same analysis after convolving the envelope with a canonical (SPM double-gamma) HRF.

Run:  python src/task0_a1_alignment.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, correlate, hilbert, sosfiltfilt
from scipy.stats import gamma

from common import AUDIO, FIGS, N_TR, RESULTS, TR, load_roi_data, subject_files, zscore

MAX_LAG = 10          # TRs
N_SURR = 2000         # phase-randomised surrogates (the circular-shift null is exhaustive, see circ_shift)
SCAN_START_TR = 9     # scan TR 0 = 13.5 s into the .wav (end of the ~13-s music intro); same crop
                      # as the course example code (t[9:289]).  It is also the offset at which a
                      # canonical-HRF model fits best (offset scan below), but that scan presumes the
                      # canonical delay, so the crop and the hemodynamic lag are not separately identified.
rng = np.random.default_rng(0)


def audio_envelope_tr():
    """Envelope of the whole .wav in consecutive 1.5-s bins (291 complete bins for 438 s)."""
    fs, y = wavfile.read(AUDIO)
    y = y.astype(float).mean(axis=1)                       # stereo -> mono
    env = np.abs(hilbert(y, N=1 << int(np.ceil(np.log2(len(y))))))[: len(y)]
    env = sosfiltfilt(butter(4, 8, fs=fs, output="sos"), env)  # 8-Hz low-pass
    n = int(round(fs * TR))
    nb = len(env) // n
    return env[: nb * n].reshape(nb, n).mean(axis=1), fs, y


def spm_hrf(tr, duration=32.0):
    """Canonical double-gamma HRF (SPM defaults) sampled every `tr` seconds."""
    t = np.arange(0, duration, tr)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    return h / h.sum()


def xcorr(env, bold, max_lag=MAX_LAG):
    """MATLAB-style xcorr(...,'coeff') of z-scored signals.

    r[lag] = (1/N) sum_t env[t] * bold[t + lag]; a POSITIVE lag means BOLD follows the sound.
    `env` may be (n_surrogates, N); `bold` is (N,). Returns lags and r (..., 2*max_lag+1).
    """
    env = np.atleast_2d(zscore(env, axis=-1))
    bold = zscore(bold)
    n = len(bold)
    lags = np.arange(-max_lag, max_lag + 1)
    r = np.stack([(env[:, : n - L] * bold[L:]).sum(1) if L >= 0 else (env[:, -L:] * bold[: n + L]).sum(1)
                  for L in lags], axis=1) / n
    return lags, r.squeeze()


def phase_randomise(x, n):
    """Surrogates with the same power spectrum as x but random Fourier phases."""
    f = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, (n, len(f)))
    ph[:, 0] = 0
    if len(x) % 2 == 0:
        ph[:, -1] = 0
    return np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=len(x), axis=1)


def circ_shift(x, n=None):
    """All admissible circularly shifted copies of x: every shift that is at least 2*MAX_LAG TRs away
    from zero in either direction (len(x) - 4*MAX_LAG = 240 distinct surrogates for 280 TRs).

    The null is enumerated exhaustively instead of sampled, so the smallest attainable p-value is
    1 / (number of shifts + 1) = 1/241.  `n` is ignored (kept so both null generators share a signature).
    """
    return np.stack([np.roll(x, s) for s in range(2 * MAX_LAG, len(x) - 2 * MAX_LAG)])


def peak(lags, r):
    k = int(np.argmax(r))
    return int(lags[k]), float(r[k])


def main():
    roi = load_roi_data()
    ia = [k for k, l in enumerate(roi["labels"]) if l.split()[1] in ("A1_L", "A1_R")]
    a1 = np.concatenate([roi["X"][k] for k in ia], axis=2)          # (18, 280, V_A1)
    subj = zscore(a1.mean(axis=2), axis=1)                          # (18, 280): mean over voxels, then z
    group = zscore(subj.mean(axis=0))
    print(f"A1 ROIs: {[roi['labels'][k] for k in ia]}, {a1.shape[2]} voxels")

    env_full, fs, y = audio_envelope_tr()
    onset = SCAN_START_TR
    env = zscore(env_full[onset: onset + N_TR])
    env_hrf = zscore(np.convolve(env_full, spm_hrf(TR))[: len(env_full)][onset: onset + N_TR])

    res = {"n_voxels_A1": int(a1.shape[2]), "scan_start_TR": onset, "audio_s": len(y) / fs,
           "n_tr_bins_audio": int(len(env_full)), "max_lag_TR": MAX_LAG,
           # subject order = sorted(glob) = order of every per-subject list in this file
           "subjects": [Path(f).stem for f in subject_files()]}
    # each hemisphere separately (raw envelope): the group peak lies between lags 3 and 4
    res["hemispheres"] = {}
    for k in ia:
        gh = zscore(zscore(roi["X"][k].mean(axis=2), axis=1).mean(axis=0))
        lags_h, r_h = xcorr(env, gh)
        res["hemispheres"][roi["labels"][k]] = {"peak_lag_TR": peak(lags_h, r_h)[0], "peak_r": peak(lags_h, r_h)[1],
                                                "r_by_lag": r_h.tolist()}
    curves = {}
    for name, e in (("raw", env), ("hrf", env_hrf)):
        lags, rg = xcorr(e, group)
        rs = np.stack([xcorr(e, s)[1] for s in subj])
        pk_g = peak(lags, rg)
        pk_s = [peak(lags, r) for r in rs]
        out = {"group_peak_lag_TR": pk_g[0], "group_peak_lag_s": pk_g[0] * TR, "group_peak_r": pk_g[1],
               "group_r_at_lag0": float(rg[MAX_LAG]), "lags_TR": lags.tolist(), "group_r_by_lag": rg.tolist(),
               "subj_peak_lag_TR": [p[0] for p in pk_s], "subj_peak_r": [p[1] for p in pk_s]}
        # null distributions of the peak (max over lags) correlation
        for null, make in (("phase", phase_randomise), ("circ", circ_shift)):
            surr = make(e, N_SURR)
            n_surr = len(surr)                                  # 2000 (phase) or 240 (all circular shifts)
            out[f"n_surr_{null}"] = n_surr
            null_g = xcorr(surr, group)[1].max(axis=1)
            out[f"group_n_exceed_{null}"] = int((null_g >= pk_g[1]).sum())
            out[f"group_p_{null}"] = float((1 + (null_g >= pk_g[1]).sum()) / (1 + n_surr))
            out[f"null95_{null}"] = float(np.quantile(null_g, 0.95))
            null_s = np.stack([xcorr(surr, s)[1].max(axis=1) for s in subj])   # (18, N_SURR)
            p_s = (1 + (null_s >= np.array([p[1] for p in pk_s])[:, None]).sum(1)) / (1 + n_surr)
            out[f"subj_p_{null}"] = p_s.tolist()
            if name == "raw" and null == "phase":
                null_plot = null_g
        res[name] = out
        curves[name] = (lags, rg, rs)
        print(f"[{name}] group peak lag {pk_g[0]} TR ({pk_g[0]*TR:.1f} s), r={pk_g[1]:.3f}, "
              f"p_phase={out['group_p_phase']:.4f}, p_circ={out['group_p_circ']:.4f}; "
              f"subjects: lag {np.mean(out['subj_peak_lag_TR']):.2f}±{np.std(out['subj_peak_lag_TR']):.2f} TR, "
              f"r {np.mean(out['subj_peak_r']):.3f}, n sig (p<.05, phase) "
              f"{int((np.array(out['subj_p_phase']) < .05).sum())}/18")

    # Where does the scan start inside the .wav?  Slide the 280-TR window over the HRF-convolved
    # envelope of the whole file and correlate at zero lag.  NOT an independent check of the crop: it
    # assumes the canonical HRF delay, so a different true delay would shift the best offset.
    conv_full = np.convolve(env_full, spm_hrf(TR))[: len(env_full)]
    offs = np.arange(0, len(env_full) - N_TR + 1)
    r_off = [float(np.corrcoef(conv_full[o: o + N_TR], group)[0, 1]) for o in offs]
    res["window_offset_scan"] = {"offset_TR": offs.tolist(), "r": r_off, "best_offset_TR": int(offs[np.argmax(r_off)])}
    print("best scan-start offset in wav (HRF model, zero lag):", res["window_offset_scan"]["best_offset_TR"], "TR")
    # The same raw-envelope analysis if the scan had started one TR later in the .wav (10 TRs = 15 s,
    # the onset of the story itself): shows how the crop trades against the estimated lag.
    alt = SCAN_START_TR + 1
    lags_a, r_a = xcorr(zscore(env_full[alt: alt + N_TR]), group)
    res["alt_scan_start"] = {"scan_start_TR": alt, "raw_group_peak_lag_TR": peak(lags_a, r_a)[0],
                             "raw_group_peak_r": peak(lags_a, r_a)[1]}
    # Check: the course example notebook z-scores every voxel before averaging (equal weight per voxel
    # instead of weight proportional to the voxel's signal amplitude).  Raw envelope, group level.
    subj_vz = zscore(zscore(a1, axis=1).mean(axis=2), axis=1)
    lags_v, r_v = xcorr(env, zscore(subj_vz.mean(axis=0)))
    res["voxel_zscore_first"] = {
        "raw_group_peak_lag_TR": peak(lags_v, r_v)[0], "raw_group_peak_r": peak(lags_v, r_v)[1],
        "min_subj_corr_with_primary": float(min(np.corrcoef(x, y)[0, 1] for x, y in zip(subj, subj_vz))),
        "subj_peak_lag_TR": [peak(*xcorr(env, s))[0] for s in subj_vz]}
    print("check, voxel-wise z-score before averaging:", {k: v for k, v in res["voxel_zscore_first"].items()})
    json.dump(res, open(RESULTS / "task0_alignment.json", "w"), indent=1)

    # ---------------- figures ----------------
    t = np.arange(N_TR) * TR
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
    L = res["raw"]["group_peak_lag_TR"]
    ax[0].plot(t, env, color="0.45", lw=1, label="audio envelope (TR-binned, z)")
    ax[0].plot(t, group, color="C3", lw=1.3, label="group A1 BOLD (z)")
    ax[0].set_title("A1 BOLD and audio envelope (envelope cropped to the scan window, no shift)")
    ax[1].plot(t, env_hrf, color="0.45", lw=1, label="envelope * canonical HRF (z)")
    ax[1].plot(t, group, color="C3", lw=1.3, label="group A1 BOLD (z)")
    ax[1].set_title(f"After convolving the envelope with the canonical HRF "
                    f"(r = {res['hrf']['group_r_at_lag0']:.2f} at lag 0)")
    ax[1].set_xlabel("Time from scan start (s)")
    for a in ax:
        a.legend(loc="upper right", ncol=2, fontsize=8, frameon=False)
        a.set_ylabel("z"); a.set_xlim(0, N_TR * TR)
    fig.tight_layout(); fig.savefig(FIGS / "task0_timecourses.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
    for a, name, ttl in ((ax[0], "raw", "Raw envelope vs. A1"), (ax[1], "hrf", "HRF-convolved envelope vs. A1")):
        lags, rg, rs = curves[name]
        a.plot(lags, rs.T, color="0.75", lw=0.7)
        a.plot([], [], color="0.75", lw=0.7, label="single subjects (n = 18)")
        a.plot(lags, rg, color="C3", lw=2.2, marker="o", ms=3.5, label="group-average A1")
        a.axhline(res[name]["null95_phase"], color="k", ls="--", lw=0.8, label="95th pct. of null peak (group)")
        a.axvline(0, color="k", lw=0.5)
        pk = res[name]["group_peak_lag_TR"]
        where = "lag 0" if pk == 0 else f"lag {pk:+d} TR ({pk * TR:+.1f} s)".replace("-", "\u2212")
        ttl = f"{ttl}\npeak: {where}, r = {res[name]['group_peak_r']:.2f}"
        a.set_xlabel("Lag (TRs; positive = BOLD follows sound)"); a.set_ylabel("Cross-correlation (r)")
        a.set_title(ttl, fontsize=10); a.set_xticks(lags[::2])
        sec = a.secondary_xaxis("top", functions=(lambda x: x * TR, lambda x: x / TR)); sec.set_xlabel("Lag (s)", fontsize=8)
    ax[2].hist(null_plot, bins=40, color="0.7")
    ax[2].axvline(res["raw"]["group_peak_r"], color="C3", lw=2)
    ax[2].set_title(f"Null of peak r, {N_SURR} phase-randomised envelopes\n"
                    f"observed r = {res['raw']['group_peak_r']:.2f}, p = {res['raw']['group_p_phase']:.4f}", fontsize=10)
    ax[2].set_xlabel("Peak cross-correlation over lags \u221210 to +10 TRs"); ax[2].set_ylabel("Surrogates")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.07, 1, 1]); fig.savefig(FIGS / "task0_xcorr.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
