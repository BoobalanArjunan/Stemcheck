"""The two measurements that look at leakage between stems.

1. interference(): with REFERENCE stems. Split an estimated stem into "the target", "a copy of each other
   instrument" and "everything else", by least squares on the spectrogram. Bias-corrected, so that clean
   estimates read as clean instead of showing a small random amount of leak.
2. shared_content(): with NO references. How much do two stems have in common beyond chance?
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def nfft_for(sr: int) -> int:
    return 2048 if sr >= 32000 else 1024


def stft(x: np.ndarray, sr: int) -> np.ndarray:
    """x: (..., n) -> complex64 (..., F, T). 50% overlap Hann, no padding."""
    nfft = nfft_for(sr)
    _, _, Z = signal.stft(x, nperseg=nfft, noverlap=nfft // 2, boundary=None, padded=False, window="hann")
    return Z.astype(np.complex64)


def _db(x: float) -> float | None:
    return round(10.0 * np.log10(x), 3) if x > 1e-30 else None


def _fit(R: np.ndarray, Y: np.ndarray, t0: int, t1: int, chunk: int = 64):
    """Least-squares fit of estimate Y on references R over frames t0..t1, one filter per frequency bin.

    Returns energies (all in spectrogram units): leak[j] (bias-corrected), leak_raw[j], bias[j], src[j] and the
    unexplained residual. R: (K, F, T); Y: (F, T). The caller makes sure there are enough frames.
    """
    K, F, _ = R.shape
    Tn = t1 - t0
    leak = np.zeros(K)
    leak_raw = np.zeros(K)
    bias = np.zeros(K)
    src = np.zeros(K)
    res = 0.0
    eye = np.eye(K)
    for f0 in range(0, F, chunk):
        X = np.transpose(R[:, f0:f0 + chunk, t0:t1], (1, 2, 0)).astype(np.complex128)   # (f, T, K)
        y = Y[f0:f0 + chunk, t0:t1].astype(np.complex128)                                # (f, T)
        Xh = X.conj().transpose(0, 2, 1)
        G = Xh @ X
        tr = np.trace(G, axis1=1, axis2=2).real[:, None, None] / K
        Gr = G + 1e-9 * np.maximum(tr, 1e-30) * eye           # keeps silent references from breaking the solve
        a = np.linalg.solve(Gr, (Xh @ y[..., None]))[..., 0]  # (f, K)
        e = y - (X @ a[..., None])[..., 0]
        sig2 = (np.abs(e) ** 2).sum(axis=1) / (Tn - K)        # unexplained power per observation, per bin
        ginv = np.linalg.inv(Gr)
        xj = (np.abs(X) ** 2).sum(axis=1)                     # (f, K) energy of each reference
        comp = (np.abs(a) ** 2) * xj                          # (f, K) energy of each fitted component
        b = sig2[:, None] * xj * np.einsum("fkk->fk", ginv).real   # what pure chance would add to comp
        leak_raw += comp.sum(axis=0)
        bias += b.sum(axis=0)
        src += xj.sum(axis=0)
        leak += np.maximum(comp - b, 0.0).sum(axis=0)
        res += float((np.abs(e) ** 2).sum())
    return {"leak": leak, "leak_raw": leak_raw, "bias": bias, "src": src, "res": res}


def null_shifts(T: int, sr: int, hop: int, n: int, rng) -> list[int]:
    """Random time shifts (in frames) for the chance measurements: at least about 2 seconds, so that a
    shifted instrument is no longer lined up with the estimate."""
    min_shift = max(min(int(round(2.0 * sr / hop)), T // 4), 1)
    return [int(rng.randint(min_shift, T - min_shift + 1)) for _ in range(n)]


def interference(refs: np.ndarray, est: np.ndarray, target: int, sr: int, window_s: float | None = 10.0,
                 refs_stft=None, n_null: int = 3, seed: int = 0):
    """Measure how much of each reference leaks into `est`.

    refs: (K, n, ch) reference stems; est: (n, ch) the estimate of reference `target`.
    Returns a dict with the overall numbers and (if window_s) a per-window timeline.
    Energies are spectrogram energies; callers turn them into dB ratios.

    What counts as "chance": two unrelated signals always show a little relationship in a finite recording.
    We estimate that level three ways and use the largest: a formula (assumes independent frames), `n_null`
    re-measurements with the other instruments moved in time (keeps their tones and phase structure, so it copes
    with sustained notes where neighbouring frames are not independent), and `n_null` re-measurements with their
    phases scrambled (keeps when and where they are loud, so it copes with errors that are loud at the same
    moments as an instrument). Each destroys any real leak. The timeline uses the formula only, to stay fast.
    """
    K, n, ch = refs.shape
    nfft = nfft_for(sr)
    hop = nfft // 2
    per_ch_R = refs_stft if refs_stft is not None else [stft(refs[:, :, c], sr) for c in range(ch)]
    Ys = [stft(est[:, c], sr) for c in range(ch)]
    T = per_ch_R[0].shape[2]
    if T - K < 4:
        raise ValueError("audio too short to measure (need at least a few seconds)")
    others = [j for j in range(K) if j != target]

    def pooled(t0, t1, Rs=per_ch_R):
        tot = {"leak": np.zeros(K), "leak_raw": np.zeros(K), "bias": np.zeros(K), "src": np.zeros(K), "res": 0.0}
        for R, Y in zip(Rs, Ys):
            r = _fit(R, Y, t0, t1)
            for k in tot:
                tot[k] = tot[k] + r[k]
        return tot

    def summarize(tot, null_raw=None):
        leak = tot["leak"].copy()
        chance_mean = tot["bias"].copy()
        chance_max = tot["bias"].copy()
        if null_raw is not None and len(null_raw):
            nr = np.array(null_raw)                                   # (n_null, K)
            chance_mean = np.maximum(chance_mean, nr.mean(axis=0))
            chance_max = np.maximum(chance_max, nr.max(axis=0))
            leak = np.maximum(tot["leak_raw"] - chance_mean, 0.0)
        tgt = max(tot["leak_raw"][target] - tot["bias"][target], 0.0)
        out = {"target_energy": tgt, "total_leak": float(leak[others].sum()), "unexplained": tot["res"], "per_source": {}}
        for j in others:
            out["per_source"][j] = {
                "leak_energy": float(leak[j]), "bias_energy": float(chance_mean[j]), "chance_max_energy": float(chance_max[j]),
                "raw_energy": float(tot["leak_raw"][j]), "src_energy": float(tot["src"][j]),
            }
        return out

    null_raw = []
    if n_null:
        rng = np.random.RandomState(seed)
        shifts = null_shifts(T, sr, hop, n_null, rng)
        for kind in ("shift", "phase"):
            for k in range(n_null):
                Rs = []
                if kind == "shift":
                    # move the other instruments in time: keeps their tones and phase structure, breaks real leaks
                    shift = shifts[k]
                    for R in per_ch_R:
                        Rn = R.copy()
                        for j in others:
                            Rn[j] = np.roll(R[j], shift, axis=1)
                        Rs.append(Rn)
                else:
                    # scramble their phases: keeps when they are loud and where, breaks real leaks
                    for R in per_ch_R:
                        Rn = R.copy()
                        for j in others:
                            Rn[j] = R[j] * np.exp(1j * rng.uniform(0.0, 2 * np.pi, R[j].shape)).astype(np.complex64)
                        Rs.append(Rn)
                null_raw.append(pooled(0, T, Rs)["leak_raw"])
    overall = summarize(pooled(0, T), null_raw)
    timeline = []
    if window_s:
        w = max(int(round(window_s * sr / hop)), K + 8)
        for t0 in range(0, T - w + 1, w):
            timeline.append({"start_s": round(t0 * hop / sr, 3), "seconds": round(w * hop / sr, 3),
                             **summarize(pooled(t0, t0 + w))})
    return {"overall": overall, "timeline": timeline}


def shared_content(a: np.ndarray, b: np.ndarray, sr: int, n_perm: int = 19, seed: int = 0) -> dict:
    """How much do two stems have in common beyond chance? (a, b: (n,) mono signals)

    Uses magnitude-squared coherence per frequency bin, weighted toward bins where both stems have energy.
    The 'chance' level comes from shifting one stem in time by a random amount and measuring again.
    Returns the statistic, the chance level, the excess, and a p-value from the shifts.
    """
    A = stft(a, sr).astype(np.complex128)
    B = stft(b, sr).astype(np.complex128)
    F, T = A.shape
    Pa = (np.abs(A) ** 2).sum(axis=1)
    Pb = (np.abs(B) ** 2).sum(axis=1)
    denom = np.maximum(Pa * Pb, 1e-30)
    w = Pa * Pb
    if w.sum() <= 0 or T < 8:
        return {"statistic": 0.0, "chance": 0.0, "excess": 0.0, "excess_db": None, "p_value": 1.0}
    w = w / w.sum()

    def stat(Bs):
        return float((np.abs((A * Bs.conj()).sum(axis=1)) ** 2 / denom * w).sum())

    s = stat(B)
    hop = nfft_for(sr) // 2
    min_shift = max(int(round(sr / hop)), 1)            # at least about one second
    if T <= 2 * min_shift + 2:
        min_shift = max(T // 4, 1)
    rng = np.random.RandomState(seed)
    nulls = np.array([stat(np.roll(B, rng.randint(min_shift, T - min_shift + 1), axis=1)) for _ in range(n_perm)])
    chance = float(nulls.mean())
    excess = max(s - chance, 0.0)
    p = float((1 + (nulls >= s).sum()) / (n_perm + 1))
    return {"statistic": s, "chance": chance, "excess": excess, "excess_db": _db(excess), "p_value": p}
