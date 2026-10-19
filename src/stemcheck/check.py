"""check: sanity checks on a mixture and its separated stems. No reference stems needed."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .audio import Audio, StemcheckError, center_window, match_channels, resample
from .bleed import shared_content

# Default limits. These are judgement calls, not measured standards; change them with the options.
RESIDUAL_WARN_DB = -30.0   # stems leave more than this much of the mix unexplained -> WARN
RESIDUAL_FAIL_DB = -15.0   # ... -> FAIL
SILENT_DBFS = -70.0
CLIP_FRACTION = 0.001
DC_LIMIT = 0.01
DUP_CORR = 0.99


@dataclass
class Finding:
    level: str           # FAIL, WARN or INFO
    code: str
    message: str
    stem: str | None = None


@dataclass
class CheckResult:
    findings: list[Finding]
    sum_check: dict | None
    stems: dict[str, dict]
    pairs: list[dict]
    sr: int
    seconds_analysed: float

    def verdict(self) -> str:
        if any(f.level == "FAIL" for f in self.findings):
            return "FAIL"
        if any(f.level == "WARN" for f in self.findings):
            return "WARN"
        return "PASS"


def estimate_delay(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    """How many samples `a` is delayed relative to `b` (positive: a[t] looks like b[t - lag]). Mono inputs."""
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    if n < 16 or not np.any(a) or not np.any(b):
        return 0
    c = signal.correlate(a, b, mode="full", method="fft")
    lags = signal.correlation_lags(n, n, mode="full")
    keep = np.abs(lags) <= max_lag
    return int(lags[keep][np.argmax(c[keep])])


def _shift(a: np.ndarray, lag: int) -> np.ndarray:
    """Return `a` moved earlier by `lag` samples (undoing a delay), same length, zero-padded at the end."""
    if lag == 0:
        return a
    out = np.zeros_like(a)
    if lag > 0:
        out[:-lag] = a[lag:]
    else:
        out[-lag:] = a[:lag]
    return out


def _dbfs(x: np.ndarray) -> float:
    return float(10 * np.log10(max(float(np.mean(np.square(x, dtype=np.float64))), 1e-30)))


def check_stems(mixture: Audio, stems: dict[str, Audio], *, max_seconds: float | None = 120.0,
                residual_warn_db: float = RESIDUAL_WARN_DB, residual_fail_db: float = RESIDUAL_FAIL_DB,
                max_shared_db: float | None = None, shared: bool = True, n_perm: int = 19) -> CheckResult:
    """Sanity-check separated `stems` against the `mixture` they came from."""
    if len(stems) < 2:
        raise StemcheckError("need at least 2 stems to check")
    F: list[Finding] = []
    sr, ch = mixture.sr, mixture.ch
    M = np.nan_to_num(mixture.data, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(mixture.data)):
        F.append(Finding("FAIL", "non_finite", "the mixture contains NaN or infinite values"))
    if _dbfs(M) < SILENT_DBFS:
        raise StemcheckError("the mixture is silent, nothing to check against")

    # bring every stem to the mixture's rate, channel count and length
    matched, info = {}, {}
    for name in sorted(stems):
        a = stems[name]
        data = a.data
        d = {"sample_rate": a.sr, "channels": a.ch, "seconds": round(a.n / a.sr, 3)}
        if not np.all(np.isfinite(data)):
            F.append(Finding("FAIL", "non_finite", "contains NaN or infinite values", name))
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if a.sr != sr:
            F.append(Finding("WARN", "sample_rate", f"sample rate is {a.sr} Hz, the mixture is {sr} Hz (resampled for the checks)", name))
            data = resample(data, a.sr, sr)
        if data.shape[1] != ch:
            F.append(Finding("WARN", "channels", f"has {a.ch} channel(s), the mixture has {ch} (matched up for the checks)", name))
            data = match_channels(data, ch)
        if abs(data.shape[0] - M.shape[0]) > 2:
            diff = data.shape[0] - M.shape[0]
            F.append(Finding("WARN", "length", f"is {abs(diff) / sr:.2f} s {'longer' if diff > 0 else 'shorter'} than the mixture (trimmed for the checks)", name))
        matched[name] = data
        raw_peak = float(np.abs(a.data[np.isfinite(a.data)]).max()) if a.data.size else 0.0
        d["rms_dbfs"] = round(_dbfs(data), 2)
        d["peak"] = round(raw_peak, 4)
        info[name] = d
        if d["rms_dbfs"] < SILENT_DBFS:
            F.append(Finding("WARN", "silent", f"is silent (RMS {d['rms_dbfs']} dBFS). That is fine for an instrument that is not in the song, a problem otherwise", name))
        else:
            frac = float(np.mean(np.abs(a.data[np.isfinite(a.data)]) >= 0.999)) if a.data.size else 0.0
            if frac > CLIP_FRACTION or raw_peak > 1.0:
                F.append(Finding("WARN", "clipping", f"is clipped or too loud ({frac:.2%} of samples at full scale, peak {raw_peak:.2f})", name))
            dc = float(np.abs(a.data[np.isfinite(a.data)].mean(axis=0)).max())
            info[name]["dc_offset"] = round(dc, 5)
            if dc > DC_LIMIT:
                F.append(Finding("WARN", "dc_offset", f"has a DC offset of {dc:.3f} (a constant shift of the waveform)", name))

    n = min([M.shape[0]] + [a.shape[0] for a in matched.values()])
    sl = center_window(n, sr, max_seconds)
    Mw = M[:n][sl].astype(np.float64)
    W = {k: v[:n][sl].astype(np.float64) for k, v in matched.items()}
    S = sum(W.values())

    # do the stems add up to the mixture?
    num = float(np.sum((Mw - S) ** 2))
    den = float(np.sum(Mw ** 2))
    res_db = round(10 * np.log10(max(num / den, 1e-10)), 2)     # below -100 dB is rounding noise
    g = float(np.sum(Mw * S) / max(float(np.sum(S ** 2)), 1e-30)) if np.any(S) else 0.0
    gain_db = round(20 * np.log10(g), 2) if g > 0 else None
    lag = estimate_delay(S.mean(axis=1), Mw.mean(axis=1), int(0.1 * sr))
    aligned_db = None
    if lag != 0:
        Sa = _shift(S, lag)
        aligned_db = round(10 * np.log10(max(float(np.sum((Mw - Sa) ** 2)) / den, 1e-10)), 2)
    sumc = {"residual_db": res_db, "best_gain_db": gain_db, "delay_samples": lag, "residual_after_alignment_db": aligned_db}
    if res_db > residual_fail_db:
        F.append(Finding("FAIL", "sum", f"the stems do not add up to the mixture: {res_db} dB of the mix is unexplained (limit {residual_fail_db} dB)"))
    elif res_db > residual_warn_db:
        F.append(Finding("WARN", "sum", f"the stems only roughly add up to the mixture: {res_db} dB unexplained (usual limit {residual_warn_db} dB)"))
    if gain_db is not None and abs(gain_db) > 1.0:
        F.append(Finding("WARN", "gain", f"added together the stems are about {abs(gain_db):.1f} dB {'quieter' if gain_db > 0 else 'louder'} than the mixture"))
    if lag != 0 and aligned_db is not None and aligned_db < res_db - 3.0:
        F.append(Finding("WARN", "delay", f"the stems look delayed by {lag} samples ({1000 * lag / sr:.1f} ms); aligned, the unexplained part drops from {res_db} to {aligned_db} dB"))

    # two stems that are the same audio
    names = sorted(W)
    mono = {k: v.mean(axis=1) for k, v in W.items()}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if info[a]["rms_dbfs"] < SILENT_DBFS or info[b]["rms_dbfs"] < SILENT_DBFS:
                continue
            x, y = mono[a], mono[b]
            c = float(np.sum(x * y) / max(np.sqrt(np.sum(x * x) * np.sum(y * y)), 1e-30))
            if abs(c) >= DUP_CORR:
                F.append(Finding("FAIL", "duplicate", f"{a} and {b} are almost the same audio (correlation {c:.3f})"))

    # shared content between pairs (a hint, see README)
    pairs = []
    if shared:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if info[a]["rms_dbfs"] < SILENT_DBFS or info[b]["rms_dbfs"] < SILENT_DBFS:
                    continue
                r = shared_content(mono[a], mono[b], sr, n_perm=n_perm)
                r.update({"a": a, "b": b})
                pairs.append(r)
                if max_shared_db is not None and r["excess_db"] is not None and r["excess_db"] > max_shared_db and r["p_value"] <= 0.05:
                    F.append(Finding("WARN", "shared", f"{a} and {b} share content beyond chance ({r['excess_db']} dB, p={r['p_value']:.2f})"))
    return CheckResult(F, sumc, info, pairs, sr, round(Mw.shape[0] / sr, 2))
