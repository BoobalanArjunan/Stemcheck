"""score: measure leakage in estimated stems against reference stems."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import Audio, StemcheckError, center_window, match_channels, resample
from .bleed import interference, stft
from .check import estimate_delay


DETECT_FACTOR = 3.0   # a leak counts as "detected" when the measured energy is this many times the strongest chance level
LIMIT_DB = 100.0   # below this the numbers are rounding noise of the audio itself, so they are capped


def _db(num: float, den: float) -> float | None:
    if num <= 1e-30 or den <= 1e-30:
        return None
    return round(float(min(max(10.0 * np.log10(num / den), -LIMIT_DB), LIMIT_DB)), 2)


@dataclass
class BleedEntry:
    source: str
    vs_target_db: float | None      # leak level compared with the stem's own (target) content
    vs_source_db: float | None      # leak level compared with the leaking instrument itself
    floor_vs_source_db: float | None  # what pure chance would show (the measurement floor)
    detected: bool                  # leak is above the chance level
    leak_dbfs: float | None         # absolute level of the leak (RMS, dB full scale)
    source_silent: bool = False     # the leaking instrument is silent, so there is nothing to leak


@dataclass
class WindowScore:
    start_s: float
    seconds: float
    total_bleed_vs_target_db: float | None
    worst_source: str | None
    worst_vs_target_db: float | None
    sdr_like_db: float | None


@dataclass
class StemScore:
    name: str
    target_gain_db: float | None
    sdr_like_db: float | None
    total_bleed_vs_target_db: float | None
    unexplained_vs_target_db: float | None
    bleed: list[BleedEntry]
    timeline: list[WindowScore]
    delay_samples: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class ScoreResult:
    stems: list[StemScore]
    sr: int
    seconds_analysed: float
    notes: list[str]
    max_bleed_db: float | None = None
    min_sdr_db: float | None = None

    def verdict(self) -> str | None:
        """PASS or FAIL against the thresholds you set; None if you set none (numbers only)."""
        if self.max_bleed_db is None and self.min_sdr_db is None:
            return None
        for s in self.stems:
            if self.max_bleed_db is not None and s.total_bleed_vs_target_db is not None \
                    and s.total_bleed_vs_target_db > self.max_bleed_db:
                return "FAIL"
            if self.min_sdr_db is not None and s.sdr_like_db is not None and s.sdr_like_db < self.min_sdr_db:
                return "FAIL"
        return "PASS"


def _prepare(refs: dict[str, Audio], ests: dict[str, Audio]):
    notes: list[str] = []
    if len(refs) < 2:
        raise StemcheckError("need at least 2 reference stems (leakage is measured between instruments)")
    rates = [a.sr for a in refs.values()]
    sr = max(set(rates), key=lambda r: (rates.count(r), r))
    ref_arr = {k: (resample(a.data, a.sr, sr) if a.sr != sr else a.data) for k, a in refs.items()}
    if len(set(rates)) > 1:
        notes.append(f"reference stems had different sample rates; all resampled to {sr} Hz")
    ch = max(a.shape[1] for a in ref_arr.values())
    ref_arr = {k: match_channels(a, ch) for k, a in ref_arr.items()}
    lower = {k.lower(): k for k in ref_arr}
    est_arr, extra = {}, []
    for name, a in ests.items():
        key = lower.get(name.lower())
        if key is None:
            extra.append(name)
            continue
        d = resample(a.data, a.sr, sr) if a.sr != sr else a.data
        if a.sr != sr:
            notes.append(f"estimate {name!r} was {a.sr} Hz; resampled to {sr} Hz")
        if d.shape[1] != ch:
            notes.append(f"estimate {name!r} has {d.shape[1]} channel(s), references have {ch}; matched up")
        est_arr[key] = match_channels(d, ch)
    if extra:
        notes.append(f"estimates with no matching reference were ignored: {', '.join(sorted(extra))}")
    if not est_arr:
        raise StemcheckError("no estimate matches a reference by name (file names must match, e.g. vocals.wav)")
    n = min([a.shape[0] for a in ref_arr.values()] + [a.shape[0] for a in est_arr.values()])
    for name, a in est_arr.items():
        diff = a.shape[0] - ref_arr[name].shape[0]
        if abs(diff) > 2:
            notes.append(f"estimate {name!r} is {abs(diff) / sr:.2f} s {'longer' if diff > 0 else 'shorter'} than its reference; trimmed")
    return ref_arr, est_arr, sr, ch, n, notes


def score_stems(refs: dict[str, Audio], ests: dict[str, Audio], *, window_s: float | None = 10.0,
                max_seconds: float | None = 120.0, max_bleed_db: float | None = None,
                min_sdr_db: float | None = None, null_shifts: int = 3) -> ScoreResult:
    """Measure leakage in each estimate against the reference stems.

    refs / ests: name -> Audio. Names are matched ignoring case. Only the middle `max_seconds` are analysed.
    """
    ref_arr, est_arr, sr, ch, n, notes = _prepare(refs, ests)
    sl = center_window(n, sr, max_seconds)
    names = sorted(ref_arr)
    R = np.stack([ref_arr[k][:n][sl] for k in names])          # (K, n, ch)
    nn = R.shape[1]
    if nn < sr * 2:
        raise StemcheckError("audio is too short to measure leakage (need at least about 2 seconds)")
    R_stft = [stft(R[:, :, c], sr) for c in range(ch)]
    ref_rms_dbfs = {k: float(10 * np.log10(max(np.mean(R[i] ** 2), 1e-30))) for i, k in enumerate(names)}
    out = []
    for name in sorted(est_arr):
        i = names.index(name)
        est = est_arr[name][:n][sl]
        r = interference(R, est, i, sr, window_s, refs_stft=R_stft, n_null=null_shifts)
        o = r["overall"]
        tgt = o["target_energy"]
        src_t = o["per_source"]  # source index -> energies
        total_leak = o["total_leak"]
        entries = []
        for j, e in src_t.items():
            vs_src = _db(e["leak_energy"], e["src_energy"])
            entries.append(BleedEntry(
                source=names[j], vs_target_db=_db(e["leak_energy"], tgt), vs_source_db=vs_src,
                floor_vs_source_db=_db(e["bias_energy"], e["src_energy"]),
                detected=bool(e["raw_energy"] >= DETECT_FACTOR * e["chance_max_energy"] and e["leak_energy"] > 0
                              and vs_src is not None and vs_src > -LIMIT_DB + 10),
                leak_dbfs=None if vs_src is None else round(ref_rms_dbfs[names[j]] + vs_src, 2),
                source_silent=bool(e["src_energy"] <= 1e-30)))
        entries.sort(key=lambda b: (b.vs_target_db is None, -(b.vs_target_db or 0)))
        tl = []
        for w in r["timeline"]:
            worst, worst_db = None, None
            for j, e in w["per_source"].items():
                d = _db(e["leak_energy"], w["target_energy"])
                if d is not None and (worst_db is None or d > worst_db):
                    worst, worst_db = names[j], d
            tl.append(WindowScore(w["start_s"], w["seconds"], _db(w["total_leak"], w["target_energy"]), worst, worst_db,
                                  _db(w["target_energy"], w["total_leak"] + w["unexplained"])))
        lag = estimate_delay(est.mean(axis=1), R[i].mean(axis=1), int(0.05 * sr))
        snotes = []
        if abs(lag) >= 2:
            snotes.append(f"estimate looks delayed by {lag} samples ({1000 * lag / sr:.1f} ms) relative to its reference")
        if tgt <= 1e-30:
            snotes.append("the reference for this stem is silent (or the estimate has none of it), so levels compared with the target are not defined")
        out.append(StemScore(
            name=name, target_gain_db=_db(tgt, _target_src(R_stft, i)),
            sdr_like_db=_db(tgt, total_leak + o["unexplained"]),
            total_bleed_vs_target_db=_db(total_leak, tgt), unexplained_vs_target_db=_db(o["unexplained"], tgt),
            bleed=entries, timeline=tl, delay_samples=int(lag), notes=snotes))
    return ScoreResult(out, sr, round(nn / sr, 2), notes, max_bleed_db, min_sdr_db)


def _target_src(R_stft, i) -> float:
    return float(sum((np.abs(R[i].astype(np.complex128)) ** 2).sum() for R in R_stft))
