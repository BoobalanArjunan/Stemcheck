from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .check import CheckResult
from .score import ScoreResult

CHECK_NOTE = ("Note: a PASS means none of these checks found a problem. It does not prove the separation is good. "
              "The 'shared content' numbers are a weak hint (see the README); leakage is only measured reliably "
              "with reference stems (stemcheck score).")
SCORE_NOTE = ("How to read this: each row is one estimated stem, each column one other instrument. The number is how "
              "loud that instrument's leak is compared with the stem's own content (more negative = cleaner). "
              "It measures the steady, filter-like part of the leak. Fast-changing leaks show up as 'unexplained'. "
              "For mask-based separators it is typically a little lower than the true leak (about 4 dB in our synthetic test).")


def _f(x, nd=1, none="n/a"):
    if x is None:
        return none
    if x <= -100:
        return "<-100"
    if x >= 100:
        return ">100"
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------- check

def check_to_dict(res: CheckResult) -> dict:
    return {
        "verdict": res.verdict(), "sample_rate": res.sr, "seconds_analysed": res.seconds_analysed,
        "findings": [asdict(f) for f in res.findings], "sum_check": res.sum_check,
        "stems": res.stems, "shared_content": res.pairs, "note": CHECK_NOTE,
    }


def check_to_text(res: CheckResult) -> str:
    L = [f"stemcheck verdict: {res.verdict()}",
         f"Analysed {res.seconds_analysed:g} s at {res.sr} Hz, {len(res.stems)} stems: {', '.join(res.stems)}", ""]
    s = res.sum_check or {}
    L.append(f"Do the stems add up to the mixture?  {_f(s.get('residual_db'))} dB of the mix is unexplained "
             "(closer to -inf is better)")
    if s.get("best_gain_db") is not None:
        L.append(f"  overall level of the stems versus the mixture: {s['best_gain_db']:+.2f} dB off")
    if s.get("delay_samples"):
        L.append(f"  stems look delayed by {s['delay_samples']} samples; unexplained after aligning: {_f(s.get('residual_after_alignment_db'))} dB")
    L.append("")
    probs = [f for f in res.findings if f.level != "INFO"]
    if probs:
        L.append("Findings:")
        for f in probs:
            who = f"[{f.stem}] " if f.stem else ""
            L.append(f"  {f.level:4s} {who}{f.message}")
    else:
        L.append("No problems found.")
    L.append("")
    if res.pairs:
        L.append("Shared content between stems (weak hint; 'chance' removed, dB of coherence, lower is cleaner):")
        for p in res.pairs:
            L.append(f"  {p['a']} / {p['b']}: {_f(p['excess_db'], 1, 'none above chance')}"
                     + (f" dB (p={p['p_value']:.2f})" if p['excess_db'] is not None else ""))
        L.append("")
    L.append(CHECK_NOTE)
    return "\n".join(L)


# ---------------------------------------------------------------- score

def score_to_dict(res: ScoreResult) -> dict:
    return {"verdict": res.verdict(), "sample_rate": res.sr, "seconds_analysed": res.seconds_analysed,
            "max_bleed_db": res.max_bleed_db, "min_sdr_db": res.min_sdr_db, "notes": res.notes,
            "stems": [asdict(s) for s in res.stems], "note": SCORE_NOTE}


def score_to_text(res: ScoreResult) -> str:
    v = res.verdict()
    L = [f"stemcheck score: {v if v else 'numbers only (no limits set)'}",
         f"Analysed {res.seconds_analysed:g} s at {res.sr} Hz.", ""]
    for n in res.notes:
        L.append(f"note: {n}")
    if res.notes:
        L.append("")
    sources = sorted({b.source for s in res.stems for b in s.bleed})
    head = f"{'stem':10s} {'quality':>8s} {'total leak':>10s} {'unexplained':>11s} | " + " ".join(f"{'from ' + x:>12s}" for x in sources)
    L.append(head)
    L.append("-" * len(head))
    for s in res.stems:
        row = f"{s.name:10s} {_f(s.sdr_like_db):>8s} {_f(s.total_bleed_vs_target_db):>10s} {_f(s.unexplained_vs_target_db):>11s} | "
        cells = []
        by = {b.source: b for b in s.bleed}
        for x in sources:
            b = by.get(x)
            if b is None:
                cells.append(f"{'-':>12s}")
            elif b.source_silent or (b.detected and b.vs_target_db is None):
                cells.append(f"{'n/a':>12s}")
            elif not b.detected:
                cells.append(f"{'<floor':>12s}")
            else:
                cells.append(f"{_f(b.vs_target_db):>12s}")
        L.append(row + " ".join(cells))
    L += ["", "quality = target compared with all leak and error, in dB (higher is better; a simple stand-in for SDR, use museval for published numbers).",
          "total leak, unexplained, from X = dB relative to the stem's own content (more negative is cleaner). '<floor' = below what chance would show.", ""]
    for s in res.stems:
        worst = [b for b in s.bleed if b.detected and b.vs_target_db is not None][:2]
        if worst:
            L.append(f"{s.name}: biggest leaks: " + ", ".join(f"{b.source} ({b.vs_target_db:.1f} dB vs stem, {b.leak_dbfs:.1f} dBFS)" for b in worst))
        elif any(b.vs_target_db is not None for b in s.bleed):
            L.append(f"{s.name}: no leak above the measurement floor")
        if s.timeline:
            vals = [(w.total_bleed_vs_target_db, w.start_s) for w in s.timeline if w.total_bleed_vs_target_db is not None]
            if vals:
                db, t = max(vals)
                L.append(f"{' ' * len(s.name)}  worst {s.timeline[0].seconds:g} s section starts at {t:g} s ({db:.1f} dB)")
        for n in s.notes:
            L.append(f"{' ' * len(s.name)}  note: {n}")
    L += ["", SCORE_NOTE]
    return "\n".join(L)


def _write(path: Path, text: str):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_check_reports(res: CheckResult, out: Path):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "stemcheck_report.json", json.dumps(check_to_dict(res), indent=2) + "\n")
    _write(out / "stemcheck_report.txt", check_to_text(res) + "\n")


def write_score_reports(res: ScoreResult, out: Path):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "stemcheck_score.json", json.dumps(score_to_dict(res), indent=2) + "\n")
    _write(out / "stemcheck_score.txt", score_to_text(res) + "\n")
    with open(out / "bleed_matrix.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["stem", "source", "vs_target_db", "vs_source_db", "floor_vs_source_db", "detected", "leak_dbfs", "source_silent"])
        for s in res.stems:
            for b in s.bleed:
                w.writerow([s.name, b.source, b.vs_target_db, b.vs_source_db, b.floor_vs_source_db, b.detected, b.leak_dbfs, b.source_silent])
    with open(out / "timeline.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["stem", "start_s", "seconds", "total_bleed_vs_target_db", "worst_source", "worst_vs_target_db", "sdr_like_db"])
        for s in res.stems:
            for t in s.timeline:
                w.writerow([s.name, t.start_s, t.seconds, t.total_bleed_vs_target_db, t.worst_source, t.worst_vs_target_db, t.sdr_like_db])
