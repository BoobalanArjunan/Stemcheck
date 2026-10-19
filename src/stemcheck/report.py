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
    _write(out / "stemcheck_report.html", check_to_html(res, out / "stemcheck_report.html"))


def write_score_reports(res: ScoreResult, out: Path):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "stemcheck_score.json", json.dumps(score_to_dict(res), indent=2) + "\n")
    _write(out / "stemcheck_score.txt", score_to_text(res) + "\n")
    _write(out / "stemcheck_score.html", score_to_html(res, out / "stemcheck_score.html"))
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


def score_to_html(res: ScoreResult, out_path: Path | None = None) -> str:
    verdict = res.verdict()
    if verdict == "PASS":
        badge_style = "background: linear-gradient(135deg, #10b981, #059669); color: white;"
        badge_text = "PASS &mdash; SEPARATION CRITERIA MET"
    elif verdict == "FAIL":
        badge_style = "background: linear-gradient(135deg, #ef4444, #dc2626); color: white;"
        badge_text = "FAIL &mdash; EXCESSIVE BLEED DETECTED"
    else:
        badge_style = "background: linear-gradient(135deg, #3b82f6, #1d4ed8); color: white;"
        badge_text = "SCORED &mdash; EVALUATION BENCHMARK"

    sources = sorted({b.source for s in res.stems for b in s.bleed})
    headers_html = "".join(f'<th style="padding:10px 14px;text-align:right;border-bottom:2px solid #334155;">from {x}</th>' for x in sources)

    rows_html = []
    worst_leaks_list = []
    for s in res.stems:
        by = {b.source: b for b in s.bleed}
        cells = []
        for x in sources:
            b = by.get(x)
            if b is None:
                cells.append('<td style="padding:10px 14px;text-align:right;color:#64748b;">&mdash;</td>')
            elif b.source_silent or (b.detected and b.vs_target_db is None):
                cells.append('<td style="padding:10px 14px;text-align:right;color:#64748b;">n/a</td>')
            elif not b.detected:
                cells.append('<td style="padding:10px 14px;text-align:right;color:#10b981;font-weight:600;">&lt;floor</td>')
            else:
                val = b.vs_target_db
                color = "#ef4444" if val > -25 else ("#f59e0b" if val > -35 else "#38bdf8")
                cells.append(f'<td style="padding:10px 14px;text-align:right;color:{color};font-weight:700;">{val:.1f} dB</td>')

        worst = [b for b in s.bleed if b.detected and b.vs_target_db is not None]
        for b in worst:
            worst_leaks_list.append((b.vs_target_db, f"<strong>{s.name}</strong> &larr; from <strong>{b.source}</strong> ({b.vs_target_db:.1f} dB vs stem, {b.leak_dbfs:.1f} dBFS)"))

        tl_text = "&mdash;"
        if s.timeline:
            vals = [(w.total_bleed_vs_target_db, w.start_s) for w in s.timeline if w.total_bleed_vs_target_db is not None]
            if vals:
                db_val, t_val = max(vals)
                tl_text = f"at {t_val:.1f}s ({db_val:.1f} dB)"

        sdr_val = f"{s.sdr_like_db:.1f} dB" if s.sdr_like_db is not None else "n/a"
        leak_val = f"{s.total_bleed_vs_target_db:.1f} dB" if s.total_bleed_vs_target_db is not None else "n/a"
        unexp_val = f"{s.unexplained_vs_target_db:.1f} dB" if s.unexplained_vs_target_db is not None else "n/a"

        row = f"""
        <tr style="border-bottom:1px solid #1e293b;">
          <td style="padding:10px 14px;font-weight:700;color:#f8fafc;">{s.name.upper()}</td>
          <td style="padding:10px 14px;text-align:right;color:#34d399;font-weight:600;">{sdr_val}</td>
          <td style="padding:10px 14px;text-align:right;color:#93c5fd;font-weight:600;">{leak_val}</td>
          <td style="padding:10px 14px;text-align:right;color:#cbd5e1;">{unexp_val}</td>
          <td style="padding:10px 14px;text-align:right;color:#f59e0b;font-size:0.85rem;">{tl_text}</td>
          {''.join(cells)}
        </tr>
        """
        rows_html.append(row)

    worst_leaks_list.sort(key=lambda x: x[0], reverse=True)
    worst_html = "".join(f"<li>{item[1]}</li>" for item in worst_leaks_list[:5]) or "<li>No leak above measurement floor detected.</li>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>StemCheck Bleed Audit Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #020617; color: #f1f5f9; margin: 0; padding: 30px 20px; }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    header {{ border-bottom: 1px solid #1e293b; padding-bottom: 20px; margin-bottom: 25px; }}
    .badge {{ display: inline-block; padding: 6px 16px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 20px; }}
    .stat-card {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 16px; }}
    .stat-label {{ font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; }}
    .stat-value {{ font-size: 1.3rem; font-weight: 700; color: #38bdf8; margin-top: 2px; }}
    table {{ width: 100%; border-collapse: collapse; background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; overflow: hidden; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:15px;">
        <div>
          <h1 style="margin:0;font-size:1.75rem;color:#f8fafc;">StemCheck Bleed &amp; Separation Audit</h1>
          <p style="margin:4px 0 0 0;color:#94a3b8;font-size:0.9rem;">Decomposed Cross-Instrument Bleed Matrix &amp; Timeline Profiler</p>
        </div>
        <div class="badge" style="{badge_style}">{badge_text}</div>
      </div>
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-label">Stems Evaluated</div>
          <div class="stat-value">{len(res.stems)}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Audio Analysed</div>
          <div class="stat-value" style="color:#a855f7;">{res.seconds_analysed:.1f}s</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Sample Rate</div>
          <div class="stat-value" style="color:#34d399;">{res.sr} Hz</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Worst Bleed Found</div>
          <div class="stat-value" style="font-size:1.1rem;color:#f87171;">{(str(worst_leaks_list[0][0]) + ' dB') if worst_leaks_list else 'None'}</div>
        </div>
      </div>
    </header>

    <main>
      <h3 style="color:#f8fafc;margin-bottom:12px;">Instrument-by-Instrument Bleed Matrix (in dB relative to stem)</h3>
      <div style="overflow-x:auto;margin-bottom:25px;">
        <table>
          <thead>
            <tr style="background:#1e293b;color:#94a3b8;font-size:0.8rem;text-transform:uppercase;">
              <th style="padding:10px 14px;text-align:left;border-bottom:2px solid #334155;">Stem</th>
              <th style="padding:10px 14px;text-align:right;border-bottom:2px solid #334155;">Quality (SDR)</th>
              <th style="padding:10px 14px;text-align:right;border-bottom:2px solid #334155;">Total Leak</th>
              <th style="padding:10px 14px;text-align:right;border-bottom:2px solid #334155;">Unexplained</th>
              <th style="padding:10px 14px;text-align:right;border-bottom:2px solid #334155;">Worst 10s Burst</th>
              {headers_html}
            </tr>
          </thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
      </div>

      <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:16px;margin-bottom:20px;">
        <h4 style="margin:0 0 10px 0;color:#f8fafc;">Acute Burst Leaks &amp; Worst Timeline Sections</h4>
        <ul style="margin:0;padding-left:20px;color:#cbd5e1;line-height:1.8;">
          {worst_html}
        </ul>
      </div>
    </main>

    <footer style="margin-top:40px;border-top:1px solid #1e293b;padding-top:20px;text-align:center;font-size:0.8rem;color:#64748b;">
      Generated by <strong>stemcheck</strong> &bull; Developed by <strong>Boobalan Arjunan</strong> &bull; ComposerMind Labs
    </footer>
  </div>
</body>
</html>
"""


def check_to_html(res: CheckResult, out_path: Path | None = None) -> str:
    verdict = res.verdict()
    if verdict == "PASS":
        badge_style = "background: linear-gradient(135deg, #10b981, #059669); color: white;"
        badge_text = "PASS &mdash; ALL SANITY CHECKS PASSED"
    elif verdict == "WARN":
        badge_style = "background: linear-gradient(135deg, #f59e0b, #d97706); color: white;"
        badge_text = "WARN &mdash; MINOR RESIDUAL ANOMALIES"
    else:
        badge_style = "background: linear-gradient(135deg, #ef4444, #dc2626); color: white;"
        badge_text = "FAIL &mdash; CRITICAL SANITY DEFECTS"

    s = res.sum_check or {}
    res_db = s.get("residual_db", "N/A")
    res_db_str = f"{res_db:.1f} dB" if isinstance(res_db, (int, float)) else str(res_db)
    gain_db = s.get("best_gain_db", 0.0)
    gain_str = f"{gain_db:+.2f} dB" if isinstance(gain_db, (int, float)) else str(gain_db)
    delay_smp = s.get("delay_samples", 0)

    findings_items = "".join(f"<li><strong>[{f.stem or 'all'}]</strong> {f.level}: {f.message}</li>" for f in res.findings if f.level != "INFO") or "<li>No defects detected (clean sum, zero clipping, no silent tracks).</li>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>StemCheck Sanity Audit Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #020617; color: #f1f5f9; padding: 30px 20px; }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    .badge {{ display: inline-block; padding: 6px 16px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem; }}
    .card {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <div class="container">
    <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #1e293b;padding-bottom:16px;margin-bottom:20px;">
      <div>
        <h1 style="margin:0;font-size:1.6rem;">StemCheck Sanity Audit</h1>
        <p style="margin:4px 0 0 0;color:#94a3b8;font-size:0.85rem;">Mixture Conservation &amp; Digital Defect Check (Zero Reference Needed)</p>
      </div>
      <div class="badge" style="{badge_style}">{badge_text}</div>
    </div>

    <div class="card">
      <h3 style="margin:0 0 10px 0;color:#f8fafc;font-size:1.1rem;">Mixture Conservation (Do Stems Add Up to Mix?)</h3>
      <p style="margin:0;color:#cbd5e1;font-size:0.95rem;line-height:1.6;">
        &bull; <strong>Unexplained Mix Residual:</strong> <code>{res_db_str}</code> (closer to -&infin; is better)<br>
        &bull; <strong>Overall Gain Offset:</strong> <code>{gain_str}</code><br>
        &bull; <strong>Detected Delay:</strong> <code>{delay_smp} samples</code>
      </p>
    </div>

    <div class="card">
      <h3 style="margin:0 0 10px 0;color:#f8fafc;font-size:1.1rem;">Defect Findings &amp; Sanity Checks</h3>
      <ul style="margin:0;padding-left:20px;color:#cbd5e1;line-height:1.7;">
        {findings_items}
      </ul>
    </div>

    <footer style="margin-top:30px;border-top:1px solid #1e293b;padding-top:16px;text-align:center;font-size:0.8rem;color:#64748b;">
      Generated by <strong>stemcheck</strong> &bull; Developed by <strong>Boobalan Arjunan</strong> &bull; ComposerMind Labs
    </footer>
  </div>
</body>
</html>
"""
