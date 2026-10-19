from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._version import __version__
from .audio import StemcheckError, find_stem_files, read_audio
from .check import RESIDUAL_FAIL_DB, RESIDUAL_WARN_DB, check_stems
from .report import (check_to_dict, check_to_text, score_to_dict, score_to_text, write_check_reports,
                     write_score_reports)
from .score import score_stems


def _load_folder(folder: str, exclude=("mixture",)):
    return {name: read_audio(p) for name, p in find_stem_files(Path(folder), exclude).items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stemcheck", description="Check separated music stems.")
    ap.add_argument("--version", action="version", version=f"stemcheck {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    c = sub.add_parser("check", help="sanity-check stems against the mixture (no reference stems needed)")
    c.add_argument("mixture", help="the original mixture file")
    c.add_argument("stems", help="folder with the separated stems (vocals.wav, drums.wav, ...)")
    c.add_argument("--max-seconds", type=float, default=120.0, help="analyse only the middle N seconds (default 120, 0 = all)")
    c.add_argument("--residual-warn", type=float, default=RESIDUAL_WARN_DB, help=f"WARN if unexplained mix is above this many dB (default {RESIDUAL_WARN_DB:g})")
    c.add_argument("--residual-fail", type=float, default=RESIDUAL_FAIL_DB, help=f"FAIL if above this many dB (default {RESIDUAL_FAIL_DB:g})")
    c.add_argument("--max-shared", type=float, help="WARN if two stems share more than this many dB of content beyond chance (off by default; it is only a hint)")
    c.add_argument("--no-shared", action="store_true", help="skip the shared-content hint")
    c.add_argument("--out", help="folder to write report files (json, txt)")
    c.add_argument("--json", action="store_true", help="print JSON instead of text")

    s = sub.add_parser("score", help="measure leakage in estimated stems against reference stems")
    s.add_argument("reference", help="folder with the true stems")
    s.add_argument("estimate", help="folder with the estimated stems (same file names)")
    s.add_argument("--window", type=float, default=10.0, help="section length in seconds for the timeline (default 10, 0 = off)")
    s.add_argument("--max-seconds", type=float, default=120.0, help="analyse only the middle N seconds (default 120, 0 = all)")
    s.add_argument("--null-shifts", type=int, default=3, help="how many time-shifted re-measurements estimate the chance level (default 3, 0 = formula only, faster)")
    s.add_argument("--max-bleed", type=float, help="FAIL if a stem's total leak is above this many dB relative to the stem (e.g. -25)")
    s.add_argument("--min-sdr", type=float, help="FAIL if a stem's quality number is below this many dB")
    s.add_argument("--out", help="folder to write report files (json, txt, csv)")
    s.add_argument("--json", action="store_true", help="print JSON instead of text")

    a = ap.parse_args(argv)
    if a.cmd not in ("check", "score"):
        ap.print_help()
        return 2
    try:
        if a.cmd == "check":
            res = check_stems(read_audio(Path(a.mixture)), _load_folder(a.stems), max_seconds=a.max_seconds or None,
                              residual_warn_db=a.residual_warn, residual_fail_db=a.residual_fail,
                              max_shared_db=a.max_shared, shared=not a.no_shared)
            if a.out:
                write_check_reports(res, Path(a.out))
            print(json.dumps(check_to_dict(res), indent=2) if a.json else check_to_text(res))
            return 1 if res.verdict() == "FAIL" else 0
        res = score_stems(_load_folder(a.reference), _load_folder(a.estimate), window_s=a.window or None,
                          max_seconds=a.max_seconds or None, max_bleed_db=a.max_bleed, min_sdr_db=a.min_sdr,
                          null_shifts=a.null_shifts)
        if a.out:
            write_score_reports(res, Path(a.out))
        print(json.dumps(score_to_dict(res), indent=2) if a.json else score_to_text(res))
        return 1 if res.verdict() == "FAIL" else 0
    except (StemcheckError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
