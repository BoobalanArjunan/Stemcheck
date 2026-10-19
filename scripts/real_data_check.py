"""Real-data check for stemcheck using MUSDB18-HQ (or any folder of song folders with stem files).

Questions answered with numbers, on real music:
  1. ACCURACY: we plant a leak of known size (drums into vocals) in real stems. Does `stemcheck score`
     measure the size we planted? (With and without extra unrelated error in the estimate.)
  2. FALSE ALARMS: with no leak planted, how often does it still say "leak detected"?
  3. NO-REFERENCE HINT: how often does the shared-content hint fire on clean real stems, and at what
     planted level does it start to notice?
  4. CHECK ON TRUE STEMS: `stemcheck check` on the dataset's own mixture and stems should PASS.
  5. (optional) REAL SEPARATOR OUTPUT: give --separated DIR (one folder per song with vocals.wav, ...
     as written by Demucs etc.) to score it and compare the hint with the measured leak.

Your files are never modified. Output goes to --out (default real_check).

Usage:
  python scripts/real_data_check.py --stems ~/musdb18hq/train --max-songs 5 --out real_check
  python scripts/real_data_check.py --stems ~/musdb18hq/train --separated ~/demucs_out/htdemucs --max-songs 5
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from stemcheck import Audio, check_stems, score_stems
from stemcheck.audio import find_stem_files
from stemcheck.bleed import shared_content

LEVELS = (-10.0, -20.0, -30.0, -40.0)
ARTIFACT_DB = 15.0     # unrelated error in the estimate, this many dB below the stem


def load_window(path: Path, seconds: float):
    info = sf.info(str(path))
    want = int(seconds * info.samplerate)
    start = max((info.frames - want) // 2, 0)
    a, sr = sf.read(str(path), start=start, frames=want, dtype="float32", always_2d=True)
    return a, sr


def load_song(folder: Path, seconds: float):
    files = find_stem_files(folder)
    refs, sr = {}, None
    for name, p in files.items():
        a, sr = load_window(p, seconds)
        refs[name] = a
    mix = None
    for ext in (".wav", ".flac"):
        m = folder / f"mixture{ext}"
        if m.exists():
            mix, _ = load_window(m, seconds)
    return refs, mix, sr


def lowpass(x, hz, sr):
    return signal.sosfilt(signal.butter(4, hz, "low", fs=sr, output="sos"), x, axis=0)


def planted(refs, target, source, db, sr, lp=None):
    leak = refs[source].astype(np.float64) * 10 ** (db / 20)
    if lp:
        leak = lowpass(leak, lp, sr)
    return (refs[target] + leak).astype(np.float32)


def with_artifacts(refs, target, other, sr, db_below=ARTIFACT_DB, seed=0):
    """Unrelated error: another stem with its phases scrambled (same spectrum and rhythm as real music,
    but not a copy of anything). Stands in for separator artifacts."""
    rng = np.random.RandomState(seed)
    x = refs[other].astype(np.float64)
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        _, _, Z = signal.stft(x[:, c], nperseg=2048, noverlap=1536)
        Z = np.abs(Z) * np.exp(1j * rng.uniform(0, 2 * np.pi, Z.shape))
        y = signal.istft(Z, nperseg=2048, noverlap=1536)[1][:x.shape[0]]
        out[:len(y), c] = y
    out *= 10 ** (-db_below / 20) * np.sqrt(np.mean(refs[target].astype(np.float64) ** 2) / max(np.mean(out ** 2), 1e-30))
    return out


def pick(res, stem, source):
    s = next(s for s in res.stems if s.name == stem)
    return next(b for b in s.bleed if b.source == source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", required=True, help="folder of song folders, e.g. musdb18hq/train")
    ap.add_argument("--max-songs", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=30.0, help="analyse the middle N seconds of each song (default 30)")
    ap.add_argument("--separated", help="optional: folder with one folder per song of separator output (same song folder names)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="real_check")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.stems).expanduser()
    songs = sorted(p for p in root.iterdir() if p.is_dir())[:a.max_songs]
    if not songs:
        raise SystemExit(f"no song folders in {root}")
    print(f"{len(songs)} song(s) from {root}, middle {a.seconds:g} s each")

    t0 = time.time()
    acc = {(L, art): [] for L in LEVELS for art in (False, True)}    # (level, artifacts) -> list of (measured, detected)
    acc_lp = {art: [] for art in (False, True)}
    false_det = []                                                    # detected flags with no leak planted (artifacts only)
    hint_clean, hint_levels = [], {L: [] for L in LEVELS}
    check_rows, errors = [], []
    exp_lp = None
    for song in songs:
        try:
            refs, mix, sr = load_song(song, a.seconds)
        except Exception as e:
            errors.append(f"{song.name}: {e}")
            continue
        if not {"vocals", "drums"} <= set(refs):
            errors.append(f"{song.name}: needs vocals.wav and drums.wav")
            continue
        R = {k: Audio(v, sr) for k, v in refs.items()}
        art = with_artifacts(refs, "vocals", "other" if "other" in refs else "bass", sr)
        for L in LEVELS:
            for with_art in (False, True):
                est = planted(refs, "vocals", "drums", L, sr)
                if with_art:
                    est = (est + art).astype(np.float32)
                r = score_stems(R, {"vocals": Audio(est, sr)}, window_s=None)
                b = pick(r, "vocals", "drums")
                acc[(L, with_art)].append((b.vs_source_db, b.detected))
        for with_art in (False, True):
            est = planted(refs, "vocals", "drums", -20.0, sr, lp=4000)
            if with_art:
                est = (est + art).astype(np.float32)
            b = pick(score_stems(R, {"vocals": Audio(est, sr)}, window_s=None), "vocals", "drums")
            exp = 10 * np.log10(np.sum(lowpass(10 ** (-20 / 20) * refs["drums"].astype(np.float64), 4000, sr) ** 2)
                                / np.sum(refs["drums"].astype(np.float64) ** 2))
            acc_lp[with_art].append((b.vs_source_db, b.detected, float(exp)))
        # false alarms: artifacts only
        r = score_stems(R, {"vocals": Audio((refs["vocals"] + art).astype(np.float32), sr)}, window_s=None)
        false_det += [b.detected for b in next(s for s in r.stems if s.name == "vocals").bleed]
        # no-reference hint on the stems themselves and with planted leak
        mono = {k: v.mean(axis=1) for k, v in refs.items()}
        names = sorted(mono)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                hint_clean.append(shared_content(mono[x], mono[y], sr)["p_value"] <= 0.05)
        for L in LEVELS:
            est = planted(refs, "vocals", "drums", L, sr).mean(axis=1)
            hint_levels[L].append(shared_content(est, mono["drums"], sr))
        # check on the true stems and the dataset's own mixture
        if mix is not None:
            c = check_stems(Audio(mix, sr), R, shared=False)
            check_rows.append((song.name, c.verdict(), c.sum_check["residual_db"], [f"{f.code}:{f.stem}" for f in c.findings]))
        print(f"  done {song.name}", flush=True)
    t_main = time.time() - t0

    # ---- optional: real separator output
    sep_rows = []
    if a.separated:
        sroot = Path(a.separated).expanduser()
        for song in songs:
            sd = sroot / song.name
            if not sd.is_dir():
                continue
            try:
                refs, mix, sr = load_song(song, a.seconds)
                ests = {}
                for name, p in find_stem_files(sd).items():
                    ests[name], esr = load_window(p, a.seconds)
                R = {k: Audio(v, sr) for k, v in refs.items()}
                E = {k: Audio(v, esr) for k, v in ests.items()}
                sc = score_stems(R, E, window_s=None)
                ck = check_stems(Audio(mix, sr), E, shared=True) if mix is not None else None
                for s in sc.stems:
                    mono_e = {k: v.data.mean(axis=1) for k, v in E.items()}
                    hint = [p for p in (ck.pairs if ck else []) if s.name in (p["a"], p["b"])]
                    sep_rows.append({"song": song.name, "stem": s.name, "sdr_like_db": s.sdr_like_db,
                                     "total_leak_db": s.total_bleed_vs_target_db,
                                     "hint_max_excess_db": max([p["excess_db"] for p in hint if p["excess_db"] is not None], default=None),
                                     "check_verdict": ck.verdict() if ck else None,
                                     "sum_residual_db": ck.sum_check["residual_db"] if ck else None})
            except Exception as e:
                errors.append(f"separated {song.name}: {e}")

    # ---- report
    def fmt(x, nd=2):
        return "n/a" if x is None else f"{x:.{nd}f}"
    L = ["# stemcheck real-data check", "",
         f"Stems: `{root}`. Songs: {len(songs)}. Window: middle {a.seconds:g} s. Planted leak: drums into vocals. "
         f"Artifacts = another stem with its phases scrambled, {ARTIFACT_DB:g} dB below the vocal. Time: {t_main:.0f} s.", ""]
    L += ["## 1. Accuracy of `stemcheck score` (asked level vs measured level, dB relative to the drums)", "",
          "| planted | extra error | measured (mean over songs) | spread (max-min) | detected |", "|---|---|---|---|---|"]
    for (lvl, art_on), rows in acc.items():
        v = [m for m, _ in rows if m is not None]
        if v:
            L.append(f"| {lvl:g} dB | {'-15 dB' if art_on else 'none'} | {np.mean(v):.2f} | {max(v) - min(v):.2f} | {np.mean([d for _, d in rows]):.0%} |")
    L.append("")
    L += ["Low-pass leak (-20 dB drums, 4 kHz low-pass): expected level is computed from the filtered signal itself.", "",
          "| extra error | expected | measured (mean) | detected |", "|---|---|---|---|"]
    for art_on, rows in acc_lp.items():
        if rows:
            L.append(f"| {'-15 dB' if art_on else 'none'} | {np.mean([e for _, _, e in rows]):.2f} | "
                     f"{np.mean([m for m, _, _ in rows if m is not None]):.2f} | {np.mean([d for _, d, _ in rows]):.0%} |")
    L += ["", "## 2. False alarms (no leak planted, only the unrelated error)", "",
          f"- 'Leak detected' on {sum(false_det)} of {len(false_det)} stem/source pairs ({np.mean(false_det) if false_det else 0:.1%}). Should be about 0%.", ""]
    L += ["## 3. No-reference hint (shared content)", "",
          f"- Clean real stems, all pairs: hint fired (p <= 0.05) on {sum(hint_clean)} of {len(hint_clean)} pairs ({np.mean(hint_clean) if hint_clean else 0:.1%}). About 5% would be pure chance.", "",
          "| planted leak | hint fired (p <= 0.05) | median excess (dB) |", "|---|---|---|"]
    for lvl, rs in hint_levels.items():
        if rs:
            ex = [r["excess_db"] for r in rs if r["excess_db"] is not None]
            L.append(f"| {lvl:g} dB | {np.mean([r['p_value'] <= 0.05 for r in rs]):.0%} | {fmt(float(np.median(ex)) if ex else None, 1)} |")
    L += ["", "## 4. `stemcheck check` on the dataset's true stems and its own mixture", ""]
    if check_rows:
        for name, verdict, res_db, fs in check_rows:
            L.append(f"- {name}: **{verdict}**, unexplained {res_db} dB {('(' + ', '.join(fs) + ')') if fs else ''}")
    else:
        L.append("- no mixture.wav found next to the stems")
    if sep_rows:
        L += ["", "## 5. Real separator output (scored against the true stems)", "",
              "| song | stem | quality (dB) | total leak (dB vs stem) | hint excess (dB) | check | unexplained mix (dB) |", "|---|---|---|---|---|---|---|"]
        for r in sep_rows:
            L.append(f"| {r['song']} | {r['stem']} | {fmt(r['sdr_like_db'], 1)} | {fmt(r['total_leak_db'], 1)} | {fmt(r['hint_max_excess_db'], 1)} | {r['check_verdict']} | {fmt(r['sum_residual_db'], 1)} |")
        xs = [(r["total_leak_db"], r["hint_max_excess_db"]) for r in sep_rows if r["total_leak_db"] is not None and r["hint_max_excess_db"] is not None]
        if len(xs) >= 5:
            from scipy.stats import spearmanr
            rho = spearmanr([x for x, _ in xs], [y for _, y in xs])[0]
            L += ["", f"- Rank correlation between the no-reference hint and the measured leak: **{rho:.2f}** over {len(xs)} stems "
                      "(1 = the hint ranks stems exactly like the measurement; near 0 = the hint tells you nothing)."]
    if errors:
        L += ["", "## Problems", ""] + [f"- {e}" for e in errors]
    L += ["", "## Look and listen (5 minutes)", "",
          "Run `stemcheck score <song>/ <separated song>/ --out one_song` on one real separation and open `one_song/stemcheck_score.txt` and `timeline.csv`. "
          "Then listen to the part of the song where the timeline says the leak is worst. Is the leak audible there?", ""]
    (out / "results.md").write_text("\n".join(L), encoding="utf-8")
    (out / "results.json").write_text(json.dumps({
        "accuracy": {f"{k[0]}|{'artifacts' if k[1] else 'clean'}": v for k, v in acc.items()},
        "false_detected": false_det, "hint_clean": hint_clean, "checks": check_rows, "separated": sep_rows, "errors": errors},
        indent=2, default=lambda o: None if o is None else float(o)), encoding="utf-8")
    print(f"\nwrote {out / 'results.md'}")


if __name__ == "__main__":
    main()
