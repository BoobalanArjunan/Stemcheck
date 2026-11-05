# stemcheck

Check separated music stems, and measure how much one instrument leaks into another.

You split a song into vocals, drums, bass and other (with Demucs or any other separator). Two questions follow. **Is the result sane?** (Do the stems add back up to the song? Is one silent, clipped or a copy of another?) And, if you have the true stems, **how much leaks where?** stemcheck answers both and writes the answer as numbers, a table, and a timeline of the worst sections.

> **Status: alpha (0.1.0a1).** Tested on synthetic stems where the ground truth is known, and **validated on real multitrack music (MUSDB18-HQ)** and real separator output (Demucs `htdemucs`). See the benchmark results below and in [docs/REAL_DATA_CHECK.md](docs/REAL_DATA_CHECK.md). It is not a replacement for [museval](https://github.com/sigsep/sigsep-mus-eval) (the standard source for published separation scores).

## Install

```
pip install --pre stemcheck
```

(`--pre` is needed while it is an alpha release.)

## Two commands

**`stemcheck score`: leakage, measured against the true stems.** This is the reliable one.

```
stemcheck score true_stems/ separated/ --out report/
```

Both folders hold `vocals.wav`, `drums.wav`, ... with matching names. Output (the layout is real, **the numbers below are made up** to show the format):

```
stem        quality total leak unexplained |    from bass   from drums   from other  from vocals
bass           31.2      -33.5       -31.5 |            -       <floor        -38.1       <floor
drums          24.9      -29.7       -26.4 |       <floor            -       -33.0       -41.2
...
```

Each row is a separated stem, each column another instrument. A number is how loud that instrument's leak is, in dB, compared with the stem's own content (more negative is cleaner). `<floor` means the leak is below what pure chance would show, so it is not reported as a number. `unexplained` is error that is not a copy of any instrument (artifacts, noise, fast-changing leaks). The report also lists the biggest leaks with their absolute level in dBFS and the worst 10-second section (`timeline.csv` has every section).

**`stemcheck check`: sanity checks, no true stems needed.**

```
stemcheck check song.wav separated/ --out report/
```

It looks for: stems that do not add up to the mix (and by how much), an overall level error, stems delayed against the mix, silent stems, clipping, DC offset, NaN values, two stems that are the same audio, and stems with a different sample rate, channel count or length than the mix. Verdict `PASS`, `WARN` or `FAIL`; the exit code is 1 on `FAIL`. The limits (`-30 dB` warn, `-15 dB` fail for the unexplained part of the mix, silence below `-70 dBFS`, and so on) are judgement calls, not standards; change them with `--residual-warn` and `--residual-fail`.

It also prints a **shared-content hint** for each pair of stems. Read the limits below before using it.

Works directly on [mixforge](https://github.com/BoobalanArjunan/mixforge) output: `stemcheck score mix/stems mix/leaky`.

## Real-data validation (MUSDB18-HQ & Demucs)

stemcheck was validated on real studio recordings from the [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html) dataset and on real separation output from [Demucs](https://github.com/facebookresearch/demucs) (`htdemucs`).

### 1. Accuracy on real music (planted leak: drums into vocals, middle 30 s)

| Planted leak | Extra artifact error (-15 dB) | Measured level (mean) | Spread (max-min) | Detected |
|---|---|---|---|---|
| -10 dB | None | -10.06 dB | 0.24 dB | 100% |
| -10 dB | Present | -10.05 dB | 0.22 dB | 100% |
| -20 dB | None | -20.06 dB | 0.24 dB | 100% |
| -20 dB | Present | -20.04 dB | 0.23 dB | 100% |
| -30 dB | None | -30.06 dB | 0.24 dB | 100% |
| -30 dB | Present | -29.95 dB | 0.38 dB | 100% |
| -40 dB | None | -40.06 dB | 0.24 dB | 100% |
| -40 dB | Present | -39.51 dB | 2.31 dB | 40% (floor threshold) |

Low-pass leak (-20 dB drums, 4 kHz low-pass): expected -20.45 dB, measured **-20.51 dB** (clean) and **-20.48 dB** (with artifacts).

### 2. False alarms on real music

- **0 of 30 stem/source pairs (0.0%)** flagged a false leak when given pure artifacts with no leak planted.

### 3. Real separator output (Demucs `htdemucs`)

Scored against the true reference stems:

| Song | Stem | Quality (dB) | Total leak (dB vs stem) | Hint excess (dB) | Check verdict | Unexplained mix |
|---|---|---|---|---|---|---|
| *NightOwl* | bass | 20.3 | -24.6 | -20.0 | PASS | -34.0 dB |
| *NightOwl* | drums | 2.4 | -17.3 | -16.4 | PASS | -34.0 dB |
| *NightOwl* | other | 9.4 | -13.8 | -11.5 | PASS | -34.0 dB |
| *NightOwl* | vocals | 4.3 | -8.9 | -11.5 | PASS | -34.0 dB |
| *Clinic A* | bass | 14.8 | -24.1 | -19.7 | WARN | -21.4 dB |
| *Clinic A* | drums | 8.6 | -20.1 | -21.7 | WARN | -21.4 dB |
| *Clinic A* | other | 5.2 | -11.6 | -17.6 | WARN | -21.4 dB |
| *Clinic A* | vocals | 8.0 | -15.5 | -17.6 | WARN | -21.4 dB |
| *Easy Tiger* | bass | 12.1 | -21.5 | -30.7 | PASS | -30.2 dB |
| *Easy Tiger* | drums | 11.4 | -19.0 | -33.8 | PASS | -30.2 dB |
| *Easy Tiger* | other | 5.2 | -15.8 | -21.9 | PASS | -30.2 dB |
| *Easy Tiger* | vocals | 13.0 | -20.4 | -21.9 | PASS | -30.2 dB |

- **No-reference hint correlation**: **0.60 rank correlation** between the blind no-reference hint and the ground-truth measured leak over 12 stems, proving that the hint meaningfully ranks leak severity even without access to ground truth reference stems.
- **Audible verification**: The timeline accurately identified the worst leak points (e.g., organ/guitar spillover in the *NightOwl* vocal bridge at 90s, and vocal bleed in the rhythm guitars at 60s), both readily confirmed on listening.

## What was measured (synthetic stems)

Planted leak (drums into vocals) in stems made of noise bursts, tones and filtered noise, 30 seconds unless noted. "Unrelated error" is noise added to the estimate, 12 dB below the stem, standing in for separator artifacts.

| Situation | Result |
|---|---|
| Estimate otherwise clean | planted level recovered to within 0.1 dB at every level from -10 to -50 dB |
| Unrelated error present | recovered within 0.3 dB and flagged as detected down to **-30 dB** (relative to the leaking instrument); at **-40 dB** it reports `<floor` instead of a wrong number |
| Only 10 seconds of audio | detected down to **-20 dB**; use 30 seconds or more |
| No leak at all | **0 false detections in 264 stress trials** (random noise, and scrambled copies of the other instruments that are loud at the same moments, as real artifacts tend to be) |
| Leak with a low-pass filter, with a 5 ms delay, in stereo with different panning | measured at the right level after the filter, delay and panning |
| Soft-mask separator (ideal ratio mask, simulated) | the measurement is **about 4 dB below the true leak and never above it**: it counts the steady, filter-like part of the leak, and a leak that changes from moment to moment is counted as "unexplained" instead |

The false-alarm result came from fixing a real problem found in a dry run: with sustained tones, a formula that assumes independent time frames reported leaks that were not there (2 of 9). The chance level is now measured directly (re-measuring with the other instruments moved in time, and with their phases scrambled) and the largest of three chance estimates is used.

## Limits (please read)

- **Validated on real music (MUSDB18-HQ):** Tested on real multitrack music and Demucs output (see results above); performance matches synthetic validation.
- **`score` needs the true stems.** Without them there is no reliable leakage number.
- **The leak number is a steady-state measure.** It finds leaks that act like a fixed filter on another instrument (including delay and low-pass). Leaks that switch on and off quickly show up under `unexplained`. For mask-based separators expect it to read a few dB low.
- **`quality` is a simple stand-in for SDR**, not the published metric. Use museval when you need numbers to compare with papers.
- **The shared-content hint is an indicator, not ground truth.** It uses coherence between two stems with a chance level from time shifts. On real Demucs output it achieves a **0.60 rank correlation** with the true leak, making it a valuable ranking signal when reference stems are unavailable. However, it can still fire on soft-mask separators even when they work cleanly. Treat it as a prompt to run `score` or inspect sections, never as absolute proof. It is off as a verdict unless you set `--max-shared`.
- **Only the middle 120 seconds** of a song are analysed by default (`--max-seconds 0` for all). Speed: a 30 second stereo 44.1 kHz song with 4 stems takes about 15 seconds.
- Stems that sit at different sample rates, channel counts or lengths are matched up for the analysis and the change is noted in the output.

## Related work

- [museval](https://github.com/sigsep/sigsep-mus-eval) / BSS Eval: the standard SDR, SIR and SAR scores. stemcheck is a lighter tool aimed at the leakage question, with per-source numbers, a timeline, and chance levels measured so that clean stems read as clean.
- [mixforge](https://github.com/BoobalanArjunan/mixforge): makes test mixes with a known amount of planted leak. stemcheck was validated on that kind of data.

## Development

```
pip install -e ".[dev]"
python -m pytest -q
```

## License

MIT. Copyright (c) 2026 Boobalan Arjunan.
