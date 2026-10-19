# Real-data check (MUSDB18-HQ)

This tests stemcheck on real music. It answers: does `stemcheck score` measure a leak we plant in real stems, does it stay quiet when nothing is planted, and how does it look on a real separator's output?

## What you need

1. **MUSDB18-HQ**: search "MUSDB18-HQ" on Zenodo. It is free for non-commercial research. Unzip it and use the `train` folder: one folder per song with `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav` and `mixture.wav`. stemcheck does not ship any music.
2. `pip install --pre stemcheck` (or work from a project checkout to get the `scripts/` folder).

## Part A: planted leaks (no separator needed)

```
python scripts/real_data_check.py --stems ~/musdb18hq/train --max-songs 3 --out real_check
```

Then a fuller run with `--max-songs 10`. Open `real_check/results.md`. It takes a few minutes per song.

What the sections mean:

1. **Accuracy.** We add a leak of known size (drums into vocals) to the real vocal. "Measured" should match "planted". At -40 dB with extra error present, `detected` is expected to be 0%: that is the measurement floor doing its job.
2. **False alarms.** With no leak planted, "leak detected" should be about 0%.
3. **No-reference hint.** On clean real stems the hint should fire on about 5% of pairs or fewer. It starts to notice planted leaks only at about -20 dB.
4. **`check` on the true stems.** The dataset's own mixture and stems should PASS with a tiny unexplained part.

## Part B: a real separator (optional, the most useful part)

Run Demucs (or any separator) on the mixtures. Each song needs its own file name, so copy the mixtures out first:

```
mkdir -p mixes
for d in ~/musdb18hq/train/*/; do n=$(basename "$d"); cp "$d/mixture.wav" "mixes/$n.wav"; done
pip install demucs
demucs -n htdemucs mixes/*.wav          # check demucs --help if a flag differs on your version
```

Demucs writes `separated/htdemucs/<song name>/vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`. Then:

```
python scripts/real_data_check.py --stems ~/musdb18hq/train --separated separated/htdemucs --max-songs 3 --out real_check
```

Section 5 of `results.md` lists quality and leakage for every stem, the `check` verdict, and the rank correlation between the no-reference hint and the measured leak. That correlation decides what we can honestly say about the hint (near 1: useful; near 0: it tells you nothing).

## Look and listen (5 minutes)

```
stemcheck score ~/musdb18hq/train/<song> separated/htdemucs/<song> --out one_song
```

Open `one_song/stemcheck_score.txt` and `one_song/timeline.csv`. Find the section the timeline calls worst for a stem, and listen to that part of the song. Is the leak audible there? That is the real test of whether the timeline points at the right place.

Your original files are never modified. Everything is written under `real_check/` and the folders you name.
