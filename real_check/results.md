# stemcheck real-data check

Stems: `/Users/mymac/Desktop/Beatunzip-Test/musdb18hq/train`. Songs: 3. Window: middle 30 s. Planted leak: drums into vocals. Artifacts = another stem with its phases scrambled, 15 dB below the vocal. Time: 63 s.

## 1. Accuracy of `stemcheck score` (asked level vs measured level, dB relative to the drums)

| planted | extra error | measured (mean over songs) | spread (max-min) | detected |
|---|---|---|---|---|
| -10 dB | none | -10.10 | 0.23 | 100% |
| -10 dB | -15 dB | -10.08 | 0.22 | 100% |
| -20 dB | none | -20.10 | 0.23 | 100% |
| -20 dB | -15 dB | -20.04 | 0.23 | 100% |
| -30 dB | none | -30.10 | 0.23 | 100% |
| -30 dB | -15 dB | -29.85 | 0.20 | 100% |
| -40 dB | none | -40.10 | 0.23 | 100% |
| -40 dB | -15 dB | -38.89 | 1.06 | 33% |

Low-pass leak (-20 dB drums, 4 kHz low-pass): expected level is computed from the filtered signal itself.

| extra error | expected | measured (mean) | detected |
|---|---|---|---|
| none | -20.42 | -20.51 | 100% |
| -15 dB | -20.42 | -20.43 | 100% |

## 2. False alarms (no leak planted, only the unrelated error)

- 'Leak detected' on 0 of 9 stem/source pairs (0.0%). Should be about 0%.

## 3. No-reference hint (shared content)

- Clean real stems, all pairs: hint fired (p <= 0.05) on 3 of 18 pairs (16.7%). About 5% would be pure chance.

| planted leak | hint fired (p <= 0.05) | median excess (dB) |
|---|---|---|
| -10 dB | 100% | -2.0 |
| -20 dB | 100% | -8.2 |
| -30 dB | 100% | -17.3 |
| -40 dB | 67% | -27.0 |

## 4. `stemcheck check` on the dataset's true stems and its own mixture

- A Classic Education - NightOwl: **PASS**, unexplained -71.54 dB 
- ANiMAL - Clinic A: **PASS**, unexplained -68.79 dB 
- ANiMAL - Easy Tiger: **PASS**, unexplained -67.45 dB 

## 5. Real separator output (scored against the true stems)

| song | stem | quality (dB) | total leak (dB vs stem) | hint excess (dB) | check | unexplained mix (dB) |
|---|---|---|---|---|---|---|
| A Classic Education - NightOwl | bass | 20.3 | -24.6 | -20.0 | PASS | -34.0 |
| A Classic Education - NightOwl | drums | 2.4 | -17.3 | -16.4 | PASS | -34.0 |
| A Classic Education - NightOwl | other | 9.4 | -13.8 | -11.5 | PASS | -34.0 |
| A Classic Education - NightOwl | vocals | 4.3 | -8.9 | -11.5 | PASS | -34.0 |
| ANiMAL - Clinic A | bass | 14.8 | -24.1 | -19.7 | WARN | -21.4 |
| ANiMAL - Clinic A | drums | 8.6 | -20.1 | -21.7 | WARN | -21.4 |
| ANiMAL - Clinic A | other | 5.2 | -11.6 | -17.6 | WARN | -21.4 |
| ANiMAL - Clinic A | vocals | 8.0 | -15.5 | -17.6 | WARN | -21.4 |
| ANiMAL - Easy Tiger | bass | 12.1 | -21.5 | -30.7 | PASS | -30.2 |
| ANiMAL - Easy Tiger | drums | 11.4 | -19.0 | -33.8 | PASS | -30.2 |
| ANiMAL - Easy Tiger | other | 5.2 | -15.8 | -21.9 | PASS | -30.2 |
| ANiMAL - Easy Tiger | vocals | 13.0 | -20.4 | -21.9 | PASS | -30.2 |

- Rank correlation between the no-reference hint and the measured leak: **0.60** over 12 stems (1 = the hint ranks stems exactly like the measurement; near 0 = the hint tells you nothing).

## Look and listen (5 minutes)

Run `stemcheck score <song>/ <separated song>/ --out one_song` on one real separation and open `one_song/stemcheck_score.txt` and `timeline.csv`. Then listen to the part of the song where the timeline says the leak is worst. Is the leak audible there?
