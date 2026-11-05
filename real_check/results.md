# stemcheck real-data check

Stems: `/Users/mymac/Desktop/Beatunzip-Test/musdb18hq/train`. Songs: 10. Window: middle 30 s. Planted leak: drums into vocals. Artifacts = another stem with its phases scrambled, 15 dB below the vocal. Time: 208 s.

## 1. Accuracy of `stemcheck score` (asked level vs measured level, dB relative to the drums)

| planted | extra error | measured (mean over songs) | spread (max-min) | detected |
|---|---|---|---|---|
| -10 dB | none | -10.06 | 0.24 | 100% |
| -10 dB | -15 dB | -10.05 | 0.22 | 100% |
| -20 dB | none | -20.06 | 0.24 | 100% |
| -20 dB | -15 dB | -20.04 | 0.23 | 100% |
| -30 dB | none | -30.06 | 0.24 | 100% |
| -30 dB | -15 dB | -29.95 | 0.38 | 100% |
| -40 dB | none | -40.06 | 0.24 | 100% |
| -40 dB | -15 dB | -39.51 | 2.31 | 40% |

Low-pass leak (-20 dB drums, 4 kHz low-pass): expected level is computed from the filtered signal itself.

| extra error | expected | measured (mean) | detected |
|---|---|---|---|
| none | -20.45 | -20.51 | 100% |
| -15 dB | -20.45 | -20.48 | 100% |

## 2. False alarms (no leak planted, only the unrelated error)

- 'Leak detected' on 0 of 30 stem/source pairs (0.0%). Should be about 0%.

## 3. No-reference hint (shared content)

- Clean real stems, all pairs: hint fired (p <= 0.05) on 7 of 60 pairs (11.7%). About 5% would be pure chance.

| planted leak | hint fired (p <= 0.05) | median excess (dB) |
|---|---|---|
| -10 dB | 100% | -0.8 |
| -20 dB | 100% | -4.8 |
| -30 dB | 100% | -13.2 |
| -40 dB | 90% | -23.1 |

## 4. `stemcheck check` on the dataset's true stems and its own mixture

- A Classic Education - NightOwl: **PASS**, unexplained -71.54 dB 
- ANiMAL - Clinic A: **PASS**, unexplained -68.79 dB 
- ANiMAL - Easy Tiger: **PASS**, unexplained -67.45 dB 
- ANiMAL - Rockshow: **PASS**, unexplained -67.28 dB 
- Actions - Devil's Words: **PASS**, unexplained -70.4 dB 
- Actions - One Minute Smile: **PASS**, unexplained -70.16 dB 
- Actions - South Of The Water: **PASS**, unexplained -70.25 dB 
- Aimee Norwich - Child: **PASS**, unexplained -70.71 dB 
- Alexander Ross - Goodbye Bolero: **WARN**, unexplained -68.62 dB (silent:vocals)
- Alexander Ross - Velvet Curtain: **PASS**, unexplained -60.35 dB 

## Look and listen (5 minutes)

Run `stemcheck score <song>/ <separated song>/ --out one_song` on one real separation and open `one_song/stemcheck_score.txt` and `timeline.csv`. Then listen to the part of the song where the timeline says the leak is worst. Is the leak audible there?
