# StemCheck: Master Project Plan, Architecture & Checkpoint Log
**Author:** Boobalan Arjunan (<boobalan.it@gmail.com>)  
**Entity:** ComposerMind Labs  
**Project:** StemCheck (Multi-Track Music Stem Sanity Checker & Bleed Measurement Engine)  
**Date Created / Milestone:** October 19, 2026  
**Status:** Alpha v0.1.0a1 (Ready for PyPI Packaging, CI/CD Active, 78/78 Unit Tests Passing)  
**Parent Suite:** Audio ML Data Curation Tier (WaveWash ➔ SplitSafe ➔ Mix-Forge ➔ StemCheck)

---

## 1. Executive Summary

> **The Evaluation Dilemma in Stem Separation:**  
> When deep learning models (Demucs, Spleeter, MDX-Net) split music into stems, how do engineers know if the extracted stems are physically sane and free of cross-instrument bleed? Published metrics (SDR) provide a single global number but cannot tell you *which instrument leaked into which stem*, *at what exact dB level*, or *in which 10-second section the bleed was worst*.

**StemCheck** is an open-source, local-first Python library and CLI auditing engine engineered to solve this diagnosis crisis through two complementary commands:

1. **`stemcheck check` (No Reference Needed):** Evaluates separated stems directly against the original song mixture. Diagnoses whether stems add back up to the mixture ($|\text{mix} - \sum \text{stems}|$), identifies overall gain offset, detects inter-stem time delays, and catches digital defects (silent stems, clipping, DC offset, duplicate tracks, or NaN values).
2. **`stemcheck score` (Reference-Based Bleed Measurement):** Evaluates estimated stems against true reference stems (such as those synthesized by **Mix-Forge**). Computes an instrument-by-instrument **Bleed Matrix** showing the exact leakage level of every instrument into every other stem (in dB), an SDR-like quality metric, unexplained distortion, and a timestamped timeline pinpointing the worst 10-second leakage bursts.

---

## 2. Key Links, Metadata & Ecosystem

| Resource | Official Reference / Destination | Status / Target |
| :--- | :--- | :--- |
| **PyPI Package** | `https://pypi.org/project/stemcheck/` | **v0.1.0a1** (`pip install --pre stemcheck`) |
| **GitHub Repository** | `https://github.com/BoobalanArjunan/Stemcheck` | Branch `main`, Tag `v0.1.0a1`, CI/CD Workflows |
| **License** | MIT License | Permissive open-source distribution |
| **Test Suite** | `tests/test_*.py` | **78 / 78 Tests Passing (100%)** |
| **Calibrated Companion** | **Mix-Forge** | Synthesizes verified answer keys to calibrate StemCheck |

---

## 3. Signal Processing & Bleed Measurement Mathematics

### A. Linear Projection & Filtered Bleed Model
StemCheck models acoustic bleed as an instrument-specific linear filtering process:
$$\hat{S}_{\text{target}}[t] = S_{\text{target}}[t] + \sum_{i \ne \text{target}} h_i[t] * S_i[t] + e[t]$$
Where:
- $S_i[t]$ is the reference audio of the leaking instrument.
- $h_i$ represents the estimated cross-microphone acoustic impulse response.
- $e[t]$ is the unexplained residual error (non-linear artifacts, neural distortion).

By solving optimal projection coefficients across frequency sub-bands, StemCheck calculates:
$$\text{Leakage}_{i \to \text{target}} = 10 \log_{10} \left( \frac{\sum |h_i * S_i|^2}{\sum |S_{\text{target}}|^2} \right) \,\text{dB}$$

### B. The Measurement Floor
Pure chance and acoustic correlation between instruments playing in the same musical key can cause spurious cross-correlation. StemCheck implements an automated **Measurement Floor**:
- Permutes audio phase and calculates statistical chance correlation.
- If measured bleed falls below this statistical threshold, it is reported as `<floor`, guaranteeing zero false-positive leakage alarms.

---

## 4. Software Stack & CLI Reference

### A. Technical Stack
- **Language:** Python 3.10+
- **Core Dependencies:** `numpy`, `scipy`, `soundfile` (zero heavy PyTorch/TensorFlow dependencies for fast, lightweight CLI execution).
- **CLI Entrypoint:** `stemcheck`

### B. Command Workflows
```bash
# 1. Sanity check separated stems against the original mix (zero reference needed)
stemcheck check song.wav separated_stems/ --out sanity_report/

# 2. Precision bleed audit against true reference stems
stemcheck score reference_stems/ separated_stems/ --out bleed_report/
```

---

## 5. Execution Roadmap & Task Status

- [x] **Task 1: Environment & Unit Test Suite** — Clean `.venv` configured, dependencies installed, **78/78 unit tests passing**.
- [x] **Task 2: Distribution Package Validation** — Wheel and tarball built via `python -m build`, validated with `twine check dist/*` (100% PASSED).
- [x] **Task 3: Git Repository Initialization** — Initial commit staged on `main`.
- [ ] **Task 4: Interactive Visual HTML Inspection Report** — Implement `--html report.html` for both `check` and `score` commands.
- [ ] **Task 5: Remote GitHub Push** — Push `main` to user-specified GitHub repository.
- [ ] **Task 6: PR Press Release & Documentation** — Archive in `10 Evidences/PR Works/`.
- [ ] **Task 7: PyPI Deployment & Release Tagging** — Deploy `v0.1.0a1`.
