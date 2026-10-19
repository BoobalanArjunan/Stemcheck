"""Synthetic stems with KNOWN relationships, so every test knows the right answer.

The stems are made from noise bursts, harmonic tones, filtered noise and a low sine, so they overlap in time
and frequency the way real instruments do, without any single pure tone making the maths degenerate.
"""
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 16000
KINDS = ["vocals", "drums", "other", "bass"]


def synth(kind: str, sec: float, seed: int, sr: int = SR) -> np.ndarray:
    r = np.random.RandomState(seed)
    n = int(sec * sr)
    t = np.arange(n) / sr
    if kind == "drums":
        env = np.exp(-((t * r.uniform(2, 5)) % 1) * 10)
        return r.randn(n) * env * 0.2
    if kind == "vocals":
        f0 = 120 + r.rand() * 80
        ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.05 * np.sin(2 * np.pi * r.uniform(.2, 1) * t))) / sr
        x = sum(np.sin(k * ph + r.rand() * 6) / k for k in range(1, 12))
        return 0.15 * x * (np.sin(2 * np.pi * r.uniform(.5, 1.5) * t + r.rand() * 6) > -.2)
    if kind == "other":
        x = signal.lfilter(*signal.butter(2, [200 / (sr / 2), 2000 / (sr / 2)], "band"), r.randn(n))
        return 0.2 * x * (1 + np.sin(2 * np.pi * .3 * t)) / 2
    if kind == "bass":
        return 0.2 * np.sin(2 * np.pi * (40 + r.rand() * 30) * t + r.rand() * 6) * (1 + np.sin(2 * np.pi * r.uniform(.3, 1) * t)) / 2
    raise ValueError(kind)


def make_stems(sec: float = 30.0, seed: int = 0, sr: int = SR, stereo: bool = False) -> dict:
    """name -> (n, ch) float32 array."""
    out = {}
    for i, k in enumerate(KINDS):
        x = synth(k, sec, seed * 10 + i, sr).astype(np.float32)
        out[k] = np.stack([x, x * 0.7], axis=1) if stereo else x[:, None]
    return out


def lowpass(x, hz, sr=SR):
    return signal.sosfilt(signal.butter(4, hz, "low", fs=sr, output="sos"), x, axis=0)


def leak(stems, target, source, db, lowpass_hz=None, sr=SR):
    """stems[target] + a copy of stems[source] at `db` dB (optionally low-passed). The known answer."""
    l = stems[source].astype(np.float64) * 10 ** (db / 20)
    if lowpass_hz:
        l = lowpass(l, lowpass_hz, sr)
    return (stems[target] + l).astype(np.float32)


def write_folder(folder: Path, stems: dict, sr: int = SR, subtype="FLOAT"):
    folder.mkdir(parents=True, exist_ok=True)
    for k, a in stems.items():
        sf.write(str(folder / f"{k}.wav"), a, sr, subtype=subtype)
    return folder


def write_mix(path: Path, stems: dict, sr: int = SR):
    sf.write(str(path), sum(stems.values()), sr, subtype="FLOAT")
    return path


def tree_hashes(root: Path):
    return {str(p.relative_to(root)): hashlib.sha1(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}
