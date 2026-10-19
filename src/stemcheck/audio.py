"""Reading audio and matching up sample rate, channels and length."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

AUDIO_EXT = {".wav", ".flac", ".ogg", ".aif", ".aiff", ".mp3"}


class StemcheckError(ValueError):
    """A problem the user can fix (bad input or option). The CLI shows it without a traceback."""


@dataclass
class Audio:
    data: np.ndarray   # (samples, channels), float32
    sr: int

    @property
    def n(self) -> int:
        return self.data.shape[0]

    @property
    def ch(self) -> int:
        return self.data.shape[1]


def read_audio(path: Path) -> Audio:
    try:
        a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:
        raise StemcheckError(f"cannot read {Path(path).name}: {e}") from None
    return Audio(a, int(sr))


def find_stem_files(folder: Path, exclude=("mixture",)) -> dict[str, Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise StemcheckError(f"{folder} is not a folder")
    ex = {e.lower() for e in exclude}
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in AUDIO_EXT and p.stem.lower() not in ex)
    names = Counter(p.stem.lower() for p in files)
    dup = [k for k, v in names.items() if v > 1]
    if dup:
        raise StemcheckError(f"{folder.name}: more than one file named {dup[0]!r} (different extensions); keep one")
    if not files:
        raise StemcheckError(f"no audio files found in {folder}")
    return {p.stem: p for p in files}


def resample(a: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return a
    g = gcd(sr_from, sr_to)
    return signal.resample_poly(a, sr_to // g, sr_from // g, axis=0).astype(np.float32)


def match_channels(a: np.ndarray, ch: int) -> np.ndarray:
    """Make `a` have `ch` channels: mono is copied up, anything else is averaged down to mono first."""
    if a.shape[1] == ch:
        return a
    if a.shape[1] == 1:
        return np.repeat(a, ch, axis=1)
    mono = a.mean(axis=1, keepdims=True)
    return np.repeat(mono, ch, axis=1) if ch > 1 else mono


def center_window(n: int, sr: int, max_seconds: float | None) -> slice:
    """The middle `max_seconds` of a signal of n samples (skips intros and outros)."""
    if not max_seconds:
        return slice(0, n)
    want = int(round(max_seconds * sr))
    if n <= want:
        return slice(0, n)
    start = (n - want) // 2
    return slice(start, start + want)
