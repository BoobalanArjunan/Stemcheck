"""stemcheck score: leakage measured against reference stems. Every expected number is known by construction."""
import json

import numpy as np
import pytest
from scipy import signal

from stemcheck import Audio, StemcheckError, score_stems
from stemcheck.report import score_to_dict, score_to_text
from conftest import SR, KINDS, leak, lowpass, make_stems

SEC = 20.0


def A(d, sr=SR):
    return {k: Audio(v, sr) for k, v in d.items()}


def entry(res, stem, source):
    s = next(x for x in res.stems if x.name == stem)
    return next(b for b in s.bleed if b.source == source)


def db_ratio(x, y):
    return 10 * np.log10(np.sum(np.square(x, dtype=np.float64)) / np.sum(np.square(y, dtype=np.float64)))


@pytest.fixture(scope="module")
def S():
    return make_stems(SEC, 1)


# ---------- clean in, clean out ----------

def test_perfect_estimates_show_no_leak(S):
    res = score_stems(A(S), A(S))
    for s in res.stems:
        assert s.total_bleed_vs_target_db <= -100 and s.sdr_like_db >= 100
        assert s.target_gain_db == pytest.approx(0.0, abs=0.01)
        assert not any(b.detected for b in s.bleed)
    assert res.verdict() is None                       # no limits set -> numbers only


# ---------- the known answer: planted leak is recovered ----------

@pytest.mark.parametrize("db", [-10.0, -20.0, -30.0, -40.0])
def test_planted_leak_level_is_recovered(S, db):
    est = dict(S)
    est["vocals"] = leak(S, "vocals", "drums", db)
    res = score_stems(A(S), A(est))
    b = entry(res, "vocals", "drums")
    assert b.detected
    assert b.vs_source_db == pytest.approx(db, abs=0.3)
    expect_vs_target = db_ratio(10 ** (db / 20) * S["drums"], S["vocals"])
    assert b.vs_target_db == pytest.approx(expect_vs_target, abs=0.3)
    expect_dbfs = 10 * np.log10(np.mean(S["drums"].astype(np.float64) ** 2)) + db
    assert b.leak_dbfs == pytest.approx(expect_dbfs, abs=0.3)


def test_leak_shows_up_only_where_it_was_planted(S):
    est = dict(S)
    est["vocals"] = leak(S, "vocals", "drums", -20)
    res = score_stems(A(S), A(est))
    assert [b.source for b in next(s for s in res.stems if s.name == "vocals").bleed if b.detected] == ["drums"]
    for name in ("drums", "other", "bass"):
        assert not any(b.detected for b in next(s for s in res.stems if s.name == name).bleed)


def test_two_leaks_into_one_stem_are_both_found(S):
    est = dict(S)
    est["bass"] = (S["bass"].astype(np.float64) + 10 ** (-15 / 20) * S["drums"] + 10 ** (-25 / 20) * S["other"]).astype(np.float32)
    res = score_stems(A(S), A(est))
    assert entry(res, "bass", "drums").vs_source_db == pytest.approx(-15, abs=0.3)
    assert entry(res, "bass", "other").vs_source_db == pytest.approx(-25, abs=0.3)
    assert not entry(res, "bass", "vocals").detected


def test_filtered_leak_is_measured_after_the_filter(S):
    est = dict(S)
    est["vocals"] = leak(S, "vocals", "drums", -20, lowpass_hz=1000)
    res = score_stems(A(S), A(est))
    filtered = lowpass(10 ** (-20 / 20) * S["drums"].astype(np.float64), 1000)
    expect = db_ratio(filtered, S["drums"])               # about -29 dB: the filter removed most of the drums
    assert expect < -25
    assert entry(res, "vocals", "drums").vs_source_db == pytest.approx(expect, abs=0.3)


def test_delayed_leak_is_found_at_the_same_level(S):
    est = dict(S)
    delayed = np.roll(S["drums"], 80, axis=0)             # 5 ms late, like a distant microphone
    est["vocals"] = (S["vocals"] + 10 ** (-25 / 20) * delayed).astype(np.float32)
    res = score_stems(A(S), A(est))
    b = entry(res, "vocals", "drums")
    assert b.detected and b.vs_source_db == pytest.approx(-25, abs=0.5)


def test_leak_in_stereo_with_different_panning(S):
    St = make_stems(SEC, 1, stereo=True)
    est = dict(St)
    pan = np.array([[1.0, 0.3]])
    est["vocals"] = (St["vocals"] + 10 ** (-20 / 20) * St["drums"] * pan).astype(np.float32)
    res = score_stems(A(St), A(est))
    b = entry(res, "vocals", "drums")
    assert b.detected
    expect = db_ratio(10 ** (-20 / 20) * St["drums"] * pan, St["drums"])
    assert b.vs_source_db == pytest.approx(expect, abs=0.4)


# ---------- other errors in the estimate must not be mistaken for leak ----------

def noise_like(x, db_below, seed=0):
    r = np.random.RandomState(seed)
    nz = r.randn(*x.shape)
    return (nz * 10 ** (-db_below / 20) * np.sqrt(np.mean(x.astype(np.float64) ** 2)) / nz.std()).astype(np.float32)


def test_unrelated_error_is_not_reported_as_leak(S):
    est = dict(S)
    est["vocals"] = S["vocals"] + noise_like(S["vocals"], 12)         # artifacts at -12 dB
    res = score_stems(A(S), A(est))
    assert not any(b.detected for b in next(s for s in res.stems if s.name == "vocals").bleed)
    v = next(s for s in res.stems if s.name == "vocals")
    assert v.unexplained_vs_target_db == pytest.approx(-12, abs=0.7)    # the noise is reported as unexplained error
    assert v.sdr_like_db == pytest.approx(12, abs=0.7)


def test_planted_leak_is_still_recovered_with_artifacts_present(S):
    est = dict(S)
    est["vocals"] = leak(S, "vocals", "drums", -25) + noise_like(S["vocals"], 12)
    res = score_stems(A(S), A(est))
    b = entry(res, "vocals", "drums")
    assert b.detected and b.vs_source_db == pytest.approx(-25, abs=0.5)


def phase_scrambled(x, seed=0):
    """Same spectrum and rhythm as x, but not a copy of it: random phase in every time-frequency cell."""
    rng = np.random.RandomState(seed)
    out = np.zeros(x.shape)
    for c in range(x.shape[1]):
        _, _, Z = signal.stft(x[:, c].astype(np.float64), nperseg=2048, noverlap=1536)
        Z = np.abs(Z) * np.exp(1j * rng.uniform(0, 2 * np.pi, Z.shape))
        y = signal.istft(Z, nperseg=2048, noverlap=1536)[1][: x.shape[0]]
        out[: len(y), c] = y
    return out


def test_unrelated_error_shaped_like_a_sustained_tone_is_not_a_leak(S):
    """'other' is a chord of three steady tones. An estimate carrying a scrambled copy of that chord has
    narrow-band error exactly where the reference has all its energy. A formula that assumes independent frames
    calls that a leak (it did, 2 of 9 times in a dry run); the time-shift chance level does not."""
    flagged = 0
    for seed in range(3):
        Sx = make_stems(SEC, 10 + seed)
        art = phase_scrambled(Sx["other"], seed)
        art *= 10 ** (-15 / 20) * np.sqrt(np.mean(Sx["vocals"].astype(np.float64) ** 2) / np.mean(art ** 2))
        est = {"vocals": Audio((Sx["vocals"] + art).astype(np.float32), SR)}
        res = score_stems(A(Sx), est, window_s=None)
        flagged += sum(b.detected for b in res.stems[0].bleed)
        fast = score_stems(A(Sx), est, window_s=None, null_shifts=0)       # formula only: documented as less safe
        assert fast.stems[0].bleed is not None
    assert flagged == 0


def test_unrelated_error_that_is_loud_when_an_instrument_is_loud_is_not_a_leak():
    """Separator mistakes tend to happen at the moments another instrument is loud. Scrambled copies of each
    other instrument (same rhythm and spectrum, not a copy) at -15 dB must not be reported as leaks."""
    flagged, total = 0, 0
    for seed in range(5):
        Sx = make_stems(15.0, 70 + seed)
        for src in ("drums", "bass", "other"):
            art = phase_scrambled(Sx[src], seed)
            art *= 10 ** (-15 / 20) * np.sqrt(np.mean(Sx["vocals"].astype(np.float64) ** 2) / np.mean(art ** 2))
            res = score_stems(A(Sx), {"vocals": Audio((Sx["vocals"] + art).astype(np.float32), SR)}, window_s=None)
            flagged += sum(b.detected for b in res.stems[0].bleed)
            total += len(res.stems[0].bleed)
    assert total == 45 and flagged == 0


@pytest.mark.parametrize("seed", [0, 6, 8])
def test_scrambled_sustained_bass_is_not_a_leak(seed):
    """These three songs fooled the phase-scramble chance level on its own (margins of 5-6 dB over it);
    only the time-shift chance level also keeps the tones' phase structure and sees through this."""
    Sx = make_stems(SEC, 50 + seed)
    art = phase_scrambled(Sx["bass"], seed)
    art *= 10 ** (-15 / 20) * np.sqrt(np.mean(Sx["vocals"].astype(np.float64) ** 2) / np.mean(art ** 2))
    res = score_stems(A(Sx), {"vocals": Audio((Sx["vocals"] + art).astype(np.float32), SR)}, window_s=None)
    assert not any(b.detected for b in res.stems[0].bleed)


def test_a_real_leak_of_the_same_sustained_tones_is_still_found(S):
    est = dict(S)
    est["vocals"] = leak(S, "vocals", "other", -20)
    b = entry(score_stems(A(S), A(est), window_s=None), "vocals", "other")
    assert b.detected and b.vs_source_db == pytest.approx(-20, abs=0.5)


def test_the_shifted_chance_levels_never_contain_the_real_leak():
    """If a 'shifted' copy of the leaking instrument stayed close to unshifted, the real leak would leak into
    the chance level and be hidden. Over many seeds a planted leak must stay far above every chance level."""
    from stemcheck.bleed import interference
    Sx = make_stems(12.0, 5)
    names = sorted(Sx)
    refs = np.stack([Sx[k] for k in names])
    i = names.index("vocals")
    est = leak(Sx, "vocals", "drums", -20)
    for seed in range(8):
        e = interference(refs, est, i, SR, None, n_null=2, seed=seed)["overall"]["per_source"][names.index("drums")]
        assert e["raw_energy"] > 30 * e["chance_max_energy"], seed


def test_time_shifts_for_the_chance_level_are_never_small():
    from stemcheck.bleed import null_shifts
    rng = np.random.RandomState(0)
    shifts = null_shifts(1000, 16000, 512, 400, rng)
    assert min(shifts) >= 62 and max(shifts) <= 1000 - 62          # at least about 2 s (62 frames) from "no shift"
    assert len(set(shifts)) > 100                                    # and genuinely random
    short = null_shifts(40, 16000, 512, 100, rng)                    # very short audio: still never zero
    assert min(short) >= 10 and max(short) <= 30


def test_formula_only_mode_has_its_own_bias_correction(S):
    """With null_shifts=0 only the formula protects against chance: white-noise error must not look like a leak,
    perfect estimates must read as perfect, and timeline sections must not show chance as leak."""
    noisy = dict(S)
    noisy["vocals"] = S["vocals"] + noise_like(S["vocals"], 12)
    res = score_stems(A(S), A(noisy), window_s=5.0, null_shifts=0)
    v = next(s for s in res.stems if s.name == "vocals")
    assert not any(b.detected for b in v.bleed)
    assert all(w.total_bleed_vs_target_db < -33.5 for w in v.timeline)     # corrected about -34.7; uncorrected -30.5, half-corrected -32.6
    assert v.total_bleed_vs_target_db < -40                                  # corrected about -40.8; uncorrected -36.7
    perfect = score_stems(A(S), A(S), window_s=None, null_shifts=0)
    assert not any(b.detected for s in perfect.stems for b in s.bleed)


def test_formula_only_mode_still_measures_strong_leaks(S):
    est = dict(S)
    est["vocals"] = leak(S, "vocals", "drums", -20)
    b = entry(score_stems(A(S), A(est), window_s=None, null_shifts=0), "vocals", "drums")
    assert b.detected and b.vs_source_db == pytest.approx(-20, abs=0.3)


def test_very_weak_leak_under_artifacts_is_reported_as_below_the_floor_not_as_a_number(S):
    est = dict(S)
    est["vocals"] = leak(S, "vocals", "drums", -55) + noise_like(S["vocals"], 12)
    res = score_stems(A(S), A(est))
    assert not entry(res, "vocals", "drums").detected
    assert "<floor" in score_to_text(res)


# ---------- level and timing of the estimate ----------

def test_quiet_estimate_is_reported_quiet_but_leak_ratio_is_unchanged(S):
    base = dict(S)
    base["vocals"] = leak(S, "vocals", "drums", -20)
    quiet = {k: (v * 0.5).astype(np.float32) if k == "vocals" else v for k, v in base.items()}
    r1 = score_stems(A(S), A(base))
    r2 = score_stems(A(S), A(quiet))
    v2 = next(s for s in r2.stems if s.name == "vocals")
    assert v2.target_gain_db == pytest.approx(-6.02, abs=0.1)
    assert entry(r2, "vocals", "drums").vs_target_db == pytest.approx(entry(r1, "vocals", "drums").vs_target_db, abs=0.2)


def test_delayed_estimate_is_flagged_and_leak_is_unaffected(S):
    est = dict(S)
    est["vocals"] = np.roll(leak(S, "vocals", "drums", -20), 20, axis=0)
    res = score_stems(A(S), A(est))
    v = next(s for s in res.stems if s.name == "vocals")
    assert v.delay_samples == 20 and any("delayed by 20 samples" in n for n in v.notes)
    assert entry(res, "vocals", "drums").vs_source_db == pytest.approx(-20, abs=0.5)


# ---------- where in the song? ----------

def test_timeline_points_at_the_section_with_the_leak(S):
    est = dict(S)
    half = len(S["vocals"]) // 2
    v = S["vocals"].astype(np.float64).copy()
    v[half:] += 10 ** (-15 / 20) * S["drums"][half:]                  # leak only in the second half
    est["vocals"] = v.astype(np.float32)
    res = score_stems(A(S), A(est), window_s=5.0)
    tl = next(s for s in res.stems if s.name == "vocals").timeline
    assert len(tl) == 4
    first, second = tl[:2], tl[2:]
    assert all(w.total_bleed_vs_target_db < -30 for w in first)        # nothing there (below what noise could hide)
    assert all(w.total_bleed_vs_target_db > -25 for w in second)
    assert all(w.worst_source == "drums" for w in second)
    assert "starts at 10" in score_to_text(res) or "starts at 9.9" in score_to_text(res)


def test_timeline_can_be_switched_off(S):
    res = score_stems(A(S), A(S), window_s=None)
    assert all(s.timeline == [] for s in res.stems)


# ---------- what a mask-based separator looks like (documents the README claim) ----------

def test_ideal_mask_separation_is_measured_as_a_slightly_low_estimate_of_the_true_leak(S):
    n = len(S["vocals"])
    stack = np.stack([S[k][:, 0].astype(np.float64) for k in KINDS])
    mix = stack.sum(0)

    def st(x):
        return signal.stft(x, nperseg=1024, noverlap=768, window="hann")[2]

    def ist(Z):
        return signal.istft(Z, nperseg=1024, noverlap=768, window="hann")[1][:n]

    RS = np.stack([st(x) for x in stack])
    i = KINDS.index("vocals")
    M = np.abs(RS[i]) / np.abs(RS).sum(0).clip(1e-12)                    # soft ratio mask
    est = {k: S[k] for k in KINDS}
    est["vocals"] = ist(M * st(mix)).astype(np.float32)[:, None]
    res = score_stems(A(S), A(est), window_s=None)
    checked = 0
    for j, k in enumerate(KINDS):
        if k == "vocals":
            continue
        truth = db_ratio(ist(M * RS[j]), stack[j])                        # the real leak of k into the estimate
        if truth < -30:
            continue                                                       # too weak to be measured against artifacts
        got = entry(res, "vocals", k).vs_source_db
        assert truth - 7.0 <= got <= truth + 0.5, (k, truth, got)         # never an over-estimate, a few dB low
        checked += 1
    assert checked >= 2


# ---------- inputs ----------

def test_names_match_ignoring_case_and_extras_are_ignored(S):
    refs = A(S)
    ests = {"VOCALS": Audio(S["vocals"], SR), "guitar": Audio(S["bass"], SR)}
    res = score_stems(refs, ests)
    assert [s.name for s in res.stems] == ["vocals"]
    assert any("guitar" in n for n in res.notes)


def test_no_matching_estimate_is_a_clear_error(S):
    with pytest.raises(StemcheckError, match="matches a reference"):
        score_stems(A(S), {"guitar": Audio(S["bass"], SR)})


def test_need_two_references(S):
    with pytest.raises(StemcheckError, match="at least 2"):
        score_stems({"vocals": Audio(S["vocals"], SR)}, A(S))


def test_estimate_at_another_sample_rate_is_resampled_with_a_note(S):
    from stemcheck.audio import resample
    est = A(S)
    est["vocals"] = Audio(resample(S["vocals"], SR, 8000), 8000)
    res = score_stems(A(S), est)
    assert any("resampled" in n and "vocals" in n for n in res.notes)
    assert next(s for s in res.stems if s.name == "vocals").target_gain_db == pytest.approx(0.0, abs=1.0)


def test_estimate_of_a_different_length_is_trimmed_with_a_note(S):
    est = A(S)
    est["vocals"] = Audio(S["vocals"][: SR * 15], SR)
    res = score_stems(A(S), est)
    assert any("shorter" in n and "vocals" in n for n in res.notes)
    assert res.seconds_analysed == pytest.approx(15.0, abs=0.1)


def test_too_short_audio_is_a_clear_error():
    short = make_stems(1.0, 1)
    with pytest.raises(StemcheckError, match="too short"):
        score_stems(A(short), A(short))


def test_silent_reference_does_not_crash_and_is_noted(S):
    refs = dict(S)
    refs["other"] = np.zeros_like(S["other"])
    est = dict(refs)
    est["vocals"] = leak(S, "vocals", "drums", -20)
    res = score_stems(A(refs), A(est))
    assert entry(res, "vocals", "drums").vs_source_db == pytest.approx(-20, abs=0.3)
    assert entry(res, "vocals", "other").vs_source_db is None and entry(res, "vocals", "other").source_silent
    assert "n/a" in score_to_text(res)
    silent = next(s for s in res.stems if s.name == "other")
    assert silent.total_bleed_vs_target_db is None and any("silent" in n for n in silent.notes)


def test_only_the_middle_seconds_are_analysed(S):
    res = score_stems(A(S), A(S), max_seconds=8.0)
    assert res.seconds_analysed == pytest.approx(8.0, abs=0.01)
    est = dict(S)
    v = S["vocals"].astype(np.float64).copy()
    v[: SR * 4] += 10 ** (-10 / 20) * S["drums"][: SR * 4]               # leak only in the first 4 s, outside the middle 8 s
    est["vocals"] = v.astype(np.float32)
    assert not entry(score_stems(A(S), A(est), max_seconds=8.0), "vocals", "drums").detected
    assert entry(score_stems(A(S), A(est), max_seconds=None), "vocals", "drums").detected


# ---------- verdict and output ----------

def test_limits_give_pass_or_fail(S):
    est = dict(S)
    est["vocals"] = leak(S, "vocals", "drums", -20)
    leak_db = score_stems(A(S), A(est)).stems[[s.name for s in score_stems(A(S), A(est)).stems].index("vocals")].total_bleed_vs_target_db
    assert -30 < leak_db < -20                                          # the planted leak, about -27 dB against the vocal
    assert score_stems(A(S), A(est), max_bleed_db=-30.0).verdict() == "FAIL"   # limit stricter than the leak
    assert score_stems(A(S), A(est), max_bleed_db=-20.0).verdict() == "PASS"   # limit looser than the leak
    assert score_stems(A(S), A(est), min_sdr_db=40.0).verdict() == "FAIL"
    assert score_stems(A(S), A(S), max_bleed_db=-30.0, min_sdr_db=40.0).verdict() == "PASS"


def test_results_are_repeatable_and_json_serialisable(S):
    est = dict(S)
    est["vocals"] = leak(S, "vocals", "drums", -20)
    a = json.dumps(score_to_dict(score_stems(A(S), A(est))), sort_keys=True)
    b = json.dumps(score_to_dict(score_stems(A(S), A(est))), sort_keys=True)
    assert a == b
