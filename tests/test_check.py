"""stemcheck check: sanity checks without reference stems. Each problem is planted, so we know what must be reported."""
import numpy as np
import pytest

from stemcheck import Audio, StemcheckError, check_stems
from stemcheck.check import estimate_delay
from conftest import SR, make_stems

SEC = 12.0


def A(d, sr=SR):
    return {k: Audio(v, sr) for k, v in d.items()}


def mixture_of(d, sr=SR):
    return Audio(sum(d.values()), sr)


def codes(res, level=None):
    return {(f.code, f.stem) for f in res.findings if level is None or f.level == level}


def has(res, code, stem=None):
    return any(f.code == code and (stem is None or f.stem == stem) for f in res.findings)


@pytest.fixture(scope="module")
def S():
    return make_stems(SEC, 2)


def test_stems_that_add_up_pass_cleanly(S):
    res = check_stems(mixture_of(S), A(S), shared=False)
    assert res.verdict() == "PASS" and res.findings == []
    assert res.sum_check["residual_db"] <= -100
    assert res.sum_check["best_gain_db"] == pytest.approx(0.0, abs=0.01)
    assert res.sum_check["delay_samples"] == 0


# ---------- do the stems add up? ----------

def test_a_missing_instrument_fails_the_sum_check(S):
    three = {k: v for k, v in S.items() if k != "other"}
    res = check_stems(mixture_of(S), A(three), shared=False)
    assert res.verdict() == "FAIL" and has(res, "sum")
    share = np.sum(S["other"].astype(np.float64) ** 2) / np.sum(sum(S.values()).astype(np.float64) ** 2)
    assert res.sum_check["residual_db"] == pytest.approx(10 * np.log10(share), abs=0.5)   # exactly the energy that is missing


def noise_at(mix, db_below_mix, seed=0):
    r = np.random.RandomState(seed)
    nz = r.randn(*mix.shape)
    return (nz * 10 ** (-db_below_mix / 20) * np.sqrt(np.mean(mix.astype(np.float64) ** 2)) / nz.std()).astype(np.float32)


@pytest.mark.parametrize("noise_db,verdict", [(45, "PASS"), (25, "WARN"), (10, "FAIL")])
def test_unexplained_energy_decides_pass_warn_fail(S, noise_db, verdict):
    mix = mixture_of(S)
    stems = dict(S)
    stems["vocals"] = S["vocals"] + noise_at(mix.data, noise_db)
    res = check_stems(mix, A(stems), shared=False)
    assert res.sum_check["residual_db"] == pytest.approx(-noise_db, abs=0.7)
    assert res.verdict() == verdict


def test_limits_can_be_changed(S):
    mix = mixture_of(S)
    stems = dict(S)
    stems["vocals"] = S["vocals"] + noise_at(mix.data, 25)
    assert check_stems(mix, A(stems), shared=False, residual_warn_db=-20.0).verdict() == "PASS"
    assert check_stems(mix, A(stems), shared=False, residual_fail_db=-26.0).verdict() == "FAIL"


def test_overall_level_error_is_reported(S):
    quiet = {k: (v * 0.8).astype(np.float32) for k, v in S.items()}
    res = check_stems(mixture_of(S), A(quiet), shared=False)
    assert has(res, "gain")
    assert res.sum_check["best_gain_db"] == pytest.approx(20 * np.log10(1 / 0.8), abs=0.05)
    loud = {k: (v * 1.25).astype(np.float32) for k, v in S.items()}
    assert "louder" in next(f.message for f in check_stems(mixture_of(S), A(loud), shared=False).findings if f.code == "gain")


def test_delay_is_found_and_measured(S):
    d = 50
    late = {k: np.concatenate([np.zeros((d, 1), np.float32), v[:-d]]) for k, v in S.items()}
    res = check_stems(mixture_of(S), A(late), shared=False)
    assert res.sum_check["delay_samples"] == d
    assert res.sum_check["residual_after_alignment_db"] < res.sum_check["residual_db"] - 20
    assert has(res, "delay") and has(res, "sum")


def test_estimate_delay_sign_and_size():
    r = np.random.RandomState(0).randn(5000)
    late = np.concatenate([np.zeros(37), r[:-37]])
    assert estimate_delay(late, r, 100) == 37          # late is r delayed by 37
    assert estimate_delay(r, late, 100) == -37
    assert estimate_delay(r, r, 100) == 0
    assert estimate_delay(np.zeros(100), r[:100], 10) == 0


# ---------- single-stem problems ----------

def test_silent_stem_is_a_warning_not_a_failure(S):
    stems = dict(S)
    stems["other"] = np.zeros_like(S["other"])
    res = check_stems(Audio(sum(stems.values()), SR), A(stems), shared=False)
    assert res.verdict() == "WARN" and codes(res) == {("silent", "other")}


def test_clipping_and_dc_offset_are_found(S):
    stems = dict(S)
    stems["vocals"] = np.clip(S["vocals"] * 12, -1, 1).astype(np.float32)
    stems["bass"] = (S["bass"] + 0.05).astype(np.float32)
    res = check_stems(Audio(sum(stems.values()), SR), A(stems), shared=False)
    assert has(res, "clipping", "vocals") and has(res, "dc_offset", "bass")
    assert not has(res, "clipping", "drums") and not has(res, "dc_offset", "drums")


def test_values_above_full_scale_are_flagged(S):
    stems = dict(S)
    stems["vocals"] = (S["vocals"] * 8).astype(np.float32)             # float audio can exceed 1.0
    assert np.abs(stems["vocals"]).max() > 1.0
    assert has(check_stems(Audio(sum(stems.values()), SR), A(stems), shared=False), "clipping", "vocals")


def test_nan_is_a_failure_and_does_not_crash(S):
    stems = {k: v.copy() for k, v in S.items()}
    stems["drums"][1000] = np.nan
    stems["bass"][2000] = np.inf
    res = check_stems(mixture_of(S), A(stems), shared=False)
    assert res.verdict() == "FAIL" and has(res, "non_finite", "drums") and has(res, "non_finite", "bass")


def test_duplicate_stems_fail_even_in_anti_phase(S):
    stems = dict(S)
    stems["other"] = S["vocals"].copy()
    assert has(check_stems(mixture_of(S), A(stems), shared=False), "duplicate")
    stems["other"] = -S["vocals"]
    res = check_stems(mixture_of(S), A(stems), shared=False)
    assert res.verdict() == "FAIL" and has(res, "duplicate")
    assert not has(check_stems(mixture_of(S), A(S), shared=False), "duplicate")


# ---------- format mismatches ----------

def test_other_sample_rate_is_warned_and_still_checked(S):
    from stemcheck.audio import resample
    stems = A(S)
    stems["bass"] = Audio(resample(S["bass"], SR, 8000), 8000)
    res = check_stems(mixture_of(S), stems, shared=False)
    assert has(res, "sample_rate", "bass")
    assert res.sum_check["residual_db"] < -20                          # resampling a 55 Hz tone barely changes it


def test_other_channel_count_is_warned_and_still_checked(S):
    St = make_stems(SEC, 2, stereo=True)
    stems = A(St)
    stems["bass"] = Audio(St["bass"][:, :1], SR)
    res = check_stems(mixture_of(St), stems, shared=False)
    assert has(res, "channels", "bass")


def test_other_length_is_warned_and_trimmed(S):
    stems = A(S)
    stems["bass"] = Audio(S["bass"][: SR * 10], SR)
    res = check_stems(mixture_of(S), stems, shared=False)
    assert has(res, "length", "bass")
    assert res.seconds_analysed == pytest.approx(10.0, abs=0.1)


# ---------- shared content: a weak hint, measured honestly ----------

def test_independent_stems_rarely_look_shared():
    hits, total = 0, 0
    for seed in range(6):
        St = make_stems(8.0, 40 + seed)
        res = check_stems(mixture_of(St), A(St))
        for p in res.pairs:
            total += 1
            hits += p["p_value"] <= 0.05
    assert total == 36 and hits / total <= 0.15


def test_strong_leak_is_seen_as_shared_content_and_only_flagged_when_asked(S):
    stems = dict(S)
    stems["vocals"] = (S["vocals"] + 10 ** (-10 / 20) * S["drums"]).astype(np.float32)
    mix = mixture_of(S)
    res = check_stems(mix, A(stems))
    pair = next(p for p in res.pairs if {p["a"], p["b"]} == {"vocals", "drums"})
    assert pair["p_value"] <= 0.05 and pair["excess_db"] is not None and pair["excess_db"] > -30
    assert not has(res, "shared")                                      # off by default: it is only a hint
    res2 = check_stems(mix, A(stems), max_shared_db=-30.0)
    assert has(res2, "shared") and res2.verdict() in ("WARN", "FAIL")
    clean = check_stems(mix, A(S), max_shared_db=-30.0)
    assert not has(clean, "shared")


def test_shared_content_can_be_skipped(S):
    assert check_stems(mixture_of(S), A(S), shared=False).pairs == []


# ---------- inputs ----------

def test_silent_mixture_and_too_few_stems_are_clear_errors(S):
    with pytest.raises(StemcheckError, match="silent"):
        check_stems(Audio(np.zeros((SR * 3, 1), np.float32), SR), A(S))
    with pytest.raises(StemcheckError, match="at least 2"):
        check_stems(mixture_of(S), {"vocals": Audio(S["vocals"], SR)})


def test_only_the_middle_seconds_are_analysed(S):
    assert check_stems(mixture_of(S), A(S), max_seconds=5.0, shared=False).seconds_analysed == pytest.approx(5.0, abs=0.01)
    assert check_stems(mixture_of(S), A(S), max_seconds=None, shared=False).seconds_analysed == pytest.approx(SEC, abs=0.01)


def test_shared_content_removes_the_chance_level():
    from stemcheck.bleed import shared_content
    St = make_stems(10.0, 7)
    r = shared_content(St["vocals"][:, 0], St["other"][:, 0], SR)
    assert r["chance"] > 0                                           # even unrelated signals show a little coherence
    assert r["excess"] == pytest.approx(max(r["statistic"] - r["chance"], 0.0))
    assert r["excess"] < r["statistic"]
    same = shared_content(St["vocals"][:, 0], St["vocals"][:, 0], SR)
    assert same["statistic"] == pytest.approx(1.0, abs=1e-6) and same["p_value"] <= 0.06


def test_max_shared_only_flags_pairs_that_beat_chance_by_the_p_value():
    St = make_stems(8.0, 41)
    res = check_stems(mixture_of(St), A(St), max_shared_db=-200.0)       # absurdly strict: any positive excess qualifies
    weak = [p for p in res.pairs if p["excess_db"] is not None and p["p_value"] > 0.05]
    assert weak, "the test needs at least one pair with some excess but no significance"
    flagged = {f.message.split(" share")[0] for f in res.findings if f.code == "shared"}
    for p in weak:
        assert f"{p['a']} and {p['b']}" not in flagged


def test_the_too_short_guard_in_the_measurement_itself():
    from stemcheck.bleed import interference
    tiny = make_stems(0.1, 1)
    refs = np.stack([tiny[k] for k in tiny])
    with pytest.raises(ValueError, match="too short"):
        interference(refs, tiny["vocals"], 0, SR)
