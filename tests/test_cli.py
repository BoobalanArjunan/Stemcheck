import csv
import json

import numpy as np
import pytest
import soundfile as sf

import stemcheck
from stemcheck.cli import main
from conftest import SR, leak, make_stems, tree_hashes, write_folder, write_mix


@pytest.fixture(scope="module")
def S():
    return make_stems(12.0, 3)


@pytest.fixture
def song(tmp_path, S):
    """tmp/ref (true stems), tmp/est (estimates with a planted leak), tmp/mix.wav"""
    write_folder(tmp_path / "ref", S)
    est = dict(S)
    est["vocals"] = leak(S, "vocals", "drums", -20)
    write_folder(tmp_path / "est", est)
    write_mix(tmp_path / "mix.wav", S)
    return tmp_path


def test_check_passes_on_good_stems(song, capsys):
    assert main(["check", str(song / "mix.wav"), str(song / "ref"), "--no-shared"]) == 0
    assert "verdict: PASS" in capsys.readouterr().out


def test_check_exit_code_1_on_a_failing_separation(song, capsys):
    (song / "ref" / "other.wav").unlink()
    (song / "ref" / "bass.wav").unlink()
    assert main(["check", str(song / "mix.wav"), str(song / "ref")]) == 1
    assert "verdict: FAIL" in capsys.readouterr().out


def test_check_writes_reports_and_json(song, capsys):
    out = song / "report"
    assert main(["check", str(song / "mix.wav"), str(song / "ref"), "--out", str(out), "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["verdict"] == "PASS"
    assert json.loads((out / "stemcheck_report.json").read_text())["verdict"] == "PASS"
    assert "stemcheck verdict" in (out / "stemcheck_report.txt").read_text()


def test_score_prints_the_table_and_writes_csvs(song, capsys):
    out = song / "report"
    assert main(["score", str(song / "ref"), str(song / "est"), "--out", str(out), "--window", "4"]) == 0
    text = capsys.readouterr().out
    assert "numbers only" in text and "vocals" in text and "from drums" in text
    rows = list(csv.DictReader(open(out / "bleed_matrix.csv")))
    v = next(r for r in rows if r["stem"] == "vocals" and r["source"] == "drums")
    assert v["detected"] == "True" and abs(float(v["vs_source_db"]) + 20) < 0.5
    assert all(r["detected"] == "False" for r in rows if r["stem"] != "vocals")
    tl = list(csv.DictReader(open(out / "timeline.csv")))
    assert {r["stem"] for r in tl} == {"vocals", "drums", "other", "bass"} and len(tl) >= 4
    assert (out / "stemcheck_score.json").exists() and (out / "stemcheck_score.txt").exists()


def test_score_limits_set_the_exit_code(song, capsys):
    base = ["score", str(song / "ref"), str(song / "est")]
    assert main(base + ["--max-bleed", "-30"]) == 1
    assert "FAIL" in capsys.readouterr().out
    assert main(base + ["--max-bleed", "-15"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_null_shifts_option_changes_how_chance_is_measured(song, capsys):
    base = ["score", str(song / "ref"), str(song / "est"), "--json", "--window", "0"]
    assert main(base) == 0
    default = json.loads(capsys.readouterr().out)
    assert main(base + ["--null-shifts", "0"]) == 0
    formula = json.loads(capsys.readouterr().out)

    def floors(d):
        return [b["floor_vs_source_db"] for s in d["stems"] for b in s["bleed"]]
    assert floors(default) != floors(formula)
    drums = lambda d: next(b for s in d["stems"] if s["name"] == "vocals" for b in s["bleed"] if b["source"] == "drums")
    assert abs(drums(default)["vs_source_db"] + 20) < 0.5 and abs(drums(formula)["vs_source_db"] + 20) < 0.5


def test_score_json_output(song, capsys):
    assert main(["score", str(song / "ref"), str(song / "est"), "--json", "--window", "0"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert {s["name"] for s in d["stems"]} == {"vocals", "drums", "other", "bass"}
    assert all(s["timeline"] == [] for s in d["stems"])


def test_works_directly_on_a_mixforge_style_folder(tmp_path, S, capsys):
    """mixforge writes stems/ (truth) and leaky/ (planted bleed) side by side."""
    mix = tmp_path / "0000_song_000"
    write_folder(mix / "stems", S)
    est = dict(S)
    est["bass"] = leak(S, "bass", "other", -18)
    write_folder(mix / "leaky", est)
    write_mix(mix / "mixture.wav", S)
    assert main(["score", str(mix / "stems"), str(mix / "leaky"), "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    bass = next(s for s in d["stems"] if s["name"] == "bass")
    b = next(x for x in bass["bleed"] if x["source"] == "other")
    assert b["detected"] and abs(b["vs_source_db"] + 18) < 0.5


def test_inputs_are_never_modified(song):
    before = tree_hashes(song)
    main(["check", str(song / "mix.wav"), str(song / "ref")])
    main(["score", str(song / "ref"), str(song / "est"), "--out", str(song / "report")])
    after = tree_hashes(song)
    assert {k: v for k, v in after.items() if not k.startswith("report")} == before


@pytest.mark.parametrize("argv", [
    ["check", "nope.wav", "nope"],
    ["score", "nope", "nope"],
])
def test_missing_inputs_exit_2_with_a_message(argv, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "Traceback" not in err


def test_no_command_and_missing_arguments(capsys):
    assert main([]) == 2
    with pytest.raises(SystemExit) as e:
        main(["score"])
    assert e.value.code == 2


def test_unreadable_audio_is_a_clear_error(song, capsys):
    (song / "est" / "vocals.wav").write_bytes(b"not audio")
    assert main(["score", str(song / "ref"), str(song / "est")]) == 2
    assert "cannot read vocals.wav" in capsys.readouterr().err


def test_estimates_that_match_no_reference_are_a_clear_error(song, capsys):
    for p in (song / "est").glob("*.wav"):
        p.rename(p.with_name("x_" + p.name))
    assert main(["score", str(song / "ref"), str(song / "est")]) == 2
    assert "matches a reference" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and stemcheck.__version__ in capsys.readouterr().out
