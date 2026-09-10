# ruff: noqa: S101

import numpy as np

from homr.rhythm_correction import correct_rhythm_consistency
from homr.transformer.decoder_inference import ScoreDecoder, _duration_candidates
from homr.transformer.vocabulary import EncodedSymbol


def _note(
    rhythm: str,
    candidates: tuple[tuple[str, float], ...] | None = None,
    position: str = "upper",
) -> EncodedSymbol:
    return EncodedSymbol(
        rhythm,
        pitch="C4",
        lift="_",
        articulation="_",
        position=position,
        rhythm_candidates=candidates or ((rhythm, 0.0),),
    )


def _full_measure() -> list[EncodedSymbol]:
    return [_note("note_4") for _ in range(4)] + [EncodedSymbol("barline")]


def _full_piano_measure() -> list[EncodedSymbol]:
    result = []
    for _ in range(4):
        result.extend([_note("note_4"), EncodedSymbol("chord"), _note("note_4", position="lower")])
    return result + [EncodedSymbol("barline")]


def test_corrects_low_confidence_duration_to_majority_measure_length() -> None:
    symbols = _full_measure() + _full_measure()
    symbols += [
        _note("note_8", (("note_8", 0.0), ("note_4", 0.2))),
        _note("note_4"),
        _note("note_4"),
        _note("note_4"),
        EncodedSymbol("barline"),
    ]

    corrected = correct_rhythm_consistency(symbols)

    assert corrected[10].rhythm == "note_4"
    assert symbols[10].rhythm == "note_8"


def test_does_not_force_a_low_probability_candidate() -> None:
    symbols = _full_measure() + _full_measure()
    symbols += [
        _note("note_8", (("note_8", 0.0), ("note_4", 1.5))),
        _note("note_4"),
        _note("note_4"),
        _note("note_4"),
        EncodedSymbol("barline"),
    ]

    corrected = correct_rhythm_consistency(symbols)

    assert corrected[10].rhythm == "note_8"


def test_does_not_correct_an_unclosed_last_measure() -> None:
    symbols = _full_measure() + _full_measure()
    symbols += [
        _note("note_8", (("note_8", 0.0), ("note_4", 0.1))),
        _note("note_4"),
        _note("note_4"),
    ]

    corrected = correct_rhythm_consistency(symbols)

    assert corrected[10].rhythm == "note_8"


def test_keeps_equal_duration_chord_tones_synchronized() -> None:
    symbols = _full_measure() + _full_measure()
    candidates = (("note_8", 0.0), ("note_4", 0.2))
    symbols += [
        _note("note_8", candidates),
        EncodedSymbol("chord"),
        _note("note_8", candidates),
        _note("note_4"),
        _note("note_4"),
        _note("note_4"),
        EncodedSymbol("barline"),
    ]

    corrected = correct_rhythm_consistency(symbols)

    assert corrected[10].rhythm == "note_4"
    assert corrected[12].rhythm == "note_4"


def test_corrects_piano_staff_timeline_independently() -> None:
    symbols = _full_piano_measure() + _full_piano_measure()
    symbols += [_note("note_4") for _ in range(4)]
    symbols += [
        _note("note_8", (("note_8", 0.0), ("note_12", 0.2)), "lower"),
        _note("note_12", position="lower"),
        _note("note_12", position="lower"),
        _note("note_4", position="lower"),
        _note("note_4", position="lower"),
        _note("note_4", position="lower"),
        EncodedSymbol("barline"),
    ]

    corrected = correct_rhythm_consistency(symbols)

    assert corrected[30].rhythm == "note_12"


def test_preserves_piano_staff_consensus_on_irregular_measure() -> None:
    symbols = _full_piano_measure() + _full_piano_measure()
    symbols += [_note("note_12") for _ in range(14)]
    candidates = (("note_4", 0.0), ("note_8", 0.1))
    symbols += [_note("note_4", candidates, "lower") for _ in range(5)]
    symbols.append(EncodedSymbol("barline"))

    corrected = correct_rhythm_consistency(symbols)

    assert [symbol.rhythm for symbol in corrected[-6:-1]] == ["note_4"] * 5


def test_converts_repeated_unmarked_triplet_bars_without_forcing_six_four() -> None:
    symbols: list[EncodedSymbol] = []
    for _ in range(5):
        symbols.extend(_note("note_8") for _ in range(12))
        symbols.extend(_note("note_4", position="lower") for _ in range(4))
        symbols.append(EncodedSymbol("barline"))

    corrected = correct_rhythm_consistency(symbols)

    upper = [symbol for symbol in corrected if symbol.position == "upper"]
    lower = [symbol for symbol in corrected if symbol.position == "lower"]
    assert {symbol.rhythm for symbol in upper} == {"note_12"}
    assert {symbol.rhythm for symbol in lower} == {"note_4"}


def test_converts_isolated_complete_triplet_overflows_to_stable_meter() -> None:
    symbols = _full_measure() + _full_measure() + _full_measure()
    symbols += [_note("note_8") for _ in range(12)] + [EncodedSymbol("barline")]

    corrected = correct_rhythm_consistency(symbols)

    assert [symbol.rhythm for symbol in corrected[-13:-1]] == ["note_12"] * 12


def test_corrects_repeated_missing_beam_level_when_it_exactly_fills_bar() -> None:
    symbols = _full_measure() + _full_measure()
    symbols += [_note("rest_8")]
    symbols += [_note("note_16") for _ in range(2)]
    symbols += [_note("note_8") for _ in range(5)]
    symbols += [_note("note_16") for _ in range(7)]
    symbols.append(EncodedSymbol("barline"))

    corrected = correct_rhythm_consistency(symbols)
    corrected_measure = corrected[10:-1]

    assert [symbol.rhythm for symbol in corrected_measure].count("note_8") == 0
    assert [symbol.rhythm for symbol in corrected_measure].count("note_16") == 14
    assert symbols[13:18][0].rhythm == "note_8"


def test_does_not_change_mixed_duration_rhythmic_layers() -> None:
    symbols = _full_measure() + _full_measure()
    symbols += [
        _note("note_8", (("note_8", 0.0), ("note_4", 0.1))),
        EncodedSymbol("chord"),
        _note("note_4"),
        _note("note_4"),
        _note("note_4"),
        _note("note_4"),
        EncodedSymbol("barline"),
    ]

    corrected = correct_rhythm_consistency(symbols)

    assert corrected[10].rhythm == "note_8"


def test_decoder_retains_only_same_kind_non_grace_duration_candidates() -> None:
    vocabulary = {
        0: "note_4",
        1: "note_8",
        2: "rest_4",
        3: "note_8G",
        4: "barline",
    }
    logits = np.array([4.0, 3.8, 3.9, 3.7, 5.0], dtype=np.float32)

    candidates = _duration_candidates(logits, "note_4", vocabulary)

    assert [token for token, _ in candidates] == ["note_4", "note_8"]
    assert candidates[0][1] == 0


def test_decoder_keeps_cache_initializer() -> None:
    assert callable(ScoreDecoder.init_cache)
