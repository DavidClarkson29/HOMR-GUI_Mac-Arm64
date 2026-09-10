# ruff: noqa: S101

from homr.barline_recovery import recover_missing_barlines
from homr.transformer.vocabulary import EncodedSymbol


def _note(x: float) -> EncodedSymbol:
    return EncodedSymbol("note_4", pitch="C4", position="upper", coordinates=(x, 20))


def test_recovers_paired_internal_barline() -> None:
    symbols = [_note(x) for x in (10, 20, 30, 40, 60, 70, 80, 90)]

    result = recover_missing_barlines(
        symbols, [49, 51], unit_size=8, require_staff_pair=True
    )

    assert [symbol.rhythm for symbol in result].count("barline") == 1
    assert [symbol.rhythm for symbol in result].index("barline") == 4


def test_does_not_recover_unpaired_grand_staff_candidate() -> None:
    symbols = [_note(x) for x in (10, 20, 30, 40, 60, 70, 80, 90)]

    result = recover_missing_barlines(
        symbols, [50], unit_size=8, require_staff_pair=True
    )

    assert all(symbol.rhythm != "barline" for symbol in result)


def test_does_not_duplicate_semantic_barline() -> None:
    symbols = [_note(x) for x in (10, 20, 30, 40)]
    symbols.append(EncodedSymbol("barline", coordinates=(50, 20)))
    symbols.extend(_note(x) for x in (60, 70, 80, 90))

    result = recover_missing_barlines(
        symbols, [49, 51], unit_size=8, require_staff_pair=True
    )

    assert [symbol.rhythm for symbol in result].count("barline") == 1


def test_rejects_candidate_that_conflicts_with_attention_order() -> None:
    symbols = [_note(x) for x in (10, 70, 30, 40, 60, 20, 80, 90)]

    result = recover_missing_barlines(
        symbols, [49, 51], unit_size=8, require_staff_pair=True
    )

    assert all(symbol.rhythm != "barline" for symbol in result)


def test_does_not_split_a_chord_token_sequence() -> None:
    symbols = [_note(10), _note(20), _note(40)]
    symbols.extend(
        [
            EncodedSymbol("chord", coordinates=(49, 20)),
            _note(51),
            _note(70),
            _note(80),
        ]
    )

    result = recover_missing_barlines(
        symbols, [49, 51], unit_size=4, require_staff_pair=True
    )

    assert all(symbol.rhythm != "barline" for symbol in result)
