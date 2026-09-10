import copy
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol

MAX_CANDIDATE_PENALTY = 1.25
MAX_CHANGES_PER_MEASURE = 2
MAX_TOTAL_PENALTY = 2.0


@dataclass(frozen=True)
class _Chord:
    symbol_indices: tuple[int, ...]


@dataclass(frozen=True)
class _Measure:
    chords: tuple[_Chord, ...]
    closed: bool
    signature: str


@dataclass(frozen=True)
class _ChordOption:
    duration: Fraction
    penalty: float
    changes: tuple[tuple[int, str], ...]


def _group_chords(symbols: list[EncodedSymbol]) -> list[_Chord]:
    chords: list[_Chord] = []
    for index, symbol in enumerate(symbols):
        if symbol.rhythm == "chord" and chords:
            continue
        if index > 0 and symbols[index - 1].rhythm == "chord" and chords:
            previous = chords[-1]
            chords[-1] = _Chord(previous.symbol_indices + (index,))
        else:
            chords.append(_Chord((index,)))
    return chords


def _group_measures(symbols: list[EncodedSymbol]) -> list[_Measure]:
    measures: list[_Measure] = []
    current: list[_Chord] = []
    signature = ""
    for chord in _group_chords(symbols):
        chord_symbols = [symbols[index] for index in chord.symbol_indices]
        for symbol in chord_symbols:
            if symbol.rhythm.startswith("timeSignature/"):
                signature = symbol.rhythm
        current.append(chord)
        first_rhythm = chord_symbols[0].rhythm
        if "barline" in first_rhythm or first_rhythm.startswith("repeat"):
            measures.append(_Measure(tuple(current), True, signature))
            current = []
    if current:
        measures.append(_Measure(tuple(current), False, signature))
    return measures


def _symbol_duration(symbol: EncodedSymbol, rhythm: str | None = None) -> Fraction:
    if not (rhythm or symbol.rhythm).startswith(("note_", "rest_")):
        return Fraction(0)
    if rhythm is None or rhythm == symbol.rhythm:
        return symbol.get_duration().fraction
    replacement = copy.copy(symbol)
    replacement.rhythm = rhythm
    replacement._duration = None
    return replacement.get_duration().fraction


def _chord_duration(symbols: list[EncodedSymbol], chord: _Chord) -> Fraction:
    durations = [
        _symbol_duration(symbols[index])
        for index in chord.symbol_indices
        if symbols[index].rhythm.startswith(("note_", "rest_"))
    ]
    positive = [duration for duration in durations if duration > 0]
    return min(positive, default=Fraction(0))


def _measure_duration(symbols: list[EncodedSymbol], measure: _Measure) -> Fraction:
    return sum((_chord_duration(symbols, chord) for chord in measure.chords), Fraction(0))


def _project_measure_to_staff(
    symbols: list[EncodedSymbol], measure: _Measure, position: str
) -> _Measure:
    chords = []
    for chord in measure.chords:
        indices = tuple(
            index
            for index in chord.symbol_indices
            if symbols[index].position == position
            and symbols[index].rhythm.startswith(("note_", "rest_"))
        )
        if indices:
            chords.append(_Chord(indices))
    return _Measure(tuple(chords), measure.closed, measure.signature)


def _compatible_candidate(original: str, candidate: str) -> bool:
    original_family = original.split("_", 1)[0]
    if not candidate.startswith(original_family + "_"):
        return False
    original_suffix = original.split("_", 1)[1]
    candidate_suffix = candidate.split("_", 1)[1]
    if ("G" in original_suffix) != ("G" in candidate_suffix):
        return False
    if original_suffix.endswith("m") or candidate_suffix.endswith("m"):
        return False
    return True


def _duration_ratio_is_safe(original: Fraction, candidate: Fraction) -> bool:
    if original <= 0 or candidate <= 0:
        return False
    ratio = max(original, candidate) / min(original, candidate)
    return ratio <= 2


def _chord_options(symbols: list[EncodedSymbol], chord: _Chord) -> list[_ChordOption]:
    original_duration = _chord_duration(symbols, chord)
    options = {_ChordOption(original_duration, 0.0, ())}
    rhythmic_indices = [
        index
        for index in chord.symbol_indices
        if symbols[index].rhythm.startswith(("note_", "rest_"))
    ]
    if not rhythmic_indices:
        return list(options)

    # A conventional chord should keep one duration on every tone. Mixed-duration
    # groups represent separate rhythmic layers and are left untouched.
    original_rhythms = {symbols[index].rhythm for index in rhythmic_indices}
    if len(rhythmic_indices) > 1 and len(original_rhythms) != 1:
        return list(options)

    anchor = symbols[rhythmic_indices[0]]
    for candidate, penalty in anchor.rhythm_candidates:
        if candidate == anchor.rhythm or penalty > MAX_CANDIDATE_PENALTY:
            continue
        if not _compatible_candidate(anchor.rhythm, candidate):
            continue
        candidate_duration = _symbol_duration(anchor, candidate)
        if not _duration_ratio_is_safe(original_duration, candidate_duration):
            continue

        changes: list[tuple[int, str]] = []
        total_penalty = 0.0
        valid = True
        for index in rhythmic_indices:
            symbol = symbols[index]
            matching = dict(symbol.rhythm_candidates).get(candidate)
            if matching is None or matching > MAX_CANDIDATE_PENALTY:
                valid = False
                break
            changes.append((index, candidate))
            total_penalty += matching
        if valid:
            options.add(_ChordOption(candidate_duration, total_penalty, tuple(changes)))
    return sorted(options, key=lambda option: (option.penalty, len(option.changes)))


def _best_exact_correction(
    symbols: list[EncodedSymbol], measure: _Measure, target: Fraction
) -> tuple[tuple[int, str], ...]:
    # duration -> (penalty, number of changed chord events, replacements)
    states: dict[Fraction, tuple[float, int, tuple[tuple[int, str], ...]]] = {
        Fraction(0): (0.0, 0, ())
    }
    for chord in measure.chords:
        next_states: dict[Fraction, tuple[float, int, tuple[tuple[int, str], ...]]] = {}
        for accumulated, (cost, changed_events, replacements) in states.items():
            for option in _chord_options(symbols, chord):
                duration = accumulated + option.duration
                option_changed = int(bool(option.changes))
                new_changed_events = changed_events + option_changed
                new_cost = cost + option.penalty
                if duration > target:
                    continue
                if new_changed_events > MAX_CHANGES_PER_MEASURE or new_cost > MAX_TOTAL_PENALTY:
                    continue
                proposed = (new_cost, new_changed_events, replacements + option.changes)
                existing = next_states.get(duration)
                if existing is None or proposed[:2] < existing[:2]:
                    next_states[duration] = proposed
        states = next_states
        if not states:
            return ()

    solution = states.get(target)
    if solution is None or solution[1] == 0:
        return ()
    return solution[2]


def _stable_target(symbols: list[EncodedSymbol], measures: list[_Measure]) -> Fraction | None:
    durations = [
        _measure_duration(symbols, measure)
        for measure in measures
        if measure.closed and _measure_duration(symbols, measure) > 0
    ]
    if len(durations) < 2:
        return None
    counts = Counter(durations)
    target, support = counts.most_common(1)[0]
    if support < 2 or support * 2 < len(durations):
        return None
    return target


def _uniform_triplet_replacements(  # noqa: PLR0911
    symbols: list[EncodedSymbol], measure: _Measure, target: Fraction
) -> tuple[tuple[int, str], ...]:
    """Scale a complete equal-note 3:2 overflow into explicit tuplet tokens."""
    duration = _measure_duration(symbols, measure)
    if (
        not measure.closed
        or len(measure.chords) < 6
        or len(measure.chords) % 3 != 0
        or duration != target * Fraction(3, 2)
    ):
        return ()
    chord_durations = {_chord_duration(symbols, chord) for chord in measure.chords}
    if len(chord_durations) != 1:
        return ()

    replacements: list[tuple[int, str]] = []
    for chord in measure.chords:
        rhythmic_indices = [
            index
            for index in chord.symbol_indices
            if symbols[index].rhythm.startswith(("note_", "rest_"))
        ]
        if not rhythmic_indices:
            return ()
        for index in rhythmic_indices:
            rhythm = symbols[index].rhythm
            match = re.fullmatch(r"(note|rest)_(\d+)", rhythm)
            if match is None:
                return ()
            denominator = int(match.group(2))
            if denominator <= 0 or denominator & (denominator - 1):
                return ()
            tuplet_denominator = denominator * 3 // 2
            candidate = f"{match.group(1)}_{tuplet_denominator}"
            if _symbol_duration(symbols[index], candidate) != _symbol_duration(
                symbols[index]
            ) * Fraction(2, 3):
                return ()
            replacements.append((index, candidate))
    return tuple(replacements)


def _systematic_duration_class_replacements(
    symbols: list[EncodedSymbol], measure: _Measure, target: Fraction
) -> tuple[tuple[int, str], ...]:
    """Correct a repeated missing beam level when it exactly explains overflow."""
    current = _measure_duration(symbols, measure)
    if not measure.closed or current <= target:
        return ()
    chords_by_duration: dict[Fraction, list[_Chord]] = {}
    for chord in measure.chords:
        duration = _chord_duration(symbols, chord)
        rhythmic_indices = [
            index
            for index in chord.symbol_indices
            if symbols[index].rhythm.startswith(("note_", "rest_"))
        ]
        shortest_indices = [
            index for index in rhythmic_indices if _symbol_duration(symbols[index]) == duration
        ]
        if (
            duration > 0
            and rhythmic_indices
            and shortest_indices
            and all(symbols[index].rhythm.startswith("note_") for index in shortest_indices)
        ):
            chords_by_duration.setdefault(duration, []).append(chord)

    for duration, affected in sorted(chords_by_duration.items(), reverse=True):
        shorter = duration / 2
        if len(affected) < 3 or len(chords_by_duration.get(shorter, [])) < 2:
            continue
        if current - len(affected) * (duration - shorter) != target:
            continue
        replacements: list[tuple[int, str]] = []
        valid = True
        for chord in affected:
            rhythmic_indices = [
                index
                for index in chord.symbol_indices
                if symbols[index].rhythm.startswith(("note_", "rest_"))
                and _symbol_duration(symbols[index]) == duration
            ]
            if not rhythmic_indices:
                valid = False
                break
            for index in rhythmic_indices:
                rhythm = symbols[index].rhythm
                match = re.fullmatch(r"note_(\d+)", rhythm)
                if match is None:
                    valid = False
                    break
                denominator = int(match.group(1))
                if denominator <= 0 or denominator & (denominator - 1):
                    valid = False
                    break
                candidate = f"note_{denominator * 2}"
                if _symbol_duration(symbols[index], candidate) != shorter:
                    valid = False
                    break
                replacements.append((index, candidate))
            if not valid:
                break
        if valid:
            return tuple(replacements)
    return ()


def _harmonize_triplet_targets(
    symbols: list[EncodedSymbol],
    projected: dict[str, list[tuple[int, _Measure]]],
    targets: dict[str, Fraction | None],
) -> None:
    if set(projected) != {"upper", "lower"}:
        return
    upper_target = targets["upper"]
    lower_target = targets["lower"]
    if upper_target is None or lower_target is None or upper_target == lower_target:
        return
    long_position = "upper" if upper_target > lower_target else "lower"
    long_target = max(upper_target, lower_target)
    short_target = min(upper_target, lower_target)
    if long_target != short_target * Fraction(3, 2):
        return
    complete = [
        measure
        for _, measure in projected[long_position]
        if measure.closed and _measure_duration(symbols, measure) > 0
    ]
    repairable = [
        measure
        for measure in complete
        if _uniform_triplet_replacements(symbols, measure, short_target)
    ]
    if len(repairable) >= 2 and len(repairable) * 2 >= len(complete):
        targets[long_position] = short_target


def correct_rhythm_consistency(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    """Conservatively repair low-confidence duration outliers in complete measures."""
    result = [copy.copy(symbol) for symbol in symbols]
    measures = _group_measures(result)

    # A denominator change starts a new meter section. The model vocabulary does
    # not encode the numerator, so each section derives its target only from a
    # repeated, majority-supported complete-measure duration.
    sections: list[list[tuple[int, _Measure]]] = []
    for measure_index, measure in enumerate(measures):
        if not sections or sections[-1][-1][1].signature != measure.signature:
            sections.append([])
        sections[-1].append((measure_index, measure))

    has_two_staves = any(
        symbol.position == "lower" and symbol.rhythm.startswith(("note_", "rest_"))
        for symbol in result
    )
    for section in sections:
        positions = ("upper", "lower") if has_two_staves else ("upper",)
        projected: dict[str, list[tuple[int, _Measure]]] = {
            position: [
                (measure_index, _project_measure_to_staff(result, measure, position))
                for measure_index, measure in section
            ]
            for position in positions
        }
        targets = {
            position: _stable_target(result, [measure for _, measure in measures])
            for position, measures in projected.items()
        }
        _harmonize_triplet_targets(result, projected, targets)

        # A crop can legitimately contain an irregular/cadenza measure. If both
        # piano staves independently agree on a nearby whole-beat length, preserve
        # it instead of forcing it to the section's usual meter.
        irregular_measures: set[int] = set()
        if has_two_staves and all(targets[position] is not None for position in positions):
            upper_by_index = dict(projected["upper"])
            lower_by_index = dict(projected["lower"])
            for measure_index in upper_by_index.keys() & lower_by_index.keys():
                upper_duration = _measure_duration(result, upper_by_index[measure_index])
                lower_duration = _measure_duration(result, lower_by_index[measure_index])
                if upper_duration <= 0 or lower_duration <= 0:
                    continue
                consensus = (upper_duration + lower_duration) / 2
                nearest_beat = Fraction(round(float(consensus * 4)), 4)
                if nearest_beat == targets["upper"] or nearest_beat == targets["lower"]:
                    continue
                if max(
                    abs(upper_duration - nearest_beat),
                    abs(lower_duration - nearest_beat),
                ) <= Fraction(1, 12):
                    irregular_measures.add(measure_index)

        for position, projected_measures in projected.items():
            target = targets[position]
            if target is None:
                continue
            for measure_index, measure in projected_measures:
                if measure_index in irregular_measures:
                    continue
                if not measure.closed or _measure_duration(result, measure) == target:
                    continue
                replacements = _systematic_duration_class_replacements(result, measure, target)
                if not replacements:
                    replacements = _uniform_triplet_replacements(result, measure, target)
                if not replacements:
                    replacements = _best_exact_correction(result, measure, target)
                if not replacements:
                    continue
                descriptions = []
                for symbol_index, rhythm in replacements:
                    old_rhythm = result[symbol_index].rhythm
                    result[symbol_index].rhythm = rhythm
                    result[symbol_index]._duration = None
                    descriptions.append(f"{old_rhythm}->{rhythm}")
                eprint(
                    "Rhythm consistency corrected measure",
                    measure_index + 1,
                    position,
                    "to",
                    str(target),
                    ":",
                    ", ".join(descriptions),
                )
    return result
