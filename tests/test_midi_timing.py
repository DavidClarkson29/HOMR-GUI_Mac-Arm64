# ruff: noqa: S101

from fractions import Fraction

from music21 import chord, meter, note, stream

from homr.midi_timing import reconcile_measure_boundaries, stabilize_measure_offsets


def test_overfull_measure_does_not_shift_following_measures() -> None:
    score = stream.Score()
    part = stream.Part()
    first = stream.Measure(number=1)
    first.timeSignature = meter.TimeSignature("4/4")
    first.insert(0, note.Note("C4", quarterLength=5.5))
    second = stream.Measure(number=2)
    second.insert(0, note.Note("D4", quarterLength=4))
    third = stream.Measure(number=3)
    third.insert(0, note.Note("E4", quarterLength=4))
    part.insert(0, first)
    part.insert(5.5, second)
    part.insert(9.5, third)
    score.insert(0, part)

    changed = stabilize_measure_offsets(score)

    assert changed == 2
    assert [measure.offset for measure in part.getElementsByClass(stream.Measure)] == [0, 4, 8]


def test_stable_measure_offsets_are_left_unchanged() -> None:
    score = stream.Score()
    part = stream.Part()
    for index, offset in enumerate((0, 4, 8), start=1):
        measure = stream.Measure(number=index)
        if index == 1:
            measure.timeSignature = meter.TimeSignature("4/4")
        measure.insert(0, note.Note("C4", quarterLength=4))
        part.insert(offset, measure)
    score.insert(0, part)

    assert stabilize_measure_offsets(score) == 0


def test_piano_staff_consensus_preserves_an_irregular_measure() -> None:
    score = stream.Score()
    for duration in (5, 14 / 3):
        part = stream.Part()
        first = stream.Measure(number=1)
        first.timeSignature = meter.TimeSignature("4/4")
        first.insert(0, note.Note("C4", quarterLength=duration))
        second = stream.Measure(number=2)
        second.insert(0, note.Note("D4", quarterLength=4))
        part.insert(0, first)
        part.insert(duration, second)
        score.insert(0, part)

    stabilize_measure_offsets(score)

    assert [part.measure(2).offset for part in score.parts] == [5, 5]


def test_disagreeing_piano_staves_do_not_create_a_false_irregular_measure() -> None:
    score = stream.Score()
    for duration in (3.625, 2.375):
        part = stream.Part()
        first = stream.Measure(number=1)
        first.timeSignature = meter.TimeSignature("4/4")
        first.insert(0, note.Note("C4", quarterLength=duration))
        second = stream.Measure(number=2)
        second.insert(0, note.Note("D4", quarterLength=4))
        part.insert(0, first)
        part.insert(duration, second)
        score.insert(0, part)

    stabilize_measure_offsets(score)

    assert [part.measure(2).offset for part in score.parts] == [4, 4]


def test_sparse_staff_does_not_shorten_a_full_measure() -> None:
    score = stream.Score()
    for duration in (4, 0.5):
        part = stream.Part()
        first = stream.Measure(number=1)
        first.timeSignature = meter.TimeSignature("4/4")
        first.insert(0, note.Note("C4", quarterLength=duration))
        second = stream.Measure(number=2)
        second.insert(0, note.Note("D4", quarterLength=4))
        part.insert(0, first)
        part.insert(duration, second)
        score.insert(0, part)

    stabilize_measure_offsets(score)

    assert [part.measure(2).offset for part in score.parts] == [4, 4]


def test_small_tuplet_overflow_is_repaired_inside_measure() -> None:
    score = stream.Score()
    for durations in ((1, 1, 1, 1), (1, 0.5, 1 / 3, 1 / 3, 1, 1)):
        part = stream.Part()
        measure = stream.Measure(number=1)
        measure.timeSignature = meter.TimeSignature("4/4")
        offset = 0
        for duration in durations:
            measure.insert(offset, note.Note("C4", quarterLength=duration))
            offset += duration
        part.insert(0, measure)
        score.insert(0, part)

    stabilize_measure_offsets(score)

    repaired = score.parts[1].measure(1)
    events = list(repaired.recurse().notes)
    assert repaired.duration.quarterLength == 4
    assert [event.offset for event in events] == [
        0,
        1,
        Fraction(4, 3),
        Fraction(5, 3),
        2,
        3,
    ]
    assert [event.duration.quarterLength for event in events][1] == Fraction(1, 3)


def _part_with_measures(first: stream.Measure) -> stream.Part:
    part = stream.Part()
    second = stream.Measure(number=2)
    second.insert(0, note.Note("C5", quarterLength=4))
    part.insert(0, first)
    part.insert(first.duration.quarterLength, second)
    return part


def test_cross_staff_scalar_tail_is_reconnected_at_end_of_measure() -> None:
    score = stream.Score()
    upper = stream.Measure(number=1)
    upper.timeSignature = meter.TimeSignature("4/4")
    for offset, pitch in zip(
        (3, Fraction(25, 8), Fraction(13, 4), Fraction(27, 8)),
        ("G5", "E5", "D5", "C5"),
        strict=True,
    ):
        upper.insert(offset, note.Note(pitch, quarterLength=Fraction(1, 8)))
    upper.insert(Fraction(7, 2), note.Note("G4", quarterLength=Fraction(1, 4)))

    lower = stream.Measure(number=1)
    lower.timeSignature = meter.TimeSignature("4/4")
    lower.insert(0, note.Note("C3", quarterLength=2))
    for offset, pitch in zip(
        (2, Fraction(17, 8), Fraction(9, 4)),
        ("E4", "D4", "C4"),
        strict=True,
    ):
        lower.insert(offset, note.Note(pitch, quarterLength=Fraction(1, 8)))

    score.insert(0, _part_with_measures(upper))
    score.insert(0, _part_with_measures(lower))

    stabilize_measure_offsets(score)

    repaired_upper = score.parts[0].measure(1)
    repaired_lower = score.parts[1].measure(1)
    assert repaired_upper.notes[-1].duration.quarterLength == Fraction(1, 8)
    assert [event.offset for event in repaired_lower.notes[-3:]] == [
        Fraction(29, 8),
        Fraction(15, 4),
        Fraction(31, 8),
    ]
    assert [part.measure(2).offset for part in score.parts] == [4, 4]


def test_unrelated_fast_lower_staff_tail_is_not_moved() -> None:
    score = stream.Score()
    upper = stream.Measure(number=1)
    upper.timeSignature = meter.TimeSignature("4/4")
    for offset, pitch in zip(
        (3, Fraction(25, 8), Fraction(13, 4), Fraction(27, 8)),
        ("G5", "E5", "D5", "C5"),
        strict=True,
    ):
        upper.insert(offset, note.Note(pitch, quarterLength=Fraction(1, 8)))
    upper.insert(Fraction(7, 2), note.Note("G4", quarterLength=Fraction(1, 4)))

    lower = stream.Measure(number=1)
    lower.timeSignature = meter.TimeSignature("4/4")
    lower.insert(0, note.Note("C3", quarterLength=2))
    for offset, pitch in zip(
        (2, Fraction(17, 8), Fraction(9, 4)),
        ("C3", "E3", "G3"),
        strict=True,
    ):
        lower.insert(offset, note.Note(pitch, quarterLength=Fraction(1, 8)))

    score.insert(0, _part_with_measures(upper))
    score.insert(0, _part_with_measures(lower))

    stabilize_measure_offsets(score)

    assert [event.offset for event in score.parts[1].measure(1).notes[-3:]] == [
        2,
        Fraction(17, 8),
        Fraction(9, 4),
    ]
    assert score.parts[0].measure(1).notes[-1].duration.quarterLength == Fraction(1, 4)


def test_trailing_chord_after_sustained_voice_is_moved_inside_measure() -> None:
    score = stream.Score()
    upper = stream.Measure(number=1)
    upper.timeSignature = meter.TimeSignature("4/4")
    upper.insert(0, note.Note("C5", quarterLength=1))
    upper.insert(1, note.Note("A4", quarterLength=3))
    upper.insert(4, chord.Chord(["C5", "F5", "C6"], quarterLength=2))
    lower = stream.Measure(number=1)
    lower.timeSignature = meter.TimeSignature("4/4")
    lower.insert(0, note.Note("F3", quarterLength=4))
    score.insert(0, _part_with_measures(upper))
    score.insert(0, _part_with_measures(lower))

    stabilize_measure_offsets(score)

    repaired = score.parts[0].measure(1)
    assert repaired.duration.quarterLength == 4
    assert repaired.notes[-1].offset == 2
    assert [part.measure(2).offset for part in score.parts] == [4, 4]


def test_sequential_trailing_chord_without_sustained_voice_is_not_moved() -> None:
    score = stream.Score()
    upper = stream.Measure(number=1)
    upper.timeSignature = meter.TimeSignature("4/4")
    upper.insert(0, note.Note("C5", quarterLength=4))
    upper.insert(4, chord.Chord(["C5", "F5", "C6"], quarterLength=2))
    lower = stream.Measure(number=1)
    lower.timeSignature = meter.TimeSignature("4/4")
    lower.insert(0, note.Note("F3", quarterLength=4))
    score.insert(0, _part_with_measures(upper))
    score.insert(0, _part_with_measures(lower))

    stabilize_measure_offsets(score)

    assert score.parts[0].measure(1).notes[-1].offset == 4


def test_uniform_five_note_run_is_redistributed_across_the_bar() -> None:
    score = stream.Score()
    measure = stream.Measure(number=1)
    measure.timeSignature = meter.TimeSignature("3/4")
    for index in range(5):
        measure.insert(Fraction(2, 3) * index, note.Note(60 + index, quarterLength=Fraction(2, 3)))
    score.insert(0, _part_with_measures(measure))

    stabilize_measure_offsets(score)

    repaired = score.parts[0].measure(1)
    assert [event.offset for event in repaired.notes] == [
        Fraction(0),
        Fraction(3, 5),
        Fraction(6, 5),
        Fraction(9, 5),
        Fraction(12, 5),
    ]
    assert [event.duration.quarterLength for event in repaired.notes] == [Fraction(3, 5)] * 5


def test_rest_and_three_notes_are_redistributed_as_a_complete_quadruplet() -> None:
    score = stream.Score()
    measure = stream.Measure(number=1)
    measure.timeSignature = meter.TimeSignature("3/4")
    measure.insert(0, note.Rest(quarterLength=1))
    for index in range(3):
        measure.insert(index + 1, chord.Chord([60 + index, 72 + index], quarterLength=1))
    score.insert(0, _part_with_measures(measure))

    stabilize_measure_offsets(score)

    repaired = score.parts[0].measure(1)
    events = list(repaired.recurse().getElementsByClass((note.Rest, chord.Chord)))
    assert [event.offset for event in events] == [0, Fraction(3, 4), Fraction(3, 2), Fraction(9, 4)]
    assert [event.duration.quarterLength for event in events] == [Fraction(3, 4)] * 4


def test_complete_triplet_tail_can_absorb_a_three_to_two_overflow() -> None:
    score = stream.Score()
    measure = stream.Measure(number=1)
    measure.timeSignature = meter.TimeSignature("4/4")
    measure.insert(0, note.Note("C4", quarterLength=2))
    for index in range(6):
        measure.insert(
            Fraction(2) + Fraction(index, 2), note.Note(62 + index, quarterLength=Fraction(1, 2))
        )
    score.insert(0, _part_with_measures(measure))

    stabilize_measure_offsets(score)

    run = score.parts[0].measure(1).notes[1:]
    assert [event.offset for event in run] == [
        Fraction(2),
        Fraction(7, 3),
        Fraction(8, 3),
        Fraction(3),
        Fraction(10, 3),
        Fraction(11, 3),
    ]
    assert [event.duration.quarterLength for event in run] == [Fraction(1, 3)] * 6


def test_short_irregular_tail_is_not_blindly_redistributed() -> None:
    score = stream.Score()
    measure = stream.Measure(number=1)
    measure.timeSignature = meter.TimeSignature("4/4")
    measure.insert(0, note.Rest(quarterLength=Fraction(13, 4)))
    for index in range(4):
        measure.insert(
            Fraction(13 + index, 4),
            note.Note(60 + index, quarterLength=Fraction(1, 4)),
        )
    score.insert(0, _part_with_measures(measure))

    stabilize_measure_offsets(score)

    assert [event.duration.quarterLength for event in measure.notes] == [Fraction(1, 4)] * 4


def test_appended_secondary_voice_is_folded_over_a_sustained_chord() -> None:
    score = stream.Score()
    measure = stream.Measure(number=1)
    measure.timeSignature = meter.TimeSignature("4/4")
    measure.insert(0, chord.Chord([60, 64, 67], quarterLength=4))
    for index, pitch in enumerate((69, 71, 74), start=4):
        measure.insert(index, note.Note(pitch, quarterLength=1))
    score.insert(0, _part_with_measures(measure))

    stabilize_measure_offsets(score)

    repaired = score.parts[0].measure(1)
    assert [event.offset for event in repaired.notes[1:]] == [1, 2, 3]
    assert repaired.duration.quarterLength == 4


def test_scalar_suffix_is_not_misclassified_as_an_appended_voice() -> None:
    score = stream.Score()
    measure = stream.Measure(number=1)
    measure.timeSignature = meter.TimeSignature("4/4")
    measure.insert(0, note.Note(60, quarterLength=4))
    measure.insert(4, note.Note(62, quarterLength=1))
    score.insert(0, _part_with_measures(measure))

    stabilize_measure_offsets(score)

    assert score.parts[0].measure(1).notes[-1].offset == 4


def test_slightly_late_independent_voice_is_shifted_inside_measure() -> None:
    score = stream.Score()
    measure = stream.Measure(number=1)
    measure.timeSignature = meter.TimeSignature("4/4")
    primary = stream.Voice(id="primary")
    primary.insert(0, chord.Chord([45, 57], quarterLength=4))
    secondary = stream.Voice(id="secondary")
    secondary.insert(Fraction(9, 4), chord.Chord([40, 52], quarterLength=2))
    measure.insert(0, primary)
    measure.insert(0, secondary)
    score.insert(0, _part_with_measures(measure))

    stabilize_measure_offsets(score)

    repaired = score.parts[0].measure(1)
    secondary_note = repaired.getElementById("secondary").notes[0]
    assert secondary_note.offset == 2
    assert repaired.duration.quarterLength == 4


def test_whole_measure_rest_follows_active_meter() -> None:
    score = stream.Score()
    measure = stream.Measure(number=1)
    measure.timeSignature = meter.TimeSignature("3/4")
    measure.insert(0, note.Rest(quarterLength=4))
    score.insert(0, _part_with_measures(measure))

    stabilize_measure_offsets(score)

    repaired = score.parts[0].measure(1)
    assert repaired.recurse().notesAndRests[0].duration.quarterLength == 3


def _part_with_bar_durations(durations: list[Fraction]) -> stream.Part:
    part = stream.Part()
    offset = Fraction(0)
    for number, duration in enumerate(durations, start=1):
        measure = stream.Measure(number=number)
        if number == 1:
            measure.timeSignature = meter.TimeSignature("3/4")
        measure.insert(0, note.Note(60 + number, quarterLength=duration))
        part.insert(offset, measure)
        offset += duration
    return part


def test_cross_staff_consensus_splits_shared_missing_boundary() -> None:
    score = stream.Score()
    score.insert(0, _part_with_bar_durations([Fraction(3), Fraction(6)]))
    score.insert(0, _part_with_bar_durations([Fraction(3), Fraction(6)]))
    score.insert(0, _part_with_bar_durations([Fraction(3), Fraction(3), Fraction(3)]))
    score.insert(0, _part_with_bar_durations([Fraction(3), Fraction(3), Fraction(3)]))

    restored = reconcile_measure_boundaries(score)

    assert restored == 2
    assert [len(list(part.getElementsByClass(stream.Measure))) for part in score.parts] == [
        3,
        3,
        3,
        3,
    ]
    for part in score.parts:
        assert [measure.number for measure in part.getElementsByClass(stream.Measure)] == [
            1,
            2,
            3,
        ]


def test_cross_staff_consensus_does_not_split_without_paired_evidence() -> None:
    score = stream.Score()
    score.insert(0, _part_with_bar_durations([Fraction(3), Fraction(6)]))
    score.insert(0, _part_with_bar_durations([Fraction(3), Fraction(3), Fraction(3)]))
    score.insert(0, _part_with_bar_durations([Fraction(3), Fraction(3), Fraction(3)]))

    assert reconcile_measure_boundaries(score) == 0
    assert [len(list(part.getElementsByClass(stream.Measure))) for part in score.parts] == [
        2,
        3,
        3,
    ]


def test_isolated_voice_overhang_is_trimmed_at_barline() -> None:
    score = stream.Score()
    measure = stream.Measure(number=1)
    measure.timeSignature = meter.TimeSignature("3/4")
    primary = stream.Voice(id="primary")
    primary.insert(0, note.Note("C5", quarterLength=3))
    overhang = stream.Voice(id="overhang")
    overhang.insert(0, note.Note("E5", quarterLength=4))
    measure.insert(0, primary)
    measure.insert(0, overhang)
    score.insert(0, _part_with_measures(measure))

    stabilize_measure_offsets(score)

    assert overhang.notes[0].quarterLength == 3
    assert primary.notes[0].quarterLength == 3


def test_repeated_run_repairs_one_supported_duration_outlier() -> None:
    score = stream.Score()
    measure = stream.Measure(number=1)
    measure.timeSignature = meter.TimeSignature("3/4")
    for pitch, duration in zip(
        ("C4", "D4", "E4", "F4", "G4", "A4"),
        (0.5, 0.5, 0.5, 0.5, 1.0, 0.5),
        strict=True,
    ):
        measure.append(note.Note(pitch, quarterLength=duration))
    score.insert(0, _part_with_measures(measure))

    stabilize_measure_offsets(score)

    assert measure.duration.quarterLength == 3
    assert [item.quarterLength for item in measure.notes] == [0.5] * 6
