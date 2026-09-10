# ruff: noqa: S101

from fractions import Fraction
from pathlib import Path

from music21 import chord, expressions, meter, midi, note, stream, tempo

from homr.midi_arpeggio import realize_midi_arpeggios


def _score_with_chord(source: chord.Chord) -> tuple[stream.Score, stream.Measure]:
    score = stream.Score()
    part = stream.Part()
    measure = stream.Measure(number=1)
    measure.timeSignature = meter.TimeSignature("4/4")
    measure.insert(0, tempo.MetronomeMark(number=120))
    measure.insert(1, source)
    part.insert(0, measure)
    score.insert(0, part)
    return score, measure


def test_expands_marked_chord_upward_and_preserves_end() -> None:
    source = chord.Chord(["G4", "C4", "E4"], quarterLength=2)
    source.expressions.append(expressions.ArpeggioMark())
    score, measure = _score_with_chord(source)

    assert realize_midi_arpeggios(score) == 1

    generated = list(measure.recurse().getElementsByClass(note.Note))
    assert [item.nameWithOctave for item in generated] == ["C4", "E4", "G4"]
    assert [item.offset for item in generated] == [1, Fraction(111, 100), Fraction(61, 50)]
    assert [item.offset + item.duration.quarterLength for item in generated] == [3, 3, 3]
    assert len(measure.recurse().getElementsByClass(chord.Chord)) == 0


def test_leaves_unmarked_chord_unchanged() -> None:
    source = chord.Chord(["C4", "E4", "G4"], quarterLength=1)
    score, measure = _score_with_chord(source)

    assert realize_midi_arpeggios(score) == 0
    assert list(measure.recurse().getElementsByClass(chord.Chord)) == [source]


def test_honors_explicit_downward_direction() -> None:
    source = chord.Chord(["C4", "E4", "G4"], quarterLength=1)
    mark = expressions.ArpeggioMark()
    mark.type = "down"
    source.expressions.append(mark)
    score, measure = _score_with_chord(source)

    realize_midi_arpeggios(score)

    generated = list(measure.recurse().getElementsByClass(note.Note))
    assert [item.nameWithOctave for item in generated] == ["G4", "E4", "C4"]


def test_serialized_midi_keeps_distinct_note_on_times(tmp_path: Path) -> None:
    source = chord.Chord(["C4", "E4", "G4"], quarterLength=2)
    source.expressions.append(expressions.ArpeggioMark())
    score, _measure = _score_with_chord(source)
    realize_midi_arpeggios(score)
    midi_path = tmp_path / "arpeggio.mid"
    score.write("midi", fp=midi_path)

    midi_file = midi.MidiFile()
    midi_file.open(midi_path)
    midi_file.read()
    midi_file.close()
    starts = []
    for track in midi_file.tracks:
        absolute_tick = 0
        for event in track.events:
            if event.isDeltaTime():
                absolute_tick += event.time
            elif (
                event.type == midi.ChannelVoiceMessages.NOTE_ON
                and event.velocity
                and event.pitch in (60, 64, 67)
            ):
                starts.append((event.pitch, absolute_tick))

    assert [pitch for pitch, _tick in starts] == [60, 64, 67]
    assert starts[0][1] < starts[1][1] < starts[2][1]
