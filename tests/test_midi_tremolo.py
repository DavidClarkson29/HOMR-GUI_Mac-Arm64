# ruff: noqa: S101

from music21 import converter, note, stream

from homr.midi_tremolo import realize_midi_tremolos
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import EncodedSymbol


def _double_tremolo_score() -> stream.Score:
    xml = generate_xml(
        XmlGeneratorArguments(),
        [
            [
                EncodedSymbol("note_1", "C3", "_", "tremolo", "lower"),
                EncodedSymbol("note_1", "C4", "_", "_", "lower"),
                EncodedSymbol("barline"),
            ]
        ],
        "",
    )
    return converter.parseData(xml.to_string(), format="musicxml")


def test_double_tremolo_expands_without_changing_measure_length() -> None:
    score = _double_tremolo_score()

    assert realize_midi_tremolos(score) == 1

    measure = score.parts[0].measure(1)
    notes = list(measure.notes)
    assert len(notes) == 32
    assert [event.pitch.midi for event in notes[:4]] == [48, 60, 48, 60]
    assert notes[-1].offset + notes[-1].duration.quarterLength == 4
    assert measure.duration.quarterLength == 4


def test_ordinary_duplet_shape_without_two_to_one_marker_is_untouched() -> None:
    score = stream.Score()
    part = stream.Part()
    measure = stream.Measure(number=1)
    measure.insert(0, note.Note("C4", quarterLength=2))
    measure.insert(2, note.Note("G4", quarterLength=2))
    part.insert(0, measure)
    score.insert(0, part)

    assert realize_midi_tremolos(score) == 0
    assert len(list(measure.notes)) == 2
