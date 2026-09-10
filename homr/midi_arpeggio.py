from __future__ import annotations

import copy
from fractions import Fraction

from music21 import chord, expressions, note, stream, tempo

from homr.simple_logging import eprint

DEFAULT_QUARTER_BPM = 120.0
NOTE_DELAY_SECONDS = 0.055
MIN_DELAY = Fraction(1, 32)
MAX_DELAY = Fraction(1, 8)
MAX_DURATION_SHARE = Fraction(1, 4)


def _as_fraction(value: object) -> Fraction:
    return Fraction(value).limit_denominator(960)  # type: ignore[arg-type]


def _has_arpeggio_mark(source: chord.Chord) -> bool:
    return any(
        isinstance(expression, expressions.ArpeggioMark)
        for expression in source.expressions
    )


def _arpeggio_is_down(source: chord.Chord) -> bool:
    marks = [
        expression
        for expression in source.expressions
        if isinstance(expression, expressions.ArpeggioMark)
    ]
    return any(getattr(mark, "type", "normal") == "down" for mark in marks)


def _quarter_bpm(source: chord.Chord) -> float:
    marking = source.getContextByClass(tempo.MetronomeMark)
    if isinstance(marking, tempo.MetronomeMark):
        sounding = marking.getQuarterBPM()
        if sounding is not None and sounding > 0:
            return float(sounding)
    return DEFAULT_QUARTER_BPM


def _note_delay(source: chord.Chord, note_count: int) -> Fraction:
    bpm_delay = _as_fraction(NOTE_DELAY_SECONDS * _quarter_bpm(source) / 60)
    bpm_delay = min(max(bpm_delay, MIN_DELAY), MAX_DELAY)
    duration = _as_fraction(source.duration.quarterLength)
    duration_delay = duration * MAX_DURATION_SHARE / max(note_count - 1, 1)
    return min(bpm_delay, duration_delay)


def _without_arpeggio_marks(source: note.Note) -> note.Note:
    result = copy.deepcopy(source)
    result.expressions = [
        expression
        for expression in result.expressions
        if not isinstance(expression, expressions.ArpeggioMark)
    ]
    return result


def realize_midi_arpeggios(score: stream.Score) -> int:
    """Expand notated arpeggio chords into staggered notes for MIDI playback.

    Unmarked chords are untouched. Generated notes are ordered low-to-high unless
    MusicXML explicitly supplies a downward direction, and all generated notes end
    at the original chord boundary so the following beat cannot be displaced.
    """
    expanded = 0
    arpeggio_chords = [
        source
        for source in score.recurse().getElementsByClass(chord.Chord)
        if len(source.pitches) > 1 and _has_arpeggio_mark(source)
    ]
    for source in arpeggio_chords:
        site = source.activeSite
        if site is None:
            continue
        source_offset = _as_fraction(source.offset)
        source_duration = _as_fraction(source.duration.quarterLength)
        ordered_notes = sorted(
            source.notes,
            key=lambda chord_note: chord_note.pitch.midi,
            reverse=_arpeggio_is_down(source),
        )
        delay = _note_delay(source, len(ordered_notes))
        replacements: list[tuple[Fraction, note.Note]] = []
        for index, chord_note in enumerate(ordered_notes):
            offset = source_offset + delay * index
            replacement = _without_arpeggio_marks(chord_note)
            replacement.duration.quarterLength = max(
                source_duration - delay * index,
                MIN_DELAY,
            )
            replacements.append((offset, replacement))

        site.remove(source)
        for offset, replacement in replacements:
            site.insert(offset, replacement)
        site.coreElementsChanged()
        expanded += 1

    if expanded:
        eprint("Expanded", expanded, "arpeggio chords for MIDI playback")
    return expanded
