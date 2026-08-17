from __future__ import annotations

import copy
from fractions import Fraction

from music21 import chord, duration, note, stream

from homr.simple_logging import eprint

TREMOLO_STEP = Fraction(1, 8)


def _as_fraction(value: object) -> Fraction:
    return Fraction(value).limit_denominator(960)  # type: ignore[arg-type]


def _is_double_tremolo_half(event: note.GeneralNote) -> bool:
    return any(
        tuplet.numberNotesActual == 2 and tuplet.numberNotesNormal == 1
        for tuplet in event.duration.tuplets
    )


def _site_events(site: stream.Stream) -> list[note.GeneralNote]:
    return sorted(
        (
            event
            for event in site.getElementsByClass((note.Note, chord.Chord))
            if _as_fraction(event.duration.quarterLength) > 0
        ),
        key=lambda event: _as_fraction(event.offset),
    )


def realize_midi_tremolos(score: stream.Score) -> int:
    """Expand standards-compliant double-note tremolos for MIDI playback.

    MusicXML encodes each tremolo endpoint at half its written duration with a
    2:1 time-modification. Music21 preserves that marker but does not alternate
    the pitches during MIDI export, so replace each adjacent pair with a bounded
    stream of 32nd notes while preserving the pair's exact total duration.
    """
    expanded = 0
    sites: list[stream.Stream] = []
    for measure in score.recurse().getElementsByClass(stream.Measure):
        voices = list(measure.getElementsByClass(stream.Voice))
        sites.extend(voices or [measure])

    for site in sites:
        events = _site_events(site)
        index = 0
        while index + 1 < len(events):
            first = events[index]
            second = events[index + 1]
            first_duration = _as_fraction(first.duration.quarterLength)
            second_duration = _as_fraction(second.duration.quarterLength)
            if not (
                _is_double_tremolo_half(first)
                and _is_double_tremolo_half(second)
                and first_duration == second_duration
                and _as_fraction(second.offset) == _as_fraction(first.offset) + first_duration
            ):
                index += 1
                continue

            start = _as_fraction(first.offset)
            total = first_duration + second_duration
            step = min(TREMOLO_STEP, first_duration, second_duration)
            count = int(total / step)
            if count < 2 or step * count != total:
                index += 2
                continue

            replacements: list[tuple[Fraction, note.GeneralNote]] = []
            for repetition in range(count):
                source = first if repetition % 2 == 0 else second
                replacement = copy.deepcopy(source)
                replacement.duration = duration.Duration(step)
                replacements.append((start + step * repetition, replacement))

            site.remove(first)
            site.remove(second)
            for offset, replacement in replacements:
                site.insert(offset, replacement)
            site.coreElementsChanged()
            expanded += 1
            index += 2

    if expanded:
        eprint("Expanded", expanded, "double-note tremolos for MIDI playback")
    return expanded
