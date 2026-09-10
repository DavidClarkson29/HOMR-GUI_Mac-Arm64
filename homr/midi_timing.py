from __future__ import annotations

from collections import Counter
from fractions import Fraction
from statistics import median

from music21 import chord, meter, note, stream

from homr.simple_logging import eprint


def _as_fraction(value: object) -> Fraction:
    return Fraction(value).limit_denominator(96)  # type: ignore[arg-type]


def _fallback_bar_duration(measures: list[stream.Measure]) -> Fraction:
    durations = [
        _as_fraction(measure.duration.quarterLength)
        for measure in measures
        if measure.duration.quarterLength > 0
    ]
    if not durations:
        return Fraction(4)
    inferred = _as_fraction(median(durations))
    conventional = (Fraction(2), Fraction(3), Fraction(4), Fraction(6))
    return min(conventional, key=lambda value: abs(value - inferred))


def _measure_bar_duration(measure: stream.Measure, fallback: Fraction) -> Fraction:
    signature = measure.timeSignature
    if not isinstance(signature, meter.TimeSignature):
        signature = measure.getContextByClass(meter.TimeSignature)
    if isinstance(signature, meter.TimeSignature):
        return _as_fraction(signature.barDuration.quarterLength)
    return fallback


def _split_measure_at_bar(
    part: stream.Part, measure: stream.Measure, bar_duration: Fraction
) -> None:
    """Split one overlong measure without inventing or deleting note events."""
    original_offset = _as_fraction(measure.offset)
    left, right = measure.splitAtQuarterLength(
        bar_duration,
        retainOrigin=False,
        addTies=True,
        searchContext=True,
    )
    part.remove(measure)
    part.insert(original_offset, left)
    part.insert(original_offset + bar_duration, right)


def reconcile_measure_boundaries(score: stream.Score) -> int:
    """Use ensemble consensus to split strongly evidenced missing barlines.

    A printed system shares vertical measure boundaries across every piano staff.
    If at least two tracks agree on a larger measure count and every shorter track
    has an internal measure spanning about two bars at the same ordinal position,
    split those long measures at the active meter boundary.  This deliberately does
    not merge extra measures: without matching geometric evidence, an insertion is
    safer than deleting a potentially real irregular bar.

    The rule depends only on part counts, meter, and rhythmic span; it contains no
    page, pitch, filename, or fixed measure-number knowledge.
    """
    parts = list(score.parts)
    if len(parts) < 2:
        return 0
    measures_by_part = [list(part.getElementsByClass(stream.Measure)) for part in parts]
    counts = [len(measures) for measures in measures_by_part]
    if not counts or min(counts) == max(counts):
        return 0

    count_support = Counter(counts)
    target_count = max(counts)
    if count_support[target_count] < 2 or target_count - min(counts) > 2:
        return 0

    reference_measures = next(
        measures for measures in measures_by_part if len(measures) == target_count
    )
    fallback = _fallback_bar_duration(reference_measures)
    shorter_indices = [index for index, count in enumerate(counts) if count < target_count]
    if len(shorter_indices) < 2:
        return 0

    split_count = 0
    while any(
        len(list(parts[index].getElementsByClass(stream.Measure))) < target_count
        for index in shorter_indices
    ):
        current_short = [
            list(parts[index].getElementsByClass(stream.Measure)) for index in shorter_indices
        ]
        common_length = min(len(measures) for measures in current_short)
        candidate_index: int | None = None
        for measure_index in range(1, common_length):
            long_enough = True
            for measures in current_short:
                measure = measures[measure_index]
                target = _measure_bar_duration(measure, fallback)
                duration = _as_fraction(measure.duration.quarterLength)
                if duration < target * 2 - Fraction(1, 2):
                    long_enough = False
                    break
            if long_enough:
                candidate_index = measure_index
                break
        if candidate_index is None:
            break

        for part_index, measures in zip(shorter_indices, current_short, strict=True):
            if len(measures) >= target_count:
                continue
            measure = measures[candidate_index]
            target = _measure_bar_duration(measure, fallback)
            _split_measure_at_bar(parts[part_index], measure, target)
            split_count += 1

    if split_count:
        for part in parts:
            for number, measure in enumerate(part.getElementsByClass(stream.Measure), start=1):
                measure.number = number
            part.coreElementsChanged()
        eprint(
            "Restored",
            split_count,
            "missing cross-staff measure boundaries before MIDI export",
        )
    return split_count


def _consensus_measure_span(
    measures: list[stream.Measure], notated_bar_duration: Fraction
) -> Fraction:
    """Preserve an irregular bar only when every populated staff agrees.

    Averaging incompatible staff lengths is unsafe: a dense right hand and a
    sparse left hand can accidentally average to a plausible whole number even
    though both are ordinary, incompletely decoded bars.  Irregular measures are
    therefore accepted only when each staff is close to the same whole-beat span
    and the staves agree with one another.
    """
    durations = [
        _as_fraction(measure.duration.quarterLength)
        for measure in measures
        if measure.duration.quarterLength > 0
    ]
    if len(durations) < 2:
        return notated_bar_duration
    consensus = sum(durations, Fraction(0)) / len(durations)
    nearest_beat = Fraction(round(float(consensus)))
    if nearest_beat < 1:
        return notated_bar_duration
    tolerance = Fraction(1, 3)
    staves_agree = max(durations) - min(durations) <= tolerance
    each_matches_candidate = all(
        abs(duration - nearest_beat) <= tolerance for duration in durations
    )
    if staves_agree and each_matches_candidate:
        return nearest_beat
    return notated_bar_duration


_COMMON_MIDI_DURATIONS = {
    Fraction(4),
    Fraction(3),
    Fraction(2),
    Fraction(3, 2),
    Fraction(1),
    Fraction(3, 4),
    Fraction(2, 3),
    Fraction(1, 2),
    Fraction(1, 3),
    Fraction(1, 4),
    Fraction(1, 6),
    Fraction(1, 8),
}


def _repair_small_measure_overflow(measure: stream.Measure, target: Fraction) -> bool:
    """Absorb a small overflow into one plausible rhythmic value and reflow the tail."""
    overflow = _as_fraction(measure.duration.quarterLength) - target
    if overflow <= 0 or overflow > Fraction(1, 3):
        return False
    events = list(measure.recurse().getElementsByClass((note.Note, note.Rest, chord.Chord)))
    for event in events:
        old_duration = _as_fraction(event.duration.quarterLength)
        new_duration = old_duration - overflow
        if new_duration <= 0 or new_duration not in _COMMON_MIDI_DURATIONS:
            continue
        event_offset = _as_fraction(event.getOffsetInHierarchy(measure))
        event.duration.quarterLength = new_duration
        for following in events:
            following_offset = _as_fraction(following.getOffsetInHierarchy(measure))
            if following_offset > event_offset:
                following.activeSite.setElementOffset(following, following_offset - overflow)
        measure.coreElementsChanged()
        return True
    return False


def _repair_isolated_voice_overhang(measure: stream.Measure, target: Fraction) -> int:
    """Trim a lone voice event at the barline without changing its onset.

    MusicXML represents a tied note with one in-bar event on each side of the
    barline; a positive-duration event itself must not extend beyond its measure.
    OMR can nevertheless decode a dotted/whole value for a note or rest that is
    alone in one voice. Limit only those isolated voice events, or an event whose
    simultaneous companion already ends exactly at the barline.
    """
    events = [
        event
        for event in _measure_events(measure)
        if _as_fraction(event.duration.quarterLength) > 0
    ]
    repaired = 0
    for event in events:
        offset = _event_offset(event, measure)
        end = offset + _as_fraction(event.duration.quarterLength)
        if offset >= target or end <= target:
            continue
        site = event.activeSite
        if not isinstance(site, stream.Voice):
            continue
        same_voice = [candidate for candidate in events if candidate.activeSite is site]
        companion_ends_at_bar = any(
            candidate is not event
            and _event_offset(candidate, measure) == offset
            and _event_offset(candidate, measure) + _as_fraction(candidate.duration.quarterLength)
            == target
            for candidate in events
        )
        if len(same_voice) != 1 and not companion_ends_at_bar:
            continue
        new_duration = target - offset
        if new_duration <= 0 or new_duration not in _COMMON_MIDI_DURATIONS:
            continue
        event.duration.quarterLength = new_duration
        repaired += 1
    if repaired:
        measure.coreElementsChanged()
    return repaired


def _repair_supported_duration_outlier(measure: stream.Measure, target: Fraction) -> bool:
    """Correct one doubled duration when the surrounding run supplies consensus.

    The rule applies only to a contiguous, non-polyphonic measure where replacing
    one value with a duration already used by at least two peers makes the bar
    exact. It repairs a missed beam level while rejecting ambiguous mixed rhythms.
    """
    duration = _as_fraction(measure.duration.quarterLength)
    overflow = duration - target
    if overflow <= 0 or overflow > target / 3:
        return False
    events = sorted(_measure_events(measure), key=lambda event: _event_offset(event, measure))
    if len(events) < 4:
        return False
    for left, right in zip(events, events[1:], strict=False):
        if _event_offset(right, measure) < (
            _event_offset(left, measure) + _as_fraction(left.duration.quarterLength)
        ):
            return False
    duration_support = Counter(_as_fraction(event.duration.quarterLength) for event in events)
    candidates = []
    for event in events:
        old_duration = _as_fraction(event.duration.quarterLength)
        new_duration = old_duration - overflow
        if new_duration <= 0 or duration_support[new_duration] < 2:
            continue
        candidates.append((event, new_duration))
    if len(candidates) != 1:
        return False
    event, new_duration = candidates[0]
    event_offset = _event_offset(event, measure)
    event.duration.quarterLength = new_duration
    for following in events:
        following_offset = _event_offset(following, measure)
        if following_offset > event_offset:
            _set_event_offset(following, measure, following_offset - overflow)
    measure.coreElementsChanged()
    return True


def _normalize_whole_measure_rest(measure: stream.Measure, target: Fraction) -> bool:
    """Make an isolated whole-measure rest follow the active meter.

    OMR represents an empty bar with ``rest_1`` regardless of whether the score is
    in 3/4, 4/4, or another meter.  Music21 otherwise treats that token as four
    literal quarter notes, so silent tracks can still extend the following bar.
    """
    events = _measure_events(measure)
    if len(events) != 1 or not isinstance(events[0], note.Rest):
        return False
    rest = events[0]
    if _event_offset(rest, measure) != 0:
        return False
    if _as_fraction(rest.duration.quarterLength) == target:
        return False
    rest.duration.quarterLength = target
    measure.coreElementsChanged()
    return True


def _repair_uniform_overfull_run(measure: stream.Measure, target: Fraction) -> bool:
    """Fit a uniformly spaced tuplet-like run into its notated bar.

    Some tuplets have no exact reciprocal duration in the transformer's vocabulary.
    A five-note group spanning three beats may therefore arrive as five 2/3-beat
    notes.  Shortening only one note destroys the pulse.  When a measure ends with
    at least three consecutive, equal-duration events and their overflow is bounded,
    distribute the available span equally across the complete run instead.

    This is intentionally independent of pitch, part number, and measure number.
    It also accepts a rest as one member of a tuplet, as in a rest plus three notes
    under a 4:3 bracket.
    """
    measure_duration = _as_fraction(measure.duration.quarterLength)
    overflow = measure_duration - target
    if overflow <= 0 or overflow > target / 3:
        return False

    events = sorted(_measure_events(measure), key=lambda event: _event_offset(event, measure))
    if len(events) < 3 or _as_fraction(events[-1].duration.quarterLength) <= 0:
        return False
    duration = _as_fraction(events[-1].duration.quarterLength)

    run = [events[-1]]
    cursor = _event_offset(events[-1], measure)
    for event in reversed(events[:-1]):
        event_duration = _as_fraction(event.duration.quarterLength)
        event_offset = _event_offset(event, measure)
        if event_duration != duration or event_offset + duration != cursor:
            break
        run.append(event)
        cursor = event_offset
    run.reverse()
    run_start = _event_offset(run[0], measure)
    run_end = _event_offset(run[-1], measure) + duration
    if len(run) < 3 or run_end != measure_duration or run_start >= target:
        return False
    new_duration = (target - run_start) / len(run)
    is_complete_triplet_run = len(run) % 3 == 0 and new_duration == duration * Fraction(2, 3)
    has_stable_beat_span = (target - run_start).denominator <= 2
    if (
        new_duration <= 0
        or new_duration >= duration
        or not has_stable_beat_span
        or (new_duration < duration * Fraction(3, 4) and not is_complete_triplet_run)
        or new_duration.denominator > 20
    ):
        return False

    for index, event in enumerate(run):
        event.duration.quarterLength = new_duration
        _set_event_offset(event, measure, run_start + new_duration * index)
    measure.coreElementsChanged()
    return True


def _repair_late_overlapping_voice(  # noqa: PLR0911
    measure: stream.Measure, target: Fraction
) -> bool:
    """Shift one slightly late, independently voiced event back inside the bar."""
    events = [
        event
        for event in _measure_events(measure)
        if not isinstance(event, note.Rest) and _as_fraction(event.duration.quarterLength) > 0
    ]
    if not events:
        return False
    trailing = max(
        events,
        key=lambda event: (
            _event_offset(event, measure) + _as_fraction(event.duration.quarterLength)
        ),
    )
    trailing_end = _event_offset(trailing, measure) + _as_fraction(trailing.duration.quarterLength)
    overflow = trailing_end - target
    if overflow <= 0 or overflow > Fraction(1, 2):
        return False
    trailing_site = trailing.activeSite
    if not isinstance(trailing_site, stream.Voice):
        return False
    same_voice = [event for event in events if event.activeSite is trailing_site]
    if len(same_voice) != 1:
        return False
    repaired_offset = _event_offset(trailing, measure) - overflow
    if repaired_offset < 0:
        return False
    overlaps_other_voice = any(
        event is not trailing
        and isinstance(event.activeSite, stream.Voice)
        and event.activeSite is not trailing_site
        and _event_offset(event, measure)
        < repaired_offset + _as_fraction(trailing.duration.quarterLength)
        and _event_offset(event, measure) + _as_fraction(event.duration.quarterLength)
        > repaired_offset
        for event in events
    )
    if not overlaps_other_voice:
        return False
    _set_event_offset(trailing, measure, repaired_offset)
    measure.coreElementsChanged()
    return True


def _repair_appended_voice_suffix(  # noqa: PLR0911
    measure: stream.Measure, target: Fraction
) -> bool:
    """Fold a complete voice appended after the bar back over the primary voice.

    The suffix must begin exactly at the bar boundary and fill the entire overflow.
    A small scalar continuation is rejected; those notes are more likely a duration
    error than a missing MusicXML voice backup.
    """
    events = sorted(
        (
            event
            for event in _measure_events(measure)
            if not isinstance(event, note.Rest) and _as_fraction(event.duration.quarterLength) > 0
        ),
        key=lambda event: _event_offset(event, measure),
    )
    if not events:
        return False
    end = max(
        _event_offset(event, measure) + _as_fraction(event.duration.quarterLength)
        for event in events
    )
    overflow = end - target
    suffix = [event for event in events if _event_offset(event, measure) >= target]
    if overflow <= 0 or not suffix or _event_offset(suffix[0], measure) != target:
        return False
    if (
        max(
            _event_offset(event, measure) + _as_fraction(event.duration.quarterLength)
            for event in suffix
        )
        != end
    ):
        return False
    repaired_start = target - overflow
    if repaired_start < 0:
        return False
    prefix = [event for event in events if _event_offset(event, measure) < target]
    if not prefix:
        return False
    if len(suffix) == 1 and not any(_event_offset(event, measure) > 0 for event in prefix):
        return False
    previous = max(prefix, key=lambda event: _event_offset(event, measure))
    first = suffix[0]
    if (
        isinstance(previous, note.Note)
        and isinstance(first, note.Note)
        and abs(previous.pitch.midi - first.pitch.midi) <= 7
    ):
        return False
    overlaps_primary = any(
        _event_offset(event, measure) < repaired_start + overflow
        and _event_offset(event, measure) + _as_fraction(event.duration.quarterLength)
        > repaired_start
        for event in prefix
    )
    if not overlaps_primary:
        return False
    for event in suffix:
        _set_event_offset(event, measure, _event_offset(event, measure) - overflow)
    measure.coreElementsChanged()
    return True


def _repair_overfull_trailing_chord(measure: stream.Measure, target: Fraction) -> bool:
    """Restore polyphony when a final chord was serialized after a sustained voice."""
    overflow = _as_fraction(measure.duration.quarterLength) - target
    if overflow <= 0 or overflow > target / 2:
        return False
    events = _measure_events(measure)
    if not events or not isinstance(events[-1], chord.Chord):
        return False
    trailing = events[-1]
    trailing_duration = _as_fraction(trailing.duration.quarterLength)
    trailing_offset = _event_offset(trailing, measure)
    if trailing_duration != overflow or trailing_offset != target:
        return False
    repaired_offset = target - overflow
    overlapping = [
        event
        for event in events[:-1]
        if not isinstance(event, note.Rest)
        and Fraction(0) < _event_offset(event, measure) < repaired_offset
        and _event_offset(event, measure) + _as_fraction(event.duration.quarterLength) >= target
    ]
    if not overlapping:
        return False
    _set_event_offset(trailing, measure, repaired_offset)
    measure.coreElementsChanged()
    return True


def _measure_events(measure: stream.Measure) -> list[note.GeneralNote]:
    return list(measure.recurse().getElementsByClass((note.Note, note.Rest, chord.Chord)))


def _event_offset(event: note.GeneralNote, measure: stream.Measure) -> Fraction:
    return _as_fraction(event.getOffsetInHierarchy(measure))


def _set_event_offset(
    event: note.GeneralNote, measure: stream.Measure, new_offset: Fraction
) -> None:
    site = event.activeSite
    if site is None:
        return
    site_offset = _as_fraction(site.getOffsetInHierarchy(measure))
    site.setElementOffset(event, new_offset - site_offset)


def _same_scalar_direction(pitches: list[int]) -> bool:
    if len(pitches) < 2:
        return False
    differences = [right - left for left, right in zip(pitches, pitches[1:], strict=False)]
    direction = 1 if differences[0] > 0 else -1
    return all(
        0 < abs(interval) <= 7 and (1 if interval > 0 else -1) == direction
        for interval in differences
    )


def _repair_cross_staff_run(measures: list[stream.Measure], target: Fraction) -> bool:
    """Reconnect a short scalar run that the decoder split across piano staves.

    Cross-staff beaming can put the last few right-hand notes into the lower-staff
    timeline.  The repair is deliberately strict: the source must end with at least
    four equal fast notes followed by one note decoded at twice their duration, the
    destination must end with at least two matching fast notes, the combined pitches
    must continue monotonically, and shortening/moving them must exactly fill the bar.
    """
    if len(measures) < 2:
        return False

    for source in measures:
        source_events = _measure_events(source)
        if len(source_events) < 5 or not all(
            isinstance(event, note.Note) for event in source_events[-5:]
        ):
            continue

        source_notes = [event for event in source_events if isinstance(event, note.Note)]
        final_note = source_notes[-1]
        rapid_duration = _as_fraction(source_notes[-2].duration.quarterLength)
        final_duration = _as_fraction(final_note.duration.quarterLength)
        if rapid_duration <= 0 or rapid_duration > Fraction(1, 4):
            continue
        if final_duration != rapid_duration * 2:
            continue

        rapid_prefix: list[note.Note] = []
        cursor = _event_offset(final_note, source)
        for candidate in reversed(source_notes[:-1]):
            candidate_duration = _as_fraction(candidate.duration.quarterLength)
            candidate_offset = _event_offset(candidate, source)
            if candidate_duration != rapid_duration or candidate_offset + rapid_duration != cursor:
                break
            rapid_prefix.append(candidate)
            cursor = candidate_offset
        rapid_prefix.reverse()
        if len(rapid_prefix) < 4:
            continue

        source_end = _event_offset(final_note, source) + final_duration
        continuation_start = source_end - rapid_duration
        if source_end >= target or target - source_end > Fraction(1, 2):
            continue

        for destination in measures:
            if destination is source:
                continue
            destination_events = _measure_events(destination)
            suffix: list[note.Note] = []
            suffix_cursor: Fraction | None = None
            for candidate in reversed(destination_events):
                if not isinstance(candidate, note.Note):
                    break
                candidate_duration = _as_fraction(candidate.duration.quarterLength)
                candidate_offset = _event_offset(candidate, destination)
                if candidate_duration != rapid_duration:
                    break
                if suffix_cursor is not None and candidate_offset + rapid_duration != suffix_cursor:
                    break
                suffix.append(candidate)
                suffix_cursor = candidate_offset
            suffix.reverse()
            if len(suffix) < 2:
                continue
            suffix_end = _event_offset(suffix[-1], destination) + rapid_duration
            if suffix_end > continuation_start:
                continue
            if continuation_start + rapid_duration * len(suffix) != target:
                continue

            pitch_run = [candidate.pitch.midi for candidate in rapid_prefix]
            pitch_run.append(final_note.pitch.midi)
            pitch_run.extend(candidate.pitch.midi for candidate in suffix)
            if not _same_scalar_direction(pitch_run):
                continue

            final_note.duration.quarterLength = rapid_duration
            for index, candidate in enumerate(suffix):
                _set_event_offset(
                    candidate,
                    destination,
                    continuation_start + rapid_duration * index,
                )
            source.coreElementsChanged()
            destination.coreElementsChanged()
            return True
    return False


def stabilize_measure_offsets(score: stream.Score) -> int:
    """Anchor MIDI measures to the notated meter instead of decoded overflow.

    Music21 normally places the next measure after the longest event in the previous
    one. A single OMR duration error can consequently shift the rest of the score.
    Re-anchoring measure containers contains that error to its source measure. A small
    overflow is also repaired when one common rhythmic value can absorb it exactly;
    notes are never deleted.

    Returns the number of offsets that were changed.
    """
    reconcile_measure_boundaries(score)
    parts = list(score.parts)
    measures_by_part = [list(part.getElementsByClass(stream.Measure)) for part in parts]
    populated = [measures for measures in measures_by_part if measures]
    if not populated:
        return 0
    fallback_duration = _fallback_bar_duration(populated[0])
    expected_offset = min(_as_fraction(measures[0].offset) for measures in populated)
    active_bar_duration = fallback_duration
    changed = 0
    repaired = 0
    whole_rests_normalized = 0
    uniform_runs_repaired = 0
    cross_staff_repaired = 0
    overlapping_chords_repaired = 0
    overlapping_voices_repaired = 0
    appended_voices_repaired = 0
    isolated_overhangs_repaired = 0
    duration_outliers_repaired = 0

    for index in range(max(len(measures) for measures in populated)):
        current = [measures[index] for measures in populated if index < len(measures)]
        for measure in current:
            signature = measure.timeSignature
            if isinstance(signature, meter.TimeSignature):
                active_bar_duration = _as_fraction(signature.barDuration.quarterLength)
                break

        whole_rests_normalized += sum(
            int(_normalize_whole_measure_rest(measure, active_bar_duration)) for measure in current
        )
        isolated_overhangs_repaired += sum(
            _repair_isolated_voice_overhang(measure, active_bar_duration) for measure in current
        )
        duration_outliers_repaired += sum(
            int(_repair_supported_duration_outlier(measure, active_bar_duration))
            for measure in current
        )
        uniform_runs_repaired += sum(
            int(_repair_uniform_overfull_run(measure, active_bar_duration)) for measure in current
        )
        cross_staff_repaired += int(_repair_cross_staff_run(current, active_bar_duration))
        overlapping_chords_repaired += sum(
            int(_repair_overfull_trailing_chord(measure, active_bar_duration))
            for measure in current
        )
        overlapping_voices_repaired += sum(
            int(_repair_late_overlapping_voice(measure, active_bar_duration)) for measure in current
        )
        appended_voices_repaired += sum(
            int(_repair_appended_voice_suffix(measure, active_bar_duration)) for measure in current
        )
        measure_span = _consensus_measure_span(current, active_bar_duration)
        for part, measures in zip(parts, measures_by_part, strict=True):
            if index >= len(measures):
                continue
            measure = measures[index]
            repaired += int(_repair_small_measure_overflow(measure, measure_span))
            if index > 0 and _as_fraction(measure.offset) != expected_offset:
                part.setElementOffset(measure, expected_offset)
                changed += 1

        is_pickup = index == 0 and any(
            measure.number == 0 or _as_fraction(measure.paddingLeft) > 0 for measure in current
        )
        if is_pickup:
            expected_offset += max(
                _as_fraction(measure.duration.quarterLength) for measure in current
            )
        else:
            expected_offset += measure_span

    for part in parts:
        part.coreElementsChanged()

    if changed:
        eprint("Stabilized", changed, "measure offsets before MIDI export")
    if repaired:
        eprint("Repaired", repaired, "small measure overflows before MIDI export")
    if whole_rests_normalized:
        eprint(
            "Normalized",
            whole_rests_normalized,
            "whole-measure rests to the active meter before MIDI export",
        )
    if uniform_runs_repaired:
        eprint(
            "Redistributed",
            uniform_runs_repaired,
            "uniform overfull runs before MIDI export",
        )
    if cross_staff_repaired:
        eprint("Reconnected", cross_staff_repaired, "cross-staff runs before MIDI export")
    if overlapping_chords_repaired:
        eprint(
            "Restored",
            overlapping_chords_repaired,
            "overlapping trailing chords before MIDI export",
        )
    if overlapping_voices_repaired:
        eprint(
            "Restored",
            overlapping_voices_repaired,
            "late overlapping voices before MIDI export",
        )
    if appended_voices_repaired:
        eprint(
            "Folded",
            appended_voices_repaired,
            "appended voice suffixes into their measures before MIDI export",
        )
    if isolated_overhangs_repaired:
        eprint(
            "Trimmed",
            isolated_overhangs_repaired,
            "isolated voice overhangs at measure boundaries before MIDI export",
        )
    if duration_outliers_repaired:
        eprint(
            "Corrected",
            duration_outliers_repaired,
            "consensus-supported duration outliers before MIDI export",
        )
    return changed
