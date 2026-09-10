import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction

import musicxml.xmlelement.xmlelement as mxl
import numpy as np

from homr import constants
from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol, empty, nonote, sort_token_chords


class ConversionState:
    def __init__(self, division: int, nominator: Fraction):
        self.beats = 4 * constants.duration_of_quarter
        self.division = division
        self.nominator = nominator
        self.tremolo_state = "stop"
        self.volta_number = 1
        self.last_volta_measure = -10

    def start_volta(self, measure_no: int) -> int:
        if measure_no == self.last_volta_measure + 1:
            self.volta_number += 1
        else:
            self.volta_number = 1
        return self.volta_number

    def stop_volta(self, measure_no: int) -> int:
        self.last_volta_measure = measure_no
        return self.volta_number

    def toggle_tremolo_state(self) -> str:
        if self.tremolo_state == "start":
            self.tremolo_state = "stop"
        else:
            self.tremolo_state = "start"
        return self.tremolo_state


class SymbolChord:
    def __init__(self, symbols: list[EncodedSymbol], tuplet_mark: str = "") -> None:
        self.symbols = symbols
        self.tuplet_mark = tuplet_mark

    def __str__(self) -> str:
        return str.join("&", [str(s) for s in self.symbols])

    def __repr__(self) -> str:
        return str(self)

    def is_barline(self) -> bool:
        if len(self.symbols) == 0:
            return False
        first_rhythm = self.symbols[0].rhythm
        return "barline" in first_rhythm or "repeat" in first_rhythm

    def get_duration(self) -> Fraction:
        notes_rests = [
            _sounding_symbol_duration(s)
            for s in self.symbols
            if s.rhythm.startswith(("note", "rest"))
        ]
        if len(notes_rests) == 0:
            return Fraction(0)
        return min(notes_rests)

    def into_positions(self) -> list["SymbolChord"]:
        upper = []
        lower = []
        lower_is_only_rest = True
        for symbol in self.symbols:
            if symbol.position == "upper":
                upper.append(symbol)
            else:
                lower.append(symbol)
                lower_is_only_rest = lower_is_only_rest and symbol.rhythm.startswith("rest")
        chords = (
            SymbolChord(upper, self.tuplet_mark),
            SymbolChord(lower, self.tuplet_mark),
        )
        if lower_is_only_rest:
            chords = (chords[1], chords[0])
        return [chord for chord in chords if len(chord.symbols) > 0]

    def strip_slur_ties(self) -> tuple[list[str], "SymbolChord"]:
        slurs_ties = set()

        result = []
        for symbol in self.symbols:
            stripped, result_symbol = symbol.strip_articulations(
                ["slurStart", "slurStop", "tieStart", "tieStop"]
            )
            result.append(result_symbol)
            for articulation in stripped:
                slurs_ties.add(articulation)

        return sorted(slurs_ties), SymbolChord(result, tuplet_mark=self.tuplet_mark)


class XmlGeneratorArguments:
    def __init__(
        self, large_page: bool | None = None, metronome: int | None = None, tempo: int | None = None
    ):
        self.large_page = large_page
        self.metronome = metronome
        self.tempo = tempo


def build_identification() -> mxl.XMLIdentification:
    """Identification/encoding so validators and apps (e.g. MuseScore) can attribute the file."""
    ident = mxl.XMLIdentification()
    enc = mxl.XMLEncoding()
    enc.add_child(mxl.XMLSoftware(value_="homr"))
    ident.add_child(enc)
    return ident


def generate_xml(
    args: XmlGeneratorArguments, staffs: list[list[EncodedSymbol]], title: str
) -> mxl.XMLElement:
    root = mxl.XMLScorePartwise(version="4.0")
    root.add_child(build_work(title))
    root.add_child(build_identification())
    root.add_child(build_defaults(args))
    has_two_staves_by_part = [_voice_has_two_staves(staff) for staff in staffs]
    shared_nominator = infer_shared_time_signature_nominator(staffs, has_two_staves_by_part)
    root.add_child(build_part_list(has_two_staves_by_part))
    for index, staff in enumerate(staffs):
        root.add_child(
            build_part(
                args,
                staff,
                index,
                has_two_staves_by_part[index],
                shared_nominator,
            )
        )
    return root


def _voice_has_two_staves(voice: list[EncodedSymbol]) -> bool:
    """True if any symbol uses the lower staff (e.g. piano left hand / bass clef)."""
    return any(s.position == "lower" for s in voice)


def build_part(
    args: XmlGeneratorArguments,
    voice: list[EncodedSymbol],
    index: int,
    has_two_staves: bool,
    shared_nominator: Fraction | None = None,
) -> mxl.XMLPart:
    part = mxl.XMLPart(id=get_part_id(index))
    is_first_part = index == 0
    measures = build_measures(args, voice, is_first_part, has_two_staves, shared_nominator)
    for measure in measures:
        part.add_child(measure)
    return part


def build_measures(
    args: XmlGeneratorArguments,
    voice: list[EncodedSymbol],
    is_first_part: bool,
    has_two_staves: bool = False,
    shared_nominator: Fraction | None = None,
) -> list[mxl.XMLMeasure]:
    staff_content: dict[int, list[mxl.XMLElement]] = {1: [], 2: []}
    staff_cursors = {1: 0, 2: 0}

    def flush_staff_content() -> None:
        """Serialize piano staves as independent MusicXML timelines.

        Transformer chord markers are occasionally missing between vertically aligned
        notes on a grand staff. Interleaving both staves would then serialize the left
        hand after the right hand and make the measure (and every following MIDI
        measure) too long. Each staff therefore owns its cursor; staff 2 is rewound to
        the measure start with a MusicXML backup.
        """
        for child in staff_content[1]:
            current_measure.add_child(child)
        if staff_content[2]:
            if staff_cursors[1] > 0:
                backup = mxl.XMLBackup()
                backup.add_child(mxl.XMLDuration(value_=staff_cursors[1]))
                current_measure.add_child(backup)
            for child in staff_content[2]:
                current_measure.add_child(child)
        staff_content[1].clear()
        staff_content[2].clear()
        staff_cursors[1] = 0
        staff_cursors[2] = 0

    def close_current_measure() -> None:
        flush_staff_content()
        rebalance_measure_voices(current_measure)
        measures.append(current_measure)

    measure_number = 1
    groups = add_tuplet_start_stop(group_into_chords(voice))
    _mark_double_tremolo_pairs(groups, has_two_staves)
    division, nominator = find_division_and_time_signature_nominator(groups, has_two_staves)
    if shared_nominator is not None:
        nominator = shared_nominator
    state = ConversionState(division, nominator)
    measures: list[mxl.XMLMeasure] = []
    current_measure = mxl.XMLMeasure(number=str(measure_number))
    first_attributes = build_or_get_attributes(current_measure, None)
    first_attributes.add_child(build_divisions(division))
    if not _has_initial_time_signature(voice):
        build_inferred_time_signature(first_attributes, state)
    if has_two_staves:
        first_attributes.add_child(mxl.XMLStaves(value_=2))
        first_attributes.add_child(mxl.XMLPartSymbol(value_="brace"))
    if is_first_part:
        direction = build_add_time_direction(args)
        if direction:
            current_measure.add_child(direction)
    attributes: mxl.XMLAttributes | None = first_attributes
    for group_no, group in enumerate(groups):
        symbol = group.symbols[0]
        rhythm = symbol.rhythm
        last_attributes = attributes
        attributes = None
        if rhythm.startswith(("note", "rest")):
            if len(group.symbols) == 1 and rhythm.endswith("m"):
                attributes = build_or_get_attributes(current_measure, last_attributes)
                build_multi_measure_rest(symbol, attributes)
            else:
                staff_positions = group.into_positions()
                for staff_pos in staff_positions:
                    staff_num = get_staff(staff_pos.symbols[0])
                    chord_duration = staff_pos.get_duration()
                    note_elements = build_note_chord(staff_pos, state, chord_duration)
                    staff_content[staff_num].extend(note_elements)
                    staff_cursors[staff_num] += _xml_cursor_delta(note_elements)
            continue
        if rhythm == "newline":
            is_last_measure = group_no == len(groups) - 1
            if not is_last_measure:
                current_measure.add_child(mxl.XMLPrint(new_system="yes"))
        elif rhythm.startswith("clef"):
            attributes = build_or_get_attributes(current_measure, last_attributes, force_new=True)
            for should_be_clef in group.symbols:
                if should_be_clef.rhythm.startswith("clef"):
                    build_clef(should_be_clef, attributes)
        elif rhythm.startswith("keySignature"):
            attributes = build_or_get_attributes(current_measure, last_attributes)
            build_key(symbol, attributes)
        elif rhythm.startswith("timeSignature"):
            attributes = build_or_get_attributes(current_measure, last_attributes)
            build_time_signature(symbol, attributes, state)
        elif "barline" in rhythm:
            flush_staff_content()
            if rhythm != "barline":
                # Standard barlines don't need extra handling
                barline = build_or_get_barline(current_measure, "right")
                build_barline_style(symbol, barline)

            close_current_measure()
            measure_number += 1
            current_measure = mxl.XMLMeasure(number=str(measure_number))
        elif rhythm == "repeatStart":
            close_current_measure()
            measure_number += 1
            current_measure = mxl.XMLMeasure(number=str(measure_number))

            barline = build_or_get_barline(current_measure, "right")
            build_repeat(symbol, barline)
        elif rhythm == "repeatEnd":
            flush_staff_content()
            barline = build_or_get_barline(current_measure, "right")
            build_repeat(symbol, barline)

            close_current_measure()
            measure_number += 1
            current_measure = mxl.XMLMeasure(number=str(measure_number))
        elif rhythm == "repeatEndStart":
            flush_staff_content()
            barline = build_or_get_barline(current_measure, "right")
            build_repeat(EncodedSymbol("repeatEnd"), barline)

            close_current_measure()
            measure_number += 1
            current_measure = mxl.XMLMeasure(number=str(measure_number))
            barline = build_or_get_barline(current_measure, "right")
            build_repeat(EncodedSymbol("repeatStart"), barline)
        elif rhythm.startswith("voltaStart"):
            volta_number = state.start_volta(measure_number)
            barline = build_or_get_barline(current_measure, "left")
            build_barline_ending(symbol, barline, volta_number)
        elif rhythm.startswith(("voltaStop", "voltaDiscontinue")):
            volta_number = state.stop_volta(measure_number)
            barline = build_or_get_barline(current_measure, "right")
            build_barline_ending(symbol, barline, volta_number)
        else:
            eprint("Symbol isn't supported yet ", symbol)

    if len(current_measure.get_children()) > 0:
        close_current_measure()
    return measures


def build_work(title_text: str) -> mxl.XMLWork:
    work = mxl.XMLWork()
    title = mxl.XMLWorkTitle()
    title._value = title_text
    work.add_child(title)
    return work


def build_defaults(args: XmlGeneratorArguments) -> mxl.XMLDefaults:
    if not args.large_page:
        return mxl.XMLDefaults()
    # These values are larger than a letter or A4 format so that
    # we only have to break staffs with every new detected staff
    # This works well for electronic formats, if the results are supposed
    # to get printed then they might need to be scaled down to fit the page
    page_width = 110  # Unit is in tenths: https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/page-height/
    page_height = 300
    defaults = mxl.XMLDefaults()
    page_layout = mxl.XMLPageLayout()
    page_height = mxl.XMLPageHeight(value_=page_height)
    page_width = mxl.XMLPageWidth(value_=page_width)
    page_layout.add_child(page_height)
    page_layout.add_child(page_width)
    defaults.add_child(page_layout)
    return defaults


def get_part_id(index: int) -> str:
    return "P" + str(index + 1)


def _part_metadata(has_two_staves: bool) -> tuple[str, str, str, int]:
    """Return (part_name, instrument_name, instrument_sound, midi_program) for a part.

    We classify by staff layout only:
    - single-staff parts -> Voice
    - two-staff parts -> Piano
    midi_program is 1-based (MusicXML: 1-128; 1 = Acoustic Grand Piano, 54 = Voice Oohs).
    """
    if has_two_staves:
        return ("Piano", "Piano", "keyboard.piano", 1)
    return ("Voice", "Voice", "voice", 54)


def build_part_list(has_two_staves_by_part: list[bool]) -> mxl.XMLPartList:
    part_list = mxl.XMLPartList()
    for part, has_two_staves in enumerate(has_two_staves_by_part):
        part_id = get_part_id(part)
        part_name_str, instrument_name_str, instrument_sound_str, midi_program = _part_metadata(
            has_two_staves
        )
        score_part = mxl.XMLScorePart(id=part_id)
        part_name = mxl.XMLPartName(value_=part_name_str)
        score_part.add_child(part_name)
        score_instrument = mxl.XMLScoreInstrument(id=part_id + "-I1")
        instrument_name = mxl.XMLInstrumentName(value_=instrument_name_str)
        score_instrument.add_child(instrument_name)
        instrument_sound = mxl.XMLInstrumentSound(value_=instrument_sound_str)
        score_instrument.add_child(instrument_sound)
        score_part.add_child(score_instrument)
        midi_instrument = mxl.XMLMidiInstrument(id=part_id + "-I1")
        midi_instrument.add_child(mxl.XMLMidiChannel(value_=part + 1))
        midi_instrument.add_child(mxl.XMLMidiProgram(value_=midi_program))
        midi_instrument.add_child(mxl.XMLVolume(value_=100))
        midi_instrument.add_child(mxl.XMLPan(value_=0))
        score_part.add_child(midi_instrument)
        part_list.add_child(score_part)
    return part_list


def build_or_get_attributes(
    measure: mxl.XMLMeasure, last_attributes: mxl.XMLAttributes | None, force_new: bool = False
) -> mxl.XMLAttributes:
    if last_attributes is not None and not force_new:
        return last_attributes

    attributes = mxl.XMLAttributes()
    measure.add_child(attributes)
    return attributes


def build_or_get_barline(measure: mxl.XMLMeasure, location: str) -> mxl.XMLBarline:
    children = measure.get_children_of_type(mxl.XMLBarline)
    for child in children:
        if child.location == location:
            return child

    barline = mxl.XMLBarline(location=location)
    measure.add_child(barline)
    return barline


def build_key(model_key: EncodedSymbol, attributes: mxl.XMLAttributes) -> None:
    key = mxl.XMLKey()
    circle_of_fifth = model_key.rhythm.split("_")[1]
    fifth = mxl.XMLFifths(value_=int(circle_of_fifth))
    attributes.add_child(key)
    key.add_child(fifth)


def get_staff(symbol: EncodedSymbol) -> int:
    return 2 if symbol.position == "lower" else 1


def get_xml_voice(staff_num: int, rhythmic_layer: int) -> int:
    """Build a stable MusicXML voice number per staff and rhythmic layer.

    Voice numbers are part-global in MusicXML, so using only the staff number can merge
    independent layers on the same staff. Reserve 4 voices per staff:
    staff 1 -> voices 1-4, staff 2 -> voices 5-8.
    """
    return (staff_num - 1) * 4 + rhythmic_layer + 1


@dataclass
class TimedNoteEvent:
    staff_num: int
    start: int
    end: int
    notes: list[mxl.XMLNote]


def rebalance_measure_voices(measure: mxl.XMLMeasure) -> None:
    """Assign stable non-overlapping voices per staff for a whole measure."""
    timed_events: list[TimedNoteEvent] = []
    current_time = 0
    last_note_start = 0
    for child in measure.get_children():
        if isinstance(child, mxl.XMLBackup):
            durations = child.get_children_of_type(mxl.XMLDuration)
            if len(durations) > 0:
                current_time -= int(durations[0].value_)
            continue
        if child.__class__.__name__ == "XMLForward":
            durations = child.get_children_of_type(mxl.XMLDuration)
            if len(durations) > 0:
                current_time += int(durations[0].value_)
            continue
        if not isinstance(child, mxl.XMLNote):
            continue

        duration_nodes = child.get_children_of_type(mxl.XMLDuration)
        duration = int(duration_nodes[0].value_) if len(duration_nodes) > 0 else 0
        staff_nodes = child.get_children_of_type(mxl.XMLStaff)
        staff_num = int(staff_nodes[0].value_) if len(staff_nodes) > 0 else 1
        is_chord_tone = len(child.get_children_of_type(mxl.XMLChord)) > 0
        start = last_note_start if is_chord_tone else current_time
        end = start + duration
        if is_chord_tone and (
            len(timed_events) > 0
            and timed_events[-1].staff_num == staff_num
            and timed_events[-1].start == start
            and timed_events[-1].end == end
        ):
            timed_events[-1].notes.append(child)
        elif is_chord_tone:
            timed_events.append(TimedNoteEvent(staff_num, start, end, [child]))
        else:
            last_note_start = start
            current_time += duration
            timed_events.append(TimedNoteEvent(staff_num, start, end, [child]))

    by_staff: dict[int, list[TimedNoteEvent]] = defaultdict(list)
    for event in timed_events:
        by_staff[event.staff_num].append(event)

    for staff_num, events in by_staff.items():
        sorted_events = sorted(events, key=lambda e: (e.start, e.end))
        active: list[tuple[int, int]] = []  # (end, local voice number 1..n)
        for event in sorted_events:
            active = [
                (active_end, voice_no)
                for active_end, voice_no in active
                if active_end > event.start
            ]
            used_voices = {voice_no for _, voice_no in active}
            voice_no = 1
            while voice_no in used_voices:
                voice_no += 1
            active.append((event.end, voice_no))
            xml_voice = str(get_xml_voice(staff_num, voice_no - 1))
            for note in event.notes:
                voice_nodes = note.get_children_of_type(mxl.XMLVoice)
                if len(voice_nodes) > 0:
                    voice_nodes[0].value_ = xml_voice


def build_clef(model_clef: EncodedSymbol, attributes: mxl.XMLAttributes) -> None:
    sign_and_line = model_clef.rhythm.split("_")[1]
    sign = sign_and_line[0]
    line = sign_and_line[1]
    clef = mxl.XMLClef(number=get_staff(model_clef))
    attributes.add_child(clef)
    clef.add_child(mxl.XMLSign(value_=sign))
    clef.add_child(mxl.XMLLine(value_=int(line)))


def build_time_signature(
    model_time_signature: EncodedSymbol, attributes: mxl.XMLAttributes, state: ConversionState
) -> None:
    time = mxl.XMLTime()

    denominator = model_time_signature.rhythm.split("/")[1]
    attributes.add_child(time)
    beats = max(int(state.nominator * int(denominator)), 1)
    time.add_child(mxl.XMLBeats(value_=str(beats)))
    time.add_child(mxl.XMLBeatType(value_=denominator))
    state.beats = beats


def _has_initial_time_signature(voice: list[EncodedSymbol]) -> bool:
    """Whether the crop contains a time signature before its first measure content."""
    for symbol in voice:
        if symbol.rhythm.startswith("timeSignature"):
            return True
        if symbol.rhythm.startswith(("note", "rest")) or "barline" in symbol.rhythm:
            return False
    return False


def build_inferred_time_signature(attributes: mxl.XMLAttributes, state: ConversionState) -> None:
    """Write a conservative meter when a screenshot starts in the middle of a score.

    HOMR can infer the typical measure duration even when the printed time
    signature is outside the crop. Prefer a quarter-note denominator because it is
    the least surprising DAW interpretation; use eighths only when quarters cannot
    represent the inferred duration exactly.
    """
    measure_duration = Fraction(state.nominator).limit_denominator(32)
    denominator = 4
    beats_fraction = measure_duration * denominator
    if beats_fraction.denominator != 1:
        denominator = 8
        beats_fraction = measure_duration * denominator
    if beats_fraction.denominator != 1:
        return
    beats = int(beats_fraction)
    if beats < 1 or beats > 12:
        return
    time = mxl.XMLTime()
    time.add_child(mxl.XMLBeats(value_=str(beats)))
    time.add_child(mxl.XMLBeatType(value_=str(denominator)))
    attributes.add_child(time)
    state.beats = beats


def build_barline_style(barline: EncodedSymbol, xml: mxl.XMLBarline) -> None:
    style_value = "heavy-heavy" if barline.rhythm == "bolddoublebarline" else "light-light"
    style = mxl.XMLBarStyle(value_=style_value)
    xml.add_child(style)


def build_barline_ending(volta: EncodedSymbol, xml: mxl.XMLBarline, volta_number: int) -> None:
    if volta.rhythm.startswith("voltaStart"):
        ending = mxl.XMLEnding(type="start", number=str(volta_number))
    elif volta.rhythm.startswith("voltaStop"):
        ending = mxl.XMLEnding(type="stop", number=str(volta_number))
    elif volta.rhythm.startswith("voltaDiscontinue"):
        ending = mxl.XMLEnding(type="discontinue", number=str(volta_number))
    else:
        raise ValueError("Unknown ending " + str(volta))
    xml.add_child(ending)


def build_repeat(barline: EncodedSymbol, xml: mxl.XMLBarline) -> None:
    if len(xml.get_children_of_type(mxl.XMLRepeat)) > 0:
        eprint("barline already has a repeat")
        return

    repeat = mxl.XMLRepeat()
    direction = "forward" if barline.rhythm == "repeatStart" else "backward"
    repeat._set_attributes({"direction": direction})
    xml.add_child(repeat)


LIFT_TO_ALTER = {
    "N": 0,
    "#": 1,
    "##": 2,
    "b": -1,
    "bb": -2,
}

DURATION_NAMES = {
    0: "breve",
    1: "whole",
    2: "half",
    4: "quarter",
    8: "eighth",
    16: "16th",
    32: "32nd",
    64: "64th",
    128: "128th",
}


def build_articulations(
    note: mxl.XMLNote,
    articualations: str,
    tuplet_mark: str,
    state: ConversionState,
    double_tremolo: bool = False,
) -> None:
    notation = mxl.XMLNotations()
    note.add_child(notation)

    xml_articulations = []
    xml_ornaments = []

    for articulation in articualations.split("_"):
        if articulation == "":
            continue
        elif articulation == nonote:
            eprint("WARNING note without valid articulation", articualations)
        elif articulation == "fermata":
            notation.add_child(mxl.XMLFermata())
        elif articulation == "arpeggiate":
            notation.add_child(mxl.XMLArpeggiate())
        elif articulation == "accent":
            xml_articulations.append(mxl.XMLAccent())
        elif articulation == "arpeggiate":
            notation.add_child(mxl.XMLArpeggiate())
        elif articulation == "fermata":
            xml_articulations.append(mxl.XMLFermata())
        elif articulation == "staccato":
            xml_articulations.append(mxl.XMLStaccato())
        elif articulation == "staccatissimo":
            xml_articulations.append(mxl.XMLStaccatissimo())
        elif articulation == "tenuto":
            xml_articulations.append(mxl.XMLTenuto())
        elif articulation == "tremolo":
            tremolo_type = state.toggle_tremolo_state() if double_tremolo else "single"
            xml_ornaments.append(mxl.XMLTremolo(value_=3, type=tremolo_type))
        elif articulation == "trill":
            xml_ornaments.append(mxl.XMLTrillMark())
        elif articulation == "breathMark":
            xml_articulations.append(mxl.XMLBreathMark())
        elif articulation == "turn":
            xml_ornaments.append(mxl.XMLInvertedTurn())
        elif articulation == "caesura":
            xml_articulations.append(mxl.XMLCaesura())
        elif articulation == "doit":
            xml_articulations.append(mxl.XMLDoit())
        elif articulation == "slurStart":
            notation.add_child(mxl.XMLSlur(type="start"))
        elif articulation == "slurStop":
            notation.add_child(mxl.XMLSlur(type="stop"))
        elif articulation == "tieStart":
            notation.add_child(mxl.XMLTied(type="start"))
        elif articulation == "tieStop":
            notation.add_child(mxl.XMLTied(type="stop"))
        else:
            raise ValueError("Unsupported articulation " + articulation)

    if tuplet_mark != "":
        tuplet_xml = mxl.XMLTuplet(type=tuplet_mark)
        notation.add_child(tuplet_xml)

    if len(xml_articulations) > 0:
        parent = mxl.XMLArticulations()
        for child in xml_articulations:
            parent.add_child(child)
        notation.add_child(parent)

    if len(xml_ornaments) > 0:
        parent = mxl.XMLOrnaments()
        for child in xml_ornaments:
            parent.add_child(child)
        notation.add_child(parent)


def build_note_or_rest(
    model_note: EncodedSymbol,
    rhythmic_layer: int,
    is_chord: bool,
    state: ConversionState,
    tuplet_mark: str,
) -> mxl.XMLNote:
    note = mxl.XMLNote()
    if is_chord:
        note.add_child(mxl.XMLChord())
    model_pitch = model_note.pitch
    model_duration = model_note.get_duration()
    if model_pitch == empty:
        if model_duration.fraction.numerator == 0:
            note.add_child(mxl.XMLRest(measure="yes"))
        else:
            note.add_child(mxl.XMLRest())
    elif model_pitch == nonote:
        eprint("WARNING note without pitch", model_note)
        note.add_child(mxl.XMLRest())
    else:
        pitch = mxl.XMLPitch()
        pitch.add_child(mxl.XMLStep(value_=model_pitch[0]))
        pitch.add_child(mxl.XMLOctave(value_=int(model_pitch[1])))
        if model_note.lift == nonote:
            eprint("WARNING note with invalid lift", model_note)
        elif model_note.lift != empty:
            pitch.add_child(mxl.XMLAlter(value_=LIFT_TO_ALTER[model_note.lift]))
        note.add_child(pitch)

    if "G" in model_note.rhythm:
        note.add_child(mxl.XMLGrace())
        base_duration = model_duration.kern
        duration_name = DURATION_NAMES[base_duration]
        note.add_child(mxl.XMLType(value_=duration_name))
    elif model_duration.fraction.numerator > 0:
        base_duration = 1 if model_duration.kern == 0 else model_duration.kern
        duration_name = DURATION_NAMES[base_duration]
        note.add_child(mxl.XMLType(value_=duration_name))
        note.add_child(
            mxl.XMLDuration(value_=int(_sounding_symbol_duration(model_note) * state.division))
        )
    else:
        duration_name = DURATION_NAMES[0]
        note.add_child(mxl.XMLType(value_=duration_name))
        note.add_child(mxl.XMLDuration(value_=state.beats))

    staff_num = get_staff(model_note)
    note.add_child(mxl.XMLStaff(value_=staff_num))
    note.add_child(mxl.XMLVoice(value_=str(get_xml_voice(staff_num, rhythmic_layer))))
    for _ in range(model_duration.dots):
        note.add_child(mxl.XMLDot())
    if model_duration.actual_notes != model_duration.normal_notes or _has_double_tremolo(
        model_note
    ):
        time_modification = mxl.XMLTimeModification()
        actual_notes = model_duration.actual_notes * (2 if _has_double_tremolo(model_note) else 1)
        time_modification.add_child(mxl.XMLActualNotes(value_=actual_notes))
        time_modification.add_child(mxl.XMLNormalNotes(value_=model_duration.normal_notes))
        note.add_child(time_modification)
        build_articulations(
            note,
            model_note.articulation,
            tuplet_mark if model_duration.actual_notes != model_duration.normal_notes else "",
            state,
            _has_double_tremolo(model_note),
        )
    else:
        build_articulations(
            note,
            model_note.articulation,
            "",
            state,
            _has_double_tremolo(model_note),
        )

    return note


def build_multi_measure_rest(
    symbol: EncodedSymbol, attributes: mxl.XMLAttributes
) -> mxl.XMLMeasureStyle:
    other_styles = attributes.get_children_of_type(mxl.XMLMeasureStyle)
    if len(other_styles) > 0:
        eprint("Measure already has a multi rest")
        return
    duration = int(symbol.rhythm.split("_")[1].replace("m", ""))
    style = mxl.XMLMeasureStyle()
    rest = mxl.XMLMultipleRest(value_=duration)
    style.add_child(rest)
    attributes.add_child(style)


def build_note_chord(
    note_chord: SymbolChord, state: ConversionState, chord_duration: Fraction
) -> list[mxl.XMLElement]:
    _slurs_ties, note_chord = note_chord.strip_slur_ties()
    by_duration = _group_notes(note_chord.symbols)
    result: list[mxl.XMLElement] = []
    final_duration = Fraction(0)
    sorted_durations = sorted(by_duration)
    for i, group_duration in enumerate(sorted_durations):
        is_first = True
        for note_loop in by_duration[group_duration]:
            note = note_loop
            result.append(build_note_or_rest(note, i, not is_first, state, note_chord.tuplet_mark))
            is_first = False
        if i != len(sorted_durations) - 1 and group_duration > Fraction(0):
            backup = mxl.XMLBackup()
            backup.add_child(mxl.XMLDuration(value_=int(group_duration * state.division)))
            result.append(backup)

        final_duration = group_duration

    # Reset the position to match the chord position
    if chord_duration < final_duration:
        backup = mxl.XMLBackup()
        backup.add_child(
            mxl.XMLDuration(value_=int((final_duration - chord_duration) * state.division))
        )
        result.append(backup)
    return result


def _xml_cursor_delta(elements: list[mxl.XMLElement]) -> int:
    """Return the net MusicXML cursor movement of a generated note group."""
    cursor = 0
    for element in elements:
        durations = element.get_children_of_type(mxl.XMLDuration)
        duration = int(durations[0].value_) if durations else 0
        if isinstance(element, mxl.XMLBackup):
            cursor -= duration
        elif element.__class__.__name__ == "XMLForward":
            cursor += duration
        elif isinstance(element, mxl.XMLNote):
            is_chord_tone = len(element.get_children_of_type(mxl.XMLChord)) > 0
            if not is_chord_tone:
                cursor += duration
    return max(cursor, 0)


def _group_notes(notes: list[EncodedSymbol]) -> dict[Fraction, list[EncodedSymbol]]:
    groups_by_duration = defaultdict(list)
    max_duration = max([_sounding_symbol_duration(n) for n in notes])
    for note in notes:
        duration = note.get_duration()
        is_grace = "G" in note.rhythm
        if is_grace:
            fraction = Fraction(0)
        elif duration.fraction.numerator == 0:
            # Whole measure rest
            fraction = max_duration
        else:
            fraction = _sounding_symbol_duration(note)
        groups_by_duration[fraction].append(note)
    return groups_by_duration


def _has_tremolo_mark(symbol: EncodedSymbol) -> bool:
    return "tremolo" in symbol.articulation.split("_")


def _has_double_tremolo(symbol: EncodedSymbol) -> bool:
    return bool(getattr(symbol, "_homr_double_tremolo", False))


def _sounding_symbol_duration(symbol: EncodedSymbol) -> Fraction:
    duration = symbol.get_duration().fraction
    if _has_double_tremolo(symbol) and duration > 0:
        return duration / 2
    return duration


def _mark_double_tremolo_pairs(groups: list[SymbolChord], independent_staves: bool) -> None:
    """Pair double-note tremolos within one printed staff and measure.

    Isolated tremolo marks remain single-note tremolos and keep their full duration.
    A missing stop mark is recovered only when the staff contains exactly two equal
    rhythmic events in that measure, which covers a common OMR omission without
    pairing an arbitrary following note.
    """
    measure_groups: list[SymbolChord] = []

    def mark_measure() -> None:
        by_staff: dict[int, list[list[EncodedSymbol]]] = {1: [], 2: []}
        for group in measure_groups:
            positions = group.into_positions() if independent_staves else [group]
            for position in positions:
                symbols = [
                    symbol
                    for symbol in position.symbols
                    if symbol.rhythm.startswith(("note_", "rest_"))
                ]
                if symbols:
                    staff = get_staff(symbols[0]) if independent_staves else 1
                    by_staff[staff].append(symbols)

        for events in by_staff.values():
            paired: set[int] = set()
            for index in range(len(events) - 1):
                if index in paired or index + 1 in paired:
                    continue
                if any(_has_tremolo_mark(symbol) for symbol in events[index]) and any(
                    _has_tremolo_mark(symbol) for symbol in events[index + 1]
                ):
                    paired.update((index, index + 1))
            if len(events) == 2 and not paired:
                marks = [any(_has_tremolo_mark(symbol) for symbol in event) for event in events]
                left_duration = min(
                    (symbol.get_duration().fraction for symbol in events[0]),
                    default=Fraction(0),
                )
                right_duration = min(
                    (symbol.get_duration().fraction for symbol in events[1]),
                    default=Fraction(0),
                )
                if sum(marks) == 1 and left_duration == right_duration > 0:
                    paired.update((0, 1))
                    marked_index = marks.index(True)
                    missing_index = 1 - marked_index
                    for symbol in events[missing_index]:
                        suffix = "" if symbol.articulation in ("", nonote) else "_"
                        symbol.articulation += suffix + "tremolo"
            for index in paired:
                for symbol in events[index]:
                    symbol._homr_double_tremolo = True  # type: ignore[attr-defined]

    for group in groups:
        if group.is_barline():
            mark_measure()
            measure_groups.clear()
        else:
            measure_groups.append(group)
    if measure_groups:
        mark_measure()


def build_add_time_direction(args: XmlGeneratorArguments) -> mxl.XMLDirection | None:
    if not args.metronome:
        return None
    direction = mxl.XMLDirection()
    direction_type = mxl.XMLDirectionType()
    direction.add_child(direction_type)
    metronome = mxl.XMLMetronome()
    direction_type.add_child(metronome)
    beat_unit = mxl.XMLBeatUnit(value_="quarter")
    metronome.add_child(beat_unit)
    per_minute = mxl.XMLPerMinute(value_=str(args.metronome))
    metronome.add_child(per_minute)
    if args.tempo:
        direction.add_child(mxl.XMLSound(tempo=args.tempo))
    else:
        direction.add_child(mxl.XMLSound(tempo=args.metronome))
    return direction


def find_common_division(durations: list[Fraction]) -> int:
    """
    Find the smallest division (denominator) so that all durations
    can be expressed as integer multiples.
    """

    def lcm(a: int, b: int) -> int:
        return abs(a * b) // math.gcd(a, b)

    denominators = [d.denominator for d in durations if d > 0]
    if not denominators:
        return 1
    common = denominators[0]
    for d in denominators[1:]:
        common = lcm(common, d)
    return common


def _is_uninformative_whole_rest(chord: SymbolChord) -> bool:
    """Whole rests adapt to the bar and must not vote for a 4/4 meter."""
    return len(chord.symbols) == 1 and chord.symbols[0].rhythm == "rest_1"


def _measure_staff_groups(
    voice: list[SymbolChord], independent_staves: bool
) -> list[dict[int, list[SymbolChord]]]:
    measures: list[dict[int, list[SymbolChord]]] = []
    current: dict[int, list[SymbolChord]] = {1: [], 2: []}

    def close_measure() -> None:
        if any(current.values()):
            measures.append({staff: list(groups) for staff, groups in current.items()})
        current[1].clear()
        current[2].clear()

    for chord_group in voice:
        if chord_group.is_barline():
            close_measure()
            continue
        positions = chord_group.into_positions() if independent_staves else [chord_group]
        for position in positions:
            if _is_uninformative_whole_rest(position):
                continue
            duration = position.get_duration()
            if duration <= 0:
                continue
            staff_num = get_staff(position.symbols[0]) if independent_staves else 1
            current[staff_num].append(position)
    close_measure()
    return measures


def _groups_duration(groups: list[SymbolChord]) -> Fraction:
    return sum((group.get_duration() for group in groups), Fraction(0))


def _looks_like_unmarked_triplet_bar(
    groups: list[SymbolChord], long_duration: Fraction, short_duration: Fraction
) -> bool:
    """Recognize a complete equal-note run decoded 3:2 too long.

    This is deliberately stricter than merely preferring the shorter piano staff:
    the long staff must consist of at least two complete groups of three equal,
    ordinary (non-tuplet) notes and must be exactly 3/2 of the other staff. That
    distinguishes a missed triplet ratio from a genuinely sparse 6/4 staff.
    """
    if len(groups) < 6 or len(groups) % 3 != 0 or long_duration != short_duration * Fraction(3, 2):
        return False
    durations = {group.get_duration() for group in groups}
    if len(durations) != 1:
        return False
    rhythmic_symbols = [
        symbol
        for group in groups
        for symbol in group.symbols
        if symbol.rhythm.startswith(("note_", "rest_"))
    ]
    return bool(rhythmic_symbols) and all(
        symbol.get_duration().actual_notes == symbol.get_duration().normal_notes
        and not symbol.rhythm.endswith("m")
        and "G" not in symbol.rhythm
        for symbol in rhythmic_symbols
    )


def _resolved_measure_duration(groups_by_staff: dict[int, list[SymbolChord]]) -> Fraction | None:
    populated = {
        staff: _groups_duration(groups)
        for staff, groups in groups_by_staff.items()
        if groups and _groups_duration(groups) > 0
    }
    if not populated:
        return None
    if len(populated) == 1:
        return next(iter(populated.values()))
    upper = populated.get(1, Fraction(0))
    lower = populated.get(2, Fraction(0))
    if upper == lower:
        return upper
    long_staff = 1 if upper > lower else 2
    long_duration = max(upper, lower)
    short_duration = min(upper, lower)
    if _looks_like_unmarked_triplet_bar(groups_by_staff[long_staff], long_duration, short_duration):
        return short_duration
    return None


def _measure_duration_votes(voice: list[SymbolChord], independent_staves: bool) -> list[Fraction]:
    return [
        duration
        for measure in _measure_staff_groups(voice, independent_staves)
        if (duration := _resolved_measure_duration(measure)) is not None
    ]


def infer_shared_time_signature_nominator(
    voices: list[list[EncodedSymbol]], has_two_staves_by_part: list[bool]
) -> Fraction | None:
    """Infer one page-wide meter without merging instruments or staves.

    Every informative measure on every piano votes for its nearest conventional bar
    duration. Whole-measure rests are excluded because ``rest_1`` adapts to the
    surrounding meter. A winner needs repeated evidence and an absolute majority;
    ambiguous pages fall back to the established per-part inference path.

    The result is a duration, not a forced denominator. An explicit ``/8`` token can
    therefore still distinguish 6/8 from 3/4 when the printed signature is visible.
    """
    if not voices:
        return None
    votes: Counter[Fraction] = Counter()
    conventional = (
        Fraction(3, 8),
        Fraction(1, 2),
        Fraction(3, 4),
        Fraction(1),
        Fraction(5, 4),
        Fraction(3, 2),
        Fraction(7, 4),
        Fraction(2),
    )
    informative_measure_count = 0
    for voice, independent_staves in zip(voices, has_two_staves_by_part, strict=True):
        groups = group_into_chords(voice)
        for duration in _measure_duration_votes(groups, independent_staves):
            nearest = min(conventional, key=lambda candidate: abs(candidate - duration))
            if abs(nearest - duration) <= Fraction(1, 8):
                votes[nearest] += 1
                informative_measure_count += 1
    if not votes:
        return None
    winner, count = votes.most_common(1)[0]
    tied = [duration for duration, votes_count in votes.items() if votes_count == count]
    if len(tied) > 1 or count < 2 or count * 2 <= informative_measure_count:
        return None
    eprint(
        "Global meter consensus across",
        len(voices),
        "piano instruments using measure duration",
        winner,
    )
    return winner


def find_division_and_time_signature_nominator(
    voice: list[SymbolChord], independent_staves: bool = False
) -> tuple[int, Fraction]:
    durations = [Fraction(1, 4)]
    for chord in voice:
        positions = chord.into_positions() if independent_staves else [chord]
        durations.extend(
            position.get_duration()
            for position in positions
            if position.get_duration() > Fraction(0)
        )

    measure_duration = _measure_duration_votes(voice, independent_staves)

    if len(measure_duration) == 0:
        # Conflicting piano staves without a visible signature are safer in the
        # conventional 4/4 fallback than in the old longest-staff inference.
        return find_common_division(durations), Fraction(1)

    nominator: Fraction = np.median(measure_duration)  # type: ignore

    return find_common_division(durations), nominator


def group_into_chords(voice: list[EncodedSymbol]) -> list[SymbolChord]:
    return [SymbolChord(s) for s in sort_token_chords(voice)]


def add_tuplet_start_stop(groups: list[SymbolChord]) -> list[SymbolChord]:
    has_tuplet_mark = False
    for group in groups:
        is_tuplet = False
        for symbol in group.symbols:
            if symbol.rhythm.startswith(("note", "rest")):
                duration = symbol.get_duration()
                is_tuplet = is_tuplet or duration.normal_notes != duration.actual_notes

        if is_tuplet != has_tuplet_mark:
            if has_tuplet_mark:
                group.tuplet_mark = "stop"
            else:
                group.tuplet_mark = "start"
            has_tuplet_mark = is_tuplet

    return groups


def build_divisions(division: int) -> mxl.XMLDivisions:
    # The divisions element indicates how many divisions per quarter(!) note are
    # used to indicate a note's duration
    # https://usermanuals.musicxml.com/MusicXML/Content/EL-MusicXML-divisions.htm
    return mxl.XMLDivisions(value_=division // 4)


if __name__ == "__main__":
    import sys

    from training.transformer.training_vocabulary import read_tokens

    file = "tabi_measure.tokens"
    if len(sys.argv) > 1:
        file = sys.argv[1]
    tokens = read_tokens(file)
    xml = generate_xml(XmlGeneratorArguments(True), [tokens], "")
    xml.write(file.replace(".tokens", ".musicxml"))
