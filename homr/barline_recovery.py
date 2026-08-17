from __future__ import annotations

from statistics import median

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol


def _coordinate_x(symbol: EncodedSymbol) -> float | None:
    coordinates = symbol.coordinates
    if coordinates is None:
        return None
    try:
        return float(coordinates[0])
    except (IndexError, TypeError, ValueError):
        return None


def _cluster_positions(positions: list[float], tolerance: float) -> list[list[float]]:
    clusters: list[list[float]] = []
    for position in sorted(positions):
        if not clusters or position - clusters[-1][-1] > tolerance:
            clusters.append([position])
        else:
            clusters[-1].append(position)
    return clusters


def _rhythmic_x(symbol: EncodedSymbol) -> float | None:
    if not symbol.rhythm.startswith(("note_", "rest_")):
        return None
    return _coordinate_x(symbol)


def _candidate_crosses_chord(symbols: list[EncodedSymbol], candidate_x: float) -> bool:
    chord_positions: list[float] = []
    join_next = False

    def crosses() -> bool:
        return (
            len(chord_positions) > 1
            and min(chord_positions) < candidate_x < max(chord_positions)
        )

    for symbol in symbols:
        if symbol.rhythm.startswith(("note_", "rest_")):
            value = _coordinate_x(symbol)
            if value is None:
                if crosses():
                    return True
                chord_positions = []
                join_next = False
                continue
            if join_next and chord_positions:
                chord_positions.append(value)
            else:
                if crosses():
                    return True
                chord_positions = [value]
            join_next = False
        elif symbol.rhythm == "chord":
            join_next = True
        else:
            if crosses():
                return True
            chord_positions = []
            join_next = False
    return crosses()


def _find_insertion_gap(
    symbols: list[EncodedSymbol], candidate_x: float, tolerance: float
) -> int | None:
    rhythmic_count = sum(_rhythmic_x(symbol) is not None for symbol in symbols)
    if rhythmic_count < 4:
        return None

    best: tuple[int, float, int] | None = None
    for gap in range(1, len(symbols)):
        # Never split the note/chord/note representation of one vertical chord.
        if symbols[gap - 1].rhythm == "chord" or symbols[gap].rhythm == "chord":
            continue
        left = [
            value
            for symbol in symbols[:gap]
            if (value := _rhythmic_x(symbol)) is not None
        ]
        right = [
            value
            for symbol in symbols[gap:]
            if (value := _rhythmic_x(symbol)) is not None
        ]
        if len(left) < 2 or len(right) < 2:
            continue

        violations = sum(value > candidate_x + tolerance for value in left) + sum(
            value < candidate_x - tolerance for value in right
        )
        if violations > 0:
            continue
        distance = abs(max(left) - candidate_x) + abs(min(right) - candidate_x)
        proposal = (violations, distance, gap)
        if best is None or proposal < best:
            best = proposal
    return None if best is None else best[2]


def recover_missing_barlines(
    symbols: list[EncodedSymbol],
    geometric_barline_x: list[float],
    unit_size: float,
    *,
    require_staff_pair: bool,
) -> list[EncodedSymbol]:
    """Insert only strongly corroborated geometric barlines into decoder output.

    The segmentation pipeline already detects tall vertical staff anchors, while
    the transformer independently emits semantic barline tokens.  For a grand
    staff, a candidate must occur on both staves; it must also divide at least two
    rhythmic events on each side without contradicting the transformer's monotonic
    attention order.  These constraints deliberately reject system edges, stems,
    and uncertain single detections.
    """
    if not geometric_barline_x:
        return symbols

    cluster_tolerance = max(float(unit_size), 6.0)
    attention_tolerance = max(float(unit_size) * 2.0, 12.0)
    clusters = _cluster_positions(geometric_barline_x, cluster_tolerance)
    candidates = [
        float(median(cluster))
        for cluster in clusters
        if not require_staff_pair or len(cluster) >= 2
    ]
    semantic_indices = [
        index
        for index, symbol in enumerate(symbols)
        if "barline" in symbol.rhythm or symbol.rhythm.startswith("repeat")
    ]

    # Attention coordinates are useful for ordering notes but are not accurate
    # enough to spatially pair barlines.  Compare the two independent detectors by
    # count and only repair a single omission.  This prevents an uncertain page
    # from manufacturing several measure boundaries at once.
    if len(candidates) != len(semantic_indices) + 1:
        return symbols

    rhythmic_indices = [
        index for index, symbol in enumerate(symbols) if _rhythmic_x(symbol) is not None
    ]
    if not rhythmic_indices:
        return symbols
    first_rhythmic = rhythmic_indices[0]
    last_rhythmic = rhythmic_indices[-1]
    unmatched = list(candidates)
    remaining_semantic = list(semantic_indices)
    if remaining_semantic and remaining_semantic[-1] > last_rhythmic:
        unmatched.pop()
        remaining_semantic.pop()
    if remaining_semantic and remaining_semantic[0] < first_rhythmic:
        unmatched.pop(0)
        remaining_semantic.pop(0)
    if len(unmatched) != 1 or remaining_semantic:
        return symbols

    result = list(symbols)
    restored = 0
    for candidate_x in unmatched:
        if _candidate_crosses_chord(result, candidate_x):
            continue
        gap = _find_insertion_gap(result, candidate_x, attention_tolerance)
        if gap is None:
            continue
        result.insert(gap, EncodedSymbol("barline", coordinates=(candidate_x, 0.0)))
        restored += 1

    if restored:
        eprint("Recovered", restored, "missing barlines from staff geometry")
    return result
