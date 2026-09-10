# ruff: noqa: S101

import numpy as np

from homr.model import MultiStaff, Staff, StaffPoint
from homr.staff_parsing import (
    _ensure_same_number_of_staffs,
    _get_number_of_voices,
    _is_close_to_image_top_or_bottom,
)


def _staff(y: float, x_start: float = 0, x_end: float = 900) -> Staff:
    lines = [y + offset for offset in range(0, 50, 10)]
    return Staff(
        [
            StaffPoint(x_start, lines, 0),
            StaffPoint(x_end, lines, 0),
        ]
    )


def test_page_edge_check_uses_vertical_not_horizontal_coordinates() -> None:
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    horizontally_edge_aligned = MultiStaff([_staff(400, x_start=0)], [])
    near_bottom = MultiStaff([_staff(940, x_start=100)], [])

    assert not _is_close_to_image_top_or_bottom(horizontally_edge_aligned, image)
    assert _is_close_to_image_top_or_bottom(near_bottom, image)


def test_structurally_different_last_system_is_preserved() -> None:
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    first = MultiStaff([_staff(100)], [])
    second = MultiStaff([_staff(400)], [])
    last_upper = _staff(880)
    last_lower = _staff(940)
    last = MultiStaff([last_upper, last_lower], [])

    result = _ensure_same_number_of_staffs([first, second, last], image)

    assert len(result) == 4
    assert [system.staffs[0] for system in result] == [
        first.staffs[0],
        second.staffs[0],
        last_upper,
        last_lower,
    ]


def test_uniform_system_structure_is_unchanged() -> None:
    image = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    systems = [MultiStaff([_staff(100)], []), MultiStaff([_staff(400)], [])]

    result = _ensure_same_number_of_staffs(systems, image)

    assert result is systems


def test_automatically_counts_one_piano_without_changing_systems() -> None:
    systems = [
        MultiStaff([_staff(100).merge(_staff(220))], []),
        MultiStaff([_staff(500).merge(_staff(620))], []),
    ]

    assert _get_number_of_voices(systems) == 1


def test_automatically_counts_two_pianos_as_four_staves() -> None:
    systems = [
        MultiStaff(
            [
                _staff(100).merge(_staff(220)),
                _staff(340).merge(_staff(460)),
            ],
            [],
        ),
        MultiStaff(
            [
                _staff(700).merge(_staff(820)),
                _staff(940).merge(_staff(1060)),
            ],
            [],
        ),
    ]

    assert _get_number_of_voices(systems) == 2
