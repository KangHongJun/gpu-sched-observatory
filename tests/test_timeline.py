import pytest

from gso.tui.widgets import render_timeline


def test_render_timeline():
    assert render_timeline({1421: [(0, 5)]}, 0, 10, width=10)[1421] == "=====     "


def test_partial_cell_overlap_marks_the_cell():
    # cells of 2.5 ns over [0, 10): (3, 4) lies inside cell 1 only
    assert render_timeline({1: [(3, 4)]}, 0, 10, width=4)[1] == " =  "


def test_touching_a_cell_boundary_does_not_mark_the_next_cell():
    # [0, 5) ends exactly at the start of cell 5 (half-open)
    assert render_timeline({1: [(0, 5)]}, 0, 10, width=10)[1][5] == " "


def test_intervals_outside_window_are_ignored():
    out = render_timeline({1: [(-20, -1), (10, 30), (12, 13)]}, 0, 10, width=5)
    assert out[1] == "     "


def test_interval_crossing_window_edge_is_clipped():
    assert render_timeline({1: [(-5, 3), (8, 40)]}, 0, 10, width=10)[1] == "===     =="


def test_empty_or_reversed_window_gives_all_spaces():
    assert render_timeline({1: [(0, 5)]}, 5, 5, width=4)[1] == "    "
    assert render_timeline({1: [(0, 5)]}, 10, 0, width=4)[1] == "    "


def test_width_one():
    assert render_timeline({1: [(9, 10)]}, 0, 10, width=1)[1] == "="
    assert render_timeline({1: []}, 0, 10, width=1)[1] == " "


@pytest.mark.parametrize("w", [0, -3])
def test_bad_width_raises(w):
    with pytest.raises(ValueError):
        render_timeline({1: [(0, 1)]}, 0, 10, width=w)


def test_zero_length_interval_marks_nothing():
    assert render_timeline({1: [(4, 4)]}, 0, 10, width=10)[1] == " " * 10


def test_multiple_pids_keep_their_own_rows():
    out = render_timeline({1: [(0, 2)], 2: [(2, 4)]}, 0, 4, width=4)
    assert out == {1: "==  ", 2: "  =="}
