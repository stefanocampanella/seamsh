"""Domain topology: merging points, chaining boundary curves into closed loops, splitting curves."""

import numpy as np
import pytest
from domains import polygon, rectangle, square_with_island, utm

from seamsh.geometry import Domain, _generate_unique_points


def test_unique_points_merges_coincident_points():
    """Coincident points get one index, and the merged ones are flagged as break points: this is how
    curves that share a point are recognised as connected."""
    x = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 0.0], [3.0, 1.0]])
    unique, index, is_break = _generate_unique_points(x)
    assert unique.shape == (4, 2)
    np.testing.assert_array_equal(unique[index], x)
    assert index[1] == index[2] and index[0] == index[4]
    # The merged points come first and are the only break points.
    assert is_break.tolist() == [True, True, False, False]
    assert set(index[[0, 1]].tolist()) == {0, 1}


def test_unique_points_without_duplicates():
    """With no coincident points nothing is merged (the empty list of pairs is a special case)."""
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 5.0]])
    unique, index, is_break = _generate_unique_points(x)
    np.testing.assert_array_equal(unique[index], x)
    assert not is_break.any()


def test_loops_are_closed_and_the_largest_comes_first():
    """Each boundary curve of the square with an island is one closed loop; the loop with the largest
    bounding box is first whatever the order of insertion, because gmsh takes the first loop of a
    surface as its outer boundary."""
    srs = utm()
    domain = Domain(srs)
    domain.add_boundary_curve(rectangle(40e3, 40e3, 60e3, 60e3, 5e3), "island", srs)
    domain.add_boundary_curve(rectangle(0, 0, 100e3, 100e3, 5e3), "coast", srs)
    domain._build_topology()
    assert len(domain._curveloops) == 2
    tags = [[domain._curves[i].tag for i, _ in loop] for loop in domain._curveloops]
    assert tags == [["coast"], ["island"]]
    for loop in domain._curveloops:
        for i, _ in loop:
            ids = domain._curves[i].pointsid
            assert ids[0] == ids[-1]


def test_curves_sharing_end_points_form_one_loop():
    """A boundary given as two open curves with different tags is chained into one loop, with the
    second curve reversed if needed."""
    srs = utm()
    square = rectangle(0, 0, 100e3, 100e3, 10e3)
    domain = Domain(srs)
    domain.add_boundary_curve(square[:21], "south", srs)
    # Given in the opposite direction: from the first corner back to the middle of the loop.
    domain.add_boundary_curve(square[20:][::-1], "north", srs)
    domain._build_topology()
    assert len(domain._curveloops) == 1
    loop = domain._curveloops[0]
    assert [domain._curves[i].tag for i, _ in loop] == ["south", "north"]
    assert [forward for _, forward in loop] == [True, False]


def test_boundary_is_split_where_an_interior_curve_touches_it():
    """An interior curve starting on a control point of the boundary splits the boundary curve there,
    so that gmsh gets curves that meet only at their end points."""
    srs = utm()
    domain = square_with_island()
    n_before = len(domain._curves)
    domain.add_interior_curve([[50e3, 0.0], [50e3, 20e3]], "dyke", srs)
    domain._build_topology()
    assert len(domain._curves) == n_before + 1
    assert len(domain._interior_curves) == 1
    shared = domain._interior_curves[0].pointsid[0]
    assert sum(shared in (c.pointsid[0], c.pointsid[-1]) for c in domain._curves) == 2


def test_interior_points_get_a_point_id():
    """Forced mesh points are numbered with the curve points."""
    srs = utm()
    domain = square_with_island()
    domain.add_interior_points([[10e3, 10e3], [20e3, 10e3]], "station", srs)
    domain._build_topology()
    ids = [p.pointid for p in domain._interior_points]
    np.testing.assert_allclose(domain._points[ids], [[10e3, 10e3], [20e3, 10e3]])


def test_open_boundary_is_an_invalid_topology():
    """Boundary curves that do not close cannot bound a surface."""
    srs = utm()
    domain = Domain(srs)
    domain.add_boundary_curve(polygon([(0, 0), (10e3, 0), (10e3, 10e3)], 5e3)[:-1], "coast", srs)
    with pytest.raises(ValueError, match="Invalid topology"):
        domain._build_topology()
