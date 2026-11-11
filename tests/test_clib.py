"""The C library (`seamshlib/`), called through ctypes as `seamsh.geometry` and `seamsh.gmsh` do."""

import numpy as np
import pytest

from seamsh import _tools


def gen_boundaries(x, triangles, seeds, sizes, tags=None):
    """Calls the coarsening and returns the output points [n, 2] and the boundary curves as a list
    of (tag, point indices)."""
    if tags is None:
        tags = np.ones(len(x), int)
    # One linked list of tags per point: (tag, index of the next entry or -1).
    tag_lists = np.c_[tags, np.full(len(x), -1)].astype(np.int32).reshape(-1)
    points, lines = _tools.gen_boundaries(x, tag_lists, triangles, seeds, sizes)
    curves, start = [], 0
    for end in np.where(lines == -1)[0]:
        curves.append((int(lines[start]), lines[start + 1 : end].tolist()))
        start = end + 1
    return points, curves


# A 10 x 10 square with a 2 x 2 square hole, as a ring of triangles one triangle wide (0-7), and
# the two triangles of the hole (8, 9).
RING_POINTS = [[0, 0], [10, 0], [10, 10], [0, 10], [4, 4], [6, 4], [6, 6], [4, 6]]
RING_TRIANGLES = [
    [0, 1, 4],
    [1, 5, 4],
    [1, 2, 5],
    [2, 6, 5],
    [2, 3, 6],
    [3, 7, 6],
    [3, 0, 7],
    [0, 4, 7],
    [4, 5, 6],
    [4, 6, 7],
]


def test_flood_fill_stops_at_short_edges():
    """From a triangle of the ring, the fill crosses the long edges of the ring but not the edges of
    the hole, shorter than the size: the result has the outer and the inner boundary, each a closed
    loop of four points, moved by at most size/50 in each direction."""
    size = 2.5
    points, curves = gen_boundaries(RING_POINTS, RING_TRIANGLES, 0, np.full(8, size))
    assert len(curves) == 2
    loops = [points[ids] for _, ids in curves]
    for tag, ids in curves:
        assert tag == 1
        assert len(ids) == 5 and ids[0] == ids[-1]
    outer, inner = loops
    for loop, corners in ((outer, RING_POINTS[:4]), (inner, RING_POINTS[4:])):
        for corner in corners:
            assert np.abs(loop - corner).max(axis=1).min() <= size / 50 + 1e-12


def test_flood_fill_crosses_long_edges():
    """With a size smaller than the edges of the hole the fill takes every triangle: only the outer
    boundary is left."""
    points, curves = gen_boundaries(RING_POINTS, RING_TRIANGLES, 0, np.full(8, 1.0))
    assert len(curves) == 1
    assert len(curves[0][1]) == 5
    assert np.abs(points).max() < 10.1 and points.max() > 9.9


def test_tags_split_the_boundary():
    """Points carry the tags of their curves; a loop is cut where the tag of its segments changes."""
    tags = [1, 1, 2, 2, 1, 1, 1, 1]
    _, curves = gen_boundaries(RING_POINTS, RING_TRIANGLES, 0, np.full(8, 1.0), tags)
    assert sorted(tag for tag, _ in curves) == [1, 2]
    assert sum(len(ids) - 1 for _, ids in curves) == 4


def test_polymesh_triangulates_points():
    """`PolyMesh` (gmsh's Delaunay kernel, used for the background size mesh) triangulates the
    points added to it: a square and its centre give four triangles."""
    mesh = _tools.PolyMesh(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    mesh.add_points(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]))
    triangles = mesh.get_triangles()
    assert triangles.shape == (4, 3)
    assert (triangles == 4).any(axis=1).all()
    assert set(triangles.reshape(-1).tolist()) == {0, 1, 2, 3, 4}


def test_errors_are_returned_not_fatal():
    """An inconsistent input must raise an exception. It used to end the Python process
    (`exit(1)`) or to loop forever."""
    # A seed that is not a triangle.
    with pytest.raises(RuntimeError, match="inconsistent triangulation"):
        gen_boundaries(RING_POINTS, RING_TRIANGLES, 99, np.full(8, 1.0))
    # Three triangles on one edge (0, 1): not a triangulation of the plane. Whatever comes out, it
    # must come back.
    points = [[0, 0], [10, 0], [5, 8], [5, -8], [5, 3]]
    triangles = [[0, 1, 2], [1, 0, 3], [0, 1, 4]]
    try:
        gen_boundaries(points, triangles, 0, np.full(5, 1.0))
    except RuntimeError as error:
        assert "inconsistent triangulation" in str(error)


def test_several_seeds():
    """Each seed fills its own region. Seeds in the ring and in the hole of the ring take the whole
    square (one boundary); two seeds in the ring are the same as one."""
    size = np.full(8, 2.5)
    _, both = gen_boundaries(RING_POINTS, RING_TRIANGLES, [0, 8], size)
    assert len(both) == 1 and len(both[0][1]) == 5
    points_one, one = gen_boundaries(RING_POINTS, RING_TRIANGLES, 0, size)
    points_two, two = gen_boundaries(RING_POINTS, RING_TRIANGLES, [0, 4], size)
    assert one == two and len(one) == 2
    np.testing.assert_array_equal(points_one, points_two)
    # The hole alone: its own boundary only.
    _, hole = gen_boundaries(RING_POINTS, RING_TRIANGLES, [8], size)
    assert len(hole) == 1 and len(hole[0][1]) == 5


def test_inside_out_slivers_do_not_break_the_walks():
    """Three points almost on a line give a triangle of almost no area, which may be listed inside
    out relative to its neighbours. Triangles are oriented from their neighbours, not one by one, so
    that walking around a point still visits each triangle once."""
    points = [[0, 0], [5, 1e-9], [10, 0], [10, 10], [0, 10]]
    # (0, 1, 2) is the sliver, here with the same orientation as its neighbours...
    triangles = [[0, 2, 1], [0, 1, 4], [1, 2, 3], [1, 3, 4]]
    reference, curves = gen_boundaries(points, triangles, 1, np.full(5, 1.0))
    assert len(curves) == 1
    # ... and here with every triangle listed in the other direction, or in mixed directions.
    for flipped in ([0, 1, 2, 3], [1, 3], [0]):
        mixed = [t[::-1] if i in flipped else t for i, t in enumerate(triangles)]
        result, curves = gen_boundaries(points, mixed, 1, np.full(5, 1.0))
        assert len(curves) == 1
        assert sorted(map(tuple, np.round(result, 6))) == sorted(map(tuple, np.round(reference, 6)))
