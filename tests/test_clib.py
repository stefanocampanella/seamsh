"""The C library (`seamshlib/`), called through ctypes as `seamsh.geometry` and `seamsh.gmsh` do."""

import ctypes as c

import numpy as np

from seamsh import _tools


def gen_boundaries(x, triangles, first, sizes, tags=None):
    """Calls `gen_boundaries_from_points` and returns the output points [n, 2] and the boundary
    curves as a list of (tag, point indices)."""
    x = np.array(x, float)
    triangles = np.array(triangles, np.int32)
    if tags is None:
        tags = np.ones(len(x), int)
    # One linked list of tags per point: (tag, index of the next entry or -1).
    tag_lists = np.c_[tags, np.full(len(x), -1)].astype(np.int32).reshape(-1)
    n_lines, n_points = c.c_int(), c.c_int()
    p_points, p_lines = c.POINTER(c.c_double)(), c.POINTER(c.c_int)()
    _tools.lib.gen_boundaries_from_points(
        c.c_int(len(x)),
        _tools.np2c(x),
        _tools.np2c(tag_lists, np.int32),
        c.c_int(len(triangles)),
        _tools.np2c(triangles, np.int32),
        c.c_int(first),
        _tools.np2c(np.array(sizes, float)),
        c.byref(p_points),
        c.byref(n_points),
        c.byref(p_lines),
        c.byref(n_lines),
    )
    points = np.ctypeslib.as_array(p_points, (n_points.value * 2,)).reshape(-1, 2).copy()
    lines = np.ctypeslib.as_array(p_lines, (n_lines.value,)).copy()
    _tools.lib.libcfree(p_points)
    _tools.lib.libcfree(p_lines)
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
