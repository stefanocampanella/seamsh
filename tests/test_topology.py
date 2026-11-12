# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: GPL-3.0-or-later

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


def test_missing_shapefile_is_reported(tmp_path):
    """A file that OGR cannot open is named in the error (it used to be an AttributeError on None,
    as in example 5, whose data archive lacks one file)."""
    domain = Domain(utm())
    for name in ("missing.shp", "missing.gpkg"):
        with pytest.raises(FileNotFoundError, match=name):
            domain.add_boundary_curves_shp(str(tmp_path / name), None)


def test_open_boundary_is_an_invalid_topology():
    """Boundary curves that do not close cannot bound a surface."""
    srs = utm()
    domain = Domain(srs)
    domain.add_boundary_curve(polygon([(0, 0), (10e3, 0), (10e3, 10e3)], 5e3)[:-1], "coast", srs)
    with pytest.raises(ValueError, match="Invalid topology"):
        domain._build_topology()


def test_loops_are_sorted_into_regions_by_nesting():
    """Loops inside an even number of loops bound regions, the others are holes of the smallest
    loop around them: here a sea (0) with an island (1) that has a lake (2) with an islet (3), and
    a separate pond (4)."""
    from seamsh.geometry import _nest_loops

    loops = [
        rectangle(0, 0, 100, 100, 10),
        rectangle(20, 20, 80, 80, 10),
        rectangle(30, 30, 70, 70, 10),
        rectangle(45, 45, 55, 55, 5),
        rectangle(200, 0, 250, 50, 10),
    ]
    regions, containers = _nest_loops(loops)
    assert regions == [[0, 1], [2, 3], [4]]
    assert [sorted(c) for c in containers] == [[], [0], [0, 1], [0, 1, 2], []]
    # The order of the loops does not matter.
    regions, _ = _nest_loops(loops[::-1])
    assert regions == [[0], [2, 1], [4, 3]]


def test_region_of_a_point():
    """Points are located in the region that holds them; land and the outside are in none."""
    srs = utm()
    domain = Domain(srs)
    domain.add_boundary_curve(rectangle(0, 0, 100e3, 100e3, 10e3), "sea", srs)
    domain.add_boundary_curve(rectangle(20e3, 20e3, 80e3, 80e3, 10e3), "island", srs)
    domain.add_boundary_curve(rectangle(30e3, 30e3, 70e3, 70e3, 10e3), "lake", srs)
    domain._build_topology()
    assert domain._regions == [[0, 1], [2]]
    assert domain._region_of((10e3, 10e3)) == 0
    assert domain._region_of((50e3, 50e3)) == 1
    assert domain._region_of((25e3, 50e3)) is None
    assert domain._region_of((150e3, 50e3)) is None


def test_three_boundary_curves_at_a_point_are_refused():
    """Where more than two boundary curves meet, nothing tells which one follows which: an error
    that names the point. (The third curve used to replace the second without notice.)"""
    srs = utm()
    domain = Domain(srs)
    domain.add_boundary_curve([[0, 0], [10e3, 0], [10e3, 10e3]], "a", srs)
    domain.add_boundary_curve([[10e3, 10e3], [0, 10e3], [0, 0]], "b", srs)
    domain.add_boundary_curve([[0, 0], [-10e3, -10e3]], "c", srs)
    with pytest.raises(
        ValueError, match=r"more than two boundary curves meet at \(.*0\.0.*0\.0.*\)"
    ):
        domain._build_topology()


def test_transformations_are_cached_by_definition():
    """Two projection objects with the same definition use the same cached transformation, and
    the cache does not grow without bound."""
    from osgeo import osr

    from seamsh import _tools

    _tools._transform_cache.clear()
    lonlat = [osr.SpatialReference() for _ in range(2)]
    for projection in lonlat:
        projection.ImportFromEPSG(4326)
        projection.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    x = np.array([[3.0, 45.0]])
    a = _tools.project_points(x, lonlat[0], utm())
    b = _tools.project_points(x, lonlat[1], utm())
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, [[500000.0, 4982950.4]], atol=0.1)
    assert len(_tools._transform_cache) == 1
    # The same projection with the other axis order is another transformation.
    lonlat[1].SetAxisMappingStrategy(osr.OAMS_AUTHORITY_COMPLIANT)
    np.testing.assert_allclose(_tools.project_points(x[:, ::-1], lonlat[1], utm()), a)
    assert len(_tools._transform_cache) == 2
    for zone in range(1, 61):
        target = osr.SpatialReference()
        target.ImportFromEPSG(32600 + zone)
        _tools.project_points(x, lonlat[0], target)
    assert len(_tools._transform_cache) <= _tools._TRANSFORM_CACHE_SIZE


def test_transformations_of_the_same_objects_skip_the_definitions(monkeypatch):
    """Exporting the two definitions (WKT) took about half of each call, and a quarter of a
    coarsening: the transformation of a pair of objects is found by their identity, and only their
    axis order is checked."""
    from osgeo import osr

    from seamsh import _tools

    lonlat = osr.SpatialReference()
    lonlat.ImportFromEPSG(4326)
    lonlat.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target = utm()
    x = np.array([[3.0, 45.0]])
    first = _tools.project_points(x, lonlat, target)
    calls = []
    real = _tools._projection_key
    monkeypatch.setattr(_tools, "_projection_key", lambda p: calls.append(p) or real(p))
    for _ in range(10):
        np.testing.assert_array_equal(_tools.project_points(x, lonlat, target), first)
    assert calls == []
    lonlat.SetAxisMappingStrategy(osr.OAMS_AUTHORITY_COMPLIANT)
    np.testing.assert_allclose(_tools.project_points(x[:, ::-1], lonlat, target), first)
    assert len(calls) == 2


def test_points_in_polygon_by_blocks(monkeypatch):
    """The test is done by blocks of points x segments, so that its memory does not grow with the
    size of the polygon (300 points against a loop of a million vertices took 4 GB): the result
    does not depend on the blocks."""
    from seamsh import geometry

    t = np.linspace(0, 2 * np.pi, 2001)
    star = np.c_[np.cos(t), np.sin(t)] * (1 + 0.3 * np.sin(7 * t))[:, None]
    points = np.random.default_rng(0).random((500, 2)) * 3 - 1.5
    whole = geometry._points_in_polygon(points, star)
    for block in (1, 7, 1000):
        monkeypatch.setattr(geometry, "_BLOCK", block)
        np.testing.assert_array_equal(geometry._points_in_polygon(points, star), whole)
    r = np.hypot(*points.T)
    assert whole[r < 0.69].all() and not whole[r > 1.31].any()


def test_regions_of_many_points():
    """The regions of many points are found at once, as for one point at a time."""
    srs = utm()
    domain = Domain(srs)
    domain.add_boundary_curve(rectangle(0, 0, 100e3, 100e3, 5e3), "sea", srs)
    domain.add_boundary_curve(rectangle(20e3, 20e3, 80e3, 80e3, 5e3), "island", srs)
    domain.add_boundary_curve(rectangle(35e3, 35e3, 65e3, 65e3, 5e3), "lake", srs)
    domain.add_boundary_curve(rectangle(120e3, 0, 160e3, 40e3, 5e3), "pond", srs)
    domain._build_topology()
    points = np.random.default_rng(1).random((400, 2)) * [170e3, 110e3] - 5e3
    regions = domain._regions_of(points)
    one_by_one = [domain._region_of(p) for p in points]
    assert [None if r < 0 else r for r in regions] == one_by_one
    assert set(regions) == {-1, 0, 1, 2}
