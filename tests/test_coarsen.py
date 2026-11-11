"""Boundary coarsening (`seamsh.geometry.coarsen_boundaries`) on synthetic domains."""

import gmsh
import numpy as np
import pytest
from domains import bounds, rectangle, square_with_island, two_basins, uniform, utm

from seamsh.geometry import Domain, coarsen_boundaries


def distance_to_polyline(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Distance of each point [n, 2] to the closest segment of the polyline [m, 2]."""
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.einsum("nsk,sk->ns", points[:, None, :] - a[None], ab) / np.einsum("sk,sk->s", ab, ab)
    closest = a[None] + np.clip(t, 0, 1)[..., None] * ab[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def test_square_with_island(model):
    """Both boundaries of the square with an island survive a coarsening finer than the island, as
    closed curves with their tags, resampled close to the original curves and at about the size."""
    size = 10e3
    domain = square_with_island(island=20e3)
    coarse = coarsen_boundaries(domain, (10e3, 10e3), utm(), uniform(size))
    assert sorted(c.tag for c in coarse._curves) == ["coast", "island"]
    originals = {c.tag: c.points for c in domain._curves}
    for curve in coarse._curves:
        np.testing.assert_allclose(curve.points[0], curve.points[-1])
        # The boundary is resampled, simplified and moved by size/50 towards the water.
        assert distance_to_polyline(curve.points, originals[curve.tag]).max() < size / 2
        steps = np.linalg.norm(np.diff(curve.points, axis=0), axis=1)
        assert 0.5 * size < np.median(steps) < 1.5 * size
    n_points = {c.tag: len(c.points) for c in coarse._curves}
    assert 30 <= n_points["coast"] <= 45 and 7 <= n_points["island"] <= 11


def test_basin_behind_a_narrow_channel_is_dropped(model):
    """A channel narrower than the coarsening size closes: only the basin holding the seed is kept.
    This is the behaviour that loses e.g. the Black Sea."""
    domain = two_basins(half_width=1e3)
    left = coarsen_boundaries(domain, (20e3, 20e3), utm(), uniform(5e3))
    assert len(left._curves) == 1
    lo, hi = bounds(left._curves[0].points)
    assert lo[0] > -1e3 and hi[0] < 45e3
    right = coarsen_boundaries(two_basins(half_width=1e3), (80e3, 20e3), utm(), uniform(5e3))
    lo, hi = bounds(right._curves[0].points)
    assert lo[0] > 55e3 and hi[0] < 101e3


def test_basin_behind_a_wide_channel_is_kept(model):
    """A channel wider than the coarsening size stays open: one boundary around both basins."""
    coarse = coarsen_boundaries(two_basins(half_width=5e3), (20e3, 20e3), utm(), uniform(3e3))
    assert len(coarse._curves) == 1
    lo, hi = bounds(coarse._curves[0].points)
    assert lo[0] < 1e3 and hi[0] > 99e3


def test_tags_follow_the_boundary(model):
    """A loop made of two curves with different tags is cut again where the tag changes."""
    srs = utm()
    square = rectangle(0, 0, 100e3, 100e3, 5e3)
    domain = Domain(srs)
    domain.add_boundary_curve(square[:41], "south", srs)
    domain.add_boundary_curve(square[40:], "north", srs)
    coarse = coarsen_boundaries(domain, (50e3, 50e3), srs, uniform(10e3))
    assert sorted(c.tag for c in coarse._curves) == ["north", "south"]
    by_tag = {c.tag: c.points for c in coarse._curves}
    # "south" holds the bottom and right sides, "north" the top and left ones.
    assert by_tag["south"][:, 1].min() < 1e3 and by_tag["north"][:, 1].max() > 99e3
    assert by_tag["south"][:, 0].max() > 99e3 and by_tag["north"][:, 0].min() < 1e3


def test_seed_outside_the_points(model):
    """A seed outside the convex hull of the boundary points is in no triangle: an error."""
    with pytest.raises(ValueError, match="outside"):
        coarsen_boundaries(square_with_island(), (-50e3, 50e3), utm(), uniform(10e3))


def test_seed_in_another_projection(model):
    """The seed is given with its own reference system and reprojected to the domain's."""
    from osgeo import osr

    lonlat = osr.SpatialReference()
    lonlat.ImportFromEPSG(4326)
    lonlat.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(utm(), lonlat)
    lon, lat, _ = transform.TransformPoint(20e3, 20e3)
    coarse = coarsen_boundaries(two_basins(half_width=1e3), (lon, lat), lonlat, uniform(5e3))
    assert bounds(coarse._curves[0].points)[1][0] < 45e3


def test_current_gmsh_model_is_kept(model):
    """Curves are sampled in temporary gmsh models; the caller's model is current again afterwards
    (moby-dick creates its model before coarsening)."""
    coarsen_boundaries(square_with_island(), (10e3, 10e3), utm(), uniform(10e3))
    assert gmsh.model.getCurrent() == model


def test_straight_boundaries_finely_sampled(model):
    """Points sampled on straight lines give a triangulation with triangles of almost no area, some
    of them inside out. The coarsening used to loop forever on it; it must give the boundary of
    both basins, joined by the 2 km channel that a 1 km coarsening keeps open."""
    coarse = coarsen_boundaries(two_basins(half_width=1e3), (20e3, 20e3), utm(), uniform(1e3))
    assert len(coarse._curves) == 1
    points = coarse._curves[0].points
    lo, hi = bounds(points)
    assert lo[0] < 1e3 and hi[0] > 99e3
    in_channel = points[(points[:, 0] > 45e3) & (points[:, 0] < 55e3)]
    assert len(in_channel) >= 10
    assert (np.abs(np.abs(in_channel[:, 1] - 20e3) - 1e3) < 100.0).all()


def test_mesh_size_must_be_positive(model):
    """A size of zero (or NaN) would sample the curves forever (or silently too little)."""
    for bad in (0.0, np.nan):
        with pytest.raises(ValueError, match="positive"):
            coarsen_boundaries(square_with_island(), (10e3, 10e3), utm(), uniform(bad))
    assert gmsh.model.getCurrent() == model


def test_points_shared_by_two_curves_have_both_tags():
    """Where two curves meet, the point carries both tags, so that the coarsened boundary is cut
    exactly there. (The second tag used to be attached to an unrelated entry.)"""
    from seamsh.geometry import _point_tag_lists

    # Four samples: curve 1 on points 0 and 1, curve 2 on points 1 and 2.
    lists = _point_tag_lists([0, 1, 1, 2], [1, 1, 2, 2], 3).reshape(-1, 2)
    assert lists.tolist() == [[1, -1], [1, 3], [2, -1], [2, -1]]
    # A tag seen twice on a point is stored once.
    lists = _point_tag_lists([0, 0, 0], [5, 5, 7], 1).reshape(-1, 2)
    assert lists.tolist() == [[5, 1], [7, -1]]


def test_global_random_state_is_untouched(model):
    """The jitter that avoids cocircular points must not reset numpy's global random generator."""
    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)
    coarsen_boundaries(square_with_island(), (10e3, 10e3), utm(), uniform(10e3))
    np.testing.assert_array_equal(np.random.random(3), expected)


def test_coarsening_is_deterministic(model):
    """Two runs give the same points (the jitter is seeded)."""
    runs = [
        coarsen_boundaries(square_with_island(), (10e3, 10e3), utm(), uniform(10e3))
        for _ in range(2)
    ]
    for a, b in zip(runs[0]._curves, runs[1]._curves):
        np.testing.assert_array_equal(a.points, b.points)
