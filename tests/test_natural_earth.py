# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: GPL-3.0-or-later

"""The whole chain on a real coastline: Natural Earth 1:110m (public domain, tests/data), in a polar
stereographic plane centred on the North Pole, as moby-dick uses seamsh.

Reference numbers, measured with gmsh 4.12.1: coarsening at 100 km from a seed at the North Pole,
then a 300 km mesh, gives 5,532 nodes in 10,323 triangles and 783 boundary edges (5,512, 10,284 and
782 before version 0.5.0+fork, which moves the coarsened boundary points along other normals).
"""

import gmsh
import numpy as np
import pytest
from domains import NE_110M, current_mesh, inside_curves, n_components
from osgeo import osr

import seamsh.gmsh
from seamsh.geometry import CurveType, Domain, coarsen_boundaries

STEREOGRAPHIC = osr.SpatialReference()
STEREOGRAPHIC.ImportFromProj4(
    "+proj=stere +lat_0=90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
)
LONLAT = osr.SpatialReference()
LONLAT.ImportFromEPSG(4326)
LONLAT.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

# (name, latitude, longitude, in the coarsened ocean reached from the North Pole at 100 km).
PROBES = [
    ("Arctic Ocean", 89.0, 0.0, True),
    ("North Atlantic", 40.0, -40.0, True),
    ("South Pacific", -40.0, -140.0, True),
    ("Indian Ocean", -20.0, 80.0, True),
    ("Southern Ocean", -60.0, 0.0, True),
    ("Hudson Bay", 60.0, -85.0, True),
    ("Sea of Japan", 40.0, 135.0, True),
    # Behind straits that a 100 km coastline closes.
    ("Mediterranean Sea", 35.0, 18.0, False),
    ("Black Sea", 43.5, 34.0, False),
    ("Baltic Sea", 58.0, 20.0, False),
    ("Red Sea", 20.0, 38.5, False),
    ("Persian Gulf", 27.0, 51.0, False),
    # Land.
    ("Sahara", 23.0, 13.0, False),
    ("Siberia", 62.0, 100.0, False),
    ("Antarctic interior", -80.0, 0.0, False),
]


class UniformOnEarth:
    """A uniform size on the Earth, as a size in the stereographic plane: divided by the scale factor
    4R^2 / (4R^2 + x^2 + y^2) of the projection (Lambrechts et al. 2008, section 3.1)."""

    def __init__(self, size: float):
        self.size = size
        self.r2 = STEREOGRAPHIC.GetSemiMajor() * STEREOGRAPHIC.GetSemiMinor()

    def __call__(self, x, projection):
        assert projection.IsSame(STEREOGRAPHIC), "sizes are asked in the domain's projection"
        return self.size * (4 * self.r2 + x[:, 0] ** 2 + x[:, 1] ** 2) / (4 * self.r2)


def to_stereographic(lat, lon):
    transform = osr.CoordinateTransformation(LONLAT, STEREOGRAPHIC)
    return np.array(transform.TransformPoint(lon, lat)[:2])


@pytest.fixture(scope="module")
def coarse():
    domain = Domain(STEREOGRAPHIC)
    domain.add_boundary_curves_shp(str(NE_110M), "featurecla", CurveType.BSPLINE)
    return coarsen_boundaries(domain, (0.0, 0.0), STEREOGRAPHIC, UniformOnEarth(100e3))


def test_coarsened_ocean(coarse):
    """The coarsened domain is the ocean connected to the North Pole: closed curves tagged from the
    shapefile, with the open oceans inside, and land and the basins behind narrow straits outside."""
    curves = [c.points for c in coarse._curves]
    assert {c.tag for c in coarse._curves} == {"Coastline"}
    for points in curves:
        np.testing.assert_allclose(points[0], points[-1])
    assert 35 <= len(curves) <= 60
    points = np.array([to_stereographic(lat, lon) for _, lat, lon, _ in PROBES])
    inside = inside_curves(points, curves)
    assert {p[0]: bool(i) for p, i in zip(PROBES, inside)} == {p[0]: p[3] for p in PROBES}


def test_ocean_mesh(coarse, model):
    """The 300 km mesh of the coarsened ocean, written in Earth-centred coordinates: one component of
    well-oriented triangles on the ellipsoid, of the expected number and size."""
    cartesian = osr.SpatialReference()
    cartesian.ImportFromProj4("+proj=cart +ellps=WGS84 +units=m")
    seamsh.gmsh.mesh(coarse, UniformOnEarth(300e3), output_srs=cartesian, smoothness=0.3)
    mesh = current_mesh()
    triangles = mesh["triangles"]
    assert len(triangles) == pytest.approx(10323, rel=0.05)
    assert len(np.unique(triangles)) == pytest.approx(5532, rel=0.05)
    assert n_components(mesh) == 1
    assert mesh["groups"] == ["Coastline", "domain"]
    # Boundary edges: those of one triangle only.
    edges = np.sort(np.r_[triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert counts.max() == 2
    assert (counts == 1).sum() == pytest.approx(783, rel=0.05)
    # On the ellipsoid, with outward normals.
    a, b, c = (mesh["x"][triangles[:, i]] for i in range(3))
    radius = np.linalg.norm(mesh["x"][np.unique(triangles)], axis=1)
    assert radius.min() > 6.35e6 and radius.max() < 6.38e6
    assert (np.einsum("ij,ij->i", np.cross(b - a, c - a), a + b + c) > 0).all()
    lengths = np.concatenate([np.linalg.norm(p - q, axis=1) for p, q in ((a, b), (b, c), (c, a))])
    assert np.median(lengths) == pytest.approx(300e3, rel=0.15)
    assert osr.SpatialReference(gmsh.model.getAttribute("Projection")[1]).IsSame(cartesian)


def test_seeded_basins(model):
    """With points in the Mediterranean, the Black Sea and the Red Sea, these basins, cut off from
    the ocean by the 100 km coastline, are kept and meshed as separate pieces."""
    basins = {"Mediterranean Sea": (35.0, 18.0), "Black Sea": (43.5, 34.0), "Red Sea": (20.0, 38.5)}
    seeds = [(0.0, 0.0)] + [tuple(to_stereographic(lat, lon)) for lat, lon in basins.values()]
    domain = Domain(STEREOGRAPHIC)
    domain.add_boundary_curves_shp(str(NE_110M), "featurecla", CurveType.BSPLINE)
    coarse = coarsen_boundaries(domain, seeds, STEREOGRAPHIC, UniformOnEarth(100e3))
    curves = [c.points for c in coarse._curves]
    points = np.array([to_stereographic(lat, lon) for _, lat, lon, _ in PROBES])
    inside = {p[0]: bool(i) for p, i in zip(PROBES, inside_curves(points, curves))}
    expected = {p[0]: p[3] or p[0] in basins for p in PROBES}
    assert inside == expected
    seamsh.gmsh.mesh(coarse, UniformOnEarth(300e3))
    mesh = current_mesh()
    assert n_components(mesh) == 4
    assert len(gmsh.model.getEntities(2)) == 4
