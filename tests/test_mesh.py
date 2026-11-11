"""Meshing (`seamsh.gmsh.mesh`) on synthetic domains. Counts are checked within ranges: the exact
numbers depend on the gmsh version."""

import gmsh
import numpy as np
import pytest
from domains import (
    current_mesh,
    n_components,
    signed_areas,
    square_with_island,
    stray_line_nodes,
    two_squares,
    uniform,
    utm,
)
from osgeo import osr

import seamsh.gmsh


def check_square_with_island(mesh, size):
    """Triangles of about `size` fill the square but not the island."""
    assert mesh["groups"] == ["coast", "domain", "island"]
    areas = signed_areas(mesh)
    assert (areas > 0).all()
    np.testing.assert_allclose(areas.sum(), 100e3**2 - 20e3**2, rtol=0.02)
    expected = (100e3**2 - 20e3**2) / (np.sqrt(3) / 4 * size**2)
    assert 0.7 * expected < len(mesh["triangles"]) < 1.5 * expected
    centroids = mesh["x"][mesh["triangles"], :2].mean(axis=1)
    assert not ((np.abs(centroids - 50e3) < 9e3).all(axis=1)).any()
    assert n_components(mesh) == 1
    # Boundary elements: about perimeter / size on the coast and on the island.
    assert 0.7 * 480e3 / size < len(mesh["lines"]) < 1.5 * 480e3 / size


def test_square_with_island(model):
    """The default path (size field on a background mesh, smoothness 0.3)."""
    seamsh.gmsh.mesh(square_with_island(), uniform(10e3))
    check_square_with_island(current_mesh(), 10e3)


def test_square_with_island_without_smoothing(model):
    """The other path (smoothness 0: sizes set on the curves, then successive adaptations)."""
    seamsh.gmsh.mesh(square_with_island(), uniform(10e3), smoothness=0)
    check_square_with_island(current_mesh(), 10e3)


def test_size_field_is_followed(model):
    """A size growing from west to east gives smaller triangles in the west."""

    def size(x, projection):
        return 4e3 + 12e3 * x[:, 0] / 100e3

    seamsh.gmsh.mesh(square_with_island(), size)
    mesh = current_mesh()
    centroids = mesh["x"][mesh["triangles"], :2].mean(axis=1)
    areas = signed_areas(mesh)
    assert np.median(areas[centroids[:, 0] < 30e3]) < 0.5 * np.median(areas[centroids[:, 0] > 70e3])


def test_projection_attribute_and_file_version(model, tmp_path):
    """The mesh records its reference system as the model attribute `Projection` (WKT), which
    `reproject` and `convert_to_gis` read back, and is written in the requested msh version."""
    seamsh.gmsh.mesh(square_with_island(), uniform(20e3), version=2.2)
    kind, wkt = gmsh.model.getAttribute("Projection")
    assert kind == "WKT" and osr.SpatialReference(wkt).IsSame(utm())
    path = tmp_path / "mesh.msh"
    gmsh.write(str(path))
    assert path.read_text().splitlines()[1].startswith("2.2")


def test_output_projection(model):
    """With `output_srs` the nodes are reprojected and the attribute names the output system."""
    lonlat = osr.SpatialReference()
    lonlat.ImportFromEPSG(4326)
    lonlat.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    seamsh.gmsh.mesh(square_with_island(), uniform(20e3), output_srs=lonlat)
    mesh = current_mesh()
    used = mesh["x"][np.unique(mesh["triangles"])]
    # UTM 31N: x = 0..100 km is about 4.5 to 3.6 degrees west of the central meridian (3 E), y on
    # the equator.
    assert -1.6 < used[:, 0].min() < -1.4 and -0.7 < used[:, 0].max() < -0.5
    assert used[:, 1].min() > -0.01 and used[:, 1].max() < 1.0
    # The WKT does not hold the axis order chosen with SetAxisMappingStrategy.
    assert gmsh.model.getAttribute("Projection")[1] == lonlat.ExportToWkt()


def test_interior_curve_and_points_are_in_the_mesh(model):
    """Interior curves become mesh edges with their own physical group, forced points mesh nodes."""
    srs = utm()
    domain = square_with_island()
    domain.add_interior_curve([[10e3, 10e3], [30e3, 10e3]], "dyke", srs)
    domain.add_interior_points([[80e3, 80e3]], "station", srs)
    seamsh.gmsh.mesh(domain, uniform(10e3))
    mesh = current_mesh()
    assert mesh["groups"] == ["coast", "domain", "dyke", "island", "station"]
    used = mesh["x"][np.unique(mesh["triangles"]), :2]
    assert np.linalg.norm(used - [80e3, 80e3], axis=1).min() < 1.0
    on_dyke = (np.abs(used[:, 1] - 10e3) < 1.0) & (used[:, 0] > 9e3) & (used[:, 0] < 31e3)
    assert on_dyke.sum() >= 3


def test_transfinite_curve(model):
    """`transfinite_curves` fixes the number of nodes on the curves with a physical tag."""
    seamsh.gmsh.mesh(square_with_island(), uniform(10e3), transfinite_curves={"island": 13})
    for dim, tag in gmsh.model.getPhysicalGroups(1):
        if gmsh.model.getPhysicalName(dim, tag) == "island":
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            n_lines = sum(len(gmsh.model.mesh.getElementsByType(1, e)[0]) for e in entities)
    assert n_lines == 12


def test_two_separate_squares(model):
    """A domain made of two separate regions is meshed as two components, with nothing between."""
    seamsh.gmsh.mesh(two_squares(), uniform(5e3))
    mesh = current_mesh()
    assert mesh["groups"] == ["domain", "left", "right"]
    assert n_components(mesh) == 2
    centroids = mesh["x"][mesh["triangles"], :2].mean(axis=1)
    assert not ((centroids[:, 0] > 40e3) & (centroids[:, 0] < 60e3)).any()
    np.testing.assert_allclose(signed_areas(mesh).sum(), 2 * 40e3**2, rtol=0.02)


@pytest.mark.xfail(strict=True, reason="outlines of islands smaller than the mesh size stay")
def test_islands_smaller_than_the_mesh_leave_no_stray_nodes(model):
    """An island much smaller than the mesh size is covered by triangles; its outline must then
    not stay in the mesh as boundary elements whose nodes belong to no triangle."""
    seamsh.gmsh.mesh(square_with_island(island=1e3), uniform(25e3))
    mesh = current_mesh()
    assert stray_line_nodes(mesh) == 0


@pytest.mark.xfail(strict=True, reason="the background size view 'tri' is left behind")
def test_no_view_is_left_in_gmsh(model):
    """The view holding the background size field is an internal: callers that write their own
    views (moby-dick) must not find it in the file."""
    seamsh.gmsh.mesh(square_with_island(), uniform(20e3))
    assert len(gmsh.view.getTags()) == 0
