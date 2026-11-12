# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: GPL-3.0-or-later

"""Meshing (`seamsh.gmsh.mesh`) on synthetic domains. Counts are checked within ranges: the exact
numbers depend on the gmsh version."""

import gmsh
import numpy as np
import pytest
from domains import (
  current_mesh,
  n_components,
  rectangle,
  signed_areas,
  square_with_island,
  stray_line_nodes,
  two_squares,
  uniform,
  utm,
)
from osgeo import osr

import seamsh.gmsh
from seamsh.geometry import Domain


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


@pytest.mark.parametrize("smoothness", [0.3, 0.0])
def test_region_smaller_than_the_mesh_size_is_meshed(model, caplog, smoothness):
  """A region of the domain is never lost: one whose boundary gmsh would mesh with fewer than
  three segments (here a 6 km square with a 30 km mesh size) gets three, hence one triangle, and
  a message says so. Unlike an island smaller than the mesh size, which is only a hole, such a
  region is a part of the domain that was asked for (e.g. a small sea with its own point in
  `coarsen_boundaries`); it used to disappear from the mesh without a word."""
  srs = utm()
  domain = Domain(srs)
  domain.add_boundary_curve(rectangle(0, 0, 120e3, 120e3, 5e3), "large", srs)
  domain.add_boundary_curve(rectangle(150e3, 0, 156e3, 6e3, 1e3), "small", srs)
  with caplog.at_level("INFO", logger="seamsh"):
    seamsh.gmsh.mesh(domain, uniform(30e3), smoothness=smoothness)
  mesh = current_mesh()
  assert n_components(mesh) == 2
  centroids = mesh["x"][mesh["triangles"], :2].mean(axis=1)
  in_small = centroids[:, 0] > 140e3
  assert in_small.sum() == 1
  assert (signed_areas(mesh) > 0).all()
  assert mesh["groups"] == ["domain", "large", "small"]
  assert stray_line_nodes(mesh) == 0
  assert any("smaller than the mesh size" in r.getMessage() for r in caplog.records)


def test_regions_larger_than_the_mesh_size_are_not_constrained(model, caplog):
  """The minimum of three segments applies to small regions only: nothing is said, and no curve
  is made transfinite, for regions the mesh size resolves."""
  with caplog.at_level("INFO", logger="seamsh"):
    seamsh.gmsh.mesh(two_squares(), uniform(5e3))
  assert not any("smaller than the mesh size" in r.getMessage() for r in caplog.records)


def written_mesh(path) -> dict:
  """The current mesh as a reader gets it: written to a file and read back."""
  gmsh.write(str(path))
  gmsh.clear()
  gmsh.open(str(path))
  return current_mesh()


def test_islands_smaller_than_the_mesh_are_left_out_of_the_file(model, tmp_path):
  """An island much smaller than the mesh size is covered by triangles; its outline must then
  not be written as boundary elements whose nodes belong to no triangle (743 such nodes in
  moby-dick's 120 km mesh, which every reader had to drop)."""
  seamsh.gmsh.mesh(square_with_island(island=1e3), uniform(25e3))
  mesh = written_mesh(tmp_path / "mesh.msh")
  assert stray_line_nodes(mesh) == 0
  assert len(np.unique(mesh["triangles"])) == len(mesh["x"])
  assert mesh["groups"] == ["coast", "domain"]
  # The triangles cover the island.
  np.testing.assert_allclose(signed_areas(mesh).sum(), 100e3**2, rtol=0.01)


def test_islands_larger_than_the_mesh_stay_in_the_file(model, tmp_path):
  """The boundary of an island that the mesh resolves is written with its physical group."""
  seamsh.gmsh.mesh(square_with_island(island=20e3), uniform(10e3))
  mesh = written_mesh(tmp_path / "mesh.msh")
  assert stray_line_nodes(mesh) == 0
  assert mesh["groups"] == ["coast", "domain", "island"]
  assert len(np.unique(mesh["triangles"])) == len(mesh["x"])


def test_no_view_or_field_is_left_in_gmsh(model):
  """The view holding the background size field is an internal: callers that write their own
  views (moby-dick) must not find it in the file."""
  seamsh.gmsh.mesh(square_with_island(), uniform(20e3))
  assert len(gmsh.view.getTags()) == 0
  assert len(gmsh.model.mesh.field.list()) == 0


def test_mesh_size_must_be_positive(model):
  """A size of zero (or NaN) would refine the curves forever (or silently not at all)."""
  for bad in (0.0, np.nan, -1.0):
    gmsh.clear()
    gmsh.model.add("test")
    with pytest.raises(ValueError, match="positive"):
      seamsh.gmsh.mesh(square_with_island(), uniform(bad), smoothness=0)


@pytest.mark.parametrize("smoothness", [0.3, 0.0])
def test_mesh_size_must_be_positive_in_the_whole_bounding_box(model, smoothness):
  """With smoothness > 0 the size is also evaluated outside the domain, on the background grid
  that covers its bounding box. A size of zero, NaN or negative there (e.g. from bathymetry,
  which has no value on land) made the refinement of that grid loop forever; it is an error that
  names a point. With smoothness 0 the size is only asked on the nodes of the mesh, inside the
  domain, and such a size outside it does no harm."""
  srs = utm()
  triangle = Domain(srs)
  triangle.add_boundary_curve(np.array([[0, 0], [100e3, 0], [0, 100e3], [0, 0]], float), "coast", srs)
  for bad in (0.0, np.nan, -1.0):

    def size(x, projection, bad=bad):
      s = np.full(len(x), 5e3)
      s[(x[:, 0] > 95e3) & (x[:, 1] > 95e3)] = bad  # a corner of the box, not of the domain
      return s

    gmsh.clear()
    gmsh.model.add("test")
    if smoothness > 0:
      with pytest.raises(ValueError, match="positive numbers: .* at"):
        seamsh.gmsh.mesh(triangle, size, smoothness=smoothness)
    else:
      seamsh.gmsh.mesh(triangle, size, smoothness=smoothness)
      assert len(current_mesh()["triangles"]) > 100


def test_a_domain_can_be_meshed_twice(model):
  """Meshing cuts boundary curves where loops touch; meshing the same domain again (e.g. at
  another resolution) starts from the curves as given. It used to start from the pieces, four of
  whose ends meet where two islands touch: an "Invalid topology" error on the second call."""
  srs = utm()
  domain = Domain(srs)
  domain.add_boundary_curve(rectangle(0, 0, 100e3, 100e3, 5e3), "coast", srs)
  # Two islands whose outlines start elsewhere and touch at (50 km, 50 km).
  corners = [[60e3, 50e3], [60e3, 60e3], [50e3, 60e3], [50e3, 50e3], [60e3, 50e3]]
  domain.add_boundary_curve(np.array(corners), "a", srs)
  domain.add_boundary_curve(100e3 - np.array(corners), "b", srs)
  counts = []
  for size in (10e3, 5e3, 10e3):
    gmsh.model.add(f"size {size}")
    seamsh.gmsh.mesh(domain, uniform(size))
    counts.append(len(current_mesh()["triangles"]))
    gmsh.model.remove()
  assert counts[0] == counts[2] and counts[1] > counts[0]
  # Curves added after a first mesh are taken into account.
  domain.add_boundary_curve(rectangle(10e3, 10e3, 20e3, 20e3, 2e3), "c", srs)
  gmsh.model.add("with c")
  seamsh.gmsh.mesh(domain, uniform(5e3))
  assert "c" in current_mesh()["groups"]


def test_convert_to_gis_and_reproject(model, tmp_path):
  """A mesh file converts to a GIS layer with one polygon per triangle, in the projection stored
  in the file, and `reproject` rewrites it in another reference system."""
  from osgeo import ogr

  seamsh.gmsh.mesh(square_with_island(), uniform(20e3))
  n_triangles = len(current_mesh()["triangles"])
  msh = tmp_path / "mesh.msh"
  gmsh.write(str(msh))
  seamsh.gmsh.convert_to_gis(str(msh), None, str(tmp_path / "mesh.gpkg"))
  source = ogr.Open(str(tmp_path / "mesh.gpkg"))
  layer = source.GetLayer()
  assert layer.GetFeatureCount() == n_triangles
  assert layer.GetSpatialRef().IsSame(utm())
  west, east, south, north = layer.GetExtent()
  np.testing.assert_allclose([west, east, south, north], [0, 100e3, 0, 100e3], atol=1.0)

  lonlat = osr.SpatialReference()
  lonlat.ImportFromEPSG(4326)
  lonlat.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
  seamsh.gmsh.reproject(str(msh), None, str(tmp_path / "lonlat.msh"), lonlat)
  gmsh.clear()
  gmsh.open(str(tmp_path / "lonlat.msh"))
  mesh = current_mesh()
  assert len(mesh["triangles"]) == n_triangles
  assert -1.6 < mesh["x"][:, 0].min() < -1.4 and mesh["x"][:, 1].max() < 1.0


def test_projection_stored_on_several_lines_is_read(model, tmp_path):
  """A projection attribute written on several lines (e.g. a pretty-printed WKT, by another tool) comes back from
  gmsh as several strings: they are joined. Only the first line was read, and reproject failed (upstream's c873f34).
  """
  seamsh.gmsh.mesh(square_with_island(), uniform(20e3))
  gmsh.model.set_attribute("Projection", ["WKT", utm().ExportToPrettyWkt()])
  msh = str(tmp_path / "pretty.msh")
  gmsh.write(msh)
  gmsh.clear()
  gmsh.open(msh)
  assert len(gmsh.model.get_attribute("Projection")) > 2
  lonlat = osr.SpatialReference()
  lonlat.ImportFromEPSG(4326)
  lonlat.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
  seamsh.gmsh.reproject(msh, None, str(tmp_path / "lonlat.msh"), lonlat)
  seamsh.gmsh.convert_to_gis(msh, None, str(tmp_path / "pretty.gpkg"))
  from osgeo import ogr

  assert ogr.Open(str(tmp_path / "pretty.gpkg")).GetLayer().GetSpatialRef().IsSame(utm())


def test_reproject_without_any_projection_is_an_error(model, tmp_path):
  """With no projection given and none stored in the file, reproject says so (it failed inside the reprojection)."""
  seamsh.gmsh.mesh(square_with_island(), uniform(20e3))
  gmsh.model.remove_attribute("Projection")
  msh = str(tmp_path / "bare.msh")
  gmsh.write(msh)
  with pytest.raises(ValueError, match="No projection"):
    seamsh.gmsh.reproject(msh, None, str(tmp_path / "out.msh"), utm())


@pytest.mark.parametrize("version", [2.2, 4.1])
def test_reproject_keeps_the_node_numbers(model, tmp_path, version):
  """Written as MSH 2, the nodes of the reprojected mesh were renumbered from 1, which breaks data kept by node
  number next to the mesh (upstream's ce6ec1a and 6442e2e); the gmsh option set for it is put back. gmsh 4.12 and
  older ignore the option. (MSH 2 holds the nodes of the elements only, MSH 4 all the nodes of the model: the nodes of
  the triangles are compared.)"""
  if version == 2.2 and tuple(int(v) for v in gmsh.__version__.split(".")[:2]) < (4, 13):
    pytest.skip("gmsh < 4.13 renumbers the nodes of MSH 2 files whatever the option")

  def triangle_nodes():
    return np.unique(gmsh.model.mesh.getElementsByType(2)[1])

  seamsh.gmsh.mesh(square_with_island(), uniform(20e3))
  tags = gmsh.model.mesh.getNodes()[0]
  gmsh.model.mesh.renumberNodes(tags, tags + 1000)
  expected = triangle_nodes()
  assert expected.min() > 1000
  gmsh.write(str(tmp_path / "in.msh"))
  before = gmsh.option.getNumber("Mesh.PreserveNumberingMsh2")
  seamsh.gmsh.reproject(str(tmp_path / "in.msh"), None, str(tmp_path / "out.msh"), utm(), output_version=version)
  assert gmsh.option.getNumber("Mesh.PreserveNumberingMsh2") == before
  gmsh.clear()
  gmsh.open(str(tmp_path / "out.msh"))
  np.testing.assert_array_equal(triangle_nodes(), expected)


def test_features_without_a_tag_are_meshed_in_no_group(model, tmp_path):
  """Features of a shapefile whose tag field is unset or empty are meshed, and are in no physical group. Their
  physical name was None, and mesh() failed (upstream's 5b45b37); an empty name made a group without a name."""
  from osgeo import ogr

  path = str(tmp_path / "tags.gpkg")
  source = ogr.GetDriverByName("GPKG").CreateDataSource(path)
  layer = source.CreateLayer("coast", utm(), ogr.wkbLineString)
  layer.CreateField(ogr.FieldDefn("phys", ogr.OFTString))
  corners = [(0, 0), (100e3, 0), (100e3, 100e3), (0, 100e3), (0, 0)]
  for value, (a, b) in zip(["coast", None, "", "coast"], zip(corners[:-1], corners[1:])):
    feature = ogr.Feature(layer.GetLayerDefn())
    if value is not None:
      feature.SetField("phys", value)
    line = ogr.Geometry(ogr.wkbLineString)
    line.AddPoint_2D(*a)
    line.AddPoint_2D(*b)
    feature.SetGeometry(line)
    layer.CreateFeature(feature)
  source = None
  domain = Domain(utm())
  domain.add_boundary_curves_shp(path, "phys")
  domain.add_interior_points(np.array([[50e3, 50e3]]), "", utm())
  seamsh.gmsh.mesh(domain, uniform(20e3))
  mesh = current_mesh()
  assert mesh["groups"] == ["coast", "domain"]
  np.testing.assert_allclose(signed_areas(mesh).sum(), 100e3**2, rtol=1e-6)
  # The group "coast" is the two tagged sides: the bottom (y = 0) and the left (x = 0).
  (coast,) = [t for d, t in gmsh.model.getPhysicalGroups(1) if gmsh.model.getPhysicalName(d, t) == "coast"]
  x = gmsh.model.mesh.getNodesForPhysicalGroup(1, coast)[1].reshape(-1, 3)
  assert np.all((np.abs(x[:, 0]) < 1) | (np.abs(x[:, 1]) < 1))
  assert np.abs(x[:, :2] - [100e3, 0]).sum(axis=1).min() < 1 and np.abs(x[:, :2] - [0, 100e3]).sum(axis=1).min() < 1


def test_size_callback_scaling_between_projections():
  """Curves are sampled in their own projection, where the size (given in the domain's) is divided
  by the local scale of the projection, estimated by finite differences. The estimate must hold
  for coordinates in degrees and in metres (where a step of 1e-8 is lost in rounding), and is
  exactly 1 when both projections are the same."""
  lonlat = osr.SpatialReference()
  lonlat.ImportFromEPSG(4326)
  lonlat.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
  utm32 = osr.SpatialReference()
  utm32.ImportFromEPSG(32632)
  size = uniform(1000.0)

  def scale(source, x, step):
    to_utm = osr.CoordinateTransformation(source, utm())
    a = np.array(to_utm.TransformPoint(*x)[:2])
    b = np.array(to_utm.TransformPoint(x[0] + step, x[1] + step)[:2])
    return np.linalg.norm(b - a) / np.hypot(step, step)

  same = seamsh.gmsh._lcproj(size, utm())(np.array([[5e5, 5e6]]), utm())
  np.testing.assert_array_equal(same, [1000.0])
  degrees = seamsh.gmsh._lcproj(size, lonlat)(np.array([[3.0, 45.0]]), utm())
  np.testing.assert_allclose(degrees, 1000.0 / scale(lonlat, (3.0, 45.0), 1e-5), rtol=1e-4)
  metres = seamsh.gmsh._lcproj(size, utm32)(np.array([[2e5, 5e6]]), utm())
  np.testing.assert_allclose(metres, 1000.0 / scale(utm32, (2e5, 5e6), 1.0), rtol=1e-4)


def test_one_surface_per_region(model):
  """Two separate squares are two gmsh surfaces, both in the physical group "domain"."""
  seamsh.gmsh.mesh(two_squares(), uniform(5e3))
  assert len(gmsh.model.getEntities(2)) == 2
  (group,) = gmsh.model.getPhysicalGroups(2)
  assert gmsh.model.getPhysicalName(*group) == "domain"
  assert len(gmsh.model.getEntitiesForPhysicalGroup(*group)) == 2


def test_lake_on_an_island_is_meshed(model):
  """Nested boundaries: the sea around an island, and the lake on the island, are two regions;
  the island between them is not meshed."""
  from domains import rectangle

  from seamsh.geometry import Domain

  srs = utm()
  domain = Domain(srs)
  domain.add_boundary_curve(rectangle(0, 0, 100e3, 100e3, 5e3), "sea", srs)
  domain.add_boundary_curve(rectangle(20e3, 20e3, 80e3, 80e3, 5e3), "island", srs)
  domain.add_boundary_curve(rectangle(35e3, 35e3, 65e3, 65e3, 5e3), "lake", srs)
  domain.add_interior_points([[50e3, 50e3]], "gauge", srs)
  seamsh.gmsh.mesh(domain, uniform(5e3))
  mesh = current_mesh()
  assert n_components(mesh) == 2
  np.testing.assert_allclose(signed_areas(mesh).sum(), 100e3**2 - 60e3**2 + 30e3**2, rtol=0.02)
  centroids = mesh["x"][mesh["triangles"], :2].mean(axis=1)
  on_island = (np.abs(centroids - 50e3) < 29e3).all(axis=1) & ~((np.abs(centroids - 50e3) < 16e3).all(axis=1))
  assert not on_island.any()
  # The forced point, in the lake, is a node of the mesh of the lake.
  used = mesh["x"][np.unique(mesh["triangles"]), :2]
  assert np.linalg.norm(used - [50e3, 50e3], axis=1).min() < 1.0


def test_coarsened_basins_are_meshed_separately(model):
  """The whole chain for basins joined by a channel too narrow for the coarsening: one seed per
  basin, and a mesh in two pieces with no element in the channel."""
  from domains import two_basins

  from seamsh.geometry import coarsen_boundaries

  seeds = [(20e3, 20e3), (80e3, 20e3)]
  coarse = coarsen_boundaries(two_basins(half_width=1e3), seeds, utm(), uniform(5e3))
  seamsh.gmsh.mesh(coarse, uniform(5e3))
  mesh = current_mesh()
  assert n_components(mesh) == 2
  assert mesh["groups"] == ["coast", "domain"]
  centroids = mesh["x"][mesh["triangles"], :2].mean(axis=1)
  assert not ((centroids[:, 0] > 42e3) & (centroids[:, 0] < 58e3)).any()
  assert (centroids[:, 0] < 40e3).sum() > 50 and (centroids[:, 0] > 60e3).sum() > 50
  assert (signed_areas(mesh) > 0).all()


def test_gmsh_options_are_left_as_they_were(model):
  """gmsh options are global: those that `mesh` sets for itself must be put back, so that they do
  not change what the caller meshes next. Only the file version is meant to stay."""
  names = seamsh.gmsh._MESH_OPTIONS
  defaults = {name: gmsh.option.getNumber(name) for name in [*names, "Mesh.MshFileVersion"]}
  try:
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e21)
    before = {name: gmsh.option.getNumber(name) for name in names}
    for smoothness in (0, 0.3):
      gmsh.clear()
      gmsh.model.add("test")
      seamsh.gmsh.mesh(square_with_island(), uniform(20e3), smoothness=smoothness, version=2.2)
      assert {name: gmsh.option.getNumber(name) for name in names} == before
    assert gmsh.option.getNumber("Mesh.MshFileVersion") == 2.2
  finally:
    for name, value in defaults.items():
      gmsh.option.setNumber(name, value)


def test_size_smoothing_does_not_depend_on_the_units(model):
  """The same domain and sizes in kilometres and in metres give the same mesh. The size is small
  in a disc and large elsewhere, so the mesh depends on the smoothing of the size; its stopping
  tolerance used to be 1 whatever the units, too coarse for sizes around 1 (682 triangles
  against 687)."""
  from domains import rectangle

  from seamsh.geometry import Domain

  def n_triangles(scale):
    srs = utm()
    domain = Domain(srs)
    domain.add_boundary_curve(rectangle(0, 0, 100 * scale, 100 * scale, 5 * scale), "c", srs)

    def size(x, projection):
      near = np.linalg.norm(x[:, :2] - 50 * scale, axis=1) < 5 * scale
      return np.where(near, 1.0, 30.0) * scale

    gmsh.clear()
    gmsh.model.add("test")
    seamsh.gmsh.mesh(domain, size)
    return len(current_mesh()["triangles"])

  kilometres, metres = n_triangles(1.0), n_triangles(1000.0)
  assert kilometres == pytest.approx(metres, rel=0.003)
  assert metres > 500


def test_progress_bars_can_be_turned_off(model, capfd):
  """`seamsh.show_progress(False)` silences the progress bars (they go to the standard error)."""
  from seamsh.geometry import coarsen_boundaries

  try:
    seamsh.show_progress(False)
    coarsen_boundaries(square_with_island(), (10e3, 10e3), utm(), uniform(10e3))
    assert "(seamsh)" not in capfd.readouterr().err
    seamsh.show_progress(True)
    coarsen_boundaries(square_with_island(), (10e3, 10e3), utm(), uniform(10e3))
    assert "(seamsh) Sampling curves" in capfd.readouterr().err
  finally:
    seamsh.show_progress(True)


def test_convert_to_ugrid(model, tmp_path):
  """The UGRID file holds the nodes, the triangles as indices of the nodes (whatever the node
  tags of the mesh), the coordinates with the units of the projection, and the variables."""
  xr = pytest.importorskip("xarray")
  pytest.importorskip("netCDF4")
  seamsh.gmsh.mesh(square_with_island(), uniform(20e3))
  # Node tags that do not start from 1.
  gmsh.model.mesh.renumberNodes()
  tags, coordinates, _ = gmsh.model.mesh.getNodes()
  gmsh.model.mesh.renumberNodes(tags, tags + 1000)
  mesh = current_mesh()
  gmsh.write(str(tmp_path / "mesh.msh"))
  variables = {
    "depth": {
      "callable_func": lambda x, projection: x[:, 0] / 1e3,
      "long_name": "depth",
      "standard_name": "sea_floor_depth_below_geoid",
      "units": "m",
    }
  }
  seamsh.gmsh.convert_to_ugrid_mesh_file(str(tmp_path / "mesh.msh"), variables, utm(), str(tmp_path / "mesh.nc"))
  ds = xr.open_dataset(tmp_path / "mesh.nc")
  assert ds["element"].shape == (len(mesh["triangles"]), 3)
  assert int(ds["element"].min()) >= 0 and int(ds["element"].max()) < ds.sizes["node"]
  x, y = ds["longitude"].values, ds["latitude"].values
  corners = np.stack([x[ds["element"].values], y[ds["element"].values]], axis=-1)
  a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
  areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
  np.testing.assert_allclose(areas.sum(), 100e3**2 - 20e3**2, rtol=0.02)
  assert ds["longitude"].attrs["units"] == "m"
  assert ds["longitude"].attrs["standard_name"] == "projection_x_coordinate"
  np.testing.assert_allclose(ds["depth"].values, x / 1e3)


def test_ugrid_global_attributes_name_this_software(model, tmp_path):
  """The global attributes of the UGRID file say which software made it, and hold no placeholder
  ("Institution Name", "REF" were written into every file)."""
  xr = pytest.importorskip("xarray")
  pytest.importorskip("netCDF4")
  seamsh.gmsh.mesh(square_with_island(), uniform(20e3))
  gmsh.write(str(tmp_path / "m.msh"))
  seamsh.gmsh.convert_to_ugrid_mesh_file(str(tmp_path / "m.msh"), {}, utm(), str(tmp_path / "m.nc"))
  attrs = xr.open_dataset(tmp_path / "m.nc").attrs
  assert attrs["source"] == "seamsh (https://github.com/stefanocampanella/seamsh)"
  assert "institution" not in attrs and "references" not in attrs
  assert attrs["Conventions"] == "CF-1.6, UGRID-1.0"


def test_convert_to_ugrid_with_the_stored_projection(model, tmp_path):
  """With projection=None the projection stored in the file by `mesh` is used, as in
  `convert_to_gis` (it was an AttributeError on None); the caller's model is left current."""
  xr = pytest.importorskip("xarray")
  pytest.importorskip("netCDF4")
  seamsh.gmsh.mesh(square_with_island(), uniform(20e3))
  gmsh.write(str(tmp_path / "mesh.msh"))
  models = gmsh.model.list()
  seamsh.gmsh.convert_to_ugrid_mesh_file(str(tmp_path / "mesh.msh"), {}, None, str(tmp_path / "m.nc"))
  assert xr.open_dataset(tmp_path / "m.nc")["longitude"].attrs["units"] == "m"
  assert gmsh.model.list() == models or len(gmsh.model.list()) == len(models) + 1


def test_quad_mesh(model):
  """`quad=True` gives quadrangles only, about half the prescribed size, with the boundary
  elements kept. Builds of gmsh that cannot do it (they crash: see `_gmsh_has_quad_algorithm`)
  are refused with an error instead."""
  if not seamsh.gmsh._gmsh_has_quad_algorithm():
    with pytest.raises(RuntimeError, match="ANN"):
      seamsh.gmsh.mesh(square_with_island(), uniform(20e3), quad=True)
    return
  seamsh.gmsh.mesh(square_with_island(), uniform(20e3), quad=True)
  types = [int(t) for t in gmsh.model.mesh.getElements(2)[0]]
  assert types == [3]
  n_quads = len(gmsh.model.mesh.getElementsByType(3)[0])
  # Area / (size / 2)^2, roughly.
  assert 50 < n_quads < 200
  assert len(gmsh.model.mesh.getElementsByType(1)[0]) > 30
  assert sorted(gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups()) == [
    "coast",
    "domain",
    "island",
  ]
