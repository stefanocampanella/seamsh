# seamsh - Copyright (C) <2010-2020>
# <Universite catholique de Louvain (UCL), Belgium
#
# List of the contributors to the development of seamsh: see AUTHORS file.
# Description and complete License: see LICENSE file.
#
# This program (seamsh) is free software:
# you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program (see COPYING file).  If not,
# see <http://www.gnu.org/licenses/>.

"""The geometry of a domain (boundary curves, interior curves and points) and its coarsening."""

import logging
from enum import Enum
from itertools import chain
from typing import Callable

import numpy as np
from osgeo import ogr, osr
from scipy.spatial import Delaunay, cKDTree

from . import _tools

__all__ = ["Domain", "CurveType", "coarsen_boundaries"]

MeshSizeCallback = Callable[[np.ndarray, osr.SpatialReference], np.ndarray]
"""A mesh size field: called with points [n, 2] and their projection, returns the desired element
size at each point [n], in the units of that projection."""

logger = logging.getLogger(__name__)


class CurveType(Enum):
  """Determine how curves control points are interpolated."""

  POLYLINE = 1
  """ linear interpolation """
  STRICTPOLYLINE = 2
  """ linear interpolation with a  mesh point forced on each curve point """
  SPLINE = 3
  """ cubic spline interpolation """
  BSPLINE = 4
  """ cubic b-spline interpolation (smoother than spline) """


def _generate_unique_points(x: np.ndarray) -> tuple:
  """Merges the points that coincide (closer than 1e-8 times the size of their bounding box).

  Args:
      x: points [n, 2].
  Returns:
      The merged points [m, 2]; for each input point the index of its merged point [n]; and for
      each merged point, whether several input points were merged into it [m] (these come first).
  """
  tree = cKDTree(x)
  unique_id = np.full([x.shape[0]], -1, np.int32)
  bbmin = np.min(x, axis=0)
  bbmax = np.max(x, axis=0)
  eps = np.linalg.norm(bbmax - bbmin) * 1e-8
  n_unique = 0
  pairs = np.array(list(tree.query_pairs(eps)))
  if pairs.shape[0] == 0:
    pairs = pairs.reshape(-1, 2)
  # Sorting the pairs is necessary to handle points connecting more than 2 lines.
  pairs.sort(axis=1)
  pairs_key = pairs[:, 0].astype(np.int64) * 2**32 + pairs[:, 1].astype(np.int64)
  pairs = pairs[pairs_key.argsort(), :]
  for p0, p1 in pairs:
    uid = max(unique_id[p0], unique_id[p1])
    if uid == -1:
      uid = n_unique
      n_unique += 1
    unique_id[p0] = uid
    unique_id[p1] = uid
  n_merged = n_unique
  for i in range(x.shape[0]):
    if unique_id[i] == -1:
      unique_id[i] = n_unique
      n_unique += 1
  unique = np.ndarray([n_unique, 2])
  unique[unique_id, :] = x
  is_merged = np.zeros([n_unique], dtype=bool)
  is_merged[:n_merged] = True
  return unique, unique_id, is_merged


class _Curve:
  """A curve through control points, with a physical tag."""

  def __init__(self, points, tag, curve_type, projection):
    self.points = np.array(points)[:, :2]  # control points, in `projection`
    self.mesh_size = None
    self.tag = tag
    self.curve_type = curve_type
    self.projection = projection
    self.pointsid = None  # indices of the points in Domain._points, set by _build_topology


class _Point:
  """A forced mesh point, with a physical tag."""

  def __init__(self, x, tag):
    self.tag = tag
    self.x = x  # in the projection of the domain
    self.pointid = None  # index in Domain._points, set by _build_topology


def _split_curves(curves: list, is_break: np.ndarray, loops=()) -> list:
  """Cuts the curves at their inner control points flagged in `is_break` (points shared with
  another curve or forced point), so that curves meet only at their end points.

  `loops` (lists of (curve index, forward) made of these curves) are updated in place.
  """
  new_curves = []
  new_indices = []  # for each curve, the indices of its pieces in new_curves
  for curve in curves:
    breaks = np.where(is_break[curve.pointsid[1:-1]])[0]
    if len(breaks) == 0:
      new_curves.append(curve)
      new_indices.append([len(new_curves) - 1])
    else:
      breaks = [0] + list(breaks + 1) + [len(curve.points) - 1]
      pieces = []
      for i, j in zip(breaks[:-1], breaks[1:]):
        piece = _Curve(curve.points[i : j + 1], curve.tag, curve.curve_type, curve.projection)
        piece.pointsid = curve.pointsid[i : j + 1]
        new_curves.append(piece)
        pieces.append(len(new_curves) - 1)
      new_indices.append(pieces)
  for iloop, loop in enumerate(loops):
    new_loop = []
    for curve_index, forward in loop:
      pieces = new_indices[curve_index]
      for j in pieces if forward else reversed(pieces):
        new_loop.append((j, forward))
    loops[iloop] = new_loop
  return new_curves


def _chain_into_loops(curves: list, points: np.ndarray) -> list:
  """Chains boundary curves that share end points into closed loops.

  Returns:
      The loops, each a list of (curve index, forward): the curves in order along the loop, with
      whether each is followed from its first to its last point.
  Args:
      curves: the boundary curves, with their `pointsid`.
      points: the points of the domain [n, 2].
  Raises:
      ValueError: if the curves do not form closed loops, or more than two of them meet at a
          point (the boundary would not tell which curve follows which).
  """
  # For each point, the (curve, end) that touch it (end 0: first point, 1: last point): two in a
  # valid boundary. For each curve, its two end points.
  point_curves = np.full([points.shape[0], 2, 2], -1, int)
  curve_points = np.ndarray([len(curves), 2], int)
  for icurve, curve in enumerate(curves):
    for end, point in ((0, curve.pointsid[0]), (1, curve.pointsid[-1])):
      curve_points[icurve, end] = point
      if point_curves[point][1][0] != -1:
        raise ValueError(f"Invalid topology: more than two boundary curves meet at {tuple(points[point])}")
      slot = 0 if point_curves[point][0][0] == -1 else 1
      point_curves[point][slot] = (icurve, end)
  touched = np.full((len(curves)), False, bool)
  loops = []
  for first in range(len(curves)):
    if touched[first]:
      continue
    loop = []
    icurve, end = first, 0
    while (not loop) or loop[0][0] != icurve:
      loop.append((icurve, end == 0))
      # Leave the curve by its other end, and take the other curve that touches that point.
      point = curve_points[icurve, (end + 1) % 2]
      if point_curves[point][0][0] != icurve:
        icurve, end = point_curves[point][0]
      else:
        icurve, end = point_curves[point][1]
      if icurve == -1 or (icurve != loop[0][0] and touched[icurve]):
        raise ValueError("Invalid topology")
      touched[icurve] = True
    loops.append(loop)
  return loops


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
  """Whether each point [n, 2] is inside the closed polyline `polygon` [m, 2] (even-odd rule: a
  half-line from the point crosses the polygon an odd number of times)."""
  points = np.asarray(points, float)
  inside = np.zeros(len(points), bool)
  # Only the points in the bounding box of the polygon can be inside.
  lower, upper = polygon.min(axis=0), polygon.max(axis=0)
  candidates = np.flatnonzero(((points >= lower) & (points <= upper)).all(axis=1))
  if len(candidates) == 0:
    return inside
  a, b = polygon[:-1], polygon[1:]
  # By blocks of points x segments, so that the temporaries hold at most _BLOCK values whatever
  # the size of the polygon.
  n_points = max(1, min(len(candidates), _BLOCK // max(1, len(a))))
  n_segments = max(1, _BLOCK // n_points)
  for start in range(0, len(candidates), n_points):
    chunk = candidates[start : start + n_points]
    px, py = points[chunk, 0, None], points[chunk, 1, None]
    crossings = np.zeros(len(chunk), np.int64)
    for first in range(0, len(a), n_segments):
      sa, sb = a[first : first + n_segments], b[first : first + n_segments]
      straddles = (sa[None, :, 1] > py) != (sb[None, :, 1] > py)
      with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = sa[None, :, 0] + (py - sa[None, :, 1]) * (sb[None, :, 0] - sa[None, :, 0]) / (
          sb[None, :, 1] - sa[None, :, 1]
        )
      crossings += np.count_nonzero(straddles & (px < x_cross), axis=1)
    inside[chunk] = crossings % 2 == 1
  return inside


# Number of values (points x segments) of the temporaries of _points_in_polygon: 8 MB each.
_BLOCK = 2**20


def _nest_loops(polygons: list) -> tuple:
  """Sorts closed loops that do not cross into regions, by containment.

  A loop inside an even number of other loops (0, 2, ...) is the outer boundary of a region; a
  loop inside an odd number of loops is a hole, of the region whose outer boundary is the
  smallest loop that contains it. An island in a lake on an island is thus a hole of the lake,
  and a lake on an island is a region of its own.

  Args:
      polygons: the loops, as closed polylines [m, 2].
  Returns:
      The regions, as lists of loop indices (the outer boundary, then the holes), ordered by the
      index of their outer boundary; and for each loop the indices of the loops that contain it.
  """
  n = len(polygons)
  lower = np.array([p.min(axis=0) for p in polygons])
  upper = np.array([p.max(axis=0) for p in polygons])
  # A loop is inside another when most of a few of its points are (the loops are polylines
  # through the control points of curves, which splines only follow approximately).
  samples = [p[np.linspace(0, len(p) - 1, 5, endpoint=False).astype(int)] for p in polygons]
  containers = [[] for _ in range(n)]
  for outer in range(n):
    inside_box = np.where((lower >= lower[outer]).all(axis=1) & (upper <= upper[outer]).all(axis=1))[0]
    inner = [i for i in inside_box if i != outer]
    if not inner:
      continue
    inside = _points_in_polygon(np.vstack([samples[i] for i in inner]), polygons[outer])
    for i, votes in zip(inner, inside.reshape(len(inner), -1)):
      if 2 * votes.sum() > len(votes):
        containers[i].append(outer)
  regions = {}
  for i in range(n):
    if len(containers[i]) % 2 == 0:
      regions.setdefault(i, [i])
  for i in range(n):
    if len(containers[i]) % 2 == 1:
      # Its region is bounded by its innermost container: the one inside all the others.
      parent = max(containers[i], key=lambda c: len(containers[c]))
      regions[parent].append(i)
  return [regions[i] for i in sorted(regions)], containers


class Domain:
  """List the domain boundaries, forced mesh points and
  forced mesh lines."""

  def __init__(self, projection: osr.SpatialReference):
    """
    Args:
        projection: coordinate system of the geometry and mesh
    """

    self._projection = projection
    self._interior_curves = []
    self._interior_points = []
    self._curves = []
    # Set by _build_topology: (boundary pieces, interior pieces, boundary curves as given,
    # interior curves as given), so that the topology is rebuilt from the curves as given.
    self._topology_inputs = None

  def _build_topology(self) -> None:
    """Works out how the curves are connected, for the mesher.

    Sets `_points` (all control points, in the projection of the domain, coincident ones
    merged), the `pointsid` of the curves and `pointid` of the forced points (indices in
    `_points`), cuts the curves where they touch each other, and sets `_curveloops`: the closed
    loops of boundary curves, as lists of (index in `_curves`, forward), the loop with the
    largest bounding box first; and `_regions`: the connected parts of the domain, as lists of
    indices in `_curveloops` (the outer boundary, then the holes: see `_nest_loops`).
    """
    logger.info("Build topology")
    # Start again from the curves as given (and those added since), not from the pieces of a
    # previous build: two loops that touch at a point inside their curves are cut there, and
    # four piece ends would then meet at that point.
    if self._topology_inputs is not None:
      pieces, interior_pieces, curves, interior_curves = self._topology_inputs
      built = {id(c) for c in chain(pieces, interior_pieces)}
      self._curves = curves + [c for c in self._curves if id(c) not in built]
      self._interior_curves = interior_curves + [c for c in self._interior_curves if id(c) not in built]
    curves_as_given = list(self._curves)
    interior_curves_as_given = list(self._interior_curves)
    all_curves = list(chain(self._curves, self._interior_curves))
    all_points = np.vstack(
      [_tools.project_points(c.points, c.projection, self._projection) for c in all_curves]
      + [point.x for point in self._interior_points]
    )
    self._points, unique_id, is_break = _generate_unique_points(all_points)
    # Points used by several curves or forced points are where curves must be cut.
    offset = 0
    touched = np.zeros((self._points.shape[0],), bool)
    for curve in all_curves:
      n = curve.points.shape[0]
      curve.pointsid = unique_id[offset : offset + n]
      touched[curve.pointsid] = True
      offset += n
    for point in self._interior_points:
      point.pointid = unique_id[offset]
      touched[point.pointid] = True
      offset += 1
    is_break = np.logical_and(touched, is_break)

    self._curveloops = _chain_into_loops(self._curves, self._points)
    self._curves = _split_curves(self._curves, is_break, self._curveloops)
    self._interior_curves = _split_curves(self._interior_curves, is_break)
    self._topology_inputs = (
      list(self._curves),
      list(self._interior_curves),
      curves_as_given,
      interior_curves_as_given,
    )
    bbox_area = np.zeros([len(self._curveloops)])
    for i, loop in enumerate(self._curveloops):
      loop_points = np.vstack([self._curves[j].points for j, _ in loop])
      bbox = np.max(loop_points, axis=0) - np.min(loop_points, axis=0)
      bbox_area[i] = bbox[0] * bbox[1]
    self._curveloops.insert(0, self._curveloops.pop(np.argmax(bbox_area)))
    self._loop_polygons = [
      np.vstack([self._points[self._curves[j].pointsid][:: 1 if forward else -1] for j, forward in loop])
      for loop in self._curveloops
    ]
    self._regions, self._loop_containers = _nest_loops(self._loop_polygons)

  def _region_of(self, point) -> int:
    """The index in `_regions` of the region that holds `point` (in the projection of the
    domain), or None if the point is outside the domain. Needs `_build_topology`."""
    region = self._regions_of(np.asarray(point, float)[None, :2])[0]
    return None if region < 0 else int(region)

  def _regions_of(self, points: np.ndarray) -> np.ndarray:
    """For each point [n, 2] (in the projection of the domain), the index in `_regions` of the
    region that holds it, or -1 if it is outside the domain. Needs `_build_topology`."""
    points = np.asarray(points, float).reshape(-1, 2)
    n_loops = len(self._loop_polygons)
    # Depth of each loop: the number of loops that contain it. The loop that holds a point
    # most closely is the deepest of the loops that contain the point.
    depth = np.array([len(c) for c in self._loop_containers])
    head_of = {region[0]: i for i, region in enumerate(self._regions)}
    n_inside = np.zeros(len(points), int)
    deepest = np.full(len(points), -1)
    for loop in range(n_loops):
      inside = _points_in_polygon(points, self._loop_polygons[loop])
      n_inside += inside
      deeper = inside & ((deepest < 0) | (depth[loop] > depth[np.maximum(deepest, 0)]))
      deepest[deeper] = loop
    regions = np.full(len(points), -1)
    for i in np.flatnonzero(n_inside % 2 == 1):
      region = head_of.get(int(deepest[i]))
      if region is None:
        # The exact test disagrees with the nesting of the loops (loops that touch, or a
        # spline loop sampled at a few points by _nest_loops): treat as outside.
        logger.warning(
          "Cannot tell which region holds %s: the loops around it are ambiguous",
          tuple(points[i]),
        )
      else:
        regions[i] = region
    return regions

  def _add_geometry(self, geometry, tag, projection, curve_type, interior, onlypoints=False):
    if geometry.GetGeometryCount() != 0:
      for subg in geometry:
        self._add_geometry(subg, tag, projection, curve_type, interior, onlypoints)
      return
    if onlypoints:
      self.add_interior_points(geometry.GetPoints(), tag, projection)
    else:
      if interior:
        self.add_interior_curve(geometry.GetPoints(), tag, projection, curve_type)
      else:
        self.add_boundary_curve(geometry.GetPoints(), tag, projection, curve_type)

  def _add_shapefile(self, filename, physical_name_field, interior, points, curve_type):
    if filename[-5:] == ".gpkg":
      driver = ogr.GetDriverByName("GPKG")
    else:
      driver = ogr.GetDriverByName("ESRI Shapefile")
    # Open returns None, or raises if the caller enabled OGR exceptions.
    try:
      data = driver.Open(filename, 0)
    except RuntimeError:
      data = None
    if data is None:
      raise FileNotFoundError(f"cannot open '{filename}' as a shapefile or geopackage")
    layer = data.GetLayer()
    layerdef = layer.GetLayerDefn()
    physfield = None
    count = 0
    if physical_name_field is not None:
      for i in range(layerdef.GetFieldCount()):
        field_name = layerdef.GetFieldDefn(i).GetName()
        if field_name == physical_name_field:
          physfield = i
      if physfield is None:
        raise ValueError("field '" + physical_name_field + "' not found in shapefile")
    layerproj = layer.GetSpatialRef()
    for i in _tools.progress(layer, desc=f"(seamsh) Import features from '{filename}'"):
      if i.geometry() is None:
        continue
      phys = i.GetField(physfield) if physfield is not None else "boundary"
      self._add_geometry(i.geometry(), phys, layerproj, curve_type, interior, points)
      count += 1
    logger.info("%d features imported", count)

  def add_interior_points(self, points: np.ndarray, physical_tag: str, projection: osr.SpatialReference) -> None:
    """Add forced interior mesh points

    Args:
        points: the points [n,2]
        physical_tag: the points physical tag
        projection: the points coordinate system
    """
    x = _tools.project_points(points, projection, self._projection)
    self._interior_points += [_Point(p, physical_tag) for p in x]

  def add_boundary_curve(
    self,
    points: np.ndarray,
    physical_tag: str,
    projection: osr.SpatialReference,
    curve_type: CurveType = CurveType.POLYLINE,
  ) -> None:
    """Add a tagged curve to the domain boundary.

    Args:
        points: the curve control points [n,2]
        physical_tag: the curve physical tag
        projection: the points coordinate system
        curve_type: curve interpolation
    """
    curve = _Curve(points, physical_tag, curve_type, projection)
    self._curves.append(curve)

  def add_interior_curve(
    self,
    points: np.ndarray,
    physical_tag: str,
    projection: osr.SpatialReference,
    curve_type: CurveType = CurveType.POLYLINE,
  ) -> None:
    """Adds a tagged curve inside the domain. The curve
    is not part of the domain boundary and will be meshed on both sides.

    Args:
        points: the curve control points [n,2]
        physical_tag: the curve physical tag
        projection: the points coordinate system
        curve_type: curve interpolation
    """
    curve = _Curve(points, physical_tag, curve_type, projection)
    self._interior_curves.append(curve)

  def add_interior_points_shp(self, filename: str, physical_name_field: str = None) -> None:
    """Adds all points of a shape file as forced mesh points.

    Args:
        filename: path to a shapefile.
        physical_name_field: name of an attribute string field with the
            curves physical tags
    """
    self._add_shapefile(filename, physical_name_field, None, True, None)

  def add_interior_curves_shp(
    self,
    filename: str,
    physical_name_field: str = None,
    curve_type: CurveType = CurveType.POLYLINE,
  ) -> None:
    """Adds all lines, polylines and polygons of a shape file as forced
    interior mesh lines

    Args:
        filename: path to a shapefile.
        physical_name_field: name of and attribute string field with the
             curves physical tags
        curve_type: curves interpolation
    """
    self._add_shapefile(filename, physical_name_field, True, False, curve_type)

  def add_boundary_curves_shp(
    self,
    filename: str,
    physical_name_field: str = None,
    curve_type: CurveType = CurveType.POLYLINE,
  ) -> None:
    """Adds all lines, polylines and polygons of a shapefile as domain
    boundaries

    Args:
        filename: path to a shapefile.
        physical_name_field: name of an attribute string field containing
            the curves physical tags
        curve_type: curves interpolation
    """
    self._add_shapefile(filename, physical_name_field, False, False, curve_type)


# Here, not at the top: gmsh.py imports this module for Domain and CurveType.
from .gmsh import _curve_sample  # noqa: E402


def _point_tag_lists(point_ids, tags, n_points: int) -> np.ndarray:
  """The tags of each point, as the linked lists that the C library reads.

  Args:
      point_ids: for each sampled point, the index of its (merged) point.
      tags: for each sampled point, the integer tag of its curve.
      n_points: the number of merged points.
  Returns:
      A flat int array of (tag, next) pairs. Pair i < n_points is the first tag of point i (-1 if
      it has none); `next` is the index of the pair holding its next tag, or -1. Points where
      several curves meet have several tags.
  """
  lists = [[-1, -1] for _ in range(n_points)]
  for i, tag in zip(point_ids, tags):
    if lists[i][0] == -1:
      lists[i][0] = tag
      continue
    last = i
    found = lists[last][0] == tag
    while not found and lists[last][1] != -1:
      last = lists[last][1]
      found = lists[last][0] == tag
    if not found:
      lists[last][1] = len(lists)
      lists.append([tag, -1])
  return np.array(lists, dtype=np.int32).reshape([-1])


def coarsen_boundaries(
  domain: Domain,
  x0: tuple,
  x0_projection: osr.SpatialReference,
  mesh_size: MeshSizeCallback,
) -> Domain:
  """Creates a new Domain with the same projection and coarsened
  boundaries.

  The boundary curves are replaced by simpler ones, with points about one mesh size apart, that
  bound the part of the domain reachable from `x0` without going through a passage narrower than
  the mesh size: features smaller than the size (small islands, inlets, narrow channels and
  whatever lies behind them) disappear. Interior curves and points are not kept.

  To keep a basin that lies behind a passage narrower than the mesh size (e.g. the Black Sea
  behind the Bosporus), give several points as `x0`, one in each basin: the result is then a
  domain made of several regions that do not touch, which `seamsh.gmsh.mesh` meshes as separate
  pieces. The boundary curves only say where the shore is, not on which side the water is: a
  point given on land gives a mesh of that land.

  How it works: every boundary curve is sampled at half the mesh size; the samples are
  triangulated (Delaunay); starting from the triangle that holds `x0`, the triangles are
  flood-filled through the edges longer than the mesh size (an edge between two shores closer
  than the size is not crossed, nor is an edge between two consecutive samples of a curve); the
  boundary of the filled triangles, thinned to one point per mesh size, is the new boundary. Its
  points are moved by a fiftieth of the size towards the inside.

  Args:
      domain: the domain to coarsen
      x0: the coordinates of one point inside the domain, or a sequence of such points.
      x0_projection: the coordinates system of x0.
      mesh_size: a function returning the desired mesh element size for given
          coordinates
  Returns:
      The coarsened domain: closed boundaries made of POLYLINE curves, with the physical tags of
      the original curves.
  Raises:
      ValueError: if a point of `x0` is outside the convex hull of the boundary points, or the
          mesh size is not positive.
  """
  logger.info("Coarsen boundaries")
  x0 = np.atleast_2d(np.asarray(x0, dtype=float))
  seed_points = _tools.project_points(x0, x0_projection, domain._projection)
  sampled = []
  sample_tags = []
  tag_number = {}  # physical tag (str) -> integer tag, from 1, for the C library

  # Two samples per mesh size: the edge between two consecutive samples of a curve is then
  # shorter than the size, and the flood fill does not cross the curve.
  def half_mesh_size(x, projection):
    return mesh_size(x, projection) * 0.5

  for curve in _tools.progress(domain._curves, desc="(seamsh) Sampling curves for coarsening"):
    samples = _curve_sample(curve, half_mesh_size, domain._projection)
    sampled.append(samples)
    number = tag_number.setdefault(curve.tag, len(tag_number) + 1)
    sample_tags.append(np.full(samples.shape[0], number, dtype=np.int32))
  logger.info("%d curves sampled", len(domain._curves))
  x = np.vstack(sampled)
  x, unique_id, _ = _generate_unique_points(x)
  tag_lists = _point_tag_lists(unique_id, np.concatenate(sample_tags), x.shape[0])
  x = np.copy(x[:, :2])
  # Move the points by a tiny random amount, to avoid cocircular points in the triangulation.
  # A generator of our own, seeded for reproducibility: the caller's global random state is left
  # alone. (RandomState(0) gives the numbers that np.random.seed(0) used to give here.)
  eps = (np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True)) * 1e-12
  x = x + np.random.RandomState(0).random_sample(x.shape) * eps
  logger.info("Delaunay mesh of sampled points")
  triangulation = Delaunay(x)
  logger.info("Extract boundaries")
  seeds = triangulation.find_simplex(seed_points)
  for point, seed in zip(x0, seeds):
    if seed == -1:
      raise ValueError(
        f"Point {tuple(point.tolist())} of x0 is outside the domain (outside the convex hull of the boundary points)"
      )
  sizes = mesh_size(x, domain._projection)
  points, lines = _tools.gen_boundaries(x, tag_lists, triangulation.simplices, seeds, sizes)

  coarse = Domain(domain._projection)
  tag_name = {number: name for name, number in tag_number.items()}
  start = 0
  for end in np.where(lines == -1)[0]:
    # One curve: its tag, the indices of its points, -1.
    coarse.add_boundary_curve(
      points[lines[start + 1 : end]],
      tag_name[lines[start]],
      domain._projection,
      curve_type=CurveType.POLYLINE,
    )
    start = end + 1
  return coarse
