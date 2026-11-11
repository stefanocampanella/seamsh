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

"""Meshing with gmsh, and conversions of the meshes.

gmsh keeps its state in the process: the caller initializes it (`gmsh.initialize()`), `mesh` works
on the current model, and the caller writes the file (`gmsh.write`).
"""

import logging
import os
import struct
import uuid

import gmsh
import numpy as np
import tqdm.contrib
from osgeo import ogr, osr

from . import _tools
from . import geometry as _geometry

try:
    import xarray as xr

    xarray_available = True
except ImportError:
    xarray_available = False

__all__ = ["mesh", "convert_to_gis", "convert_to_ugrid_mesh_file", "merge_meshes", "reproject"]

logger = logging.getLogger(__name__)


def _gmsh_curve_geo(curve_type: _geometry.CurveType, pointsid) -> list:
    """Adds a curve through the points `pointsid` (indices in Domain._points, i.e. gmsh point tags
    minus one) to the gmsh model and returns the tags of its gmsh curves (one, or one per segment
    for a STRICTPOLYLINE)."""
    pts = [i + 1 for i in pointsid]
    if curve_type == _geometry.CurveType.SPLINE:
        return [gmsh.model.geo.addSpline(pts)]
    elif curve_type == _geometry.CurveType.BSPLINE:
        return [gmsh.model.geo.addBSpline(pts)]
    elif curve_type == _geometry.CurveType.STRICTPOLYLINE:
        pairs = zip(pts[:-1], pts[1:])
        return [gmsh.model.geo.addLine(p0, p1) for p0, p1 in pairs]
    elif curve_type == _geometry.CurveType.POLYLINE:
        return [gmsh.model.geo.addPolyline(pts)]
    else:
        raise ValueError(f"unknown curve_type '{curve_type}'")


def _check_sizes(size):
    """Sampling refines a curve until its segments are shorter than the size: a size that is zero,
    negative or not a number would make it loop forever, or stop without notice."""
    if not np.all(size > 0):
        raise ValueError("The mesh size callback must return positive numbers")
    return size


def _curve_sample_gmsh_tag(tag: int, lc, projection: osr.SpatialReference) -> tuple:
    """Samples gmsh curve `tag` so that consecutive points are no farther apart than the size.

    Starting from five points, a point is added in the middle (in the parameter of the curve) of
    every segment longer than the smaller size of its two ends, until none is: the spacing ends up
    between half the size and the size. (The article integrates 1/size along the curve; this
    bisection needs only point evaluations of the size.)

    Args:
        tag: the gmsh curve, in the current model.
        lc: size callback, called with points in the coordinates of the gmsh model and `projection`.
    Returns:
        The points [n, 2], their parameters on the curve [n] and the sizes there [n].
    """
    bounds = gmsh.model.getParametrizationBounds(1, tag)
    xi = np.linspace(bounds[0][0], bounds[1][0], 5)
    x = gmsh.model.getValue(1, tag, xi).reshape([-1, 3])
    size = _check_sizes(lc(x, projection))
    while True:
        dist = np.linalg.norm(x[1:, :] - x[:-1, :], axis=1)
        target = np.minimum(size[1:], size[:-1])
        too_long = target < dist
        if not np.any(too_long):
            break
        new_xi = (xi[1:][too_long] + xi[:-1][too_long]) / 2
        new_x = gmsh.model.getValue(1, tag, new_xi).reshape([-1, 3])
        new_size = _check_sizes(lc(new_x, projection))
        xi = np.hstack([xi, new_xi])
        order = np.argsort(xi)
        xi = xi[order]
        x = np.vstack([x, new_x])[order, :]
        size = np.hstack([size, new_size])[order]
    return x[:, :2], xi, size


class _lcproj:
    """A size callback seen from another projection.

    Curves are sampled in their own projection (e.g. longitude and latitude), while sizes are
    lengths in the projection of the domain. This wraps a size callback `lc` of the domain so that
    it takes points in the curve's projection `proj` and returns sizes in the units of `proj`: the
    size divided by the local scale of the transformation, as for the stereographic plane in
    Lambrechts et al. 2008, section 3.1 (exact for conformal transformations).
    """

    def __init__(self, lc, proj: osr.SpatialReference):
        self.lc = lc
        self.projection = proj

    def __call__(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
        """
        Args:
            x: points in the curve's projection.
            projection: the projection of the domain, in which `lc` is called.
        """
        xlc = _tools.project_points(x, self.projection, projection)
        slc = self.lc(xlc, projection)
        if self.projection.IsSame(projection):
            return slc
        # Local scale of the projection, by finite differences: how far the projected point moves
        # when the point moves by (eps, eps). The step is 1e-8 for coordinates up to about a
        # thousand (degrees), and grows in proportion beyond (metres), where 1e-8 would be lost in
        # the rounding of the coordinates.
        eps = max(1e-8, 1e-11 * float(np.abs(x).max()))
        x2 = np.copy(x)
        x2[:, : projection.GetAxesCount()] += eps
        xlc2 = _tools.project_points(x2, self.projection, projection)
        enorm = np.hypot(eps, eps)
        h = np.linalg.norm(xlc2 - xlc, axis=1) / enorm
        return slc / h


def _curve_sample(curve, lc, projection: osr.SpatialReference) -> np.ndarray:
    """Points [n, 2] along `curve`, in `projection`, no farther apart than the size given by the
    callback `lc` (called with points in `projection`)."""
    # The curve is built in a temporary gmsh model; the caller's model is made current again.
    current_model = gmsh.model.getCurrent()
    gmsh.model.add(str(uuid.uuid4()))
    try:
        tags = [gmsh.model.geo.addPoint(*x, 0) - 1 for x in curve.points[:-1, :]]
        if np.linalg.norm(curve.points[0, :] - curve.points[-1, :]) < 1e-8:
            tags.append(tags[0])  # a closed curve
        else:
            tags.append(gmsh.model.geo.addPoint(*curve.points[-1, :], 0) - 1)
        curve_tags = _gmsh_curve_geo(curve.curve_type, tags)
        gmsh.model.geo.synchronize()
        samples = _curve_sample_gmsh_tag(curve_tags[0], _lcproj(lc, curve.projection), projection)[
            0
        ]
    finally:
        gmsh.model.remove()
        gmsh.model.setCurrent(current_model)
    return _tools.project_points(samples, curve.projection, projection)


def _create_gmsh_geometry(domain: _geometry.Domain) -> None:
    """Builds the geometry of `domain` (after `_build_topology`) in the current gmsh model: its
    points, curves, one plane surface per region of the domain (bounded by the curve loops of the
    region), the interior curves and points embedded in the surface that holds them, and the
    physical groups: one per curve or point tag, and "domain" for the surfaces."""
    logger.info("Build gmsh model")
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.LcIntegrationPrecision", 1e-5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromParametricPoints", 1)
    curve_groups = {}  # physical tag -> gmsh curves
    for i, x in enumerate(domain._points):
        gmsh.model.geo.addPoint(*x, 0, tag=i + 1)
    loop_tags = []
    for loop in domain._curveloops:
        loop_curves = []
        for curve_index, forward in loop:
            curve = domain._curves[curve_index]
            tags = _gmsh_curve_geo(curve.curve_type, curve.pointsid)
            curve_groups.setdefault(curve.tag, []).extend(tags)
            # gmsh takes a curve backwards when its tag is negative.
            loop_curves.extend(tags if forward else [-t for t in reversed(tags)])
        loop_tags.append(gmsh.model.geo.addCurveLoop(loop_curves))
    # One surface per region; in each, the first loop is the outer boundary, the others are holes.
    surfaces = [
        gmsh.model.geo.addPlaneSurface([loop_tags[i] for i in region]) for region in domain._regions
    ]

    def region_of(x, what):
        region = domain._region_of(x)
        if region is None:
            logger.warning("%s is outside the domain: embedded in its first region", what)
            region = 0
        return region

    embedded_curves = [[] for _ in surfaces]
    for curve in domain._interior_curves:
        tags = _gmsh_curve_geo(curve.curve_type, curve.pointsid)
        curve_groups.setdefault(curve.tag, []).extend(tags)
        # The region of the middle of the curve (its ends may be on the boundary).
        middle = len(curve.pointsid) // 2
        x = domain._points[curve.pointsid[middle - 1 : middle + 1]].mean(axis=0)
        embedded_curves[region_of(x, f"An interior curve tagged '{curve.tag}'")].extend(tags)
    point_groups = {}  # physical tag -> gmsh points
    embedded_points = [[] for _ in surfaces]
    for point in domain._interior_points:
        point_groups.setdefault(point.tag, []).append(point.pointid + 1)
        region = region_of(domain._points[point.pointid], f"A point tagged '{point.tag}'")
        embedded_points[region].append(point.pointid + 1)
    gmsh.model.geo.synchronize()
    for surface, curves, points in zip(surfaces, embedded_curves, embedded_points):
        gmsh.model.mesh.embed(1, curves, 2, surface)
        gmsh.model.mesh.embed(0, points, 2, surface)
    for name, tags in curve_groups.items():
        tag = gmsh.model.addPhysicalGroup(1, tags)
        gmsh.model.setPhysicalName(1, tag, name)
    for name, tags in point_groups.items():
        tag = gmsh.model.addPhysicalGroup(0, tags)
        gmsh.model.setPhysicalName(0, tag, name)
    tag = gmsh.model.addPhysicalGroup(2, surfaces)
    gmsh.model.setPhysicalName(2, tag, "domain")


def _mesh_bgrid(
    domain: _geometry.Domain, mesh_size: _geometry.MeshSizeCallback, smoothness: float
) -> None:
    """Meshes the current model with the mesh size given to gmsh on a background mesh.

    The size is evaluated on the nodes of a triangulation of the bounding box of the domain:
    a 100 x 100 grid, refined where the size is small, on which the size is limited so that it grows
    by at most `smoothness` per unit of distance. gmsh interpolates the size in these triangles.
    """
    logger.info("Build mesh size field")
    x0 = np.min(domain._points, axis=0)
    x1 = np.max(domain._points, axis=0)
    d = np.max(x1 - x0) / 100
    x = np.arange(x0[0], x1[0] + d, d)
    y = np.arange(x0[1], x1[1] + d, d)
    xx, yy = np.meshgrid(x, y)
    xy = np.stack((xx.T, yy.T), axis=2).reshape(-1, 2)
    v = mesh_size(xy, domain._projection)
    poly = _tools.PolyMesh(x0, x1)
    poly.add_points(xy)
    while True:
        xyz = np.c_[xy.reshape(-1, 2), np.zeros((xy.size // 2, 1))]
        tri = poly.get_triangles()
        # The edges of the triangles, each once: pairs of nodes packed in one integer to sort them.
        edges = np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]
        edges.sort(axis=1)
        shift = 2**32
        edges = edges.astype(np.uint64)
        ekey = np.unique(edges[:, 0] * shift + edges[:, 1])
        edges = np.c_[ekey // shift, ekey % shift]
        dedges = xy[edges[:, 0]] - xy[edges[:, 1]]
        elength = np.hypot(dedges[:, 0], dedges[:, 1])
        # Smoothing: along each edge, the size at one end is at most the size at the other end plus
        # smoothness x length. Repeated until no size changes by 1 or more (an absolute tolerance,
        # in the units of the projection: fine for sizes in metres).
        while True:
            vo = v.copy()
            cond = v[edges[:, 0]] > v[edges[:, 1]] + elength * smoothness
            v[edges[cond, 0]] = v[edges[cond, 1]] + elength[cond] * smoothness
            cond = v[edges[:, 1]] > v[edges[:, 0]] + elength * smoothness
            v[edges[cond, 1]] = v[edges[cond, 0]] + elength[cond] * smoothness
            if np.max(np.abs(v - vo)) < 1:
                break
        # Refinement: an edge more than twice as long as the smaller size at its ends is cut in
        # two, so that the background mesh resolves the variations of the size.
        eminsize = np.minimum(v[edges[:, 0]], v[edges[:, 1]])
        tocut = np.where(eminsize * 2 < elength)[0]
        if tocut.size == 0:
            break
        newp = (xy[edges[tocut, 0]] + xy[edges[tocut, 1]]) / 2
        xy = np.r_[xy, newp]
        v = np.r_[v, mesh_size(newp, domain._projection)]
        poly.add_points(newp)
    # A gmsh view of scalar triangles ("ST": 9 coordinates and 3 values each), used as size field.
    xtri = xyz[tri]
    data = np.c_[xtri.swapaxes(1, 2).reshape(-1, 9), v.flatten()[tri]]
    view = gmsh.view.add("tri")
    gmsh.view.add_list_data(view, "ST", tri.shape[0], data.flatten())
    field = gmsh.model.mesh.field.add("PostView")
    gmsh.model.mesh.field.setNumber(field, "ViewTag", view)
    gmsh.model.mesh.field.setAsBackgroundMesh(field)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.LcIntegrationPrecision", 1e-5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromParametricPoints", 0)
    logger.info("Mesh with gmsh")
    gmsh.model.mesh.generate(2)
    # The size field and its view are internals: do not leave them in the caller's model, where
    # the view would be written to files with the mesh.
    gmsh.model.mesh.field.remove(field)
    gmsh.view.remove(view)


def _mesh_successive(
    domain: _geometry.Domain,
    mesh_size: _geometry.MeshSizeCallback,
    intermediate_file_name: str = None,
) -> None:
    """Meshes the current model without smoothing the mesh size: the size is set on the curves, the
    curves are meshed, then the surface is meshed with a uniform size and remeshed three times, each
    time with the size evaluated on the nodes of the previous mesh."""
    for curve in domain._curves:
        curve.mesh_size = mesh_size(curve.points, domain._projection)

    # 1D mesh
    curves = gmsh.model.getEntities(1)
    for dim, tag in tqdm.tqdm(curves, desc="(seamsh) Sample curves for mesh size"):
        _, xi, size = _curve_sample_gmsh_tag(
            tag, lambda x, p: mesh_size(x, p) / 2, domain._projection
        )
        gmsh.model.mesh.setSizeAtParametricPoints(dim, tag, xi, size * 2)
    logger.info("%d curve sampled", len(curves))
    logger.info("Generate 1D mesh")
    gmsh.model.mesh.generate(1)
    if intermediate_file_name is not None:
        if intermediate_file_name == "-":
            gmsh.fltk.run()
        else:
            gmsh.write(intermediate_file_name + "_1d.msh")
    # 2D mesh
    logger.info("Generate initial 2D mesh")
    _, x, _ = gmsh.model.mesh.getNodes()
    x = x.reshape([-1, 3])
    domain_size = np.max(x, axis=0) - np.min(x, axis=0)
    initlc = np.linalg.norm(domain_size) / 100
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromParametricPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", initlc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", initlc)
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e22)
    bg_field = gmsh.model.mesh.field.add("PostView")
    gmsh.model.mesh.field.setAsBackgroundMesh(bg_field)
    nadapt = 3
    for i in range(nadapt):
        logger.info("Generate refined 2D mesh (pass %d/%d)", i + 1, nadapt)
        node_tags, node_x, _ = gmsh.model.mesh.getNodes(-1, -1)
        node_x = node_x.reshape([-1, 3])
        nodes_map = {tag: i for i, tag in enumerate(node_tags)}
        sf_view = gmsh.view.add("mesh size field")
        node_lc = mesh_size(node_x, domain._projection)
        etypes, _, enodes = gmsh.model.mesh.getElements(2, -1)
        for etype, enode in zip(etypes, enodes):
            nn = 3 if etype == 2 else 4
            enode = np.array([nodes_map[t] for t in enode]).reshape([-1, nn])
            xnode = node_x[enode, :].swapaxes(1, 2).reshape([-1, nn * 3])
            data = np.column_stack([xnode, node_lc[enode]])
            gmsh.view.addListData(
                sf_view, "ST" if etype == 2 else "SQ", enode.shape[0], data.reshape([-1])
            )
        gmsh.model.mesh.field.setNumber(bg_field, "ViewTag", sf_view)
        if intermediate_file_name is not None:
            if intermediate_file_name == "-":
                gmsh.fltk.run()
            else:
                gmsh.view.write(sf_view, intermediate_file_name + "_2d_" + str(i) + ".pos")
        gmsh.model.mesh.generate(2)
        gmsh.view.remove(sf_view)
    gmsh.model.mesh.field.remove(bg_field)


def _drop_curves_outside_the_mesh():
    """Takes out of the physical groups the curves none of whose nodes belongs to a surface element.

    These are the outlines of islands much smaller than the mesh size: gmsh meshes the outline with
    one to three nodes, cannot use it as a hole, and covers the island with triangles. Left in their
    physical group, they would be written to the mesh file as boundary elements on nodes that no
    triangle uses. gmsh does not let us delete a curve on the boundary of a meshed surface (nor its
    mesh), but it writes only the elements of physical groups.
    """
    surface_nodes = gmsh.model.mesh.getElements(2)[2]
    used = set(np.concatenate(surface_nodes).tolist()) if surface_nodes else set()
    n_dropped = 0
    for dim, group in gmsh.model.getPhysicalGroups(1):
        curves = gmsh.model.getEntitiesForPhysicalGroup(dim, group)
        kept = []
        for curve in curves:
            curve_nodes = gmsh.model.mesh.getElements(dim, int(curve))[2]
            nodes = np.concatenate(curve_nodes).tolist() if curve_nodes else []
            if used.intersection(nodes):
                kept.append(int(curve))
        if len(kept) == len(curves):
            continue
        n_dropped += len(curves) - len(kept)
        name = gmsh.model.getPhysicalName(dim, group)
        gmsh.model.removePhysicalGroups([(dim, group)])
        if kept:
            gmsh.model.addPhysicalGroup(dim, kept, group)
            gmsh.model.setPhysicalName(dim, group, name)
    if n_dropped:
        logger.info("%d curves smaller than the mesh size left out of the mesh", n_dropped)


def _reproject(input_srs, output_srs):
    """Moves the nodes (and points) of the current model from `input_srs` (None: the projection
    stored in the model) to `output_srs`."""
    if input_srs is None:
        p = gmsh.model.get_attribute("Projection")
        if len(p) == 2 and p[0] == "WKT":
            input_srs = osr.SpatialReference()
            input_srs.ImportFromWkt(p[1])
    ntags, nx, _ = gmsh.model.mesh.get_nodes()
    nx = nx.reshape(-1, 3)
    nx = _tools.project_points(nx, input_srs, output_srs)
    for i, x in zip(ntags, nx):
        if len(x) == 3:
            gmsh.model.mesh.set_node(i, [x[0], x[1], x[2]], [])
        else:
            gmsh.model.mesh.set_node(i, [x[0], x[1], 0], [])

    for _, tag in gmsh.model.get_entities(0):
        _, x, _ = gmsh.model.mesh.get_nodes(0, tag)
        if x.size == 0:
            gmsh.model.remove_entities([(0, tag)])
        else:
            if len(x) == 3:
                gmsh.model.set_coordinates(tag, x[0], x[1], x[2])
            else:
                gmsh.model.set_coordinates(tag, x[0], x[1], 0)


def mesh(
    domain: _geometry.Domain,
    mesh_size: _geometry.MeshSizeCallback,
    version: float = 4.1,
    intermediate_file_name: str = None,
    smoothness=0.3,
    output_srs: osr.SpatialReference = None,
    transfinite_curves: dict = None,
    quad: bool = False,
) -> None:
    """Calls gmsh to generate a mesh from a geometry and a mesh size callback, it assumes that the
    gmsh is initialized and use the current model.

    The mesh is left in the current gmsh model, in the coordinate system `output_srs` (stored as
    WKT in the model attribute "Projection"); write it with `gmsh.write`. A domain made of several
    regions that do not touch (e.g. from `coarsen_boundaries` with several points) gives a mesh in
    as many pieces. Boundary and interior
    curves, and forced points, are in physical groups named by their physical tags, the surface in
    the physical group "domain". Curves that end up in no element of the surface (the outlines of
    islands much smaller than the mesh size) are left out of their physical group, hence of the
    file.

    Args:
        domain: the input geometry
        mesh_size: callable prescribing the mesh element size; it is called with points in the
            coordinate system of the domain, and must return positive sizes.
        version: msh file version (typically 2.0 or 4.0)
        intermediate_file_name: if not None, save intermediate meshes to those
            files for debugging purpose (suffixes and extensions will be
            appended), if == "-", an interactive gmsh graphical window will pop
            up after each meshing step. Only used when smoothness is 0.
        smoothness: Maximum gradation of the mesh size, any positive value
            is valid but a value in the range [0.1,0.5] is recommended. With 0, the mesh size is
            not smoothed and another meshing strategy is used (successive adaptations).
        output_srs : coordinate system of the output file (if None, the
            coordinate system of the domain is used).
        transfinite_curves : dict(str, int) prescribe a fixed number of mesh nodes
            on curves with given physical tag.
        quad: if True, generate quadrangles instead of triangles. This a achieved by
            generating a quad-dominant mesh and then splitting all elements (quads and
            triangles) into quads. So the resulting mesh size is half the prescribed one.
    """
    logger.info("Generate mesh")
    if transfinite_curves is None:
        transfinite_curves = {}
    domain._build_topology()
    _create_gmsh_geometry(domain)
    for dim, tag in gmsh.model.get_physical_groups(1):
        pname = gmsh.model.getPhysicalName(dim, tag)
        if pname in transfinite_curves:
            for etag in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                gmsh.model.mesh.set_transfinite_curve(etag, transfinite_curves[pname])
    if quad:
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)

    if smoothness > 0:
        _mesh_bgrid(domain, mesh_size, smoothness)
    else:
        _mesh_successive(domain, mesh_size, intermediate_file_name)

    if quad:
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
        gmsh.model.mesh.refine()

    _drop_curves_outside_the_mesh()
    if output_srs is not None:
        _reproject(domain._projection, output_srs)
    else:
        output_srs = domain._projection
    if output_srs is not None:
        gmsh.model.set_attribute("Projection", ["WKT", output_srs.ExportToWkt()])

    gmsh.option.setNumber("Mesh.MshFileVersion", version)


def reproject(
    input_filename: str,
    input_srs: osr.SpatialReference,
    output_filename: str,
    output_srs: osr.SpatialReference,
    output_version: float = 4.1,
):
    """Change the coordinate of an existing mesh.

    Args:
        input_filename: input mesh file (any format readable by gmsh).
        input_srs : coordinate system of the input mesh. If None, the model projection is used.
        output_filename: output mesh file (.msh)
        output_srs : coordinate system of the output file (if None, the
            coordinate system of the domain is used).
        output_version: msh file version (typically 2.0 or 4.0)
    """
    gmsh.model.add(str(uuid.uuid4()))
    gmsh.open(input_filename)
    _reproject(input_srs, output_srs)
    gmsh.option.setNumber("Mesh.MshFileVersion", output_version)
    if output_srs is not None:
        gmsh.model.set_attribute("Projection", ["WKT", output_srs.ExportToWkt()])
    gmsh.write(output_filename)
    gmsh.model.remove()


def convert_to_gis(
    input_filename: str, projection: osr.SpatialReference, output_filename: str
) -> None:
    """Converts a triangular gmsh mesh into shapefile or geopackage.

    Args:
        input_filename : any mesh file readable by gmsh (typically
            a .msh file)
        projection: the projection assigned to the output layer, mesh
            files do not store any projection so neither checks nor
            re-projections are performed (if none mesh projection will be used).
        output_filename : shape file (.shp) or geopackage (.gpkg) file
    """
    logger.info('Convert "%s" into "%s"', input_filename, output_filename)
    gmsh.model.add(str(uuid.uuid4()))
    gmsh.open(input_filename)
    if output_filename.endswith(".shp"):
        shpdriver = ogr.GetDriverByName("ESRI Shapefile")
    elif output_filename.endswith(".gpkg"):
        shpdriver = ogr.GetDriverByName("GPKG")
    else:
        raise ValueError("Unknown file extension '" + output_filename + "'")
    if os.path.exists(output_filename):
        shpdriver.DeleteDataSource(output_filename)
    out_data_source = shpdriver.CreateDataSource(output_filename)
    if projection is None:
        p = gmsh.model.get_attribute("Projection")
        if len(p) == 2 and p[0] == "WKT":
            projection = osr.SpatialReference()
            projection.ImportFromWkt(p[1])
    out_layer = out_data_source.CreateLayer("mesh", geom_type=ogr.wkbPolygon, srs=projection)
    out_layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    id_field = out_layer.FindFieldIndex("id", 1)

    tri_i, tri_n = gmsh.model.mesh.getElementsByType(2)
    node_tags, x, _ = gmsh.model.mesh.getNodesByElementType(2)
    x = x.reshape([-1, 3])
    nodes_map = {tag: i for i, tag in enumerate(node_tags)}
    tri_n = np.array([nodes_map[t] for t in tri_n]).reshape([-1, 3])

    tri_n = np.column_stack([tri_n, tri_n[:, [0]]])
    header = struct.pack("<biii", 1, 3, 1, 4)
    out_data_source.StartTransaction()
    for i, tn in zip(tri_i, tri_n):
        wkb = header + x[tn, :2].astype("<f8").tobytes()
        feat = ogr.Feature(out_layer.GetLayerDefn())
        feat.SetGeometryDirectly(ogr.CreateGeometryFromWkb(wkb))
        feat.SetField(id_field, int(i))
        out_layer.CreateFeature(feat)
        feat = None
    out_data_source.CommitTransaction()
    gmsh.model.remove()


def convert_to_ugrid_mesh_file(
    input_filename: str,
    dico_var: dict,
    projection: osr.SpatialReference,
    output_filename: str,
) -> None:
    """Converts a triangular gmsh mesh into netcdf ugrid mesh file.

    Args:
        input_filename : any mesh file readable by gmsh (typically
            a .msh file)
        dico_var : a dictionary of variable name with all required information
        projection: the projection assigned to the mnt layer
        output_filename : netcdf ugrid mesh file (.nc)
    """
    if not xarray_available:
        raise ValueError("The xarray python library is required to write ugrid files.")

    logger.info('Convert "%s" into "%s"', input_filename, output_filename)
    gmsh.model.add(str(uuid.uuid4()))
    gmsh.open(input_filename)

    _, tri_n = gmsh.model.mesh.getElementsByType(2)
    tri_n = tri_n.reshape([-1, 3])

    _, nodes, _ = gmsh.model.mesh.getNodes()
    nodes = nodes.reshape([-1, 3])
    x = nodes[:, 0]
    y = nodes[:, 1]

    element = np.subtract(tri_n, 1)
    mesh_topology = np.array([-2147483647], dtype="int32")

    ds = xr.Dataset()

    ds.coords["longitude"] = (("node"), np.double(x))
    ds["longitude"].attrs = {
        "long_name": "longitude",
        "standard_name": "longitude",
        "units": "m",
        "positive": "east",
    }

    ds.coords["latitude"] = (("node"), np.double(y))
    ds["latitude"].attrs = {
        "long_name": "latitude",
        "standard_name": "latitude",
        "units": "m",
        "positive": "north",
    }

    ds["element"] = (("nele", "nvertex"), element)
    ds["element"].attrs = {
        "long_name": "element",
        "standard_name": "face_node_connectivity",
        "units": "nondimensional",
        "start_index": 0,
    }

    ds["mesh_topology"] = ("single", mesh_topology)
    ds["mesh_topology"].attrs = {
        "long_name": "mesh topology",
        "standard_name": "mesh_topology",
        "node_coordinates": "longitude latitude",
        "face_node_connectivity": "element",
        "cf_role": "mesh_topology",
        "topology_dimension": 2,
    }

    for var in dico_var:
        var_val = dico_var[var]["callable_func"](nodes, projection)
        ds[var] = (("node"), var_val)
        ds[var].attrs = {
            "long_name": dico_var[var]["long_name"],
            "standard_name": dico_var[var]["standard_name"],
            "location": "node",
            "mesh": "mesh_topology",
            "units": dico_var[var]["units"],
        }

    encoding = {v: {"zlib": True, "complevel": 4} for v in dico_var}
    ds.attrs = {
        "title": gmsh.model.getCurrent(),
        "institution": "Institution Name",
        "source": "UCL(SEAMSH) & BRGM",
        "references": "REF",
        "Conventions": "CF-1.6, UGRID-1.0",
    }

    ds.to_netcdf(path=output_filename, format="NETCDF4", encoding=encoding)

    gmsh.model.remove()


def merge_meshes(filename_orig: str, filename_dest: str, filename_out: str, seamlines: dict):
    """Merge two meshes. No coordinate conversion are done. Projection
        should be identical on both meshes.

    Args:
        filename_orig : any mesh file readable by gmsh (typically
            a .msh file)
        filename_dest : any mesh file readable by gmsh (typically
            a .msh file).
        filename_out : any mesh file writable by gmsh (typically
            a .msh file)
        seamlines : dictionary of physicals tag of matching lines
            from orig mesh to dest mesh. There should be the same
            number of nodes on both side. Each node of orig will
            be mapped to its closest node on dest.
    """

    def get_ordered_nodes_for_physical_line(model, ptag):
        gmsh.model.set_current(model)
        for _, i in gmsh.model.get_physical_groups(1):
            if gmsh.model.get_physical_name(1, i) == ptag:
                entities = gmsh.model.get_entities_for_physical_group(1, i)
                if len(entities) != 1:
                    raise ValueError(f"Multiple entities on physical tag {ptag} on mesh {model}")
                n_, x_, _ = gmsh.model.mesh.get_nodes(1, entities[0], True, False)
                x_ = x_.reshape(-1, 3)
                xmap = dict(zip(n_, x_))
                _, enodes = gmsh.model.mesh.get_elements_by_type(1, entities[0])
                enodes = enodes.reshape(-1, 2)
                nextn = dict(enodes)
                pfrom = gmsh.model.get_boundary([(1, entities[0])])[0][1]
                points = [pfrom]
                while points[-1] in nextn:
                    points.append(nextn[points[-1]])
                return (
                    entities[0],
                    np.array(points, np.uint64),
                    np.array([xmap[i] for i in points]),
                )
        raise ValueError(f"Physical tag '{ptag}' not found in mesh '{model}'")

    gmsh.open(filename_dest)
    model_dest = gmsh.model.get_current()
    itag = gmsh.model.mesh.get_max_node_tag()
    gmsh.open(filename_orig)
    model_orig = gmsh.model.get_current()

    entitymap = {}
    nodemap = {}
    ignoreentities = []
    for tagfrom, tagto in seamlines.items():
        _, nodesto, xto = get_ordered_nodes_for_physical_line(model_dest, tagto)
        entfrom, nodesfrom, xfrom = get_ordered_nodes_for_physical_line(model_orig, tagfrom)
        if nodesto.size != nodesfrom.size:
            raise ValueError(f"Nodes do not match for tags {tagfrom}, {tagto}")
        if np.linalg.norm(xto - xfrom) > np.linalg.norm(xto - xfrom[::-1]):
            nodesto = nodesto[::-1]
        ignoreentities.append((1, entfrom))
        for i, j in zip(nodesto, nodesfrom):
            logger.debug("seam %s -> %s: node %d on node %d", tagfrom, tagto, j, i)
            nodemap[j] = i

    for entity_dim in range(4):
        gmsh.model.set_current(model_orig)
        for dim, tag in gmsh.model.get_entities(entity_dim):
            gmsh.model.set_current(model_orig)
            ntag, ncoord, npcoord = gmsh.model.mesh.get_nodes(dim, tag)
            etypes, _, enodes = gmsh.model.mesh.get_elements(dim, tag)
            keep = [t not in nodemap for t in ntag]
            n = np.sum(keep)
            dntag = np.uint64(itag) + np.arange(n, dtype=ntag.dtype) + 1
            itag += n
            if (dim, tag) in ignoreentities:
                continue
            if len(dntag) == 0 and len(enodes) == 0:
                continue
            gmsh.model.set_current(model_dest)
            dtag = gmsh.model.add_discrete_entity(dim)
            entitymap[(dim, tag)] = dtag
            gmsh.model.mesh.add_nodes(
                dim,
                dtag,
                dntag,
                ncoord.reshape(-1, 3)[keep].reshape(-1),
                npcoord.reshape(-1, dim)[keep].reshape(-1) if npcoord.size != 0 else [],
            )
            for t0, t1 in zip(ntag[keep], dntag):
                nodemap[t0] = t1
            dntag = [[nodemap[n] for n in nodes] for nodes in enodes]
            for et, dnt in zip(etypes, dntag):
                gmsh.model.mesh.add_elements_by_type(dtag, et, [], dnt)

    gmsh.model.set_current(model_orig)
    for dim, tag in gmsh.model.get_physical_groups():
        gmsh.model.set_current(model_orig)
        name = gmsh.model.get_physical_name(dim, tag)
        oe = []
        for ie in gmsh.model.get_entities_for_physical_group(dim, tag):
            if (dim, ie) in entitymap:
                oe.append(entitymap[(dim, ie)])
        if len(oe) != 0:
            gmsh.model.set_current(model_dest)
            gmsh.model.add_physical_group(dim, oe, name=name)

    gmsh.model.set_current(model_orig)
    gmsh.model.remove()
    gmsh.write(filename_out)
    gmsh.model.set_current(model_dest)
    gmsh.model.remove()
