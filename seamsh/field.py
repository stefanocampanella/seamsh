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

"""Building blocks for mesh size fields: callables `f(x[n, 2], projection) -> [n]` that users
combine in their own size function (see Lambrechts et al., "Multiscale mesh generation on the
sphere", Ocean Dynamics 58, 2008, section 3.1: sizes from the distance to the shore and from the
bathymetry, the final size being the minimum of several fields)."""

import logging
from itertools import chain

import numpy as np
from osgeo import gdal, osr
from scipy.spatial import cKDTree

from . import _tools
from .geometry import Domain as _Domain
from .geometry import _points_in_polygon
from .gmsh import _curve_sample

__all__ = ["Distance", "Raster", "Inpoly"]

logger = logging.getLogger(__name__)


class Distance:
    """Callable evaluating the distance to a set of discretized curves.

    The curves are discretized as sets of points, then the distance to the
    closest point is computed.
    """

    def __init__(
        self,
        domain: _Domain,
        sampling: float,
        tags: list[str] = None,
        projection: osr.SpatialReference = None,
    ):
        """
        Args:
            domain: a Domain object containing the set of curves
            sampling: the interval between two consecutive sampling points.
            tags: List of physical tags specifying the curve from the domain.
                if None, all curves are taken into account.
            projection: coordinates system used to search for nearest neighbours
                if None, the domain projection is used.
        """
        self._projection = domain._projection if projection is None else projection
        logger.info("Create distance field")
        points = []
        all_curves_iter = chain(domain._curves, domain._interior_curves)

        def size(x, proj):
            return np.full([x.shape[0]], sampling)

        icurve = 0
        for curve in _tools.progress(
            all_curves_iter, desc="(seamsh) Sampling features for distance computation"
        ):
            if (tags is None) or (curve.tag in tags):
                points.append(_curve_sample(curve, size, self._projection))
                icurve += 1
        for point in chain(domain._interior_points):
            if (tags is None) or (point.tag in tags):
                points.append(point.x)
        logger.info("%d features sampled", icurve)
        points = np.vstack(points)
        logger.info("Build KDTree with %d points", points.shape[0])
        self._tree = cKDTree(points)

    def __call__(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
        """Computes the distance between each point of x and the curves.

        Args:
            x: the points [n,2]
            projection: the coordinate system of the points; they are
                reprojected to the coordinate system of the field.
        Returns:
            The distance, in the unit of the coordinate system of the field
            (that of the domain by default). [n]
        """
        x = _tools.project_points(x, projection, self._projection)
        return self._tree.query(x)[0]


class Raster:
    """Callable to evaluate a raster field (the first band of a GDAL dataset).

    The value at a point is that of the pixel that contains it, without interpolation; points
    outside the raster take the value of the nearest pixel of its border.
    """

    def __init__(self, src_ds: gdal.Dataset):
        """
        Args:
            src_ds: GDAL dataset containing the raster field.
        """
        self._geo_matrix = src_ds.GetGeoTransform()
        self._data = src_ds.GetRasterBand(1).ReadAsArray()
        self._projection = osr.SpatialReference()
        self._projection.ImportFromWkt(src_ds.GetProjection())
        if int(gdal.__version__.split(".")[0]) >= 3:
            order = osr.OAMS_TRADITIONAL_GIS_ORDER
            self._projection.SetAxisMappingStrategy(order)
        if self._projection.IsSame(osr.SpatialReference()):
            logger.warning("Raster used has no spatial reference. Be sure to define one manually.")

    def __call__(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
        """Evaluates the field value on each point of x.

        Args:
            x: the points [n,2]
            projection: the coordinate system of the points; they are
                reprojected to the coordinate system of the raster.
        Returns:
            The field value on points x. [n]
        """
        x = _tools.project_points(x, projection, self._projection)
        gm = self._geo_matrix
        lon, lat = x[:, 0], x[:, 1]
        det = gm[5] * gm[1] - gm[2] * gm[4]
        pixx = ((lon - gm[0]) * gm[5] - (lat - gm[3]) * gm[2]) / det
        pixy = ((lat - gm[3]) * gm[1] - (lon - gm[0]) * gm[4]) / det
        pixy = np.clip(pixy.astype(int), 0, self._data.shape[0] - 1)
        pixx = np.clip(pixx.astype(int), 0, self._data.shape[1] - 1)
        return self._data[pixy, pixx]


class Inpoly:
    """Callable evaluating the membership to a set of polygons: the closed boundary curves of a
    domain (each polygon is the polyline through the control points of one curve)."""

    def __init__(self, domain: _Domain, tags: list[str] = None):
        """
        Args:
            domain: a Domain object containing the set of curves
            tags: List of physical tags specifying the curve from the domain.
                if None, all curves are taken into account.
        """
        logger.info("Initialisation Inpoly")
        self._projection = domain._projection

        domain._build_topology()
        self._area = []
        for loop in domain._curveloops:
            for curve_index, forward in loop:
                curve = domain._curves[curve_index]
                if (tags is None) or (curve.tag in tags):
                    ptsid = list(curve.pointsid if forward else np.flip(curve.pointsid))
                    if ptsid[-1] != ptsid[0]:
                        raise ValueError(
                            f"Inpoly needs closed curves: a curve tagged '{curve.tag}' is open"
                        )
                    self._area.append(domain._points[ptsid])

    def __call__(self, x: np.ndarray, projection: osr.SpatialReference) -> np.ndarray:
        """Indicate for each point of x if it belongs to a set of polygons.

        Args:
            x: the points [n,2]
            projection: the coordinate system of the points; they are
                reprojected to the coordinate system of the domain.
        Returns:
            True if the point belongs to a polygon / False otherwise. [n]
        """
        x = _tools.project_points(x, projection, self._projection)

        inpoly = np.full(x.shape[0], False)
        for polygon in self._area:
            inpoly |= _points_in_polygon(x, polygon)
        return inpoly
