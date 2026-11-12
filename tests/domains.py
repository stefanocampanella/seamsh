# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: GPL-3.0-or-later

"""Synthetic geometries and mesh measurements shared by the tests.

Coordinates are metres in a projected reference system (UTM 31N); the shapes are about 100 km across.
"""

import pathlib

import gmsh
import numpy as np
from osgeo import osr

from seamsh.geometry import Domain

osr.UseExceptions()

DATA = pathlib.Path(__file__).parent / "data"
NE_110M = DATA / "ne_110m_coastline" / "ne_110m_coastline.shp"


def utm() -> osr.SpatialReference:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32631)
    return srs


def uniform(size: float):
    """A mesh-size callback returning `size` everywhere."""

    def mesh_size(x, projection):
        return np.full(x.shape[0], float(size))

    return mesh_size


def polygon(corners, spacing: float) -> np.ndarray:
    """A closed polyline through `corners`, with points about every `spacing` along each side."""
    corners = [*corners, corners[0]]
    points = []
    for a, b in zip(corners[:-1], corners[1:]):
        n = max(1, int(round(np.hypot(b[0] - a[0], b[1] - a[1]) / spacing)))
        for t in np.linspace(0, 1, n, endpoint=False):
            points.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    points.append(points[0])
    return np.array(points, float)


def rectangle(x0, y0, x1, y1, spacing: float) -> np.ndarray:
    return polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], spacing)


def square_with_island(island: float = 20e3) -> Domain:
    """A 100 km square ("coast") with a centred square island ("island") of side `island`."""
    srs = utm()
    domain = Domain(srs)
    domain.add_boundary_curve(rectangle(0, 0, 100e3, 100e3, 5e3), "coast", srs)
    lo, hi = 50e3 - island / 2, 50e3 + island / 2
    domain.add_boundary_curve(rectangle(lo, lo, hi, hi, island / 8), "island", srs)
    return domain


def two_basins(half_width: float = 1e3) -> Domain:
    """Two 40 km square basins joined by a 20 km long channel of width 2 x `half_width`.

    The left basin holds (20 km, 20 km), the right one (80 km, 20 km).
    """
    srs = utm()
    lo, hi = 20e3 - half_width, 20e3 + half_width
    corners = [
        (0, 0),
        (40e3, 0),
        (40e3, lo),
        (60e3, lo),
        (60e3, 0),
        (100e3, 0),
        (100e3, 40e3),
        (60e3, 40e3),
        (60e3, hi),
        (40e3, hi),
        (40e3, 40e3),
        (0, 40e3),
    ]
    domain = Domain(srs)
    domain.add_boundary_curve(polygon(corners, 4e3), "coast", srs)
    return domain


def two_squares() -> Domain:
    """Two separate 40 km squares ("left", "right"), 20 km apart: a disconnected domain."""
    srs = utm()
    domain = Domain(srs)
    domain.add_boundary_curve(rectangle(0, 0, 40e3, 40e3, 5e3), "left", srs)
    domain.add_boundary_curve(rectangle(60e3, 0, 100e3, 40e3, 5e3), "right", srs)
    return domain


def bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def current_mesh() -> dict:
    """The mesh of gmsh's current model: `x` [n, 3], `triangles` [m, 3] and `lines` [k, 2] as indices
    into `x`, and the names of the physical groups."""
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    index = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    index[tags.astype(np.int64)] = np.arange(len(tags))
    elements = {}
    for element_type, n in ((2, 3), (1, 2)):
        _, nodes = gmsh.model.mesh.getElementsByType(element_type)
        elements[element_type] = index[nodes.astype(np.int64)].reshape(-1, n)
    groups = sorted(gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups())
    return {
        "x": coordinates.reshape(-1, 3),
        "triangles": elements[2],
        "lines": elements[1],
        "groups": groups,
    }


def signed_areas(mesh: dict) -> np.ndarray:
    """Signed areas of the triangles in the (x, y) plane."""
    a, b, c = (mesh["x"][mesh["triangles"][:, i], :2] for i in range(3))
    return 0.5 * (
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )


def n_components(mesh: dict) -> int:
    """Number of connected components of the triangles (joined through shared vertices)."""
    parent = np.arange(len(mesh["x"]))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in mesh["triangles"]:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[rc] = ra
    return len({find(i) for i in np.unique(mesh["triangles"])})


def stray_line_nodes(mesh: dict) -> int:
    """Number of nodes of boundary line elements that belong to no triangle."""
    return len(np.setdiff1d(np.unique(mesh["lines"]), np.unique(mesh["triangles"])))


def inside_curves(points: np.ndarray, curves: list) -> np.ndarray:
    """Even-odd test of `points` [n, 2] against the closed polylines `curves`."""
    crossings = np.zeros(len(points), dtype=int)
    for curve in curves:
        a, b = curve[:-1], curve[1:]
        for i, (px, py) in enumerate(points):
            straddles = (a[:, 1] > py) != (b[:, 1] > py)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_cross = a[:, 0] + (py - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
            crossings[i] += int(np.count_nonzero(straddles & (px < x_cross)))
    return crossings % 2 == 1
