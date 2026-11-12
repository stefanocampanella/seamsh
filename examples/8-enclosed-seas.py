# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Enclosed seas and a hierarchy of meshes
=======================================
This example meshes the global ocean together with the seas that a coarse
coastline cuts off from it (the Mediterranean, the Black Sea, the Sea of
Azov, the Baltic and the Red Sea), at two resolutions: a 300 km mesh and a
600 km mesh of the same domain, as needed for multigrid-like methods.
"""

# %%
# Download the Natural Earth coastline data set and extract the shapefile.

import os
import zipfile

import requests

coastline_url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
shp_file = "ne_10m_coastline/ne_10m_coastline.shp"

if not os.path.isfile(shp_file):
    req = requests.get(coastline_url, allow_redirects=True)
    open("ne_10m_coastline.zip", "wb").write(req.content)
    zipfile.ZipFile("ne_10m_coastline.zip").extractall("ne_10m_coastline")

import gmsh
import numpy as np
from osgeo import osr

import seamsh
from seamsh.geometry import CurveType

gmsh.initialize()

# %%
# As in the previous example, the domain is in the polar stereographic
# projection centred on the North Pole.

domain_srs = osr.SpatialReference()
domain_srs.ImportFromProj4("+ellps=WGS84 +proj=stere +lat_0=90")
domain = seamsh.geometry.Domain(domain_srs)
domain.add_boundary_curves_shp(shp_file, "featurecla", CurveType.BSPLINE)

# %%
# A mesh size that is uniform on the Earth is not uniform in the stereographic
# plane: lengths are stretched there by the scale factor of the projection,
# 1 + r^2 / (4 R^2) at the distance r from the pole.

R = 6371000


def uniform_on_earth(size):
    def mesh_size(x, projection):
        return size * (1 + (x[:, 0] ** 2 + x[:, 1] ** 2) / (4 * R**2))

    return mesh_size


# %%
# The coastline is coarsened to 30 km. Coarsening keeps the water that can be
# reached from a point without crossing a passage narrower than that size: from
# the North Pole alone, everything behind the Strait of Gibraltar, the
# Bosporus, the Kerch Strait, the Danish straits and Bab-el-Mandeb would be
# lost. One point is therefore given in each of these seas. The points are
# given in longitude and latitude; any coordinate system can be used, the same
# for all the points.

lonlat = osr.SpatialReference()
lonlat.ImportFromEPSG(4326)
lonlat.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

seeds = [
    (0.0, 90.0),  # Arctic Ocean: the global ocean
    (18.0, 35.0),  # Mediterranean Sea
    (34.0, 43.5),  # Black Sea
    (36.5, 46.0),  # Sea of Azov
    (20.0, 58.0),  # Baltic Sea
    (38.5, 20.0),  # Red Sea
]
coarse = seamsh.geometry.coarsen_boundaries(domain, seeds, lonlat, uniform_on_earth(30e3))

# %%
# The coarsened domain is made of six regions that do not touch: the straits
# stay closed. Each region is meshed on its own, so the mesh is in six pieces.
# A point given on land would give a mesh of that land: the coastline only
# tells where the shore is, not on which side the water is.
#
# The same coarsened domain is meshed twice, each time in a new gmsh model. In
# the 600 km mesh, the Sea of Azov is smaller than an element: seamsh keeps it
# as a single triangle rather than dropping it, so that both meshes cover the
# same seas.
#
# Each mesh is written in the stereographic plane, and its nodes and triangles
# are kept as numpy arrays (see the first example) to be drawn below. Then it
# is written again in Cartesian coordinates (x, y, z on the Earth), in which
# both meshes can be used together.

cart_srs = osr.SpatialReference()
cart_srs.ImportFromProj4("+ellps=WGS84 +proj=cart +units=m +x_0=0 +y_0=0")

meshes = {}
for name, size in (("fine", 300e3), ("coarse", 600e3)):
    gmsh.model.add(name)
    seamsh.gmsh.mesh(coarse, uniform_on_earth(size))
    gmsh.write(f"enclosed_seas_{name}_stereo.msh")

    node_tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = xyz.reshape(-1, 3)
    index = np.zeros(node_tags.max() + 1, int)
    index[node_tags] = np.arange(node_tags.size)
    _, triangle_nodes = gmsh.model.mesh.getElementsByType(2)
    meshes[name] = xyz, index[triangle_nodes.reshape(-1, 3)]

    seamsh.gmsh.reproject(
        f"enclosed_seas_{name}_stereo.msh", domain_srs, f"enclosed_seas_{name}.msh", cart_srs
    )

# %%
# Both meshes, drawn in longitude and latitude. Triangles that cross the
# antimeridian are left out of the drawing.

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

to_lonlat = osr.CoordinateTransformation(domain_srs, lonlat)
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
for ax, (name, (xyz, triangles)) in zip(axes, meshes.items()):
    lon, lat = np.array(to_lonlat.TransformPoints(xyz[:, :2]))[:, :2].T
    mask = np.ptp(lon[triangles], axis=1) > 180
    ax.triplot(Triangulation(lon, lat, triangles, mask=mask), color="k", linewidth=0.3)
    ax.set_title(f"{name}: {len(triangles)} triangles")
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_ylabel("latitude (degrees)")
axes[1].set_xlabel("longitude (degrees)")
plt.show()

# %%
# gmsh is finalized when it is no longer needed.

gmsh.finalize()
