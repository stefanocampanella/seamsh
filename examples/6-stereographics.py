"""
Stereographic projection
========================
This example illustrates the use of the stereographic projection to generate
a mesh covering most of the Earth's surface.
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
import seamsh.geometry
from seamsh.geometry import CurveType

# %%
# seamsh drives gmsh, which has to be initialized first.

gmsh.initialize()

# %%
# The mesh is generated in the polar stereographic projection.
# the singular point, at the opposite of the projection origin has to be outside the domain. Here
# the the North Pole is chosen as the origin of the stereographic projection so that the singular
# point (i.e. the South Pole) is outside the meshed domain.

domain_srs = osr.SpatialReference()
domain_srs.ImportFromProj4("+ellps=WGS84 +proj=stere +lat_0=90")
domain = seamsh.geometry.Domain(domain_srs)
domain.add_boundary_curves_shp(shp_file, "featurecla", CurveType.POLYLINE)

# %%
# "Cartesian" projection (i.e. no projection) is used to compute the distance to the coast.

cart_srs = osr.SpatialReference()
cart_srs.ImportFromProj4("+ellps=WGS84 +proj=cart +units=m +x_0=0 +y_0=0")


dist_coast = seamsh.field.Distance(domain, 10000, projection=cart_srs)


# %%
# it is necessary to convert the mesh size to stereographic coordinates


def mesh_size(x, projection):
    s_coast = np.clip((dist_coast(x, projection) - 20000) * 0.5, 20000, 100000)
    R = 6371000
    stereo_factor = 2 / (1 + x[:, 0] ** 2 / R**2 + x[:, 1] ** 2 / R**2)
    return s_coast / stereo_factor


coarse = seamsh.geometry.coarsen_boundaries(domain, (0, 0), domain_srs, mesh_size)

# %%
# Eventually the mesh is generated.
# The mesh is saved both in stereographic and cartesian coordinates

seamsh.gmsh.mesh(coarse, mesh_size, output_srs=domain_srs)
gmsh.write("natural_earth.msh")

# %%
# The nodes and triangles of the mesh, taken from the gmsh model (see the first
# example), are in the stereographic plane. The size of each triangle on the
# Earth is its mean edge length there times the scale factor used in
# ``mesh_size``.

node_tags, xyz, _ = gmsh.model.mesh.getNodes()
xyz = xyz.reshape(-1, 3)
index = np.zeros(node_tags.max() + 1, int)
index[node_tags] = np.arange(node_tags.size)
_, triangle_nodes = gmsh.model.mesh.getElementsByType(2)
triangles = index[triangle_nodes.reshape(-1, 3)]

R = 6371000
corners = xyz[triangles, :2]
edge_length = np.linalg.norm(corners - corners[:, [1, 2, 0]], axis=2).mean(axis=1)
centre = corners.mean(axis=1)
size_on_earth = edge_length * 2 / (1 + (centre**2).sum(axis=1) / R**2)

# %%
# The mesh is drawn in longitude and latitude, each triangle coloured by its
# size: ``mesh_size`` asks for 20 km along the coasts, growing to 100 km away
# from them. Triangles that cross the antimeridian are left out of the drawing.

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

lonlat = osr.SpatialReference()
lonlat.ImportFromEPSG(4326)
lonlat.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
to_lonlat = osr.CoordinateTransformation(domain_srs, lonlat)
lon, lat = np.array(to_lonlat.TransformPoints(xyz[:, :2]))[:, :2].T
triangulation = Triangulation(lon, lat, triangles, mask=np.ptp(lon[triangles], axis=1) > 180)

fig, ax = plt.subplots(figsize=(12, 6))
image = ax.tripcolor(triangulation, facecolors=size_on_earth / 1e3, cmap="viridis")
fig.colorbar(image, ax=ax, label="triangle size (km)", shrink=0.8)
ax.set_aspect("equal")
ax.set_xticks(np.arange(-180, 181, 60))
ax.set_yticks(np.arange(-90, 91, 30))
ax.set_xlabel("longitude (degrees)")
ax.set_ylabel("latitude (degrees)")
plt.show()

# %%
# The mesh file is then written again in Cartesian coordinates.

seamsh.gmsh.reproject("natural_earth.msh", domain_srs, "natural_earth_cart.msh", cart_srs)

# %%
# gmsh is finalized when it is no longer needed.

gmsh.finalize()
