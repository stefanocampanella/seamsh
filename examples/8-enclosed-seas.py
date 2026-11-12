# %%
# Enclosed seas and a hierarchy of meshes
# =======================================
# This example meshes the global ocean together with the seas that a coarse
# coastline cuts off from it (the Mediterranean, the Black Sea, the Sea of
# Azov, the Baltic and the Red Sea), at two resolutions: a 300 km mesh and a
# 600 km mesh of the same domain, as needed for multigrid-like methods.

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
# The same coarsened domain is meshed twice. In the 600 km mesh, the Sea of
# Azov is smaller than an element: seamsh keeps it as a single triangle rather
# than dropping it, so that both meshes cover the same seas.

cart_srs = osr.SpatialReference()
cart_srs.ImportFromProj4("+ellps=WGS84 +proj=cart +units=m +x_0=0 +y_0=0")

for name, size in (("fine", 300e3), ("coarse", 600e3)):
    gmsh.model.add(name)
    seamsh.gmsh.mesh(coarse, uniform_on_earth(size), output_srs=cart_srs)
    gmsh.write(f"enclosed_seas_{name}.msh")
