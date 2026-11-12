"""
GIS Data
========
This example illustrates how the library interacts with various GIS file
formats directly from numpy arrays and osr spatial references.
"""

# %%
# Let's download some data: the test data of upstream's seamsh (the original
# project, at UCLouvain), around the island of Porquerolles in the south of
# France. It is downloaded, not
# distributed with this fork.

import os
import tarfile
import urllib.request

if not os.path.isdir("data2"):
    url = "https://nextcloud.cism.ucl.ac.be/s/TfY6tCzrqiKaTTX/download/data-test-2.tar.gz"
    urllib.request.urlretrieve(url, "data-test-2.tar.gz")
    f = tarfile.open("data-test-2.tar.gz", "r:*")
    f.extractall()

# %%
# The actual example starts here.

import gmsh
import numpy as np
from osgeo import gdal, osr

import seamsh
import seamsh.geometry
from seamsh.geometry import CurveType

# %%
# seamsh drives gmsh, which has to be initialized first.

gmsh.initialize()

# %%
# First, let's create a domain object and its associated projection.
# Any projection supported by osgeo.osr can be used.

domain_srs = osr.SpatialReference()
domain_srs.ImportFromProj4("+proj=utm +ellps=WGS84 +zone=31")
domain = seamsh.geometry.Domain(domain_srs)

# %%
# We load all curves from a given ESTRI shapefile as polylines.
# In the shapefile, a field named "physical" defines the physical tag of each
# curve. If a re-projection is required, it will be done automatically.

domain.add_boundary_curves_shp("data2/data_no_duplicate.shp", "physical", CurveType.POLYLINE)

# %%
# Interior curves and interior points can be loaded in a similar way.
# Physical tags are optional

domain.add_interior_curves_shp("data2/interior.shp", None, CurveType.STRICTPOLYLINE)
domain.add_interior_points_shp("data2/interior_points.shp", "physical")

# %%
# Seamsh provides helper classes to compute the element size field.
#
# field.Raster evaluates a gdal raster dataset (e.g. a geotiff file).

bath_field = seamsh.field.Raster(gdal.Open("data2/medit.tiff"))

# %%
# field.Distance can be used to compute the distance from given (tagged)
# boundaries. A first field returns the distance from boundaries with physical
# tag "coast", or "island". The curves are sampled at regular intervals and
# the computed distance is actually the distance from the closest sampling
# point. The second argument sets the length of the interval between the
# sampling points.

dist_coast = seamsh.field.Distance(domain, 100, ["coast", "island"])

# %%
# A second distance field returns the distance from the island tagged
# "porquerolles", in the shp file.

dist_porquerolles = seamsh.field.Distance(domain, 20, ["porquerolles"])


# %%
# The mesh size is defined based on those fields, smaller elements are
# prescribed around the Porquerolles island.


def mesh_size(x, projection):
    s_coast = np.clip((dist_coast(x, projection) - 400) * 0.5, 200, 5000)
    s_porq = np.clip((dist_porquerolles(x, projection) - 200) * 0.5, 50, 5000)
    s_dist = np.minimum(s_coast, s_porq)
    return s_dist


# %%
# Another option would be to take a mesh size proportional to the square root
# of the (clipped) bathymetry :

# def mesh_size(x,projection) :
#     bath = -bath_field(x,projection)
#     s_bath = np.sqrt(np.clip(bath,100,4000))*10
#     return  s_bath*10

# %%
# The mesh is generated in the current gmsh model, in the coordinate system
# "output_srs", then written to a file by gmsh.

output_srs = osr.SpatialReference()
output_srs.ImportFromProj4("+proj=latlon +ellps=WGS84")

seamsh.gmsh.mesh(domain, mesh_size, output_srs=output_srs)
gmsh.write("gis_mesh.msh")

# %%
# The mesh, drawn from the nodes and triangles of the gmsh model (see the first
# example). The mesh is finer around the island of Porquerolles.

import matplotlib.pyplot as plt

node_tags, xyz, _ = gmsh.model.mesh.getNodes()
xyz = xyz.reshape(-1, 3)
index = np.zeros(node_tags.max() + 1, int)
index[node_tags] = np.arange(node_tags.size)
_, triangle_nodes = gmsh.model.mesh.getElementsByType(2)
triangles = index[triangle_nodes.reshape(-1, 3)]

fig, ax = plt.subplots(figsize=(9, 6))
ax.triplot(xyz[:, 0], xyz[:, 1], triangles, color="k", linewidth=0.1)
ax.set_aspect(1 / np.cos(np.radians(xyz[:, 1].mean())))
ax.set_xlabel("longitude (degrees)")
ax.set_ylabel("latitude (degrees)")
plt.show()

# %%
# seamsh.gmsh.reproject writes an existing mesh file in another coordinate
# system (here the same one).

seamsh.gmsh.reproject("gis_mesh.msh", None, "gis_mesh_lonlat.msh", output_srs)

# %%
# The gmsh.convert_to_gis function can be used to convert a gmsh .msh file
# into a shape file or into a geo package file.

seamsh.gmsh.convert_to_gis("gis_mesh.msh", None, "gis_mesh.gpkg")

# %%
# gmsh is finalized when it is no longer needed.

gmsh.finalize()
