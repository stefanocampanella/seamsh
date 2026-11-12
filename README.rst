Seamsh is a Python library wrapping gmsh_, gdal_  and scipy_ to simplify the generation of unstructured meshes. While primarily developed for coastal ocean simulations, it can be used in other GIS contexts.

Main Features :

- Import ESRI shapefiles to define tagged domain boundaries, interior lines and interior points.
- Define arbitrary mesh elements size fields based on distances from lines or raster files.
- Create a low-resolution valid topology from high-resolution non-conformal (i.e. intersecting) data.
- Mesh domains made of several disconnected regions, such as seas cut off from the ocean by straits narrower than the mesh size.

This is a fork of seamsh_, developed by Jonathan Lambrechts at the Université catholique de Louvain (UCLouvain). It branched from upstream's version 0.4.18 and has diverged since: see ``CHANGELOG.rst``. Its versions carry the local label ``+fork`` (e.g. ``0.5.0+fork``); the packages named seamsh on PyPI are upstream's releases, whose numbers (0.5 and later) do not refer to the same code. This fork is installed from source (see the installation instructions).

Seamsh is free software, distributed under the GPL_ (version 3 or later): upstream's files are under the LGPL_ (version 3 or later), the files added by this fork under the GPL, and the compiled library contains files from gmsh, which are under the GPL (see ``LICENSE.txt``). See the GitHub page of this fork for the `source code`_ and `bug reports`_. The documentation_ contains examples, Python API reference and installation instructions.

.. _gmsh : https://www.gmsh.info
.. _gdal : https://gdal.org
.. _scipy : https://scipy.org
.. _seamsh : https://git.immc.ucl.ac.be/jlambrechts/seamsh
.. _source code : https://github.com/stefanocampanella/seamsh
.. _documentation : https://stefanocampanella.github.io/seamsh/
.. _bug reports : https://github.com/stefanocampanella/seamsh/issues
.. _GPL : https://www.gnu.org/licenses/gpl-3.0.html
.. _LGPL : https://www.gnu.org/licenses/lgpl-3.0.html
