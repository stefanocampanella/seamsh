..
   SPDX-FileCopyrightText: 2026 Stefano Campanella
   SPDX-License-Identifier: GPL-3.0-or-later

Changes from upstream seamsh
============================

This fork (https://github.com/stefanocampanella/seamsh) starts from upstream's commit ``ca380b5`` (19 September
2025, version 0.4.18, https://git.immc.ucl.ac.be/jlambrechts/seamsh). Upstream's later releases (0.4.19 to 0.6.3, on
PyPI) are other code and are not followed; the few of their bug fixes that applied here are included (see below).
``CHANGELOG.rst`` lists every change; this page sums them up, and says what matters to code written for upstream's
0.4.18, such as moby-dick's.

Changes to the API
------------------

The functions and classes are the same; these calls change.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Function
     - Upstream 0.4.18
     - This fork
   * - ``gmsh.mesh``
     - ``mesh(domain, filename, mesh_size, ...)``: meshes in a temporary model, writes ``filename`` and removes the
       model; gmsh is initialised when seamsh is imported, and finalised at exit
     - ``mesh(domain, mesh_size, ...)``: the caller initialises gmsh and writes the file (``gmsh.write``); the mesh is
       left in the current model, with its projection in the model attribute ``Projection``; the gmsh options it sets
       are put back (except ``Mesh.MshFileVersion``)
   * - ``gmsh.mesh``, new options
     -
     - ``optimize=True`` (gmsh's Relocate2D after meshing: fewer poor triangles, **other node positions**);
       ``valence_repair=False`` (gmsh's valence repair, only in gmsh's branch ``valence-repair-2d``)
   * - ``field.Raster``
     - ``Raster(filename)``
     - ``Raster(dataset)``: an open GDAL dataset (``gdal.Open(filename)``)
   * - ``geometry.coarsen_boundaries``
     - ``x0``: one point
     - ``x0``: one point or a sequence of points, one per basin to keep (disconnected domains)
   * - ``Domain.add_*`` points, ``coarsen_boundaries`` x0
     - numpy arrays
     - anything ``numpy.asarray`` takes (lists)
   * - ``reproject``, ``convert_to_gis``, ``convert_to_ugrid_mesh_file``
     - projection may be None (documented for some)
     - None: the projection stored in the file (a WKT on several lines too); none at all is a ``ValueError``
   * - ``seamsh.show_progress``
     -
     - new: turns the progress bars (tqdm, on the standard error) off or on
   * - Python
     - 3.6 and later
     - 3.10 and later
   * - gmsh's state
     - initialised at import
     - never initialised nor finalised by seamsh: the caller does it (``gmsh.initialize()``)

Changes of behaviour
--------------------

What a caller sees without changing its code:

- **Meshes are not identical** to upstream's for the same input: the coarsened coastline is shifted along other
  normals (a corner fix), ``mesh`` optimises the triangles by default, the edge tags of short curves are kept. Same
  sizes, better triangles; regenerate meshes rather than mix them with old ones.
- **Several regions.** A domain may be made of regions that do not touch (from several seeds, or given so): each is a
  surface of its own, all in the physical group ``domain``; the mesh is in as many pieces. A region smaller than the
  mesh size is kept, as one triangle. A seed whose basin the coarsening loses is reported by a warning.
- **Errors are exceptions**: the C library no longer ends the process (``RuntimeError``); a size callback returning
  zero, a negative number or NaN is a ``ValueError`` (it looped forever); a shapefile that cannot be opened is a
  ``FileNotFoundError``; ``quad=True`` with a gmsh that would crash is a ``RuntimeError``.
- **Size callbacks** are also called at points of the domain's bounding box outside the domain (the background grid of
  the mesh size), and must return positive sizes there.
- **Mesh files** hold no node outside the triangles (outlines of islands smaller than the mesh are left out of their
  physical group), and no view of the size field.
- **Tags**: a feature whose tag field is unset or empty is meshed in no physical group (it made ``mesh`` fail); a
  short curve between two longer ones keeps its tag through the coarsening.
- ``reproject`` keeps the node numbers in MSH 2 files (with gmsh 4.13 and later).
- ``coarsen_boundaries`` leaves numpy's global random state and the current gmsh model alone.
- Messages go through Python's ``logging`` (logger names ``seamsh.*``), progress through tqdm.
- UGRID files (``convert_to_ugrid_mesh_file``) write the connectivity as node indices, the units of the coordinates,
  and ``source = "seamsh (https://github.com/stefanocampanella/seamsh)"``.

Inside
------

- The C library reports errors by return code, frees what it allocates, and walks boundaries on triangle edges (a
  region one triangle wide lost loops). It is bound with ctypes in ``seamsh/_libseamsh.py``; reprojection is in
  ``_projection.py`` (with caches of transformations), the progress bars in ``_progress.py``.
- Tests (pytest, no network), with quality targets for the triangles; GitHub CI (lint, REUSE, tests with ty and
  coverage); documentation on GitHub Pages with the examples run.
- Faster: cached coordinate transformations, point-in-polygon tests by blocks, the background grid's edges made unique
  by sorting, the triangulations of the size field freed.

Fixes taken from upstream's later releases
------------------------------------------

``f28aa56`` (a short curve keeps its tag), ``c873f34`` (a projection on several lines), ``ce6ec1a`` and ``6442e2e``
(node numbers in MSH 2), ``5b45b37`` (unset or empty tags), ``8db84ce`` and ``adddf85`` (faster removal of curves
outside the mesh, done here with a table). Not taken: upstream's new coarsening (hxt2d), its requirement of a gmsh
development build, and its new features.

For moby-dick
-------------

moby-dick calls ``Domain(projection=...)``, ``add_boundary_curves_shp``, ``coarsen_boundaries(domain, x0=...,
x0_projection=..., mesh_size=...)``, ``field.Distance``, ``field.Raster(gdal_dataset)`` and ``gmsh.mesh(boundary,
mesh_size, output_srs=..., smoothness=...)``: these calls are the fork's already, and run unchanged. What to know:

- ``x0`` may be a list of seeds: one in the open ocean (also outside the Arctic, whose passages close on coarse
  coastlines) and one per enclosed sea to keep. Keep seamsh's warnings in the logs: they name the seeds whose sea is
  lost.
- Coarsen the coastline of each level of a hierarchy for that level (at about its size): a coarse mesh on a coastline
  coarsened for a finer one has poor triangles along the coast.
- ``mesh`` optimises the triangles by default (``optimize=False`` for the previous behaviour); meshes change, and are
  to be regenerated, with the same seamsh version for all the levels of a hierarchy.
- Install from the tag of a release (``seamsh@git+https://github.com/stefanocampanella/seamsh.git@v0.6.0+fork``),
  never from PyPI, whose ``seamsh`` is upstream's.
- gmsh: released gmsh (4.12 to 4.15) is enough; options of development builds are used when present.
