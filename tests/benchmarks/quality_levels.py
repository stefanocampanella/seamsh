# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: GPL-3.0-or-later

"""Quality (and time) of global ocean meshes at several resolutions, as moby-dick makes them.

Not a test: it needs Natural Earth's 1:10m coastline (https://www.naturalearthdata.com, ne_10m_coastline) and takes
minutes. The ocean is meshed in the polar stereographic plane with a size uniform on the Earth, written in
Earth-centred coordinates; each level coarsens the coastline at a fraction of its own size, with one seed in the
open ocean and one in each enclosed sea. Run from the repository, with seamsh built in place:

    python tests/benchmarks/quality_levels.py ne_10m_coastline.shp --levels 25 100 200 400 --fraction 1.0
"""

import argparse
import sys
import time
from pathlib import Path

import gmsh
import numpy as np
from osgeo import osr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality import mesh_quality  # noqa: E402

import seamsh  # noqa: E402
from seamsh.geometry import CurveType  # noqa: E402

STEREOGRAPHIC = osr.SpatialReference()
STEREOGRAPHIC.ImportFromProj4("+proj=stere +lat_0=90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m")
CARTESIAN = osr.SpatialReference()
CARTESIAN.ImportFromProj4("+proj=cart +ellps=WGS84 +units=m")
LONLAT = osr.SpatialReference()
LONLAT.ImportFromEPSG(4326)
LONLAT.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
# The open ocean (the North Pole, and the North Atlantic for coastlines coarse enough to close the Arctic's passages)
# and the seas a coarse coastline cuts off from it, in longitude and latitude.
SEEDS = [(0.0, 90.0), (-40.0, 40.0), (18.0, 35.0), (34.0, 43.5), (36.5, 46.0), (20.0, 58.0), (38.5, 20.0)]


def uniform_on_earth(size: float):
  """A size uniform on the Earth, in the stereographic plane (divided by the scale factor of the projection)."""
  r2 = STEREOGRAPHIC.GetSemiMajor() * STEREOGRAPHIC.GetSemiMinor()

  def mesh_size(x, projection):
    return size * (4 * r2 + x[:, 0] ** 2 + x[:, 1] ** 2) / (4 * r2)

  return mesh_size


def print_worst(quality, n: int) -> None:
  """The n worst triangles: gamma, angles, longitude and latitude of the centre, whether at the boundary."""
  tags, coordinates, _ = gmsh.model.mesh.getNodes()
  index = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
  index[tags.astype(np.int64)] = np.arange(len(tags))
  triangles = index[gmsh.model.mesh.getElementsByType(2)[1].astype(np.int64)].reshape(-1, 3)
  centres = coordinates.reshape(-1, 3)[triangles].mean(axis=1)
  to_lonlat = osr.CoordinateTransformation(CARTESIAN, LONLAT)
  for i in np.argsort(quality.gamma)[:n]:
    lon, lat = to_lonlat.TransformPoint(*centres[i])[:2]
    where = "boundary" if quality.on_boundary[i] else "interior"
    print(
      f"     gamma {quality.gamma[i]:.3f}, angles {quality.min_angle[i]:.1f}-{quality.max_angle[i]:.1f}, "
      f"{lat:.2f} N {lon:.2f} E, {where}"
    )


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("shapefile")
  parser.add_argument("--levels", type=float, nargs="+", default=[25, 50, 100, 200, 400], help="mesh sizes (km)")
  parser.add_argument("--fraction", type=float, default=1.0, help="coastline coarsening size / mesh size")
  parser.add_argument("--coarsen", type=float, help="one coastline coarsening size for all levels (km)")
  parser.add_argument("--one-seed", action="store_true", help="the open ocean only, no enclosed seas")
  parser.add_argument("--no-optimize", action="store_true", help="mesh(optimize=False)")
  parser.add_argument("--valence-repair", action="store_true", help="mesh(valence_repair=True)")
  parser.add_argument("--output", type=Path, help="directory where to write the meshes (binary .msh)")
  parser.add_argument("--worst", type=int, default=0, help="list the N worst triangles (position, angles)")
  args = parser.parse_args()

  seamsh.show_progress(False)
  gmsh.initialize()
  gmsh.option.setNumber("General.Terminal", 0)
  print(f"seamsh {Path(seamsh.__file__).parent}, gmsh {gmsh.__version__}", flush=True)
  domain = seamsh.geometry.Domain(STEREOGRAPHIC)
  domain.add_boundary_curves_shp(args.shapefile, "featurecla", CurveType.BSPLINE)
  seeds = SEEDS[:1] if args.one_seed else SEEDS
  coarsened = {}
  for level in args.levels:
    size = (args.coarsen or args.fraction * level) * 1e3
    if size not in coarsened:
      t0 = time.perf_counter()
      coarsened[size] = seamsh.geometry.coarsen_boundaries(domain, seeds, LONLAT, uniform_on_earth(size))
      print(f"coastline coarsened at {size / 1e3:g} km in {time.perf_counter() - t0:.1f} s", flush=True)
    gmsh.model.add(f"level {level:g}")
    t0 = time.perf_counter()
    seamsh.gmsh.mesh(
      coarsened[size],
      uniform_on_earth(level * 1e3),
      output_srs=CARTESIAN,
      optimize=not args.no_optimize,
      valence_repair=args.valence_repair,
    )
    elapsed = time.perf_counter() - t0
    quality = mesh_quality()
    missed = quality.failures(degree=level <= 100)
    print(f"{level:g} km (coast {size / 1e3:g} km, {elapsed:.1f} s): {quality.summary()}")
    print("   targets: " + ("met" if not missed else "missed: " + "; ".join(missed)), flush=True)
    if args.worst:
      print_worst(quality, args.worst)
    if args.output:
      gmsh.option.setNumber("Mesh.Binary", 1)
      gmsh.write(str(args.output / f"ocean_{level:g}km.msh"))
    gmsh.model.remove()
  gmsh.finalize()


if __name__ == "__main__":
  main()
