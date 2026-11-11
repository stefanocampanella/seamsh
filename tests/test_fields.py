"""Mesh-size helpers of `seamsh.field`: callables `f(x[n, 2], projection) -> [n]`."""

import numpy as np
from domains import square_with_island, utm
from osgeo import gdal, osr

import seamsh.field


def test_distance_to_all_curves(model):
    """`Distance` is the distance to the nearest sampled point of the curves: exact within the
    sampling interval."""
    srs = utm()
    distance = seamsh.field.Distance(square_with_island(island=20e3), 500.0)
    x = np.array([[10e3, 50e3], [50e3, 50e3], [50e3, 25e3]])
    np.testing.assert_allclose(distance(x, srs), [10e3, 10e3, 15e3], atol=500.0)


def test_distance_to_tagged_curves(model):
    """`tags` restricts the curves the distance is measured to."""
    srs = utm()
    distance = seamsh.field.Distance(square_with_island(island=20e3), 500.0, tags=["coast"])
    np.testing.assert_allclose(distance(np.array([[50e3, 50e3]]), srs), [50e3], atol=500.0)


def test_distance_in_another_projection(model):
    """Query points are reprojected to the projection the field was built in."""
    lonlat = osr.SpatialReference()
    lonlat.ImportFromEPSG(4326)
    lonlat.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    lon, lat, _ = osr.CoordinateTransformation(utm(), lonlat).TransformPoint(10e3, 50e3)
    distance = seamsh.field.Distance(square_with_island(island=20e3), 500.0)
    np.testing.assert_allclose(distance(np.array([[lon, lat]]), lonlat), [10e3], atol=500.0)


def raster_dataset(values: np.ndarray, x0: float, y0: float, pixel: float) -> gdal.Dataset:
    """An in-memory raster in UTM with its top-left corner at (x0, y0) and square pixels."""
    rows, columns = values.shape
    dataset = gdal.GetDriverByName("MEM").Create("", columns, rows, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform((x0, pixel, 0.0, y0, 0.0, -pixel))
    dataset.SetProjection(utm().ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(values)
    return dataset


def test_raster_takes_the_value_of_the_pixel():
    """`Raster` returns the value of the pixel that contains the point (no interpolation); points
    outside the raster take the value of the nearest edge pixel."""
    values = np.arange(12.0).reshape(3, 4)
    field = seamsh.field.Raster(raster_dataset(values, x0=0.0, y0=30e3, pixel=10e3))
    x = np.array(
        [
            [5e3, 25e3],  # first row, first column
            [35e3, 25e3],  # first row, last column
            [15e3, 5e3],  # last row, second column
            [19e3, 11e3],  # still the second column, second row
            [-50e3, 25e3],  # west of the raster
            [35e3, -50e3],  # south of the raster
        ]
    )
    np.testing.assert_array_equal(field(x, utm()), [0.0, 3.0, 9.0, 5.0, 0.0, 11.0])


def test_inpoly(model):
    """`Inpoly` tells which points are inside the closed boundary curves with the given tags."""
    field = seamsh.field.Inpoly(square_with_island(island=20e3), tags=["island"])
    x = np.array([[50e3, 50e3], [10e3, 10e3], [59e3, 41e3], [61e3, 50e3]])
    np.testing.assert_array_equal(field(x, utm()), [True, False, True, False])
