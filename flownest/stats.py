"""Totals over every catchment from one pass over the numbered raster.

Read each pixel once and add its value to the bin of its channel pixel (|v|). The running sum C of the bins then gives the total
upstream of channel pixel p over [p, p + upstream_channel_pixel_count), without reading flow directions.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _sum_by_channel_pixel(raster, values, first, nbin):
    s = np.zeros(nbin + 1)
    nr, nc = raster.shape
    for r in range(nr):
        for c in range(nc):
            k = raster[r, c]
            if k == 0 or k == -2147483648:
                continue
            i = (k if k > 0 else -k) - first + 1
            if 1 <= i <= nbin:                          # a window cut from a larger raster can hold numbers of its neighbours
                s[i] += values[r, c]
    return s


def upstream_totals(raster, upstream_channel_pixel_count, values, first=1):
    """Total of values over the catchment of each channel pixel. Index i of the result belongs to number first + i.

    raster is signed (+n on channel pixels, -n on hillslope pixels draining first to n). upstream_channel_pixel_count includes each
    pixel itself, for ids first, first + 1, ... in order. Its upstream ids are [p, p + count).
    The catchment must lie inside the raster, as for a whole basin cut from the release.
    Only the raster's 0 and nodata are skipped. values are used as they are, so replace its own nodata (NaN, -9999) first.
    """
    values = np.broadcast_to(np.asarray(values, np.float64), raster.shape)
    counts = np.asarray(upstream_channel_pixel_count, np.int64)
    c = np.cumsum(_sum_by_channel_pixel(np.asarray(raster), values, int(first), counts.size))
    return c[np.arange(counts.size) + counts] - c[:-1]


def upstream_means(raster, upstream_channel_pixel_count, values, pixel_area, first=1):
    """Area-weighted mean of values over the catchment of each channel pixel, and the area, as the tuple (mean, area)."""
    a = upstream_totals(raster, upstream_channel_pixel_count, pixel_area, first)
    return upstream_totals(raster, upstream_channel_pixel_count, np.asarray(values, np.float64) * pixel_area, first) / a, a


def pixel_areas_km2(lat_top, dlat, dlon, nrow):
    """WGS84 pixel areas in km2, shape (nrow, 1), one value per row to broadcast across columns.

    All angles are degrees. lat_top is the north edge of the first row (not its centre), positive north.
    dlat > 0 is the row height southward; dlon is the positive pixel width in longitude.
    """
    a, e2 = 6378.137, 0.00669437999014
    e = np.sqrt(e2)
    edge = np.radians(lat_top - np.arange(nrow + 1) * dlat)
    s = np.sin(edge)
    q = (1 - e2) * (s / (1 - e2 * s ** 2) - np.log((1 - e * s) / (1 + e * s)) / (2 * e))
    return (a ** 2 * np.radians(dlon) / 2 * np.abs(np.diff(q)))[:, None]

