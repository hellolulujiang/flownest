"""Totals over every catchment of a basin from the released files, in one pass over its pixels.

    python upstream_statistics.py release_folder basin_id [values.tif]

Cuts a basin from the released pixel raster by its window, reads upstream_channel_pixel_count from the pixel table, and sums the
WGS84 areas upstream of every channel pixel, without the flow directions. At the reach downstream ends
the sums are compared with the upstream area of MERIT Hydro (upstream_area_km2 of the reach table). With a raster on the release
grid it also gives the area-weighted mean of that raster upstream of the basin outlet.
Needs pyarrow and rasterio. Example: python upstream_statistics.py .../flownest_merit90m_global_20261111_0314 10670   (the Dungeness, Washington)
"""
import sys
import numpy as np
from flownest import Release, upstream_totals, pixel_areas_km2

rel = Release(sys.argv[1])
b = rel.basin(int(sys.argv[2]))
first, count = b['outlet_pixel_seq_id'], b['channel_pixel_count']
w = (b['row_min'], b['row_max'], b['col_min'], b['col_max'])
v = rel.window(*w)                                                           # across 180 degrees in its two parts joined
x0, dx, _, y0, _, dy = rel.geotransform
pixel_area = np.broadcast_to(pixel_areas_km2(y0 + w[0] * dy, -dy, dx, v.shape[0]), v.shape)
upstream_counts = rel.pixels(first, count, ['upstream_channel_pixel_count'])['upstream_channel_pixel_count']

area = upstream_totals(v, upstream_counts, pixel_area, first=first)
first_reach, reach_count = rel.reach_range(b['basin_id'])
r = rel.reaches(first_reach, reach_count, columns=['downstream_end_pixel_seq_id', 'upstream_area_km2'])
ratio = area[r['downstream_end_pixel_seq_id'] - first] / r['upstream_area_km2']
print(f'{count} channel pixels, {reach_count} reaches; area from the pixels / MERIT Hydro upstream area at reach '
      f'downstream ends: median {np.median(ratio):.5f}, range {ratio.min():.5f} to {ratio.max():.5f}; at the basin outlet {area[0]:.1f} km2 '
      f'against {b["upstream_area_km2"]:.1f} km2')
if len(sys.argv) > 3:
    x = rel.window(*w, raster=sys.argv[3]).astype(np.float64)               # must be on the release grid; mind its nodata
    mean = upstream_totals(v, upstream_counts, x * pixel_area, first=first) / area
    print('area-weighted mean upstream of the basin outlet:', mean[0])
