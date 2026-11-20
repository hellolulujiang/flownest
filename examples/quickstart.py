"""Number a made-up grid and ask it the usual questions. The grid only shows the calls; its numbers mean nothing."""
import numpy as np
from flownest import number_grid, upstream_pixels, verify

d8 = np.full((6, 6), 4, np.uint8)      # everything flows south (ESRI codes: 1 east, 4 south)
d8[5, :] = 1                           # the final row flows east
d8[5, 5] = 0                           # the outlet
d8[2, 1] = 1                           # one branch joins its neighbour
area = upstream_pixels(d8)             # upstream area in pixels; a real grid comes with upa in km2

n = number_grid(d8, area, threshold=3)
print(n.raster)                                        # +p on channel pixels, -p on the hillslope pixels that drain first to p
upstream = n.upstream(n.first + 1)
print('upstream ids:', upstream)
print('pixels in that catchment:', ((np.abs(n.raster) >= upstream.start) & (np.abs(n.raster) < upstream.stop)).sum())
print('the same from one pass over the raster:', n.totals(1.0)[1])
print('reach table columns:', list(n.reaches(3)))
print('unit catchments at 5 pixels, two ranges each:', n.unit_catchments(5))
print('checks failed:', sum(verify(n, d8, area, 3).values()))
