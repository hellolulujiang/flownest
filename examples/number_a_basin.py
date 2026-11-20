"""Number a D8 GeoTIFF and write the signed raster, the reach table and the area-weighted mean of a third raster upstream of every
channel pixel.

    python number_a_basin.py dir.tif upa.tif elv.tif out_folder [threshold_km2]

dir.tif uses the codes 1, 2, 4 ... 128 (MERIT Hydro, HydroSHEDS) and its own nodata value (247 in MERIT Hydro, where 0 is a river
mouth and 255 an inland sink, both outlets); upa.tif holds upstream area in km². The three rasters are on one latitude-longitude grid,
and elv.tif has a value on every pixel with a flow direction, since the means do not mask it. Needs rasterio.
"""
import csv, os, sys
import numpy as np, rasterio
from flownest import number_grid, pixel_areas_km2, verify

d8_file, upa_file, elv_file, out = sys.argv[1:5]
threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
os.makedirs(out, exist_ok=True)
with rasterio.open(d8_file) as f:
    d8, profile, grid = f.read(1), f.profile, f.transform
    nodata = 255 if f.nodata is None else f.nodata
with rasterio.open(upa_file) as f:
    upa = f.read(1).astype(np.float64)
with rasterio.open(elv_file) as f:
    elv = f.read(1).astype(np.float64)

n = number_grid(d8, upa, threshold, nodata=nodata)
bad = verify(n, d8, upa, threshold, nodata=nodata)
print('channel pixels', n.grid_index.size, 'checks failed', sum(bad.values()), bad)

profile.update(dtype='int32', nodata=-2147483648, compress='ZSTD', predictor=2, tiled=True, blockxsize=512, blockysize=512)
with rasterio.open(f'{out}/numbers.tif', 'w', **profile) as o:
    no_direction = np.isnan(d8) if np.isnan(nodata) else d8 == nodata
    o.write(np.where(no_direction, np.int32(-2147483648), n.raster), 1)   # the nodata of dir.tif stays nodata

t = n.reaches(threshold)
with open(f'{out}/reaches.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(t)
    w.writerows(zip(*[c.tolist() for c in t.values()]))

pixel_km2 = pixel_areas_km2(grid.f, -grid.e, grid.a, d8.shape[0])
mean, catchment_km2 = n.means(elv, pixel_km2)          # the same two passes answer any raster on this grid
np.savetxt(f'{out}/mean_upstream.csv', np.column_stack((np.arange(n.grid_index.size) + n.first, catchment_km2, mean)), fmt=['%d', '%.3f', '%.3f'],
           delimiter=',', header='pixel_seq_id,catchment_km2,mean_upstream', comments='')
