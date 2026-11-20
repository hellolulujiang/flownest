"""Spot check of flownest.catchments on the demo window (upper White): the label rasters of gauges, SWORD reaches and SWORD
nodes, the unit and upstream windows of the location tables, and the nodes' unit pixel counts, against the crosswalks_with_catchments package of the release.
usage: python -I check_catchment_rasters_on_demo_window.py <package dir> <global release> <GRDC crosswalk parquet> <SWORD crosswalk dir> <crosswalks_with_catchments package>
"""
import sys
import time
import numpy as np
import pyarrow.parquet as pq

pkg, release, grdc_table, sword_dir, s14 = sys.argv[1:6]
sys.path.insert(0, pkg)
from flownest.release import Release, read_window      # noqa: E402
from flownest import catchments                        # noqa: E402

t0 = time.time()
rel = Release(release)
OUTLET, UPC, BASIN = 1039545713, 402853, 6
W = (56741, 59783, 102928, 107802)                     # the demo's raster window, half-open
basin = rel.table('basin.parquet')
b = BASIN - 1
first, count = int(basin['outlet_pixel_seq_id'][b]), int(basin['channel_pixel_count'][b])
values = rel.window(*W)
print(f'window {values.shape}, basin {BASIN} pixels [{first}, {first + count})', flush=True)
bad = 0


def report(what, n_diff, n):
    global bad
    bad += n_diff
    print(f'{what}: {n_diff} of {n} differ', flush=True)


def objects(pixel, seq):
    """the objects of the basin, in seq order, with their ends"""
    k = np.flatnonzero((pixel >= first) & (pixel < first + count))
    k = k[np.argsort(seq[k])]
    p = pixel[k].astype(np.int64)
    u, back = np.unique(p, return_inverse=True)
    upc = rel.rows_at(f'pixel_{rel.base}.parquet', u, ['upstream_channel_pixel_count'])['upstream_channel_pixel_count']
    return k, p, p + upc[back].astype(np.int64)


def check(kind, pixel, seq, tif, location=None, key=None, unit_count=None):
    k, p, end = objects(pixel, seq)
    parent, upstream_count = catchments.nest(p, end)
    local = catchments.label_window(values, p, end, parent)            # labels 1..len(k) within the basin
    to_seq = np.r_[0, seq[k]]
    lab = np.where(local == catchments.NODATA, 0, local)
    mine = (np.sign(lab) * to_seq[np.abs(lab)]).astype(np.int64)
    mine[local == catchments.NODATA] = catchments.NODATA
    theirs = read_window(tif, *W, (rel.nrow, rel.ncol), rel.geotransform)
    report(f'{kind} label raster (window)', int((mine != theirs).sum()), mine.size)
    inside = (p >= OUTLET) & (p < OUTLET + UPC)                         # their land lies wholly in the window
    print(f'  {kind}: {len(k)} in the basin, {int(inside.sum())} upstream of the demo outlet', flush=True)
    unit = catchments.unit_windows(local, W[0], W[2])
    if location is not None:
        loc = pq.read_table(location).to_pandas().set_index(key)
        n = 0
        diff = 0
        for j in np.flatnonzero(inside):
            row = loc.loc[seq[k[j]]]
            got = unit.get(j + 1)
            want = None if np.isnan(float(row.unit_catchment_row_min)) else (int(row.unit_catchment_row_min), int(row.unit_catchment_row_max),
                                                                               int(row.unit_catchment_col_min), int(row.unit_catchment_col_max))
            up = catchments.upstream_window(j + 1, int(upstream_count[j]), unit)
            want_up = (int(row.upstream_catchment_row_min), int(row.upstream_catchment_row_max), int(row.upstream_catchment_col_min),
                       int(row.upstream_catchment_col_max))
            diff += (got != want) + (up != want_up)
            n += 2
        report(f'{kind} unit and upstream windows', diff, n)
    if unit_count is not None:
        loc = pq.read_table(unit_count).to_pandas().set_index('sword_node_seq_id')
        s = np.abs(local[local != catchments.NODATA].astype(np.int64))
        counts = np.bincount(s, minlength=len(k) + 1)
        j = np.flatnonzero(inside)
        want = loc.loc[seq[k[j]], 'unit_catchment_pixel_count'].to_numpy()
        report(f'{kind} unit_catchment_pixel_count', int((counts[j + 1] != want).sum()), j.size)
    print(f'  {time.time() - t0:.0f} s', flush=True)


g = pq.read_table(grdc_table, columns=['station_seq_id', 'pixel_seq_id']).to_pandas()
g = g[g.station_seq_id > 0]
G = 'flownest_merit90m_crosswalk_grdc_burek2023'
check('gauge', g.pixel_seq_id.to_numpy(), g.station_seq_id.to_numpy(), f'{s14}/{G}/{G}_station_seq_id.tif',
      f'{s14}/{G}/{G}_location.parquet', 'station_seq_id')
r = pq.read_table(f'{sword_dir}/flownest_merit90m_crosswalk_sword_v17c_reach.parquet',
                  columns=['sword_reach_seq_id', 'most_downstream_pixel_seq_id']).to_pandas()
r = r[r.sword_reach_seq_id > 0]
SW = 'flownest_merit90m_crosswalk_sword_v17c'
check('reach', r.most_downstream_pixel_seq_id.to_numpy(), r.sword_reach_seq_id.to_numpy(), f'{s14}/{SW}/{SW}_reach_seq_id.tif',
      f'{s14}/{SW}/{SW}_reach_location.parquet', 'sword_reach_seq_id')
n = pq.read_table(f'{sword_dir}/{SW}_node.parquet', columns=['sword_node_seq_id', 'pixel_seq_id']).to_pandas()
n = n[n.sword_node_seq_id > 0]
check('node', n.pixel_seq_id.to_numpy(), n.sword_node_seq_id.to_numpy(), f'{s14}/{SW}/{SW}_node_seq_id.tif',
      unit_count=f'{s14}/{SW}/{SW}_node_location.parquet')
print('ALL SAME' if bad == 0 else f'{bad} DIFFERENCES', f'{time.time() - t0:.0f} s')
sys.exit(1 if bad else 0)  # exit status 1 when anything differs
