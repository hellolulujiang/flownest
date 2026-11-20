"""Check flownest.merit_basins against the release: every column of the MERIT-Basins crosswalk at 25 km2, globally (pfaf 1 .. 9 in
the order the release read them). The river systems against the FullBasin basins: check_merit_basins_river_systems_on_fullbasin.py.
usage: python -I check_merit_basins_against_release.py <package dir> <global release> <coarser thresholds folder> <MERIT-Basins dir> <upa tif> <MERIT-Basins crosswalk parquet of the release>
"""
import sys
import time
import numpy as np
import pyarrow.parquet as pq

pkg, release, coarser, mb_dir, upa, c_table = sys.argv[1:7]
sys.path.insert(0, pkg)
from flownest.release import Release          # noqa: E402
from flownest import merit_basins             # noqa: E402

t0 = time.time()
rel = Release(release, coarser)
lookup = merit_basins.ReleaseReaches(rel, 25)
bad = 0

# the base reaches cover the channel pixels in runs: so a base reach's last pixel is outlet + count - 1 = next outlet - 1
base = rel.table(f'reach_{rel.base}.parquet', ['downstream_end_pixel_seq_id', 'channel_pixel_count'])
o, cnt = base['downstream_end_pixel_seq_id'].astype(np.int64), base['channel_pixel_count'].astype(np.int64)
runs = int((o[1:] != o[:-1] + cnt[:-1]).sum()) + int(o[0] != 1)
print(f'base reaches not in back-to-back runs: {runs}; last pixel {o[-1] + cnt[-1] - 1}', flush=True)
bad += runs

theirs_all = pq.read_table(c_table)
print(f'C table: {theirs_all.num_rows} rows, {theirs_all.num_columns} columns', flush=True)
offset = 0
counts = np.zeros(len(merit_basins.MATCH_CLASSES), np.int64)
no_area = 0
for k in range(1, 10):
    path = f'{mb_dir}/pfaf_{k}_MERIT_Hydro_v07_Basins_v01.zip'
    reaches = merit_basins.read_merit_basins(path)
    mine = merit_basins.place_reaches(rel, lookup, reaches, upa)
    n = reaches['comid'].size
    theirs = theirs_all.slice(offset, n)
    if list(mine) != theirs.column_names:
        print('column names or order differ:', list(mine), theirs.column_names)
        bad += 1
    diff_here = 0
    for name, v in mine.items():
        t = theirs.column(name).to_numpy(zero_copy_only=False)
        if v.dtype.kind == 'f':
            if v.dtype != t.dtype:
                print(f'  {name}: dtype {v.dtype} vs {t.dtype}')
                bad += 1
            d = int((~((v == t) | (np.isnan(v) & np.isnan(t)))).sum())
        else:
            d = int((v != t).sum())
        if d:
            print(f'  pfaf {k} {name}: {d} of {n} differ', flush=True)
        diff_here += d
    bad += diff_here
    for j, c in enumerate(merit_basins.MATCH_CLASSES):
        counts[j] += int((mine['match_class'] == c).sum())
    no_area += int(np.isnan(reaches['upstream_area_km2']).sum())
    offset += n
    print(f'pfaf {k}: {n} reaches, {reaches["x"].size} vertices, {diff_here} differences, {time.time() - t0:.0f} s', flush=True)
if offset != theirs_all.num_rows:
    print(f'row count: {offset} here, {theirs_all.num_rows} in C')
    bad += 1
print('classes:', dict(zip(merit_basins.MATCH_CLASSES, counts.tolist())))
print('reaches with no upstream area:', no_area)
print('ALL SAME' if bad == 0 else f'{bad} DIFFERENCES', f'{time.time() - t0:.0f} s')
sys.exit(1 if bad else 0)  # exit status 1 when anything differs
