"""Check the numbering of flownest against the release, basin by basin: the three basins of the README and a sample of basins
drawn by area. For each basin the package numbers the basin again from MERIT Hydro's D8 and upstream area (Release.number_basin)
and compares, with the release:
  [N1] the signed pixel raster of the basin's window
  [N2] at every threshold of the release and its coarser thresholds folder: the reach topology (ids, pixels, links, paths), the
       upstream area (float32), the unit and upstream catchment windows, and the unit catchment area (the release rounds it to six
       decimals per base unit catchment)
  [N3] at every coarser threshold: the reaches the package derives from the base tables alone (Release.reaches_at and
       reach_at_threshold without the coarser folder) against the coarser tables and their crosswalk
usage: python -I check_numbering_against_release.py <package dir> <global release> <coarser thresholds folder> <d8 tif> <upa tif>
       [basins drawn a class below 1e4 km2 (a sixth of it from 1e4 to 1e5 km2), default 30] [seed, default 0]
"""
import sys
import time
import numpy as np

pkg, release, coarser, d8, upa = sys.argv[1:6]
per_class = int(sys.argv[6]) if len(sys.argv) > 6 else 30
seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
sys.path.insert(0, pkg)
from flownest.release import Release, label_of          # noqa: E402

TOPOLOGY = ['reach_seq_id', 'downstream_end_pixel_seq_id', 'channel_pixel_count', 'upstream_channel_pixel_count',
            'next_down_reach_seq_id', 'upstream_reach_count', 'path_seq_id', 'path_level']
WINDOWS = [p + s for p in ('unit_catchment_', 'upstream_catchment_') for s in ('row_min', 'row_max', 'col_min', 'col_max')]
NAMED = {10670: 'Dungeness', 4422: 'Nisqually', 8206: 'Siletz'}
CLASSES = ((1, 1e2), (1e2, 1e3), (1e3, 1e4), (1e4, 1e5))           # km2, half-open; a basin's window is held in memory

t0 = time.time()
rel = Release(release, coarser)                                   # with the coarser tables
derived = Release(release)                                        # the base tables only
bt = rel.table('basin.parquet', ['basin_id', 'upstream_area_km2'])
rng = np.random.default_rng(seed)
chosen = list(NAMED)
for lo, hi in CLASSES:
    pool = bt['basin_id'][(bt['upstream_area_km2'] >= lo) & (bt['upstream_area_km2'] < hi)]
    pool = pool[~np.isin(pool, chosen)]
    take = per_class if hi <= 1e4 else max(per_class // 6, 1)
    chosen += sorted(rng.choice(pool, size=min(take, pool.size), replace=False).tolist())
print(f'{len(chosen)} basins: {list(NAMED.values())} and {per_class} a class in {CLASSES[:3]} km2, {max(per_class // 6, 1)} in '
      f'{CLASSES[3]}; thresholds {rel.thresholds}', flush=True)
bad = 0
checked = {'pixels': 0, 'reaches': 0, 'derived reaches': 0}


def differ(what, n_diff, basin_id):
    global bad
    if n_diff:
        bad += int(n_diff)
        print(f'  basin {basin_id} {what}: {int(n_diff)} differ', flush=True)


for i, basin_id in enumerate(chosen):
    n, window = rel.number_basin(int(basin_id), d8, upa)
    raster = rel.window(*window).astype(np.int64)
    first, count = n.first, n.grid_index.size
    mine = (raster != rel.nodata) & (np.abs(raster) >= first) & (np.abs(raster) < first + count)
    differ('[N1] pixels of the basin', int(((n.raster != 0) != mine).sum()), basin_id)
    differ('[N1] signed raster', int((n.raster[mine] != raster[mine]).sum()), basin_id)
    checked['pixels'] += count
    base_ends = n.reaches(rel.thresholds[0])['downstream_end_pixel_seq_id']
    for threshold in rel.thresholds:
        selection = rel.reach_range(int(basin_id), threshold)
        if selection is None:
            differ(f'[N2] {label_of(threshold)} reaches where the release has none', n.reaches(threshold)['reach_seq_id'].size,
                   basin_id)
            continue
        want = rel.reaches(*selection, threshold)
        got = n.reaches(threshold, first_reach=selection[0], first_path=int(want['path_seq_id'][0]))
        if got['reach_seq_id'].size != want['reach_seq_id'].size:
            differ(f'[N2] {label_of(threshold)} reach count', 1, basin_id)
            continue
        for column in TOPOLOGY:
            differ(f'[N2] {label_of(threshold)} {column}', (got[column] != want[column]).sum(), basin_id)
        differ(f'[N2] {label_of(threshold)} upstream_area_km2',
               (got['upstream_area_km2'].astype(np.float32) != want['upstream_area_km2']).sum(), basin_id)
        box = rel.reach_windows(n, window, threshold, first_reach=selection[0])
        location = rel.rows(f'reach_location_{label_of(threshold)}.parquet', *selection)
        for column in WINDOWS:
            differ(f'[N2] {label_of(threshold)} {column}', (box[column] != location[column]).sum(), basin_id)
        unit = n.unit_catchments(threshold, first_reach=selection[0])
        k = sum(np.searchsorted(base_ends, unit[f'first_{j}'] + unit[f'count_{j}']) -
                np.searchsorted(base_ends, unit[f'first_{j}']) for j in (1, 2))
        area = want['unit_catchment_area_km2'].astype(np.float64)
        differ(f'[N2] {label_of(threshold)} unit_catchment_area_km2 beyond the rounding',
               (np.abs(box['unit_catchment_area_km2'] - area) > k * 5e-7 + 1e-9 * area).sum(), basin_id)
        checked['reaches'] += selection[1]
        if threshold == rel.thresholds[0]:
            continue
        label = label_of(threshold)
        d = derived.reaches_at(threshold, basin_id=int(basin_id))
        for column in ('downstream_end_reach_seq_id_1km2', 'stem_reach_count_1km2', *TOPOLOGY, 'basin_id', 'upstream_area_km2'):
            differ(f'[N3] {label} {column}', (d[column] != want[column]).sum(), basin_id)
        for column in ('length_km', 'distance_to_basin_outlet_km'):
            differ(f'[N3] {label} {column} beyond 1e-6 km', (np.abs(d[column] - want[column]) > 1e-6).sum(), basin_id)
        first_base, target = derived.reach_at_threshold(threshold, basin_id=int(basin_id))
        cross = rel.rows(f'crosswalk_reach_{rel.base}_to_coarser.parquet', first_base, target.size,
                         [f'reach_seq_id_{label}'])[f'reach_seq_id_{label}']
        differ(f'[N3] {label} crosswalk', (target != cross).sum(), basin_id)
        checked['derived reaches'] += selection[1]
    name = NAMED.get(int(basin_id), '')
    if name or (i + 1) % 10 == 0 or i + 1 == len(chosen):
        print(f'{i + 1}/{len(chosen)} basin {basin_id} {name} {count} channel pixels; {time.time() - t0:.0f} s', flush=True)
print(f'checked: {checked}')
print('ALL SAME' if bad == 0 else f'{bad} DIFFERENCES', f'{time.time() - t0:.0f} s')
sys.exit(1 if bad else 0)  # exit status 1 when anything differs
