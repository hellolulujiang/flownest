"""Check flownest.stations against the release: place the GRDC stations (Burek and Smilovic 2023 list) with the Python package
and compare every row and column with the release's GRDC crosswalk (flownest_merit90m_crosswalk_grdc_burek2023.parquet).
usage: python -I check_grdc_stations_against_release.py <package dir> <global release> <station list csv> <upa tif> <GRDC crosswalk parquet of the release>
"""
import sys
import time
import numpy as np
import pyarrow.parquet as pq

pkg, release, station_csv, upa, c_table = sys.argv[1:6]
sys.path.insert(0, pkg)
from flownest import Release                                       # noqa: E402
from flownest.stations import read_station_list, place_stations   # noqa: E402

t0 = time.time()
rel = Release(release)
st = read_station_list(station_csv, 'burek2023')
mine_all = place_stations(rel, st, upa)
c = pq.read_table(c_table).to_pydict()
print(f'{st["station_id"].size} stations here, {len(c["station_seq_id"])} rows in C, {time.time() - t0:.0f} s', flush=True)
bad = 0
for name, mine in mine_all.items():
    theirs = np.array(c[name], dtype=object if name == 'pixel_class' else None)
    if mine.dtype.kind == 'f':
        same = (mine.astype(np.float64) == theirs) | (np.isnan(mine.astype(np.float64)) & np.isnan(theirs))
    else:
        same = mine == theirs
    n = int((~same).sum())
    bad += n
    print(f'{name:32s} {n} differ' + ('' if n == 0 else f'; first rows {np.flatnonzero(~same)[:5].tolist()}'))
print('ALL SAME' if bad == 0 else f'{bad} DIFFERENCES', f'{time.time() - t0:.0f} s')
sys.exit(1 if bad else 0)  # exit status 1 when anything differs
