"""The MERIT-Basins crosswalk at the level of basins first (polygon, then line, then point): each MERIT-Basins river system (the
reaches linked by NextDownID to one outlet reach) against the FullBasin basin it lies in, from the crosswalk's own rows.
  [B1] the placed reaches of a river system are all in one FullBasin basin
  [B2] where its outlet reach (NextDownID 0) begins: on the FullBasin outlet pixel, on a pixel next to it (8 neighbours), elsewhere
  [B3] MERIT-Basins uparea of the outlet reach against the FullBasin basin's area (several river systems may end in one basin)
usage: python -I check_merit_basins_river_systems_on_fullbasin.py <package dir> <global release> <MERIT-Basins crosswalk parquet>
"""
import sys
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

pkg, release, table = sys.argv[1:4]
sys.path.insert(0, pkg)
from flownest.release import Release          # noqa: E402

t0 = time.time()
rel = Release(release)
x = pq.read_table(table, columns=['merit_basins_comid', 'merit_basins_next_down_comid', 'merit_basins_upstream_area_km2',
                                  'pixel_seq_id', 'basin_id', 'match_class']).to_pandas()
comid = x.merit_basins_comid.to_numpy(np.int64)
down = x.merit_basins_next_down_comid.to_numpy(np.int64)
basin = x.basin_id.to_numpy(np.int64)
pixel = x.pixel_seq_id.to_numpy(np.int64)
m = comid.size
order = np.argsort(comid, kind='stable')
sc = comid[order]
j = np.minimum(np.searchsorted(sc, down), m - 1)
found = (down != 0) & (sc[j] == down)
print(f'{m} reaches; COMID unique {bool((sc[1:] != sc[:-1]).all())}; NextDownID not 0 and not found: {int(((down != 0) & ~found).sum())}')
root = np.where(found, order[j], np.arange(m))
for step in range(200):
    nxt = root[root]
    if (nxt == root).all():
        break
    root = nxt
is_outlet = root == np.arange(m)
print(f'river systems (outlet reaches): {int(is_outlet.sum())}; of them NextDownID 0: {int((is_outlet & (down == 0)).sum())}; '
      f'settled after {step} steps')
print('match_class of the outlet reaches:', x.match_class[is_outlet].value_counts().to_dict())

placed = basin > 0
g = pd.DataFrame({'root': root[placed], 'basin': basin[placed]}).groupby('root').basin.nunique()
print(f'[B1] river systems with a reach on a channel pixel: {g.size}; all in one FullBasin basin: {int((g == 1).sum())}; '
      f'in more than one: {int((g > 1).sum())}')
if (g > 1).any():
    print('     e.g. outlet COMIDs', comid[g[g > 1].index.to_numpy()[:10]].tolist())
# the basin of a river system: that of its outlet reach
bt = rel.table('basin.parquet', ['basin_id', 'outlet_pixel_seq_id', 'upstream_area_km2', 'is_endorheic'])
nb = bt['basin_id'].size
o = np.flatnonzero(is_outlet & placed)
ob = basin[o]
in_table = ob <= nb
print(f'[B2] outlet reaches on a channel pixel: {o.size}; basin in the basin table: {int(in_table.sum())}; under 1 km2: {int((~in_table).sum())}')
o, ob = o[in_table], ob[in_table]
outlet_pixel = bt['outlet_pixel_seq_id'][ob - 1].astype(np.int64)
u, back = np.unique(np.r_[pixel[o], outlet_pixel], return_inverse=True)
rc = rel.rows_at(f'pixel_{rel.base}.parquet', u, ['row', 'col'])
r, c = rc['row'][back].astype(np.int64), rc['col'][back].astype(np.int64)
dr = np.abs(r[:o.size] - r[o.size:])
dc = np.abs(c[:o.size] - c[o.size:])
dc = np.minimum(dc, rel.ncol - dc)                                                # across 180 degrees
on = pixel[o] == outlet_pixel
nextto = ~on & (dr <= 1) & (dc <= 1)
print(f'[B2] outlet reach begins on the FullBasin outlet pixel: {int(on.sum())}; on a pixel next to it: {int(nextto.sum())}; '
      f'elsewhere: {int((~on & ~nextto).sum())}')
if (~on & ~nextto).any():
    k = np.flatnonzero(~on & ~nextto)[:10]
    print('     e.g. COMID', comid[o[k]].tolist(), 'basin', ob[k].tolist(), 'rows off', dr[k].tolist(), 'cols off', dc[k].tolist())
fb = pd.DataFrame({'basin': ob, 'mb_area': x.merit_basins_upstream_area_km2.to_numpy()[o]})
per = fb.groupby('basin').mb_area.agg(['size', 'sum'])
area = bt['upstream_area_km2'][per.index.to_numpy() - 1].astype(np.float64)
endo = bt['is_endorheic'][per.index.to_numpy() - 1].astype(bool)
rel_diff = (per['sum'].to_numpy() - area) / area
print(f'[B3] FullBasin basins holding a MERIT-Basins river system: {per.index.size} of {nb} (to the sea {int((~endo).sum())}, '
      f'endorheic {int(endo.sum())}); holding more than one: {int((per["size"] > 1).sum())}')
for name, k in (('to the sea', ~endo), ('endorheic', endo)):
    d = np.abs(rel_diff[k])
    d = d[np.isfinite(d)]
    print(f'[B3] {name}: |sum of MB outlet uparea - FullBasin area| / FullBasin area: median {np.median(d):.4%}; '
          f'< 1 %: {np.mean(d < 0.01):.1%}; < 5 %: {np.mean(d < 0.05):.1%}; > 10 %: {int((d > 0.10).sum())}')
big = np.argsort(-area)[:30]
print(f'[B3] the 30 largest: max |difference| {np.nanmax(np.abs(rel_diff[big])):.3%}')
print(f'{time.time() - t0:.0f} s')
