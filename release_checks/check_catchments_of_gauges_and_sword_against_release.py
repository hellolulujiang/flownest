"""Check flownest.catchments against the release: the nesting of gauges, SWORD reaches and SWORD nodes (parent, upstream count)
and the gauge-node pairs of the crosswalks_with_catchments package, globally.
usage: python -I check_catchments_of_gauges_and_sword_against_release.py <package dir> <global release> <GRDC crosswalk parquet> <SWORD crosswalk dir> <gauge-node pair parquet>
"""
import sys
import time
import numpy as np
import pyarrow.parquet as pq

pkg, release, grdc_table, sword_dir, pair_table = sys.argv[1:6]
sys.path.insert(0, pkg)
from flownest.release import Release          # noqa: E402
from flownest import catchments               # noqa: E402

t0 = time.time()
rel = Release(release)
bad = 0


def report(what, n_diff, n):
    global bad
    bad += n_diff
    print(f'{what}: {n_diff} of {n} differ', flush=True)


def pixel_rows(ids):
    """upstream channel pixel count and upstream area of channel pixels, read once for unique ids."""
    u, back = np.unique(ids, return_inverse=True)
    rows = rel.rows_at(f'pixel_{rel.base}.parquet', u, ['upstream_channel_pixel_count', 'upstream_area_km2'])
    return rows['upstream_channel_pixel_count'][back].astype(np.int64), rows['upstream_area_km2'][back]


# ---------------------------------------------------------------------------------------------------- gauges
st = pq.read_table(grdc_table, columns=['station_seq_id', 'pixel_seq_id', 'upstream_area_km2', 'next_down_station_seq_id',
                                        'upstream_station_count', 'grdc_station_id']).to_pandas()
st = st[st.station_seq_id > 0].sort_values('station_seq_id')
assert (st.pixel_seq_id > 0).all(), 'a numbered gauge off a channel pixel: this check covers channel gauges only'
g_pixel = st.pixel_seq_id.to_numpy(np.int64)
g_upc, g_area = pixel_rows(g_pixel)
report('gauge area = pixel table area', int((g_area != st.upstream_area_km2.to_numpy(np.float32)).sum()), len(st))
g_end = g_pixel + g_upc
g_parent, g_count = catchments.nest(g_pixel, g_end)
seq = st.station_seq_id.to_numpy()
next_down = np.where(g_parent > 0, g_parent, seq)                  # the table names the gauge itself when none is downstream
report('gauge next_down_station_seq_id', int((next_down != st.next_down_station_seq_id.to_numpy()).sum()), len(st))
report('gauge upstream_station_count', int((g_count != st.upstream_station_count.to_numpy()).sum()), len(st))
print(f'  gauges done {time.time() - t0:.0f} s', flush=True)

# ---------------------------------------------------------------------------------------------------- SWORD reaches
re = pq.read_table(f'{sword_dir}/flownest_merit90m_crosswalk_sword_v17c_reach.parquet',
                   columns=['sword_reach_seq_id', 'most_downstream_pixel_seq_id', 'upstream_sword_reach_count']).to_pandas()
re = re[re.sword_reach_seq_id > 0].sort_values('sword_reach_seq_id')
print(f'  {len(re)} numbered reaches', flush=True)
r_pixel = re.most_downstream_pixel_seq_id.to_numpy(np.int64)
r_upc, _ = pixel_rows(r_pixel)
_, r_count = catchments.nest(r_pixel, r_pixel + r_upc)
report('reach upstream_sword_reach_count', int((r_count != re.upstream_sword_reach_count.to_numpy()).sum()), len(re))
print(f'  reaches done {time.time() - t0:.0f} s', flush=True)

# ---------------------------------------------------------------------------------------------------- SWORD nodes
nd = pq.read_table(f'{sword_dir}/flownest_merit90m_crosswalk_sword_v17c_node.parquet',
                   columns=['sword_node_seq_id', 'pixel_seq_id', 'upstream_area_km2', 'sword_node_id', 'sword_reach_id',
                            'upstream_sword_node_count']).to_pandas()
nd = nd[nd.sword_node_seq_id > 0].sort_values('sword_node_seq_id')
print(f'  {len(nd)} numbered nodes', flush=True)
n_pixel = nd.pixel_seq_id.to_numpy(np.int64)
n_upc, n_area = pixel_rows(n_pixel)
report('node area = pixel table area', int((n_area != nd.upstream_area_km2.to_numpy(np.float32)).sum()), len(nd))
n_end = n_pixel + n_upc
n_parent, n_count = catchments.nest(n_pixel, n_end)
report('node upstream_sword_node_count', int((n_count != nd.upstream_sword_node_count.to_numpy()).sum()), len(nd))
print(f'  nodes done {time.time() - t0:.0f} s', flush=True)

# ---------------------------------------------------------------------------------------------------- gauge-node pairs
side, node_seq, ratio = catchments.pair_gauges_with_nodes(g_pixel, g_area, g_end, n_pixel, n_area,
                                                          nd.sword_node_id.to_numpy(), n_end, n_parent)
pr = pq.read_table(pair_table).to_pandas().set_index('station_seq_id').loc[seq]
k = np.maximum(node_seq - 1, 0)
has = node_seq > 0
mine = {
    'pair_side': side.astype(str),
    'pair_on_gauge_pixel': has & (n_pixel[k] == g_pixel),
    'pair_pixel_seq_id': np.where(has, n_pixel[k], -1),
    'pair_sword_node_seq_id': np.where(has, node_seq, -1),
    'pair_sword_node_id': np.where(has, nd.sword_node_id.to_numpy()[k], -1),
    'pair_sword_reach_id': np.where(has, nd.sword_reach_id.to_numpy()[k], -1),
    'pair_area_ratio': ratio,
    'pair_within_10pct': np.abs(ratio - 1) <= 0.10,
}
for c, v in mine.items():
    theirs = pr[c].to_numpy()
    if c in ('pair_pixel_seq_id', 'pair_sword_node_seq_id', 'pair_sword_node_id', 'pair_sword_reach_id'):
        theirs = pr[c].fillna(-1).to_numpy(np.int64)
    if c == 'pair_area_ratio':
        theirs = theirs.astype(np.float64)
        diff = ~((v == theirs) | (np.isnan(v) & np.isnan(theirs)))
        near = np.abs(v - theirs) <= 1e-12 * np.abs(theirs)
        print(f'  pair_area_ratio: {int(diff.sum())} not bit-equal, {int((diff & ~near).sum())} beyond 1e-12 relative')
        report('pair_area_ratio (1e-12)', int((diff & ~near & ~(np.isnan(v) & np.isnan(theirs))).sum()), len(pr))
        continue
    report(c, int((v != theirs).sum()), len(pr))
for s in ('down', 'up', 'none'):
    print(f'  {s}: {int((side == s).sum())}')
w = mine['pair_within_10pct']
print(f'  within 5/10/20 %: {int((np.abs(ratio - 1) <= 0.05).sum())} / {int(w.sum())} / {int((np.abs(ratio - 1) <= 0.20).sum())}; '
      f'of the within 10 %: down {int((w & (side == "down")).sum())} (on the gauge pixel {int((w & mine["pair_on_gauge_pixel"]).sum())}),'
      f' up {int((w & (side == "up")).sum())}')
print('ALL SAME' if bad == 0 else f'{bad} DIFFERENCES', f'{time.time() - t0:.0f} s')
sys.exit(1 if bad else 0)  # exit status 1 when anything differs
