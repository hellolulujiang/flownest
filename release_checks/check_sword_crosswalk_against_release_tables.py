"""Check the SWORD crosswalk package against the release it names, from the release's own tables (not SWORD, not the C code):
what each placed node and each reach says about its pixel, path, reach and basin must be what the release says of that pixel.

  [P1] a placed node (pixel_seq_id > 0) has placed_by matched_path or continued_from_neighbor; an unplaced one has row and col
       empty, reach = path = 0 and placed_by empty
  [P2] a placed node's row, col, reach_seq_id_1km2, path_seq_id_1km2 and upstream_area_km2 are the pixel table's
  [P3] a node placed by matched_path lies on its reach's path or path2
  [Q1] a reach's nodes and nodes_placed count its node rows; most_downstream_pixel_seq_id is the smallest pixel of its placed
       nodes (0 when none)
  [Q2] a reach's basin_id is the pixel table's basin of most_downstream_pixel_seq_id (0 when none)
  [Q3] sword_downstream_end / sword_upstream_end_pixel_seq_id are the pixels of its nodes of smallest / largest sword_node_order
       (SWORD's order within the reach, which rises with SWORD dist_out; dist_out itself is not in the package)
  [S1] a subnetwork's basin_id is 0 or a basin (the basins under 1 km2 are not in the basin table, so only counted)
usage: python -I check_sword_crosswalk_against_release_tables.py <package dir> <global release> <SWORD crosswalk dir>
"""
import sys
import time
import numpy as np
import pyarrow.parquet as pq

pkg, release, sword = sys.argv[1:4]
sys.path.insert(0, pkg)
from flownest.release import Release          # noqa: E402

t0 = time.time()
rel = Release(release)
P = f'{sword}/flownest_merit90m_crosswalk_sword_v17c_'
bad = 0


def report(what, n_diff, n):
    global bad
    bad += n_diff
    print(f'{what}: {n_diff} of {n} differ', flush=True)


nd = pq.read_table(P + 'node.parquet', columns=['sword_node_id', 'sword_reach_id', 'sword_node_order', 'pixel_seq_id', 'row', 'col',
                                               'reach_seq_id_1km2', 'path_seq_id_1km2', 'placed_by',
                                               'upstream_area_km2']).to_pandas()
re = pq.read_table(P + 'reach.parquet', columns=['sword_reach_id', 'sword_subnetwork_id', 'basin_id',
                                                'most_downstream_pixel_seq_id', 'path_seq_id_1km2', 'path2_seq_id_1km2', 'nodes',
                                                'nodes_placed', 'sword_upstream_end_pixel_seq_id',
                                                'sword_downstream_end_pixel_seq_id']).to_pandas()
sn = pq.read_table(P + 'subnetwork.parquet', columns=['sword_subnetwork_id', 'basin_id']).to_pandas()
print(f'{len(nd)} nodes, {len(re)} reaches, {len(sn)} subnetworks', flush=True)

# ------------------------------------------------------------------------------------------------ nodes
placed = nd.pixel_seq_id.to_numpy() > 0
by = nd.placed_by.fillna('').to_numpy()
report('[P1] placed nodes without a row or col', int((placed & (nd.row.isna() | nd.col.isna()).to_numpy()).sum()), int(placed.sum()))
report('[P1] placed_by of placed nodes', int((placed & ~np.isin(by, ['matched_path', 'continued_from_neighbor'])).sum()), int(placed.sum()))
off = ~placed
report('[P1] unplaced nodes with a row, col, reach, path or placed_by',
       int((off & (nd.row.notna().to_numpy() | nd.col.notna().to_numpy() | (nd.reach_seq_id_1km2.to_numpy() != 0) |
                   (nd.path_seq_id_1km2.to_numpy() != 0) | (by != ''))).sum()), int(off.sum()))
k = np.flatnonzero(placed)
pix = nd.pixel_seq_id.to_numpy(np.int64)[k]
u, back = np.unique(pix, return_inverse=True)
rows = rel.rows_at(f'pixel_{rel.base}.parquet', u, ['row', 'col', 'reach_seq_id', 'path_seq_id', 'basin_id', 'upstream_area_km2'])
print(f'  pixel table read for {u.size} pixels, {time.time() - t0:.0f} s', flush=True)
for mine, theirs in (('row', 'row'), ('col', 'col'), ('reach_seq_id_1km2', 'reach_seq_id'), ('path_seq_id_1km2', 'path_seq_id')):
    report(f'[P2] node {mine}', int((nd[mine].to_numpy()[k].astype(np.int64) != rows[theirs][back].astype(np.int64)).sum()), k.size)
area = nd.upstream_area_km2.to_numpy()[k]
report('[P2] node upstream_area_km2', int((area != rows['upstream_area_km2'][back]).sum()), k.size)

re_index = re.set_index('sword_reach_id')
m = k[by[k] == 'matched_path']
own_path = re_index.loc[nd.sword_reach_id.to_numpy()[m], ['path_seq_id_1km2', 'path2_seq_id_1km2']].to_numpy(np.int64)
node_path = nd.path_seq_id_1km2.to_numpy(np.int64)[m]
report('[P3] matched_path node off its reach path(s)', int(((node_path != own_path[:, 0]) & (node_path != own_path[:, 1])).sum()), m.size)

# ------------------------------------------------------------------------------------------------ reaches
g = nd.groupby('sword_reach_id', sort=False)
count = g.size()
pl = nd[placed]
gp = pl.groupby('sword_reach_id', sort=False)
n_placed = gp.size()
lowest = gp.pixel_seq_id.min()
ids = re.sword_reach_id.to_numpy()
report('[Q1] nodes', int((count.reindex(ids, fill_value=0).to_numpy() != re.nodes.to_numpy()).sum()), len(re))
report('[Q1] nodes_placed', int((n_placed.reindex(ids, fill_value=0).to_numpy() != re.nodes_placed.to_numpy()).sum()), len(re))
report('[Q1] most_downstream_pixel_seq_id',
       int((lowest.reindex(ids, fill_value=0).to_numpy(np.int64) != re.most_downstream_pixel_seq_id.to_numpy(np.int64)).sum()), len(re))
md = re.most_downstream_pixel_seq_id.to_numpy(np.int64)
has = md > 0
b = np.zeros(len(re), np.int64)
b[has] = rel.rows_at(f'pixel_{rel.base}.parquet', md[has], ['basin_id'])['basin_id']
report('[Q2] reach basin_id', int((b != re.basin_id.to_numpy(np.int64)).sum()), len(re))
# the end nodes: smallest and largest sword_node_order of the reach (all its nodes, placed or not); their pixel, 0 when not placed
srt = nd.sort_values(['sword_reach_id', 'sword_node_order'], kind='stable')
assert not srt.duplicated(['sword_reach_id', 'sword_node_order']).any(), 'two nodes of a reach share a node_order'
down_end = srt.groupby('sword_reach_id', sort=False).pixel_seq_id.first()
up_end = srt.groupby('sword_reach_id', sort=False).pixel_seq_id.last()
report('[Q3] sword_downstream_end_pixel_seq_id',
       int((down_end.reindex(ids, fill_value=0).to_numpy(np.int64) != re.sword_downstream_end_pixel_seq_id.to_numpy(np.int64)).sum()), len(re))
report('[Q3] sword_upstream_end_pixel_seq_id',
       int((up_end.reindex(ids, fill_value=0).to_numpy(np.int64) != re.sword_upstream_end_pixel_seq_id.to_numpy(np.int64)).sum()), len(re))

# ------------------------------------------------------------------------------------------------ subnetworks
n_basins = rel.table('basin.parquet', ['basin_id'])['basin_id'].size
sb = sn.basin_id.to_numpy(np.int64)
print(f'  subnetworks with a basin in the basin table: {int(((sb >= 1) & (sb <= n_basins)).sum())}; with basin 0: '
      f'{int((sb == 0).sum())}; basin under 1 km2 (not in the table): {int((sb > n_basins).sum())}')
report('[S1] subnetwork basin_id below 0', int((sb < 0).sum()), len(sn))
print('ALL SAME' if bad == 0 else f'{bad} DIFFERENCES', f'{time.time() - t0:.0f} s')
sys.exit(1 if bad else 0)  # exit status 1 when anything differs
