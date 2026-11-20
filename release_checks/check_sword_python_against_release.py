"""Check flownest.sword against the release: compare the three SWORD crosswalk tables that the Python package wrote for
some SWORD subnetworks with the rows of those subnetworks in the release (crosswalk_sword_v17c tables), column by column. The node
table has no subnetwork id, so its release rows are those of the SWORD reaches of these subnetworks.

The seq ids (sword_reach_seq_id, upstream_sword_reach_count, sword_node_seq_id, upstream_sword_node_count) number all
SWORD reaches and nodes of the world, so on a few subnetworks they differ: the seq ids are compared by order only (the
rows of one subnetwork must come in the same order), and an upstream count can only be smaller here, since SWORD reaches
of subnetworks not run here can lie upstream on FlowNest (here <= release is checked, and how many are equal is shown; a count is 0 here exactly where it is 0 in the release).
usage: python -I check_sword_python_against_release.py <folder written by flownest.sword> <SWORD crosswalk folder of the release> [max rows shown]
"""
import glob
import sys
import numpy as np
import pandas as pd

mine_dir, c_dir = sys.argv[1].rstrip('/') + '/', sys.argv[2].rstrip('/') + '/'
show = int(sys.argv[3]) if len(sys.argv) > 3 else 3
PREFIX = 'flownest_merit90m_crosswalk_sword_v17c_'
GLOBAL = {'sword_reach_seq_id', 'upstream_sword_reach_count', 'sword_node_seq_id', 'upstream_sword_node_count'}
KEY = {'subnetwork': 'sword_subnetwork_id', 'reach': 'sword_reach_id', 'node': 'sword_node_id'}


def same_values(a, b):
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    na = a.isna().values & b.isna().values
    if a.dtype.kind in 'fiub' and b.dtype.kind in 'fiub':
        x, y = a.astype(np.float64).values, b.astype(np.float64).values
        return na | (x == y) | (np.isnan(x) & np.isnan(y))
    return na | (a.astype(str).values == b.astype(str).values)


bad = 0
reach_ids = None
for t, key in KEY.items():   # subnetwork, then reach, then node
    mine = pd.read_parquet(glob.glob(mine_dir + PREFIX + t + '.parquet')[0])
    if t != 'node':
        subnetworks = sorted(set(mine.sword_subnetwork_id))
        theirs = pd.read_parquet(c_dir + PREFIX + t + '.parquet', filters=[('sword_subnetwork_id', 'in', subnetworks)])
    else:   # the node table has no subnetwork id: its rows in C are those of the reaches of the subnetworks run here
        theirs = pd.read_parquet(c_dir + PREFIX + t + '.parquet', filters=[('sword_reach_id', 'in', reach_ids)])
    if t == 'reach':
        reach_ids = sorted(set(theirs.sword_reach_id))
    print(f'== {t}: {len(mine)} rows here, {len(theirs)} rows in the release for the same {len(subnetworks)} subnetworks')
    if list(mine.columns) != list(theirs.columns):
        print('  columns differ:', [c for c in theirs if c not in mine], [c for c in mine if c not in theirs])
        bad += 1
    assert mine[key].is_unique and set(mine[key]) == set(theirs[key]), 'the rows differ'
    th0 = theirs.set_index(key).reindex(mine[key])
    for c in [c for c in theirs if c in GLOBAL and c.startswith('upstream_')]:
        x, y = mine[c].values, th0[c].values
        n = int((x > y).sum())
        z = int(((x >= 1) != (y >= 1)).sum())                   # each placed object counts itself: >= 1 where the release has >= 1
        bad += n + z
        print(f'  {c:40s} here <= release: {"yes" if n == 0 else f"{n} larger"}, >= 1 where the release is: '
              f'{"yes" if z == 0 else f"{z} differ"} ({int((x == y).sum())} of {len(x)} equal)')
    for c in [c for c in theirs if c in GLOBAL and not c.startswith('upstream_')]:
        sm = mine[mine[c] > 0].sort_values(c)[key].values
        st = theirs[theirs[c] > 0].sort_values(c)[key].values
        n = int(len(sm) != len(st) or (sm != st).sum())
        bad += n
        print(f'  {c:40s} order: {"same" if n == 0 else f"{n} differ"} ({len(sm)} numbered)')
    m = mine.set_index(key)
    th = theirs.set_index(key).reindex(m.index)
    for c in [c for c in theirs.columns if c != key and c not in GLOBAL and c in m]:
        ok = same_values(m[c], th[c])
        n = int((~ok).sum())
        bad += n
        if n:
            i = np.flatnonzero(~ok)[:show]
            print(f'  {c:40s} {n} differ; e.g. {key} {m.index[i].tolist()}: here {m[c].values[i].tolist()} | release {th[c].values[i].tolist()}')
print('ALL SAME' if bad == 0 else f'{bad} DIFFERENCES')
sys.exit(1 if bad else 0)  # exit status 1 when anything differs
