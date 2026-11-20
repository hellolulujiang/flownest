"""Step 1: every SWORD reach sampled on the 90 m FlowNest basin grid (all basins, also those under 1 km2), and every SWORD
system (subnetwork) given its main FlowNest basin. Systems that also run into another large basin are classified by how
SWORD links the two basins (estuary or delta, lake, canal, bifurcation, divide, cut inside a reach)."""
import json, sqlite3
from multiprocessing import Pool

import numpy as np, pandas as pd, pyogrio, shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import inputs as IN


LEVELS = ['flowdivide_level1_id', 'flowdivide_level2_id', 'flowdivide_level3_id']


def read_outlet_reaches(fn):
    """1 km2 reach table -> (outlet reach of each basin, channel pixel count of each basin). The outlet reach is the one
    whose next_down points to itself."""
    T = pd.read_parquet(fn + 'flownest_merit90m_reach_1km2.parquet',
                        columns=['reach_seq_id', 'downstream_end_pixel_seq_id', 'next_down_reach_seq_id', 'basin_id',
                                 'channel_pixel_count', 'upstream_area_km2'])
    o = T[T.next_down_reach_seq_id == T.reach_seq_id]
    assert o.basin_id.is_unique and o.downstream_end_pixel_seq_id.is_unique
    assert o.basin_id.nunique() == T.basin_id.nunique(), 'a basin without an outlet reach'
    return o, T.groupby('basin_id').channel_pixel_count.sum(), T


def level_ids_from_groups(fn, outlet_pixel):
    """Outlet pixel seq id -> L1, L2, L3 group ids (the 313 L3 pixel ranges in metadata.json follow one another in pixel
    order)."""
    G = pd.DataFrame(json.load(open(fn + 'metadata.json'))['flowdivide_level3_groups']).sort_values('first_pixel_seq_id')
    start = G.first_pixel_seq_id.values.astype(np.int64)
    end = start + G.channel_pixel_count.values.astype(np.int64)
    assert (start[1:] == end[:-1]).all() and start[0] == 1, 'level-3 groups do not tile the pixel ids'
    i = np.searchsorted(start, outlet_pixel, 'right') - 1
    assert (i >= 0).all() and (outlet_pixel < end[i]).all()
    return {c: G[c].values[i].astype(np.int64) for c in LEVELS}


def build_all_basin_table(fn):
    """One table of all basins, in outlet pixel order: basin_id, L1-L3, outlet_pixel_seq_id, channel_pixel_count,
    upstream_area_km2, is_endorheic."""
    o, npix, T = read_outlet_reaches(fn)
    A = pd.DataFrame({'basin_id': o.basin_id.values.astype(np.int64),
                      'outlet_pixel_seq_id': o.downstream_end_pixel_seq_id.values.astype(np.int64),
                      'upstream_area_km2': o.upstream_area_km2.values.astype(np.float32)})
    A = A.sort_values('outlet_pixel_seq_id').reset_index(drop=True)
    A['channel_pixel_count'] = npix.reindex(A.basin_id).values.astype(np.int64)
    st = A.outlet_pixel_seq_id.values
    assert st[0] == 1 and (st[1:] == st[:-1] + A.channel_pixel_count.values[:-1]).all(), 'basin pixel ranges do not tile'
    i = np.searchsorted(st, T.downstream_end_pixel_seq_id.values.astype(np.int64), 'right') - 1
    assert (A.basin_id.values[i] == T.basin_id.values).all(), 'reach basin_id disagrees with the pixel ranges'
    for c, v in level_ids_from_groups(fn, st).items():
        A[c] = v
    B = pd.read_parquet(fn + 'flownest_merit90m_basin.parquet', columns=['basin_id', 'is_endorheic']).set_index('basin_id')
    A['is_endorheic'] = pd.array(B.is_endorheic.reindex(A.basin_id).values, dtype='Int8')
    return A[['basin_id'] + LEVELS + ['outlet_pixel_seq_id', 'channel_pixel_count', 'upstream_area_km2', 'is_endorheic']]


def check_against_basin_file(A, fn):
    """The basins of at least 1 km2 must equal the release basin table, column by column."""
    B = pd.read_parquet(fn + 'flownest_merit90m_basin.parquet').set_index('basin_id')
    J = A.set_index('basin_id').reindex(B.index)
    assert J.outlet_pixel_seq_id.notna().all(), 'a published basin missing from the outlet reaches'
    bad = {c: int((J[c].values != B[c].values).sum()) for c in J.columns if c in B.columns}
    assert not any(bad.values())
    small = ~A.basin_id.isin(B.index)
    assert A.basin_id[small].min() == len(B) + 1 and A.upstream_area_km2[small].max() < 1


STEP_M = 45.0
COLS = ['reach_id', 'subnetwork_id', 'reach_length', 'best_outlet', 'best_headwater',
        'dist_out', 'n_rch_up', 'n_rch_down', 'rch_id_up_main', 'rch_id_dn_main', 'rch_id_dn_1', 'rch_id_dn_2',
        'rch_id_dn_3', 'main_side', 'is_mainstem', 'end_reach', 'lakeflag', 'type', 'x', 'y', 'river_name']

def sample_continent(cont):
    """One continent: SWORD reach attributes, and the length of each reach in each FlowNest basin (basin -1 = sea),
    from points sampled densely along the reach lines and read on the 90 m pixel grid."""
    f = IN.gpkg_path(cont)
    df = pyogrio.read_dataframe(f, layer='reaches', columns=COLS)
    geoms = df.geometry.values
    xy, gi = shapely.get_coordinates(geoms, return_index=True)
    ok = gi[:-1] == gi[1:]
    a, b, ri = xy[:-1][ok], xy[1:][ok], gi[:-1][ok]
    dlon = b[:, 0] - a[:, 0]
    dlat = b[:, 1] - a[:, 1]
    assert np.abs(dlon).max() < 180, 'segment crosses 180'
    latm = np.radians((a[:, 1] + b[:, 1]) / 2)
    seg_m = np.hypot(dlon * 111320.0 * np.cos(latm), dlat * 110574.0)
    k = np.maximum(1, np.ceil(seg_m / STEP_M)).astype(np.int64)
    s = np.repeat(np.arange(len(k)), k)
    start = np.cumsum(k) - k
    t = (np.arange(len(s)) - start[s] + 0.5) / k[s]
    lon = a[s, 0] + dlon[s] * t
    lat = a[s, 1] + dlat[s] * t
    w = (seg_m / k / 1000.0)[s]
    rr = ri[s]
    del s, t
    ds = IN.Grid(IN.FN + 'flownest_merit90m_pixel_seq_id_1km2.tif')
    gt = ds.gt; band = ds
    W, H = ds.width, ds.height
    col = np.floor((lon - gt[0]) / gt[1]).astype(np.int64) % W
    row = np.floor((lat - gt[3]) / gt[5]).astype(np.int64)
    val = np.full(len(lon), -2147483648, np.int64)
    inside = (row >= 0) & (row < H)
    key = np.where(inside, (row >> 9) * 1000 + (col >> 9), -1)
    order = np.argsort(key, kind='stable')
    ks = key[order]
    ub, first = np.unique(ks, return_index=True)
    last = np.append(first[1:], len(ks))
    for kk, f0, f1 in zip(ub, first, last):
        if kk < 0:
            continue
        br, bc = divmod(int(kk), 1000)
        y0, x0 = br * 512, bc * 512
        arr = band.read(x0, y0, min(512, W - x0), min(512, H - y0))
        idx = order[f0:f1]
        val[idx] = arr[row[idx] - y0, col[idx] - x0]
    bas = pd.read_parquet(IN.BALL, columns=['basin_id', 'outlet_pixel_seq_id'])
    assert (np.diff(bas.outlet_pixel_seq_id.values) > 0).all()
    o = bas.outlet_pixel_seq_id.values
    code = np.full(len(val), -1, np.int64)
    code[val == 0] = 0
    land = (val != -2147483648) & (val != 0)
    code[land] = bas.basin_id.values[np.searchsorted(o, np.abs(val[land]), 'right') - 1]
    g = pd.DataFrame({'ri': rr, 'basin_id': code, 'len_km': w}).groupby(['ri', 'basin_id'], as_index=False).len_km.sum()
    g['reach_id'] = df.reach_id.values[g.ri.values]
    g = g[['reach_id', 'basin_id', 'len_km']]
    att = pd.DataFrame(df.drop(columns='geometry'))
    att['continent'] = cont
    att['geom_len_km'] = np.bincount(ri, weights=seg_m, minlength=len(df)) / 1000.0
    return att, g


def name_of_system(g):
    """System name: the river_name of most reaches; ties go to the longest total length, then to alphabetical order."""
    t = g.groupby('river_name').reach_length.agg(['size', 'sum']).reset_index()
    return t.sort_values(['size', 'sum', 'river_name'], ascending=[False, False, True]).river_name.iloc[0]


def system_to_basin():
    """Each system's FlowNest basin: the basin holding most of its land length (share), the second basin, other large basins
    (over 1000 km2 and at least 5% of its length), the basin of its main outlet, and a status: no_flownest_land,
    also_in_other_large_basin, one_basin (share at least 0.95) or main_basin_plus_small_bits."""
    a = pd.read_parquet(IN.FACE + 'sword_reach_attributes.parquet')
    rb = pd.read_parquet(IN.FACE + 'sword_reach_x_flownest_basin_90m.parquet')
    b = pd.read_parquet(IN.BALL, columns=['basin_id', 'flowdivide_level3_id', 'upstream_area_km2', 'is_endorheic']).set_index('basin_id')
    rs = rb.merge(a[['reach_id', 'subnetwork_id']], on='reach_id')
    allkm = rs.groupby('subnetwork_id').len_km.sum()
    seakm = rs[rs.basin_id == -1].groupby('subnetwork_id').len_km.sum()
    land = rs[rs.basin_id > 0]
    sb = land.groupby(['subnetwork_id', 'basin_id']).len_km.sum().reset_index()
    L = rs[rs.basin_id >= 0].groupby('subnetwork_id').len_km.sum()
    sb['share'] = sb.len_km / sb.subnetwork_id.map(L)
    sb['area_km2'] = sb.basin_id.map(b.upstream_area_km2)
    sb = sb.sort_values(['subnetwork_id', 'len_km', 'basin_id'], ascending=[True, False, True])
    sb['rank'] = sb.groupby('subnetwork_id').cumcount() + 1
    sb.to_parquet(IN.FACE + 'sword_system_x_flownest_basin.parquet', index=False)
    p1 = sb[sb['rank'] == 1].set_index('subnetwork_id')
    p2 = sb[sb['rank'] == 2].set_index('subnetwork_id')
    ol = sb[(sb['rank'] >= 2) & (sb.area_km2 > 1000) & (sb.share >= 0.05)]
    ol_top = ol.drop_duplicates('subnetwork_id').set_index('subnetwork_id')
    m = a[a.main_side == 0]
    outlet = (m.groupby(['subnetwork_id', 'best_outlet']).reach_length.sum().reset_index()
              .sort_values(['subnetwork_id', 'reach_length', 'best_outlet'], ascending=[True, False, True])
              .drop_duplicates('subnetwork_id', keep='first').set_index('subnetwork_id').best_outlet)
    rmaj = (land.sort_values(['reach_id', 'len_km', 'basin_id'], ascending=[True, False, True])
            .drop_duplicates('reach_id', keep='first').set_index('reach_id').basin_id)
    up = a.set_index('reach_id').rch_id_up_main
    def outlet_basin(r):
        for k in range(50):
            if r in rmaj.index: return rmaj[r], k
            r = up.get(r, 0)
            if not r: break
        return np.nan, np.nan
    ob = pd.DataFrame([outlet_basin(r) for r in outlet.values], index=outlet.index, columns=['outlet_basin_id', 'outlet_steps_up'])
    name = a[a.river_name.ne('NODATA')].groupby('subnetwork_id')[['river_name', 'reach_length']].apply(name_of_system)
    S = pd.DataFrame(index=sorted(a.subnetwork_id.unique()))
    S.index.name = 'subnetwork_id'
    S['river_name'] = name
    S['continent'] = a.groupby('subnetwork_id').continent.first()
    S['reach_count'] = a.groupby('subnetwork_id').size()
    S['sword_km'] = a.groupby('subnetwork_id').reach_length.sum() / 1000
    S['sea_fraction'] = (seakm.reindex(S.index).fillna(0) / allkm.reindex(S.index))
    S['basin_id'] = p1.basin_id.reindex(S.index).astype('Int64')
    S['flowdivide_level3_id'] = S.basin_id.map(b.flowdivide_level3_id).astype('Int64')
    S['basin_area_km2'] = S.basin_id.map(b.upstream_area_km2)
    S['is_endorheic'] = S.basin_id.map(b.is_endorheic).astype('Int64')
    S['share'] = p1.share.reindex(S.index)
    S['second_basin_id'] = p2.basin_id.reindex(S.index).astype('Int64')
    S['second_share'] = p2.share.reindex(S.index)
    S['second_area_km2'] = p2.area_km2.reindex(S.index)
    S['other_large_basin_count'] = ol.groupby('subnetwork_id').size().reindex(S.index).fillna(0).astype(int)
    S['other_large_basin_id'] = ol_top.basin_id.reindex(S.index).astype('Int64')
    S['other_large_share'] = ol_top.share.reindex(S.index)
    S['other_large_area_km2'] = ol_top.area_km2.reindex(S.index)
    S['best_outlet_reach'] = outlet.reindex(S.index).astype('Int64')
    S = S.join(ob)
    S['outlet_basin_id'] = S.outlet_basin_id.astype('Int64')
    S['systems_in_basin'] = S.basin_id.map(S.basin_id.value_counts()).astype('Int64')
    S['status'] = np.select([S.basin_id.isna(), S.other_large_basin_count > 0, S.share.round(4) >= 0.95],
                            ['no_flownest_land', 'also_in_other_large_basin', 'one_basin'], 'main_basin_plus_small_bits')
    S.reset_index().to_csv(IN.FACE + 'sword_system_to_flownest_basin.csv', index=False)


INT = ['basin_id', 'flowdivide_level3_id', 'is_endorheic', 'second_basin_id', 'other_large_basin_id', 'best_outlet_reach',
       'outlet_basin_id', 'outlet_steps_up', 'systems_in_basin']


def classify_two_basin_systems():
    """For systems also in another large basin: how SWORD links the two basins, by reach pairs that cross directly or by
    reaches in neither basin that touch both (link_kind estuary_delta, inland_lake, inland_canal, inland_bifurcation,
    inland_divide, cut_inside_reach or unclear); and each reach's own basin."""
    S = pd.read_csv(IN.FACE + 'sword_system_to_flownest_basin.csv', float_precision='round_trip', keep_default_na=False, na_values=[''],
                    dtype={c: 'Int64' for c in INT}).set_index('subnetwork_id')
    S = S.drop(columns=[c for c in S.columns if c.startswith('link_') or c in ('wse_rise_m', 'in_79_part_list')])
    a = pd.read_parquet(IN.FACE + 'sword_reach_attributes.parquet')
    rb = pd.read_parquet(IN.FACE + 'sword_reach_x_flownest_basin_90m.parquet')
    b = pd.read_parquet(IN.BALL, columns=['basin_id', 'flowdivide_level3_id']).set_index('basin_id')

    landtot = rb[rb.basin_id >= 0].groupby('reach_id').len_km.sum()
    top = (rb[rb.basin_id > 0].sort_values(['reach_id', 'len_km', 'basin_id'], ascending=[True, False, True])
           .drop_duplicates('reach_id', keep='first').set_index('reach_id'))
    A = a.set_index('reach_id')
    A['bas'] = top.basin_id
    A['wse'] = pd.concat([pd.read_sql('select reach_id, wse from reaches', sqlite3.connect(f'file:{f}?mode=ro', uri=True)) for f in map(IN.gpkg_path, IN.CONTINENTS)]).set_index('reach_id').wse
    DN = ['rch_id_dn_1', 'rch_id_dn_2', 'rch_id_dn_3']

    links, bridges, rows = [], [], []
    for s in S.index[S.status == 'also_in_other_large_basin']:
        p, q = S.at[s, 'basin_id'], S.at[s, 'other_large_basin_id']
        R = A[A.subnetwork_id == s]
        ids = R.index.values
        pos = pd.Series(np.arange(len(ids)), index=ids)
        e = np.array([(pos[r], pos[d]) for r, dd in zip(ids, R[DN].values) for d in dd if d and d in pos.index]).reshape(-1, 2)
        bas = R.bas.values
        m = np.array([{bas[u], bas[v]} == {p, q} for u, v in e], dtype=bool) if len(e) else np.zeros(0, bool)
        if m.any():
            u, v = e[m, 0], e[m, 1]
            K = pd.DataFrame(dict(subnetwork_id=s, up=ids[u], dn=ids[v], up_basin=bas[u], dn_basin=bas[v],
                                  dist_out_dn_km=R.dist_out.values[v] / 1000, lake_up=R.lakeflag.values[u], lake_dn=R.lakeflag.values[v],
                                  n_dn_up=R.n_rch_down.values[u], wse_up=R.wse.values[u], wse_dn=R.wse.values[v],
                                  x=R.x.values[u], y=R.y.values[u], name_up=R.river_name.values[u], name_dn=R.river_name.values[v]))
            links.append(K)
            lf = set(K.lake_up) | set(K.lake_dn)
            rows.append(dict(subnetwork_id=s, link_direct=True, link_count=len(K), link_dist_to_sea_km=K.dist_out_dn_km.min(),
                             lakeflags=lf, bif=(K.n_dn_up > 1).any(), wse_rise_m=(K.wse_dn - K.wse_up).max()))
            continue
        if not (bas == q).any():
            rows.append(dict(subnetwork_id=s, link_direct=False, link_count=0, cut_inside=True))
            continue
        other = ~np.isin(bas, [p, q])
        if len(e):
            k = other[e[:, 0]] & other[e[:, 1]]
            n, lab = connected_components(coo_matrix((np.ones(k.sum()), (e[k, 0], e[k, 1])), shape=(len(ids),) * 2), directed=False)
            touch = {}
            for x, y in np.vstack([e, e[:, ::-1]]):
                if other[x] and not other[y]:
                    touch.setdefault(lab[x], set()).add(bas[y])
            br = [c for c, t in touch.items() if {p, q} <= t]
        else:
            br = []
        if not br:
            rows.append(dict(subnetwork_id=s, link_direct=False, link_count=0))
            continue
        Rb = R[np.isin(lab, br) & other]
        bridges.append(Rb.reset_index()[['subnetwork_id', 'reach_id', 'bas', 'dist_out', 'lakeflag', 'n_rch_down', 'type', 'river_name']])
        rows.append(dict(subnetwork_id=s, link_direct=False, link_count=len(br), link_dist_to_sea_km=Rb.dist_out.min() / 1000,
                         lakeflags=set(Rb.lakeflag), bif=(Rb.n_rch_down > 1).any(), wse_rise_m=np.nan))

    T = pd.DataFrame(rows).set_index('subnetwork_id')
    has = lambda f: T.lakeflags.apply(lambda z: isinstance(z, set) and f in z)
    d = T.link_dist_to_sea_km
    T['link_kind'] = np.select([T.get('cut_inside', pd.Series(False, index=T.index)).eq(True), d.isna(), (d <= 50) | has(3), has(2), has(1), T.bif.eq(True)],
                               ['cut_inside_reach', 'unclear', 'estuary_delta', 'inland_canal', 'inland_lake', 'inland_bifurcation'], 'inland_divide')
    T['link_count'] = T.link_count.astype('Int64')
    T['link_lakeflags'] = T.lakeflags.apply(lambda z: ';'.join(str(int(i)) for i in sorted(z)) if isinstance(z, set) else '')
    for c in ['link_kind', 'link_direct', 'link_count', 'link_lakeflags']:
        S[c] = T[c]
    S['link_dist_to_sea_km'] = T.link_dist_to_sea_km.round(1)
    S['wse_rise_m'] = T.wse_rise_m.round(1)
    S.reset_index().to_csv(IN.FACE + 'sword_system_to_flownest_basin.csv', index=False)
    pd.concat(links).to_csv(IN.FACE + '_links_to_other_large_basin.csv', index=False)
    pd.concat(bridges).to_csv(IN.FACE + '_bridges_to_other_large_basin.csv', index=False)

    R = a[['reach_id', 'subnetwork_id']].set_index('reach_id')
    R['basin_id'] = top.basin_id.reindex(R.index).astype('Int64')
    R['share'] = (top.len_km / landtot).reindex(R.index).round(4)
    R['system_basin_id'] = R.subnetwork_id.map(S.basin_id).astype('Int64')
    R['flowdivide_level3_id'] = R.basin_id.map(b.flowdivide_level3_id).astype('Int64')
    R.reset_index().to_parquet(IN.FACE + 'sword_reach_to_flownest_basin.parquet', index=False)


LINK_CAUSE = {
    'estuary_delta': 'the two rivers share an estuary, delta or coastal lagoon in SWORD, and MERIT treats the estuary as sea '
                     'so each river reaches the sea in its own basin',
    'inland_lake': 'SWORD links the two rivers through a lake, and MERIT drains the lake to one side',
    'inland_canal': 'SWORD links the two rivers through a canal, which MERIT does not draw',
    'inland_bifurcation': 'the river splits into two branches that reach different outlets, and MERIT has one flow direction per pixel',
    'inland_divide': 'MERIT and SWORD draw the divide differently here (or SWORD links two rivers that are not connected)',
    'cut_inside_reach': 'the divide of another large basin cuts through the middle of one or two reaches (1 to 4 km in total)'}


def add_issue_columns():
    """The four issue columns of the subnetwork table (issue, cause, action, why_not_fixed), one entry per issue."""
    S = pd.read_csv(IN.FACE + 'sword_system_to_flownest_basin.csv', float_precision='round_trip', keep_default_na=False, na_values=[''])
    assert len(S) == 3027 and S.subnetwork_id.is_unique
    iss = [[] for _ in range(len(S))]

    def add(mask, issue, cause, action, why):
        for i in np.flatnonzero(np.asarray(mask)):
            iss[i].append((issue, cause, action, why))

    st = S.status.values
    matched = S.basin_id.notna().values
    assert (matched == (st != 'no_flownest_land')).all()

    add(st == 'no_flownest_land', 'not_matched',
        'all reaches are tidal channels (lakeflag 3) in the sea beyond the MERIT coastline', 'not_matched',
        'a channel in the sea drains no FlowNest land, and the nearest land belongs to many small coastal basins, so any basin would be a guess')
    add(st == 'main_basin_plus_small_bits', 'small_bits_in_other_basins',
        'some reaches or parts of reaches fall in small FlowNest basins beside the main basin (coastal basins at the mouth, '
        'or a divide drawn a little differently)', 'kept: basin_id is the main basin',
        'not needed: each other basin is at most 1000 km2 or holds less than 5% of the river length')
    for k, txt in LINK_CAUSE.items():
        add((st == 'also_in_other_large_basin') & (S.link_kind.values == k), 'also_in_other_large_basin', txt,
            'kept: basin_id is the basin with the most river length, and each reach keeps its own basin in the reach table',
            'SWORD and MERIT are not edited, and the line step matches each reach within its own basin')
    q = (st == 'also_in_other_large_basin')
    assert set(S.link_kind[q]) <= set(LINK_CAUSE), set(S.link_kind[q]) - set(LINK_CAUSE)
    add(S.wse_rise_m.fillna(0).values > 5, 'water_surface_rises_at_link',
        'the SWORD water surface rises by more than 5 m going downstream across the link to the other basin, '
        'so SWORD may link the two rivers wrongly', 'kept',
        'SWORD is not edited, and the line step matches each reach within its own basin so the link does not change any match')
    add(matched & (S.sea_fraction.round(4).values >= 0.05), 'part_in_sea',
        'at least 5% of the river length lies beyond the MERIT coastline (estuary or tidal reaches)', 'kept: basin_id from the land part',
        'MERIT is not edited, and the reaches in the sea are handled in the line step')
    ob = S.outlet_basin_id.values
    add(matched & ~np.isnan(ob) & (ob != S.basin_id.values), 'outlet_in_other_basin',
        'the SWORD main outlet reach lies in another FlowNest basin (mostly a small coastal basin at the mouth)',
        'kept: basin_id by river length', 'the basin with the most river length is more robust than the basin of one outlet reach')
    add(matched & np.isnan(ob), 'outlet_in_sea', 'the SWORD main outlet reach lies beyond the MERIT coastline', 'kept: basin_id by river length',
        'the outlet reach has no basin, so the basin with the most river length is used')
    add(matched & (S.systems_in_basin.fillna(1).values > 1), 'basin_holds_several_systems',
        'several SWORD systems drain into this one FlowNest basin (e.g. rivers into the Caspian Sea, or the Uruguay River into the La Plata)',
        'kept', 'normal: SWORD splits a river network into systems differently from MERIT basins')

    for v in iss:
        for t in v:
            assert all(x and ';' not in x for x in t), t
    for c, k in zip(('issue', 'cause', 'action', 'why_not_fixed'), range(4)):
        S[c] = ['; '.join(t[k] for t in v) for v in iss]
    n = S.issue.str.count('; ') + (S.issue != '')
    for c in ('cause', 'action', 'why_not_fixed'):
        assert (S[c].str.count('; ') + (S[c] != '') == n).all(), c
    f = IN.FACE + 'sword_system_to_flownest_basin_with_issues.csv'
    S.to_csv(f, index=False)


def write_all_basin_table():
    """The table of all basins (also those under 1 km2), checked against the release basin table."""
    A = build_all_basin_table(IN.FN)
    check_against_basin_file(A, IN.FN)
    A.to_parquet(IN.BALL, index=False)


def sample_reaches(processes=6):
    """sample_continent for the six continents, in `processes` worker processes (one after another when processes is 1)."""
    conts = ['AS', 'SA', 'NA', 'EU', 'AF', 'OC']
    if processes <= 1:
        res = [sample_continent(c) for c in conts]
    else:
        with Pool(processes, initializer=IN.configure, initargs=IN.ARGS) as p:
            res = p.map(sample_continent, conts)
    pd.concat([r[0] for r in res], ignore_index=True).to_parquet(IN.FACE + 'sword_reach_attributes.parquet', index=False)
    pd.concat([r[1] for r in res], ignore_index=True).to_parquet(IN.FACE + 'sword_reach_x_flownest_basin_90m.parquet', index=False)


def run(processes=6):
    """Step 1 in order: all basins, reaches on the basin grid, systems to basins, two-basin systems, issue columns."""
    write_all_basin_table()
    sample_reaches(processes)
    system_to_basin()
    classify_two_basin_systems()
    add_issue_columns()
