"""Step 4: the per-system reach and node tables joined into global tables; reach and node seq ids numbered in FlowNest
pixel order with the count of SWORD reaches or nodes upstream of each; global summary tables written."""
import os, json, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyogrio

from . import inputs as IN
from . import reaches


CONTS = ['AF', 'AS', 'EU', 'NA', 'OC', 'SA']



def stack(frames):
    """The non-empty tables one under another; when all are empty (a few small subnetworks), the first one, with its
    columns and no rows."""
    frames = list(frames)
    full = [f for f in frames if len(f)]
    return pd.concat(full, ignore_index=True) if full else frames[0].iloc[:0].reset_index(drop=True)

def step_of(D):
    """The step at which each reach was matched (the steps of match_steps_<sid>.csv)."""
    s = np.full(len(D), 'not_matched', object)
    st, k = D.match_stage.values, D.match_kind.values
    s[st == 1] = D.found_window[st == 1].astype(str).values + 'x' + D.found_window[st == 1].astype(str).values
    s[st == 2] = 'beyond_7x7'
    s[(st == 3) & (k == 'between_neighbors')] = 'between_neighbors'
    s[(st == 3) & (k == 'two_paths_against_flow')] = 'against_flow'
    for nm in ('land_part_cover', 'partly_on_neighbor_path', 'within_water_surface', 'short_path_into_neighbor'):
        s[(st == 3) & (D.neighbor_relation.values == nm)] = nm
    ag = D.area_guided.values if 'area_guided' in D else np.full(len(D), '', object)
    s[ag == 'switch'] = 'area_guided_switch'
    s[ag == 'rescue'] = 'area_guided_rescue'
    return s


def merge_reaches(tmp, out):
    """Join the per-system reach and direction tables of tmp/systems/ into global tables in out, with summary tables."""
    os.makedirs(out, exist_ok=True)
    P = pd.read_csv(tmp + 'progress.csv', float_precision='round_trip', keep_default_na=False, na_values=['']).drop_duplicates('subnetwork_id', keep='last')
    S = pd.read_csv(IN.FACE + 'sword_system_to_flownest_basin.csv', float_precision='round_trip', keep_default_na=False, na_values=[''])
    fail = P[P.status != 'ok']
    ok = P.subnetwork_id[P.status == 'ok'].astype(int).tolist()
    sysd = tmp + 'systems/'
    D = pd.concat([pd.read_parquet(sysd + f'sword_reach_to_flownest_path_{s}.parquet') for s in ok], ignore_index=True)
    D = D.rename(columns={'second_path_seq_id': 'runner_up_path_seq_id', 'second_cover': 'runner_up_cover'})
    X = stack(pd.read_csv(sysd + f'flow_direction_reach_{s}.csv', float_precision='round_trip') for s in ok)
    Y = pd.concat([pd.read_csv(sysd + f'flow_direction_neighbor_pairs_{s}.csv', float_precision='round_trip') for s in ok], ignore_index=True)
    W = pd.concat([pd.read_csv(sysd + f'match_steps_{s}.csv', float_precision='round_trip').assign(subnetwork_id=s) for s in ok], ignore_index=True)
    miss = S[S.basin_id.isna()].subnetwork_id.tolist() + fail.subnetwork_id.astype(int).tolist()
    add = []
    for c in CONTS:
        ids = S[(S.continent == c) & S.subnetwork_id.isin(miss)].subnetwork_id.tolist()
        if not ids:
            continue
        g = pyogrio.read_dataframe(IN.gpkg_path(c), layer='reaches', read_geometry=False,
                                   columns=['reach_id', 'subnetwork_id', 'reach_length', 'width', 'lakeflag', 'type', 'main_side', 'facc_flag', 'river_name'],
                                   where=f"subnetwork_id in ({','.join(map(str, ids))})")
        add.append(g)
    if add:
        g = pd.concat(add, ignore_index=True)
        E = pd.DataFrame({c: pd.Series([], dtype=D[c].dtype) for c in D.columns}).reindex(range(len(g)))
        for c in D.columns:
            if D[c].dtype == object:
                E[c] = ''
            elif str(D[c].dtype) in ('float64', 'float32'):
                E[c] = np.nan
            else:
                E[c] = 0
        E['reach_id'], E['subnetwork_id'] = g.reach_id.values, g.subnetwork_id.values
        E['reach_length_km'], E['width_m'] = g.reach_length.values / 1000, g.width.values
        for c in ('lakeflag', 'type', 'main_side', 'river_name'):
            E[c] = g[c].values
        E['sword_facc_flag'] = g.facc_flag.values
        E['match_kind'] = 'not_found'
        nob = g.subnetwork_id.isin(S[S.basin_id.isna()].subnetwork_id).values
        E['not_found_reason'] = np.where(nob, 'no_flownest_land', 'system_run_failed')
        E['own_basin_id'] = pd.array([pd.NA] * len(E), dtype='Int64')
        E = E.astype(D.dtypes.to_dict())
        D = pd.concat([D, E], ignore_index=True)
    D = D.sort_values('reach_id').reset_index(drop=True)
    D['continent'] = D.subnetwork_id.map(S.set_index('subnetwork_id').continent)
    D['matched_at_step'] = step_of(D)
    assert D.reach_id.is_unique
    D.drop(columns=['continent']).to_parquet(out + 'sword_reach_to_flownest_path_global.parquet', index=False)
    D.drop(columns=['continent']).to_csv(out + 'sword_reach_to_flownest_path_global.csv', index=False)
    X.to_csv(out + 'flow_direction_reach_global.csv', index=False)
    Y.to_csv(out + 'flow_direction_neighbor_pairs_global.csv', index=False)

    m = D.match_stage > 0
    L = D.reach_length_km
    rows = []
    for c, g in list(D.groupby('continent')) + [('all', D)]:
        mm = g.match_stage > 0
        rows.append(dict(continent=c, systems=g.subnetwork_id.nunique(), reaches=len(g), matched=int(mm.sum()),
                         matched_pct=round(100 * mm.mean(), 2), length_km=round(g.reach_length_km.sum(), 1),
                         matched_length_pct=round(100 * g.reach_length_km[mm].sum() / g.reach_length_km.sum(), 2),
                         not_matched=int((~mm).sum()),
                         river_reaches=int(((g.type == 1) & (g.lakeflag == 0)).sum()),
                         river_matched_pct=round(100 * mm[(g.type == 1) & (g.lakeflag == 0)].mean(), 2)))
    C = pd.DataFrame(rows)
    C.to_csv(out + 'global_summary_by_continent.csv', index=False)
    order = ['3x3', '5x5', '7x7', 'beyond_7x7', 'between_neighbors', 'against_flow',
             'land_part_cover', 'partly_on_neighbor_path', 'within_water_surface', 'short_path_into_neighbor',
             'area_guided_switch', 'area_guided_rescue']
    st = D[m].groupby('matched_at_step').agg(new_matched=('reach_id', 'size'), new_length_km=('reach_length_km', 'sum')).reindex(order).fillna(0)
    st['new_two_paths'] = D[m & D.match_kind.isin(['two_paths', 'two_paths_against_flow'])].groupby('matched_at_step').size().reindex(order).fillna(0).astype(int)
    st['cum_matched'] = st.new_matched.cumsum().astype(int)
    st['cum_matched_pct'] = (100 * st.cum_matched / len(D)).round(2)
    st['cum_length_pct'] = (100 * st.new_length_km.cumsum() / L.sum()).round(2)
    st['new_length_km'] = st.new_length_km.round(1)
    st = st.reset_index().rename(columns={'matched_at_step': 'step'})
    W['step2'] = np.where(W.step.str.startswith('beyond_7x7'), 'beyond_7x7', W.step)
    chk = W.groupby('step2').new_matched.sum().reindex(order).fillna(0).astype(int)
    st['check_sum_of_system_tables'] = chk.values
    st.to_csv(out + 'global_match_steps.csv', index=False)
    it = W[W.step.str.startswith('beyond_7x7')].groupby('step').new_matched.sum()
    it.index = it.index.str.replace('beyond_7x7_iteration_', '').astype(int)
    it.sort_index().rename('new_matched').rename_axis('stage2_iteration').reset_index().to_csv(out + 'global_beyond_7x7_iterations.csv', index=False)
    R = D[~m].groupby('not_found_reason').agg(reaches=('reach_id', 'size'), length_km=('reach_length_km', 'sum'),
                                               lakeflag_not_0=('lakeflag', lambda x: int((x != 0).sum())),
                                               type_not_1=('type', lambda x: int((x != 1).sum())))
    R['length_km'] = R.length_km.round(1)
    R.sort_values('reaches', ascending=False).reset_index().to_csv(out + 'global_not_found_reasons.csv', index=False)
    g = D.groupby('subnetwork_id')
    Q = pd.DataFrame({'reaches': g.size(), 'matched': g.match_stage.apply(lambda x: int((x > 0).sum())),
                      'length_km': g.reach_length_km.sum().round(1),
                      'matched_length_km': D[m].groupby('subnetwork_id').reach_length_km.sum().reindex(g.size().index).fillna(0).round(1),
                      'neighbor_not_connected': D[m & (D.neighbors_connected < D.neighbors_matched)].groupby('subnetwork_id').size().reindex(g.size().index).fillna(0).astype(int),
                      'reach_direction_reversed': X[X.reach_direction == 'reversed'].assign(s=lambda x: x.reach_id.map(D.set_index('reach_id').subnetwork_id)).groupby('s').size().reindex(g.size().index).fillna(0).astype(int)})
    Q['matched_pct'] = (100 * Q.matched / Q.reaches).round(2)
    Q['matched_length_pct'] = (100 * Q.matched_length_km / Q.length_km).round(2)
    Q = Q.join(S.set_index('subnetwork_id')[['river_name', 'continent', 'basin_id', 'basin_area_km2', 'status']].rename(columns={'status': 'face_status'}))
    Q = Q.join(P.set_index('subnetwork_id')[['status']].rename(columns={'status': 'run_status'}))
    Q.reset_index().to_csv(out + 'global_systems.csv', index=False)
    rd = X.reach_direction.value_counts()
    pdir = Y.pair_direction.value_counts()
    pd.concat([rd.rename('count').rename_axis('value').reset_index().assign(table='reach_direction'),
               pdir.rename('count').rename_axis('value').reset_index().assign(table='pair_direction')])[['table', 'value', 'count']].to_csv(
        out + 'global_flow_direction_summary.csv', index=False)
    T = D[m].groupby('matched_at_step').area_check.value_counts().unstack(fill_value=0).reindex(order).fillna(0).astype(int)
    T.to_csv(out + 'global_area_check_by_step.csv')


NOLAND = ('not_placed', 'face step: the system has no FlowNest land (all reaches are tidal channels in the sea beyond the MERIT coastline)',
          'not_placed', 'a channel in the sea drains no FlowNest land, so any pixel would be a guess')


def no_land_nodes(S, schema):
    """Systems with no FlowNest basin: their nodes read from SWORD and recorded as not placed."""
    ids = S[S.basin_id.isna()]
    out = []
    for cont, g in ids.groupby('continent'):
        con = reaches.open_sword_gpkg(cont)
        q = ','.join(map(str, g.subnetwork_id))
        N = pd.read_sql(f'select subnetwork_id, node_id, reach_id, node_order, dist_out, x, y, width, max_width, lakeflag, n_chan_max, main_side, '
                        f'facc_total from nodes where subnetwork_id in ({q})', con)
        con.close()
        out.append(N)
    N = pd.concat(out, ignore_index=True).sort_values(['reach_id', 'dist_out'], ascending=[True, False]).reset_index(drop=True)
    E = {}
    for f in schema:
        if pa.types.is_string(f.type) or pa.types.is_large_string(f.type):
            E[f.name] = [''] * len(N)
        elif pa.types.is_boolean(f.type):
            E[f.name] = [False] * len(N)
        elif pa.types.is_floating(f.type):
            E[f.name] = [np.nan] * len(N)
        else:
            E[f.name] = [0] * len(N)
    E = pd.DataFrame(E)
    E['subnetwork_id'], E['node_id'], E['reach_id'], E['node_order'] = N.subnetwork_id.values, N.node_id.values, N.reach_id.values, N.node_order.values
    E['dist_out_m'], E['lon'], E['lat'] = N.dist_out.values, N.x.values, N.y.values
    E['width_m'], E['max_width_m'], E['lakeflag'], E['n_chan_max'], E['main_side'] = N.width.values, N.max_width.values, N.lakeflag.values, N.n_chan_max.values, N.main_side.values
    E['sword_facc_total_km2'] = reaches.sword_fill_to_nan(N.facc_total.values)   # SWORD facc unrounded, as in C
    for c, t in zip(('issue', 'cause', 'action', 'why_not_fixed'), NOLAND):
        E[c] = t
    return pa.Table.from_pandas(E, preserve_index=False).cast(schema)


def upstream_pixel_count(pix):
    """upstream_channel_pixel_count from the FlowNest pixel table, which is stored in pixel_seq_id order (row k is pixel
    k + 1); only the row groups needed are read."""
    pf = pq.ParquetFile(IN.FN + 'flownest_merit90m_pixel_1km2.parquet')
    assert pf.schema_arrow.names[0] == 'pixel_seq_id'
    u = np.unique(pix); U = np.zeros(len(u), np.int64); start = 0
    for g in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        a, b = np.searchsorted(u, [start + 1, start + n + 1])
        if b > a:
            t = pf.read_row_group(g, columns=['pixel_seq_id', 'upstream_channel_pixel_count'])
            ids = t.column(0).to_numpy()
            assert ids[0] == start + 1 and ids[-1] == start + n and (np.diff(ids) == 1).all()
            U[a:b] = t.column(1).to_numpy()[u[a:b] - start - 1]
        start += n
    assert (U >= 1).all()
    return u, U


def seq_and_upstream_count(key, tie, ident, u, U):
    """key = FlowNest pixel_seq_id (0 = none). Rows are numbered 1..n in key order, within one key the smaller tie
    (downstream) first, then ident. The upstream count of a row is the number of rows at or after it whose key lies in
    [key, key + upstream channel pixels of that pixel): the FlowNest pixel ranges make these one block."""
    o = np.flatnonzero(key > 0)
    o = o[np.lexsort((ident[o], tie[o], key[o]))]
    ks = key[o]
    j = np.searchsorted(u, ks); assert (u[j] == ks).all()
    seq = np.zeros(len(key), np.int64); seq[o] = np.arange(1, len(o) + 1)
    cnt = np.zeros(len(key), np.int64); cnt[o] = np.searchsorted(ks, ks + U[j], 'left') - np.arange(len(o))
    assert (cnt[o] >= 1).all()
    return seq, cnt


def merge_nodes(tmp, out, all_systems=True):
    """Join the per-system node tables into a global node table in out, add the nodes of systems with no FlowNest land,
    number reach and node seq ids. all_systems=False (a trial on a few systems) skips the global checks."""
    os.makedirs(out, exist_ok=True)
    P = pd.read_csv(tmp + 'progress.csv', float_precision='round_trip', keep_default_na=False, na_values=['']).drop_duplicates('subnetwork_id', keep='last')
    S = pd.read_csv(IN.FACE + 'sword_system_to_flownest_basin.csv', float_precision='round_trip', keep_default_na=False, na_values=[''])
    fail = P[P.status != 'ok']
    ok = P.subnetwork_id[P.status == 'ok'].astype(int).tolist()
    assert not len(fail), fail.to_string()
    if all_systems:
        assert set(ok) == set(S.subnetwork_id[S.basin_id.notna()]), 'some systems not run'
    sysd = tmp + 'systems/'
    ref = pq.read_table(sysd + f'sword_node_to_flownest_pixel_{int(P.sort_values("nodes").subnetwork_id.iloc[-1])}.parquet')
    schema = pa.schema([pa.field('subnetwork_id', pa.int64())] + list(ref.schema))
    fN = out + 'sword_node_to_flownest_pixel_global.parquet'
    w = pq.ParquetWriter(fN, schema)
    for s in sorted(ok):
        t = pq.read_table(sysd + f'sword_node_to_flownest_pixel_{s}.parquet')
        t = t.add_column(0, 'subnetwork_id', pa.array(np.full(t.num_rows, s, np.int64)))
        w.write_table(t.select(schema.names).cast(schema))
    w.write_table(no_land_nodes(S, schema))
    w.close()
    RR = pd.concat([pd.read_csv(sysd + f'sword_reach_node_placement_{s}.csv', float_precision='round_trip', keep_default_na=False, na_values=['']).assign(subnetwork_id=s)
                    for s in ok], ignore_index=True)
    SM = stack(pd.read_csv(sysd + f'seam_between_neighbor_reaches_{s}.csv', float_precision='round_trip', keep_default_na=False, na_values=['']).assign(subnetwork_id=s)
               for s in ok)
    for c in ('match_kind', 'orientation', 'line_reach_direction'):
        RR[c] = RR[c].fillna('').astype(str)
    cont = S.set_index('subnetwork_id').continent
    N = pq.read_table(fN, columns=['subnetwork_id', 'node_id', 'reach_id', 'dist_out_m', 'lakeflag', 'pixel_seq_id', 'placed_by', 'orientation',
                                   'along_flownest_flow', 'distance_m', 'nodes_on_this_pixel', 'issue']).to_pandas()
    assert N.node_id.is_unique, 'duplicate node_id'
    nl = N[N.subnetwork_id.isin(S.subnetwork_id[S.basin_id.isna()])].groupby(['subnetwork_id', 'reach_id']).size().rename('nodes').reset_index()
    E = pd.DataFrame({c: pd.Series([np.nan] * len(nl)) if RR[c].dtype == float else pd.Series([0] * len(nl)) if RR[c].dtype != object
                      else pd.Series([''] * len(nl)) for c in RR.columns})
    E['reach_id'], E['subnetwork_id'], E['nodes'], E['match_kind'] = nl.reach_id.values, nl.subnetwork_id.values, nl.nodes.values, 'not_found'
    RR = pd.concat([RR, E.astype(RR.dtypes.to_dict())], ignore_index=True).sort_values('reach_id').reset_index(drop=True)
    assert RR.reach_id.is_unique and RR.nodes.sum() == len(N)
    pix = N.pixel_seq_id.to_numpy(np.int64)
    u, U = upstream_pixel_count(pix[pix > 0])
    nseq, ncnt = seq_and_upstream_count(pix, N.dist_out_m.to_numpy(float), N.node_id.to_numpy(np.int64), u, U)
    K = N.loc[pix > 0, ['reach_id', 'pixel_seq_id', 'dist_out_m']].sort_values(['reach_id', 'pixel_seq_id', 'dist_out_m'])
    K = K.drop_duplicates('reach_id').set_index('reach_id')
    rkey = K.pixel_seq_id.reindex(RR.reach_id).fillna(0).to_numpy(np.int64)
    rseq, rcnt = seq_and_upstream_count(rkey, K.dist_out_m.reindex(RR.reach_id).fillna(0).to_numpy(float), RR.reach_id.to_numpy(np.int64), u, U)
    i = RR.columns.get_loc('reach_id') + 1
    for k, (c, v) in enumerate((('flownest_downstream_pixel_seq_id', rkey), ('sword_reach_seq_id', rseq), ('upstream_sword_reach_count', rcnt))):
        RR.insert(i + k, c, v)
    tw = fN + '.tmp'
    pf = pq.ParquetFile(fN)
    sch = pf.schema_arrow
    j = sch.get_field_index('node_id') + 1
    sch2 = sch.insert(j, pa.field('sword_node_seq_id', pa.int64())).insert(j + 1, pa.field('upstream_sword_node_count', pa.int64()))
    w = pq.ParquetWriter(tw, sch2); a = 0
    for bt in pf.iter_batches(batch_size=1 << 20):
        t = pa.Table.from_batches([bt])
        t = t.add_column(j, 'sword_node_seq_id', pa.array(nseq[a:a + t.num_rows])).add_column(j + 1, 'upstream_sword_node_count',
                                                                                              pa.array(ncnt[a:a + t.num_rows]))
        w.write_table(t); a += t.num_rows
    w.close(); assert a == len(N)
    os.replace(tw, fN)
    m = json.load(open(IN.FN + 'metadata.json'))
    json.dump(dict(flownest_release=os.path.basename(IN.FN.rstrip('/')), numbering_version=m['numbering_version'],
                   numbering_sha256=m['numbering_sha256'], sword_nodes_numbered=int((nseq > 0).sum()), sword_reaches_numbered=int((rseq > 0).sum()),
                   ranges='[first, first + count)', note='sword_node_seq_id and sword_reach_seq_id follow FlowNest pixel_seq_id order, so they '
                   'hold only with this FlowNest numbering; upstream means upstream along FlowNest (MERIT) flow, not SWORD topology'),
              open(out + 'sword_seq_id_numbering.json', 'w'), indent=1)
    cols = ['subnetwork_id'] + [c for c in RR.columns if c != 'subnetwork_id']
    RR[cols].to_csv(out + 'sword_reach_node_placement_global.csv', index=False)
    RR[cols].to_parquet(out + 'sword_reach_node_placement_global.parquet', index=False)
    cols = ['subnetwork_id'] + [c for c in SM.columns if c != 'subnetwork_id']
    SM[cols].to_csv(out + 'seam_between_neighbor_reaches_global.csv', index=False)

    N['continent'] = N.subnetwork_id.map(cont)
    rows = []
    for c, g in list(N.groupby('continent')) + [('all', N)]:
        pl = g.pixel_seq_id > 0
        d = g.distance_m[pl]
        rows.append(dict(continent=c, systems=g.subnetwork_id.nunique(), nodes=len(g), placed=int(pl.sum()), placed_pct=round(100 * pl.mean(), 2),
                         placed_on_matched_path=int((g.placed_by == 'matched_path').sum()),
                         continued_from_neighbor=int((g.placed_by == 'continued_from_neighbor').sum()),
                         not_placed=int((~pl).sum()),
                         lakeflag0_placed_pct=round(100 * pl[g.lakeflag == 0].mean(), 2),
                         distance_median_m=round(d.median(), 1), distance_p90_m=round(d.quantile(.9), 1), distance_p99_m=round(d.quantile(.99), 1),
                         along_flownest_flow_pct_of_placed=round(100 * g.along_flownest_flow[pl].mean(), 2),
                         shares_pixel=int((pl & (g.nodes_on_this_pixel > 1)).sum()),
                         nodes_with_no_issue=int((g.issue == '').sum())))
    C = pd.DataFrame(rows)
    C.to_csv(out + 'global_point_summary_by_continent.csv', index=False)
    flat = N.issue.str.split('; ').explode()
    flat = flat[flat != '']
    I = pd.crosstab(flat, N.continent.reindex(flat.index), margins=True, margins_name='all').drop(index='all')
    I = I.sort_values('all', ascending=False).rename_axis('issue').reset_index()
    I.to_csv(out + 'global_point_issue_counts.csv', index=False)
    F = pq.read_table(fN, columns=['pixel_seq_id', 'cause']).to_pandas()
    nc = F.cause[F.pixel_seq_id == 0].str.split('; ').str[0].value_counts().rename_axis('cause').rename('nodes').reset_index()
    nc.to_csv(out + 'global_point_not_placed_causes.csv', index=False)
    G = SM.groupby('seam').agg(seams=('seam', 'size'), pixels_between_min=('pixels_between', 'min'),
                               pixels_between_median=('pixels_between', 'median'), pixels_between_max=('pixels_between', 'max'))
    G.sort_values('seams', ascending=False).reset_index().to_csv(out + 'global_seam_types.csv', index=False)
    O = RR.orientation.replace('', 'no_node_placed').value_counts().rename_axis('orientation').rename('reaches').reset_index()
    O.to_csv(out + 'global_reach_orientation.csv', index=False)
