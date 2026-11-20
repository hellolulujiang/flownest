"""Step 5: reaches rescued from their placed nodes (stage 4), area disagreements explained (area_note, distance to a pixel
of the same area), the issue columns finished, and global summary tables written."""
import numpy as np, pandas as pd, pyarrow.parquet as pq

from . import inputs as IN
from . import reaches


COVER = 0.8
_FB = []


def sword_node_facc_total_once():
    """SWORD v17c (build 0.0.15) facc_total of every node, read once per run -> DataFrame[node_id, reach_id, facc_total]."""
    if not _FB:
        _FB.append(reaches.sword_node_facc_total())
    return _FB[0]
NO_RESCUE = {'mostly_sea', 'not_in_any_basin', 'no_flownest_land'}
NOT_FOUND = {
    'mostly_sea': ('more than half of the reach lies in the sea beyond the MERIT coastline (estuary or tidal channel)',
                   'MERIT has no channel there, so any path would be a guess'),
    'not_in_any_basin': ('the reach has no FlowNest basin of its own (most sample points fall outside every basin, though less than half in the sea)',
                         'there is no basin to search in, so any path would be a guess'),
    'no_flownest_land': ('face step: the system has no FlowNest land (tidal channels in the sea)',
                         'a channel in the sea drains no FlowNest land, so any path would be a guess'),
    'no_matched_neighbor': ('a long path covers at least 80% of the reach beyond the 7x7 window, but no SWORD neighbour reach is matched to confirm it',
                            'beyond 7x7 only paths confirmed by matched neighbour reaches are taken, otherwise a parallel channel could be taken'),
    'candidate_not_connected': ('a long path covers at least 80% of the reach beyond the 7x7 window, but it is neither the path of a matched '
                                'neighbour reach nor joined to it at a confluence',
                                'an unconnected path beyond 7x7 may be a parallel channel, so it is not taken'),
    'only_short_path_nearby': ('within 7x7 only paths of 8 km or shorter cover the reach (MERIT draws this channel as short paths)',
                               'short paths are not used, since in tests they matched the middle of short reaches to side gullies'),
    'partly_on_path': ('only 50% to 80% of the reach lies on one long path (the MERIT channel leaves the SWORD centerline for part of the reach)',
                       'below 80% the match is not certain'),
    'far_from_path': ('the reach is far from every long path (mostly a lake centerline, or a channel MERIT does not draw)',
                      'no path is near enough to match')}


STAGE4_WHY = {
    'few_nodes_continued': 'fewer than 80% of its nodes could be continued from neighbour reaches',
    'continued_onto_short_path': 'its continued nodes lie on paths of 8 km or shorter for more than 20% of the reach '
                                 '(short paths are not used, as in the line step)',
    'path_in_other_basin': 'its continued nodes lie on a path of another basin (the line step searches only in the reach\'s own basin)',
    'two_paths_out_of_order': 'the two paths its continued nodes lie on are in the wrong order along the FlowNest flow',
    'disagrees_with_neighbor': 'the path its continued nodes lie on is out of FlowNest flow order with a matched neighbour reach'}


AREA_WHY = {
    'no_path_cover>=0.5': 'no path in the stage 2 window covers half the reach',
    'no_path_with_same_area': 'no path covering half the reach has an upstream area within 2 times of the SWORD facc_total',
    'two_paths_with_same_area': 'two paths covering half the reach both have an upstream area within 2 times of the SWORD facc_total '
                                '(parallel channels), so the water cannot be told apart',
    'short_path': 'the only path with the same upstream area is 8 km or shorter',
    'same_as_current': 'the path with the same upstream area is already the matched path (the area differs on the part of the reach off it)',
    'direction_unknown': 'the reach has one node only, so the flow direction along it cannot be checked',
    'path_not_along_whole_reach': 'the path with the same upstream area does not reach both ends of the reach',
    'against_flow': 'on the path with the same upstream area, MERIT upa at the upstream quarter of the reach is more than 1.1 times '
                    'that at the downstream quarter (the path runs against the SWORD flow)',
    'neighbor_conflict': 'the path with the same upstream area is neither the same path as nor joined to a trusted neighbour reach',
    'no_trusted_neighbor': 'the path with the same upstream area covers less than 80% of the reach and no matched neighbour reach backs it '
                           '(one whose area agrees, or one already switched or matched by upstream area)',
    'far_without_witness': 'the path with the same upstream area lies off the river corridor (median distance more than half the river '
                           'width plus 180 m, and more than 3 pixels) and no matched neighbour reach backs it (one whose area agrees, '
                           'or one already switched or matched by upstream area)',
    'neighbor_switch_conflict': 'a neighbour reach could be switched or matched in the same round to a path that neither is the same as '
                                'nor joins this one, so neither of the two is taken'}
AREA_NOTE = {
    'side_channel': 'a SWORD side channel (main_side not 0), whose SWORD facc_total is that of the whole river',
    'ghost': 'a SWORD ghost reach (type 6)', 'lake': 'a lake reach (lakeflag 1)',
    'sword_facc_jump_in_reach': 'the SWORD node facc_total changes more than 2 times within the reach (a confluence inside the reach)',
    'sword_facc_not_monotone': 'the SWORD facc_total is out of order with the neighbour reaches',
    'sword_facc_flag_suspect': 'SWORD flags its own facc of the reach suspect (facc_flag)',
    'sword_facc_flag_same_river': 'SWORD flags that the inflows may share drainage area (facc_flag same_river)',
    'sword_facc_flag_braid_share': 'SWORD gives a braided channel a width-based share (facc_flag braid_share), while MERIT D8 draws one channel',
    'sword_facc_flag_delta_arm': 'SWORD gives a distributary an unvalidated share (facc_flag delta_arm), while MERIT D8 draws one channel',
    'sword_facc_flag_filled': 'SWORD filled its facc from neighbours or a prior (facc_flag filled)',
    'no_same_area_pixel_within_11km': 'no MERIT pixel within 11 km has an upstream area within 2 times of the SWORD facc_total'}


SAME_AREA_KM = 11.0


def read_across_180(b, c0, r0, w, h):
    """A window of w columns from column c0 (which may lie west of column 0 or run past the last column: those columns are
    read from the other side of the grid), as float64."""
    idx = np.arange(c0, c0 + w) % b.width
    cut = np.flatnonzero(np.diff(idx) != 1) + 1
    return np.hstack([b.read(int(p[0]), r0, len(p), h) for p in np.split(idx, cut)]).astype(np.float64)


def same_area_distance(D, ix):
    """Reaches whose area disagrees: for each node (v17c 0.0.15 coordinates), the nearest MERIT pixel whose upa is within
    2 times the node's facc_total; the reach gets the median distance (km). A node with none within SAME_AREA_KM is NaN,
    and so is a reach whose nodes all have none."""
    FB = sword_node_facc_total_once()
    rid = D.reach_id.values[ix]
    FB = FB[FB.reach_id.isin(rid)]
    ds = IN.Grid(IN.UPA)
    b, gt, W, H = ds, ds.gt, ds.width, ds.height
    cont = {1: 'AF', 2: 'EU', 3: 'AS', 4: 'AS', 5: 'OC', 6: 'SA', 7: 'NA', 8: 'NA', 9: 'NA'}
    cc_ = pd.Series(rid).astype(str).str[0].astype(int).map(cont).values
    out, edge = {}, [0]
    for c in sorted(set(cc_)):
        ids = rid[cc_ == c]
        con = reaches.open_sword_gpkg(c)
        N = pd.read_sql(f"select node_id, reach_id, x, y from nodes where reach_id in ({','.join(map(str, ids))})", con).merge(
            FB[['node_id', 'facc_total']], on='node_id')
        con.close()
        dd = np.full(len(N), np.nan)
        for k, (x, y, f) in enumerate(zip(N.x.values, N.y.values, N.facc_total.values)):
            col, row = int((x - gt[0]) / gt[1]), int((y - gt[3]) / gt[5])
            kr = int(np.ceil(SAME_AREA_KM * 1000 / reaches.PX_NS_M))
            kc = int(np.ceil(kr / max(np.cos(np.radians(y)), 0.05)))
            if col - kc < 0 or col + kc + 1 > W:
                edge[0] += 1
            c0, r0 = col - kc, max(row - kr, 0)     # the columns go on across 180 degrees, as if the grid went on
            a = read_across_180(b, c0, r0, 2 * kc + 1, min(row + kr + 1, H) - r0)
            rr, cc = np.nonzero((a >= f / reaches.AREA_TOL) & (a <= reaches.AREA_TOL * f))
            if len(rr):
                dk = np.hypot((gt[3] + (rr + r0 + 0.5) * gt[5] - y) * reaches.M_LAT,
                              (gt[0] + (cc + c0 + 0.5) * gt[1] - x) * reaches.M_LON * np.cos(np.radians(y))).min() / 1000
                if dk <= SAME_AREA_KM:
                    dd[k] = dk
        out.update(pd.Series(dd).groupby(N.reach_id.values).median().to_dict())
    return np.array([out.get(r, np.nan) for r in rid])


def area_why(note, code, sea, st, own):
    """Why a reach whose area disagrees was not rematched by area. Stage 4 is checked first (the node step runs after the
    area step, so the reason left by a failed area rematch in the line step is cleared)."""
    if st == 4:
        return 'not tried: matched in the point step (stage 4), after the same-water rule was applied'
    if 'side_channel' in note:
        return 'not switched: MERIT D8 draws one channel, and the SWORD facc_total of a side channel is that of the whole river'
    if 'sword_facc_not_monotone' in note:
        return 'not switched: the SWORD facc_total is out of order with the neighbour reaches, so it is not used to switch'
    if round(sea, 4) >= 0.5:
        return 'not switched: half or more of the reach lies in the sea'
    if code:
        return 'not switched: ' + AREA_WHY[code]
    if not own:
        return 'not switched: the reach has no basin of its own in FlowNest (own_basin_id 0, searched in its subnetwork basin instead), so the same-water rule is not used'
    return 'not tried'


def downstream_of(Pd, a, b):
    """Whether path b is a, or downstream of a along path_down."""
    for _ in range(100000):
        if a == b:
            return True
        n = int(Pd.get(a, a))
        if n == a:
            return False
        a = n
    raise RuntimeError('path_down loop')


def rescue(D, RR, N, Y):
    """Stage 4: returns ({reach_id: (path_seq_id, path2_seq_id, cover, continued_from_reach_id)}, {reach_id: why not
    rescued}, path table). Only long paths (> 8 km, as in the line step) in the reach's own basin; with two paths the
    upstream one flows into the downstream one; and the paths agree with the flow of the neighbours matched in stages 1-3
    (the upstream neighbour's path upstream or the same, the downstream neighbour's path downstream or the same)."""
    cand = D[(D.match_stage == 0) & ~D.not_found_reason.isin(NO_RESCUE)].reach_id
    r = RR.set_index('reach_id').reindex(cand)
    ok = r.index[(r.nodes_continued_from_neighbor >= COVER * r.nodes) & (r.nodes > 0)]
    why = {int(x): 'few_nodes_continued' for x in cand if x not in set(ok)}
    g = N[N.reach_id.isin(ok) & (N.placed_by == 'continued_from_neighbor')]
    M = D[D.match_stage.between(1, 3)].set_index('reach_id')
    nb = Y[Y.reach_id.isin(ok) | Y.dn_reach_id.isin(ok)]
    paths = set(g.path_seq_id) | set(M.path_seq_id.reindex(nb.reach_id).dropna()) | set(M.path2_seq_id.reindex(nb.reach_id).dropna()) \
        | set(M.path_seq_id.reindex(nb.dn_reach_id).dropna())
    paths = sorted(int(x) for x in paths if x > 0)
    if not paths:
        return {}, why, pd.DataFrame(columns=['path_length_km', 'path_level', 'basin_id', 'path_down'])
    bas = pq.read_table(IN.FN + 'flownest_merit90m_reach_1km2.parquet', filters=[('path_seq_id', 'in', paths)],
                        columns=['basin_id']).column('basin_id').unique().to_pylist()
    _, P = reaches.load_paths(bas)
    Pd = P.path_down.to_dict()
    longp = set(P.index[P.path_length_km > reaches.MIN_PATH_KM])
    bid = D.set_index('reach_id').basin_id
    out = {}
    for rid, x in g.groupby('reach_id'):
        n = int(r.nodes[rid])
        x = x[x.path_seq_id.isin(longp)]
        c = x.path_seq_id.value_counts()
        c = c.rename_axis('p').reset_index(name='n').sort_values(['n', 'p'], ascending=[False, True], kind='stable').set_index('p').n
        if len(c) and c.iloc[0] >= COVER * n:
            p, p2, cov = int(c.index[0]), 0, c.iloc[0] / n
        elif len(c) > 1 and c.iloc[:2].sum() >= COVER * n:
            a, b = int(c.index[0]), int(c.index[1])
            up = x.groupby('path_seq_id').dist_out_m.mean()
            p, p2 = (a, b) if up[a] >= up[b] else (b, a)
            cov = c.iloc[:2].sum() / n
            if not downstream_of(Pd, p, p2):
                why[rid] = 'two_paths_out_of_order'
                continue
        else:
            why[rid] = 'continued_onto_short_path'
            continue
        if any(P.basin_id[q] != bid[rid] for q in (p, p2) if q):
            why[rid] = 'path_in_other_basin'
            continue
        pdn = p2 or p
        bad = False
        for u in nb.reach_id[nb.dn_reach_id == rid]:
            if u in M.index:
                ue = int(M.path2_seq_id[u]) or int(M.path_seq_id[u])
                bad |= not (ue in Pd and downstream_of(Pd, ue, p))
        for d in nb.dn_reach_id[nb.reach_id == rid]:
            if d in M.index:
                bad |= not downstream_of(Pd, pdn, int(M.path_seq_id[d]))
        if bad:
            why[rid] = 'disagrees_with_neighbor'
            continue
        out[rid] = (p, p2, cov, int(x.continued_from_reach_id.mode().iloc[0]))
    return out, why, P


def main(ld, pd_):
    """Step 5 on the merged reach table (folder ld) and node table (folder pd_): stage 4 rescue, area notes and issue
    columns of the reaches, written as sword_reach_to_flownest_path_global_with_issues (the node table is read only)."""
    D = pd.read_parquet(ld + 'sword_reach_to_flownest_path_global.parquet')
    X = pd.read_csv(ld + 'flow_direction_reach_global.csv', float_precision='round_trip')
    Y = pd.read_csv(ld + 'flow_direction_neighbor_pairs_global.csv', float_precision='round_trip')
    RR = pd.read_parquet(pd_ + 'sword_reach_node_placement_global.parquet')
    N = pq.read_table(pd_ + 'sword_node_to_flownest_pixel_global.parquet',
                      columns=['reach_id', 'node_id', 'path_seq_id', 'placed_by', 'dist_out_m', 'continued_from_reach_id', 'merit_upa_km2',
                               'merit_upa_float32_km2']).to_pandas()
    assert len(D) == len(RR) and set(D.reach_id) == set(RR.reach_id)
    sw = (D.switch_window.values > 0) & (D.area_guided.values == '')
    assert D.match_stage[sw].isin([1, 2]).all()
    assert (D.match_stage[D.area_guided != ''] == 3).all()
    w = D.switch_window.values[sw].astype(int)
    D.loc[sw, 'found_window'] = w
    D.loc[sw, 'match_stage'] = np.where(w <= 2 * reaches.K_AUTO + 1, 1, 2)
    D.loc[sw, 'matched_at_step'] = np.where(w <= 2 * reaches.K_AUTO + 1, [f'{k}x{k}' for k in w], 'beyond_7x7')
    res, s4why, P = rescue(D, RR, N, Y)
    D['not_found_reason_before_stage4'] = ''
    D['continued_from_reach_id'] = 0
    D['stage4_not_done'] = D.reach_id.map(s4why).fillna('')
    q = D.reach_id.isin(res).values
    v = np.array([res[x] for x in D.reach_id[q]], dtype=np.int64) if q.any() else np.zeros((0, 4), np.int64)
    cov = np.array([res[x][2] for x in D.reach_id[q]], dtype=np.float64)
    D.loc[q, 'not_found_reason_before_stage4'] = D.not_found_reason[q]
    D.loc[q, ['match_stage', 'match_kind', 'neighbor_relation', 'matched_at_step', 'not_found_reason']] = \
        [4, 'continued_from_neighbor', 'continued_from_neighbor', 'continued_from_neighbor', '']
    D.loc[q, 'path_seq_id'], D.loc[q, 'path2_seq_id'], D.loc[q, 'cover'], D.loc[q, 'continued_from_reach_id'] = v[:, 0], v[:, 1], cov, v[:, 3]
    if q.any():
        D.loc[q, 'path_length_km'] = P.path_length_km.reindex(v[:, 0]).values.round(2)
        D.loc[q, 'path_level'] = P.path_level.reindex(v[:, 0]).values
        pp = pd.DataFrame(v[:, :2], index=D.reach_id[q].values, columns=['p', 'p2'])
        g4 = N[N.reach_id.isin(pp.index) & N.merit_upa_float32_km2.notna()]
        on = (g4.path_seq_id.values == pp.p.reindex(g4.reach_id).values) | (g4.path_seq_id.values == pp.p2.reindex(g4.reach_id).values)
        med = g4[on].merit_upa_float32_km2.astype(np.float64).groupby(g4[on].reach_id).median().reindex(D.reach_id[q])
        assert med.notna().all()
        D['merit_upa_on_path_km2'] = D.merit_upa_on_path_km2.astype(np.float64)
        D.loc[q, 'merit_upa_on_path_km2'] = med.round(1).values
        fb = sword_node_facc_total_once().set_index('node_id').facc_total
        g5 = g4[on]
        qn = (g5.merit_upa_float32_km2.astype(np.float64) / g5.node_id.map(fb)).groupby(g5.reach_id).median().reindex(D.reach_id[q])
        D.loc[q, 'area_ratio'] = qn.values
        ok_ = qn.round(3).between(1 / reaches.AREA_TOL, reaches.AREA_TOL).values
        D.loc[q, 'area_check'] = np.where(qn.isna().values, 'no_facc', np.where(ok_, 'agrees', 'disagrees'))
        fr = sword_node_facc_total_once().groupby('reach_id').facc_total.agg(['min', 'max'])
        fmed = D.set_index('reach_id').sword_facc_total_km2.round(1)   # compared as the reach table gives it (1 decimal), as C does
        pr_ = Y[['reach_id', 'dn_reach_id']].drop_duplicates()
        fup = pr_.assign(f=pr_.reach_id.map(fmed)).groupby('dn_reach_id').f.max()
        fdn = pr_.assign(f=pr_.dn_reach_id.map(fmed)).groupby('reach_id').f.max()
        for i in np.flatnonzero(q & (D.area_check.values == 'disagrees')):
            r_, f_ = D.reach_id.iat[i], np.round(D.sword_facc_total_km2.iat[i], 1)
            nt = [('side_channel', D.main_side.iat[i] != 0), ('ghost', D['type'].iat[i] == 6), ('lake', D.lakeflag.iat[i] == 1),
                  ('sword_facc_jump_in_reach', r_ in fr.index and fr['max'][r_] > reaches.AREA_TOL * fr['min'][r_]),
                  ('sword_facc_not_monotone', fup.get(r_, 0) > reaches.AREA_TOL * f_ or fdn.get(r_, np.inf) < f_ / reaches.AREA_TOL),
                  ('sword_facc_flag_' + D.sword_facc_flag.iat[i], D.sword_facc_flag.iat[i] != 'ok')]
            D.iat[i, D.columns.get_loc('area_note')] = ';'.join(z for z, c in nt if c)
        D.loc[q, 'area_guided_not_done'] = ''
    assert ((D.stage4_not_done != '') & q).sum() == 0
    D['same_area_pixel_distance_km'] = np.nan
    ix = np.flatnonzero((D.match_stage.values > 0) & (D.area_check.values == 'disagrees'))
    D.loc[D.index[ix], 'same_area_pixel_distance_km'] = same_area_distance(D, ix).round(2)
    none_ = (D.area_check.values == 'disagrees') & (D.match_stage.values > 0) & D.same_area_pixel_distance_km.isna().values
    D.loc[none_, 'area_note'] = [';'.join(z for z in (a, 'no_same_area_pixel_within_11km') if z) for a in D.area_note[none_].fillna('')]
    e = RR.set_index('reach_id')
    for c in ('nodes', 'nodes_placed', 'upstream_end_pixel_seq_id', 'downstream_end_pixel_seq_id'):
        D[c] = e[c].reindex(D.reach_id).values
    for k, c in enumerate(('flownest_downstream_pixel_seq_id', 'sword_reach_seq_id', 'upstream_sword_reach_count')):
        D.insert(D.columns.get_loc('reach_id') + 1 + k, c, e[c].reindex(D.reach_id).values)
    iss = [[] for _ in range(len(D))]

    def add(mask, issue, cause, action, why):
        for i in np.flatnonzero(np.asarray(mask)):
            iss[i].append((issue, cause, action, why))

    st, kind, reason = D.match_stage.values, D.match_kind.values, D.not_found_reason.values
    m = st > 0
    assert set(reason[~m]) <= set(NOT_FOUND), set(reason[~m]) - set(NOT_FOUND)
    s4n = D.stage4_not_done.values
    tried = (~m) & ~np.isin(reason, list(NO_RESCUE))
    assert (tried == (s4n != '')).all()
    for k, (cau, why) in NOT_FOUND.items():
        add(~m & (reason == k) & ~tried, 'not_matched', cau, 'not_matched', why)
        for w, wt in STAGE4_WHY.items():
            add(tried & (reason == k) & (s4n == w), 'not_matched', cau, 'not_matched', why + ', and in the point step ' + wt)
    rb = D.not_found_reason_before_stage4.values
    for k, (cau, _) in NOT_FOUND.items():
        add((st == 4) & (rb == k), 'matched_by_continuing_from_neighbor', f'line stages 1 to 3: not matched ({cau})',
            'matched: in the point step at least 80% of its nodes were placed by continuing from the placed neighbour reach '
            '(continued_from_reach_id) along the FlowNest flow, and path_seq_id (with path2_seq_id when two paths are needed) holds at least 80% of its nodes', 'not needed: matched')
    add(kind == 'between_neighbors', 'matched_between_neighbors', 'stages 1 and 2 found no confirmed path covering 80% of the reach',
        'matched: the path on which both neighbour reaches end (no distance threshold, see cover and median_pixel_distance)', 'not needed: matched')
    add(kind == 'two_paths_against_flow', 'two_paths_against_flow',
        'SWORD flows from path_seq_id into path2_seq_id, but in MERIT path2_seq_id flows into path_seq_id (a SWORD distributary drawn by MERIT as a tributary)',
        'kept', 'SWORD and MERIT are not edited, and MERIT has one flow direction per pixel')
    rl = D.neighbor_relation.values
    add((st == 3) & (rl == 'land_part_cover'), 'matched_on_land_part',
        'the downstream end of the reach lies in the sea, and stages 1 and 2 counted the sea part in the cover',
        'matched: one long path covers at least 80% of the land part (land_cover) within 7x7, or within the stage 2 window when a matched '
        'neighbour reach is on it, and no other path covers half the land part', 'not needed: matched')
    add((st == 3) & (rl == 'partly_on_neighbor_path'), 'matched_partly_on_path',
        'one long path covers only 50% to 80% of the reach',
        'matched: in the first window (7x7, then half the width plus 180 m, then the stage 2 window) where one long path covers half the '
        'reach, every matched neighbour reach is on it or joins it at a confluence, and no other path covers 30%', 'not needed: matched')
    add((st == 3) & (rl == 'within_water_surface'), 'matched_within_water_surface',
        'no neighbour reach is matched, so stage 2 had no neighbour evidence for this wide river',
        'matched: the only long path covering 80% within half the SWORD width plus 3 pixels, with no other path covering half the reach '
        'and its median distance within half the width plus 180 m', 'not needed: matched')
    add((st == 3) & (rl == 'short_path_into_neighbor'), 'matched_short_path',
        'only a path of 8 km or less covers the reach (stages 1 to 3 use longer paths only)',
        'matched: the short path covers 80% within 7x7, flows into the path of every matched downstream neighbour reach, and its outlet '
        'lies within half the width plus 1.18 km of the downstream end of the reach (reaches shorter than 0.65 km are not taken)',
        'not needed: matched')
    rev = D.reach_id.isin(X.reach_id[X.reach_direction == 'reversed']).values
    add(rev, 'reversed_vs_flownest_flow', 'MERIT upa decreases along the SWORD flow on the matched path (distributary, tidal reach or SWORD direction)',
        'kept', 'SWORD and MERIT are not edited')
    rev4 = (st == 4) & (e.orientation.reindex(D.reach_id).values == 'reversed')
    add(rev4, 'reversed_vs_flownest_flow', 'the placed nodes run against the FlowNest flow (distributary, tidal reach or SWORD direction)',
        'kept', 'SWORD and MERIT are not edited')
    # stages 1 to 3 judge the direction by MERIT upa along the reach line; the placed nodes may still run against the flow.
    # A pair against the flow is left out: its part B nodes run reversed on purpose and it carries two_paths_against_flow
    revn = m & (st < 4) & ~rev & (e.orientation.reindex(D.reach_id).values == 'reversed') & (kind != 'two_paths_against_flow')
    add(revn, 'reversed_vs_flownest_flow', 'the placed nodes run against the FlowNest flow, though MERIT upa does not decrease along the '
        'SWORD line on the matched path (the node order and the line disagree, or the path bends back)', 'kept', 'SWORD and MERIT are not edited')
    pr = Y[Y.pair_direction.isin(['same_path_reversed', 'junction_reversed'])]
    add(D.reach_id.isin(set(pr.reach_id) | set(pr.dn_reach_id)).values, 'neighbor_pair_against_flownest_flow',
        'with a SWORD neighbour reach, the MERIT flow on the shared path or at the confluence runs the other way (see the neighbour pair table)',
        'kept', 'SWORD and MERIT are not edited')
    add(m & (st < 4) & (D.neighbors_connected.values < D.neighbors_matched.values), 'neighbor_not_connected',
        'the matched path is neither the same path as nor joined at a confluence with the path of at least one matched SWORD neighbour reach '
        '(e.g. a SWORD side channel, or a SWORD link across a divide)', 'kept', 'the match stands on its own cover, so it is kept and flagged')
    fx = D.fix_note.fillna('').values
    add(np.char.find(fx.astype(str), 'through_path_switch') >= 0, 'path_switched',
        'the first match differed from the path that both neighbour reaches run on',
        'fixed: switched to the path through both neighbour reaches (old path in switched_from_path_seq_id, unless switched again by '
        'upstream area)', 'not needed: fixed')
    add(np.char.find(fx.astype(str), 'upstream_continuation_switch') >= 0, 'path_switched',
        'the first match differed from the path that the downstream neighbour reach continues upstream on',
        'fixed: switched to that path (old path in switched_from_path_seq_id, unless switched again by upstream area)', 'not needed: fixed')
    add(np.char.find(fx.astype(str), 'pair_rejected_order') >= 0, 'pair_rejected_order',
        'two paths together cover 80% of the reach but in the wrong order along the flow (the tributary part lies downstream)',
        'not used', 'a pair in the wrong order is only taken with neighbour evidence (two_paths_against_flow)')
    snd = D.switch_not_done.fillna('').values != ''
    agsw = D.area_guided.values == 'switch'
    add(snd & ~agsw, 'switch_not_done',
        'the downstream neighbour reach continues upstream on another path, which covers this reach only in a window larger than 7x7',
        'kept the first match', 'a switch backed by one neighbour only is made when the other path covers the reach within 7x7')
    add(snd & agsw, 'switch_not_done',
        'the downstream neighbour reach continues upstream on another path, which covers this reach only in a window larger than 7x7',
        'not switched by this rule (the reach was switched later by upstream area, see path_switched_by_upstream_area)',
        'a switch backed by one neighbour only is made when the other path covers the reach within 7x7')
    add(m & (D.nodes_placed.values == 0), 'no_node_placed', 'the reach is matched to a path but none of its nodes could be placed on it',
        'kept', 'see the node table')
    agd, ack = D.area_guided.values, D.area_check.values
    add(agd == 'switch', 'path_switched_by_upstream_area',
        'the path matched by the geometric rules (stages 1 to 3) has a MERIT upstream area more than 2 times different from the SWORD '
        'facc_total (area_ratio_before_switch)',
        'fixed: switched to the only long path in the stage 2 window that covers half the reach, reaches both ends, does not clearly '
        'run against the SWORD flow (MERIT upa at the upstream quarter of the reach at most 1.1 times that at the downstream quarter, '
        'so a small decrease can remain and is flagged reversed_vs_flow) and has an upstream area within 2 times of the SWORD facc_total, and is the same path as or joined to every '
        'trusted neighbour reach, at least one backing it when it covers less than 80% or lies off the river corridor (the path before '
        'this switch in switched_from_path_seq_id, or the dropped one when the reach had two paths and keeps only one of them)', 'not needed: fixed')
    add(agd == 'rescue', 'matched_by_upstream_area', 'the geometric rules (stages 1 to 3) did not match the reach',
        'matched: the only long path in the stage 2 window that covers half the reach, reaches both ends, does not clearly run '
        'against the SWORD flow (MERIT upa at the upstream quarter of the reach at most 1.1 times that at the downstream quarter, so a '
        'small decrease can remain and is flagged reversed_vs_flow) and has an upstream area within 2 times of the SWORD facc_total, and is the same path as or joined to every trusted '
        'neighbour reach (at least one backing it when it covers less than 80% or lies off the river corridor)', 'not needed: matched')
    fbn = sword_node_facc_total_once().set_index('node_id').facc_total
    gq = N[N.reach_id.isin(D.reach_id[agd != '']) & N.merit_upa_km2.notna()]
    nq = (gq.merit_upa_km2 / gq.node_id.map(fbn)).groupby(gq.reach_id).median().reindex(D.reach_id).values
    nq3 = np.round(nq, 3)   # checked on 3 decimals, printed as computed
    nbad = (agd != '') & np.isfinite(nq) & ((nq3 < 1 / reaches.AREA_TOL) | (nq3 > reaches.AREA_TOL))
    for i in np.flatnonzero(nbad):
        iss[i].append(('node_area_disagrees',
                       f'the path was chosen by its upstream area, but at the pixels of its placed nodes the median of MERIT upa / SWORD node '
                       f'facc_total is {nq[i]:.6g}, more than 2 times off (the nodes may lie on a part of the path, or a tributary pixel, where the '
                       f'area differs from that at the SWORD nodes)', 'kept, flagged for review',
                       'the path itself agrees in upstream area, so the match is kept, check the node pixels'))
    an, ac, sea = D.area_note.fillna('').values, D.area_guided_not_done.fillna('').values, D.sea_fraction.values
    sd, own = D.same_area_pixel_distance_km.values.astype(float), D.own_basin_id.notna().values & (D.own_basin_id.fillna(0).values > 0)
    for i in np.flatnonzero(ack == 'disagrees'):
        nt = [AREA_NOTE[z] for z in an[i].split(';') if z]
        iss[i].append(('upstream_area_disagrees', 'the MERIT upstream area on the matched path differs more than 2 times from the SWORD facc_total'
                       + (' (possible reasons: ' + ', '.join(nt) + ')' if nt else '')
                       + (f' (the median distance from its nodes to the nearest MERIT pixel with an upstream area within 2 times of the '
                          f'node facc_total is {sd[i]:.2f} km, same_area_pixel_distance_km)' if np.isfinite(sd[i]) else ''), 'kept, flagged for review',
                       area_why(an[i], ac[i], sea[i], st[i], own[i])))
    for i in np.flatnonzero(~m & (ac != '')):
        k = [j for j, t in enumerate(iss[i]) if t[0] == 'not_matched']
        assert len(k) == 1
        t = iss[i][k[0]]
        iss[i][k[0]] = (t[0], t[1], t[2], t[3] + ', and by upstream area: ' + AREA_WHY[ac[i]])
    known = {'', 'through_path_switch', 'upstream_continuation_switch', 'pair_rejected_order', 'area_guided_switch'}
    assert all(set(s.split(';')) <= known for s in fx), set(t for s in fx for t in s.split(';')) - known
    for v_ in iss:
        for t in v_:
            assert all(x and ';' not in x for x in t), t
    for c, k in zip(('issue', 'cause', 'action', 'why_not_fixed'), range(4)):
        D[c] = ['; '.join(t[k] for t in v_) for v_ in iss]
    nn = D.issue.str.count('; ') + (D.issue != '')
    for c in ('cause', 'action', 'why_not_fixed'):
        assert (D[c].str.count('; ') + (D[c] != '') == nn).all(), c
    assert (D.issue.str.contains('not_matched') == (D.match_stage == 0)).all()
    by_rule = D.matched_at_step.isin(reaches.RESCUE).values   # matched in the end by one of the four extra rules
    D['land_cover'] = D.land_cover.where(by_rule & (D.matched_at_step.values == 'land_part_cover'))
    D['merit_water_fraction'] = D.merit_water_fraction.where(by_rule)
    D.to_parquet(ld + 'sword_reach_to_flownest_path_global_with_issues.parquet', index=False)
    D.to_csv(ld + 'sword_reach_to_flownest_path_global_with_issues.csv', index=False)

    S = pd.read_csv(IN.FACE + 'sword_system_to_flownest_basin.csv', float_precision='round_trip',
                    keep_default_na=False, na_values=['']).set_index('subnetwork_id').continent
    D['continent'] = D.subnetwork_id.map(S)
    rows = []
    for c, g in list(D.groupby('continent')) + [('all', D)]:
        mm = g.match_stage > 0
        L = g.reach_length_km
        rows.append(dict(continent=c, systems=g.subnetwork_id.nunique(), reaches=len(g), matched_stage1_3=int(g.match_stage.between(1, 3).sum()),
                         matched_stage4=int((g.match_stage == 4).sum()), matched=int(mm.sum()), matched_pct=round(100 * mm.mean(), 2),
                         matched_length_pct=round(100 * L[mm].sum() / L.sum(), 2), not_matched=int((~mm).sum()),
                         type1_lakeflag0_reaches=int(((g.type == 1) & (g.lakeflag == 0)).sum()),
                         type1_lakeflag0_matched_pct=round(100 * mm[(g.type == 1) & (g.lakeflag == 0)].mean(), 2),
                         lake_reaches_lakeflag1=int((g.lakeflag == 1).sum()), lake_reaches_matched_pct=round(100 * mm[g.lakeflag == 1].mean(), 2),
                         reaches_with_no_issue=int((g.issue == '').sum())))
    C = pd.DataFrame(rows)
    C.to_csv(ld + 'global_summary_by_continent_with_stage4.csv', index=False)
    flat = D.issue.str.split('; ').explode()
    flat = flat[flat != '']
    I = pd.crosstab(flat, D.continent.reindex(flat.index), margins=True, margins_name='all').drop(index='all')
    I = I.sort_values('all', ascending=False).rename_axis('issue').reset_index()
    I.to_csv(ld + 'global_line_issue_counts.csv', index=False)
    R = D[D.match_stage == 0].groupby('not_found_reason').agg(reaches=('reach_id', 'size'), length_km=('reach_length_km', 'sum'))
    R['length_km'] = R.length_km.round(1)
    R4 = D[D.match_stage == 4].groupby('not_found_reason_before_stage4').size().rename('rescued_in_stage4')
    R = R.join(R4, how='outer').fillna(0).astype({'reaches': int, 'rescued_in_stage4': int}).sort_values('reaches', ascending=False)
    R.rename_axis('not_found_reason').reset_index().to_csv(ld + 'global_not_found_reasons_with_stage4.csv', index=False)
