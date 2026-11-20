"""Step 2b (one system at a time): the flow direction of each matched reach, and of each pair of SWORD neighbours, checked
against the FlowNest paths they were matched to. Along a path the MERIT upstream area grows downstream, so the median upa
of the sample points on the upstream and downstream 20% of a reach tells same, reversed or equal; a reach on fewer than
5 path pixels is too_short_to_tell. SWORD facc is not used: it was taken from MERIT, so it is no independent evidence."""
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import numpy as np, pandas as pd, shapely

from . import inputs as IN
from . import reaches


DN = [f'rch_id_dn_{i}' for i in (1, 2, 3, 4)]
UP = [f'rch_id_up_{i}' for i in (1, 2, 3, 4)]
END = 0.2
MIN_PIX = 5


def main(sid, out):
    """Flow direction of the matched reaches and neighbour pairs of system sid, written to out as
    flow_direction_reach_<sid>.csv and flow_direction_neighbor_pairs_<sid>.csv."""
    p, R = reaches.load_system(sid)
    T, P = reaches.load_paths(np.r_[reaches.basins_of(p, R), p])
    pdn = P.path_down.to_dict()
    M = pd.read_parquet(out + f'sword_reach_to_flownest_path_{sid}.parquet').set_index('reach_id')
    R = R.set_index('reach_id', drop=False)
    n = len(R)
    import pyogrio
    S = pd.read_csv(IN.FACE + 'sword_system_to_flownest_basin.csv', float_precision='round_trip', keep_default_na=False, na_values=['']).set_index('subnetwork_id')
    if reaches.CACHE.get('cont') == S.continent[sid]:
        G = pd.DataFrame(reaches.CACHE['R'].loc[reaches.CACHE['R'].subnetwork_id.values == sid, ['reach_id', 'dist_out', 'wse']]).set_index('reach_id')
    else:
        G = pyogrio.read_dataframe(IN.gpkg_path(S.continent[sid]), layer='reaches', columns=['reach_id', 'dist_out', 'wse'],
                                   where=f'subnetwork_id = {sid}', read_geometry=False).set_index('reach_id')
    pairs = [(int(r), int(d)) for r in R.index for d in R.loc[r, DN] if d and d in R.index]
    S1 = pd.DataFrame(pairs, columns=['reach_id', 'dn_reach_id'])
    S1['dist_out_km'], S1['dn_dist_out_km'] = G.dist_out[S1.reach_id].values / 1000, G.dist_out[S1.dn_reach_id].values / 1000
    S1['wse_m'], S1['dn_wse_m'] = G.wse[S1.reach_id].values, G.wse[S1.dn_reach_id].values
    lon, lat, w, ri = reaches.sample_lines(R)
    L = np.bincount(ri, weights=w, minlength=n)
    at = pd.Series(w).groupby(ri).cumsum().values - w / 2
    geo = R.geometry.values
    f0 = shapely.get_coordinates(shapely.get_point(geo, 0)); f1 = shapely.get_coordinates(shapely.get_point(geo, -1))
    dd = lambda F, x, y: np.hypot((F[:, 0] - x) * reaches.M_LON * np.cos(np.radians(y)), (F[:, 1] - y) * reaches.M_LAT)
    ux, uy, vx, vy = (R[c].values for c in ('up_x', 'up_y', 'dn_x', 'dn_y'))
    starts_up = dd(f0, ux, uy) + dd(f1, vx, vy) <= dd(f1, ux, uy) + dd(f0, vx, vy)
    how = np.where(R.n_nodes.values >= 2, 'node_dist_out', '').astype(object)
    for i in np.flatnonzero(R.n_nodes.values < 2):
        r = R.index[i]
        for cols, is_dn in ((DN, True), (UP, False)):
            nb = [x for x in R.loc[r, cols] if x and x in R.index]
            if nb:
                u = shapely.union_all(geo[[R.index.get_loc(x) for x in nb]])
                a, b = shapely.distance(shapely.Point(f0[i]), u), shapely.distance(shapely.Point(f1[i]), u)
                starts_up[i] = (a > b) if is_dn else (a < b)
                how[i] = 'downstream_neighbor' if is_dn else 'upstream_neighbor'
                break
    pos_up = np.where(starts_up[ri], at, L[ri] - at) / L[ri]
    Klim = np.clip(np.ceil(np.where(R.width.values > 0, R.width.values, 0) / 2 / reaches.PX_NS_M), reaches.K_NEAR, reaches.K_CAP).astype(int)
    ds = IN.Grid(IN.FN + 'flownest_merit90m_reach_seq_id_1km2.tif'); gt = ds.gt
    col = np.floor((lon - gt[0]) / gt[1]).astype(np.int64); row = np.floor((lat - gt[3]) / gt[5]).astype(np.int64)
    Mx = int(Klim.max()) + 1
    y0, x0 = int(row.min() - Mx), int(col.min() - Mx)
    h, wd = int(row.max() - y0 + Mx + 1), int(col.max() - x0 + Mx + 1)
    A = reaches.read_window(ds, x0, y0, wd, h)
    Pr = reaches.path_raster(A, T)
    del A
    du = IN.Grid(IN.UPA); assert np.allclose(du.gt, gt)
    U = reaches.read_window(du, x0, y0, wd, h)
    rr, cc = row - y0, col - x0
    pts_of = pd.Series(np.arange(len(ri))).groupby(ri).apply(lambda s: s.values)
    upa_pt = np.full(len(ri), np.nan); path_pt = np.zeros(len(ri), np.int64)
    rows, med = [], {}
    for i in range(n):
        r = R.index[i]; d = M.loc[r]
        if d.match_stage <= 0:
            continue
        I = pts_of[i]; K = Klim[i]
        a0, a1, b0, b1 = rr[I].min() - K, rr[I].max() + K + 1, cc[I].min() - K, cc[I].max() + K + 1
        sub = Pr[a0:a1, b0:b1]
        yy, xx = np.nonzero(np.isin(sub, [x for x in (d.path_seq_id, d.path2_seq_id) if x > 0]))
        P = np.c_[rr[I] - a0, cc[I] - b0]
        nb_ = cKDTree(np.c_[yy, xx]).query_ball_point(P, r=K, p=np.inf, return_sorted=True)
        k = np.zeros(len(I), np.int64); ok = np.zeros(len(I), bool)
        for j, q in enumerate(nb_):
            if q:
                q = np.asarray(q); k[j] = q[np.argmin((yy[q] - P[j, 0]) ** 2 + (xx[q] - P[j, 1]) ** 2)]; ok[j] = True
        upa_pt[I[ok]] = U[a0 + yy[k[ok]], b0 + xx[k[ok]]]; path_pt[I[ok]] = sub[yy[k[ok]], xx[k[ok]]]
        J = I[ok]; pu = pos_up[J]; v = upa_pt[J]
        rec = dict(reach_id=r, match_stage=d.match_stage, match_kind=d.match_kind, path_seq_id=d.path_seq_id, path2_seq_id=d.path2_seq_id,
                   reach_length_km=round(L[i], 3), river_name=d.river_name, type=d.type, lakeflag=d.lakeflag, n_nodes=R.n_nodes.iloc[i],
                   orientation_from=how[i], points=len(I), points_on_path=int(ok.sum()),
                   path_pixels=len(set(zip(yy[k[ok]], xx[k[ok]]))))
        if len(J) == 0:
            rec.update(upa_upstream_end_km2=np.nan, upa_downstream_end_km2=np.nan, spearman_position_vs_upa=np.nan,
                       reach_direction='no_points_near_path')
            rows.append(rec)
            continue
        up_s, dn_s = pu <= END, pu >= 1 - END                  # the points in each end fifth; none: those at the end-most point
        up_m = np.median(v[up_s if up_s.any() else pu <= pu.min() + 1e-9])
        dn_m = np.median(v[dn_s if dn_s.any() else pu >= pu.max() - 1e-9])
        rec.update(upa_upstream_end_km2=round(float(up_m), 2), upa_downstream_end_km2=round(float(dn_m), 2),
                   spearman_position_vs_upa=round(float(spearmanr(pu, v)[0]), 3) if len(set(v)) > 1 else np.nan,
                   reach_direction='too_short_to_tell' if rec['path_pixels'] < MIN_PIX else
                   ('same' if dn_m > up_m else ('reversed' if up_m > dn_m else 'equal')))
        pp = path_pt[J]
        for p_ in np.unique(pp):
            med[(r, int(p_))] = float(np.median(v[pp == p_]))
        rows.append(rec)
    X = pd.DataFrame(rows, columns=None if rows else ['reach_id', 'reach_direction', 'orientation_from', 'reach_length_km', 'type', 'lakeflag']).set_index('reach_id')
    lo = lambda r: (M.path2_seq_id[r] or M.path_seq_id[r])
    hi = lambda r: M.path_seq_id[r]
    rel, mu, md = [], [], []
    for r, dn in pairs:
        if M.match_stage[r] <= 0 or M.match_stage[dn] <= 0:
            rel.append('not_both_matched'); mu.append(np.nan); md.append(np.nan); continue
        a, b = lo(r), hi(dn)
        ma, mb = med.get((r, a), np.nan), med.get((dn, b), np.nan)
        mu.append(ma); md.append(mb)
        if a == b:
            rel.append('same_path_ok' if ma < mb else ('same_path_reversed' if ma > mb else 'same_path_too_short_to_tell'))
        elif pdn.get(a) == b:
            rel.append('flows_into_ok')
        elif pdn.get(b) == a:
            rel.append('junction_reversed')
        else:
            Pa, Pb = {M.path_seq_id[r], M.path2_seq_id[r]} - {0}, {M.path_seq_id[dn], M.path2_seq_id[dn]} - {0}
            rel.append('other_connected' if any(x == y or pdn.get(x) == y or pdn.get(y) == x for x in Pa for y in Pb) else 'not_connected')
    S1['pair_direction'] = rel
    S1['upa_median_km2'] = np.round(mu, 2); S1['dn_upa_median_km2'] = np.round(md, 2)
    X.reset_index().to_csv(out + f'flow_direction_reach_{sid}.csv', index=False)
    S1.to_csv(out + f'flow_direction_neighbor_pairs_{sid}.csv', index=False)
