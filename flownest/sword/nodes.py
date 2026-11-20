"""Step 3 (one system at a time): every SWORD node placed on a channel pixel of the FlowNest path its reach was matched to.
Along a path the nodes of a reach only move downstream, with the least total distance; nodes of an unmatched reach are
continued from a placed neighbour reach where that is safe. The issue columns say what is doubtful and why."""
import os, numpy as np, pandas as pd, pyarrow.parquet as pq

from . import inputs as IN
from . import reaches


MARGIN_KM = 5.0
REVERSE_GAIN = 0.8
FAR_EXTRA_M = 180.0
M_LON, M_LAT = 111320.0, 110574.0


NODE_COLS = ('node_id, reach_id, x, y, node_order, dist_out, width, max_width, facc_total, '
             'lakeflag, n_chan_max, main_side, node_length')
NCACHE = {}


def use_continent_nodes(cont):
    """Read all nodes of one continent into memory once (one continent at a time)."""
    if NCACHE.get('cont') == cont:
        return
    NCACHE.clear()
    con = reaches.open_sword_gpkg(cont)
    NCACHE['N'] = pd.read_sql(f'select subnetwork_id, {NODE_COLS} from nodes', con)
    con.close()
    NCACHE['cont'] = cont


def load_nodes(cont, sid):
    """The SWORD nodes of system sid (from memory when its continent is loaded)."""
    if NCACHE.get('cont') == cont:
        return NCACHE['N'][NCACHE['N'].subnetwork_id.values == sid].drop(columns='subnetwork_id').reset_index(drop=True)
    con = reaches.open_sword_gpkg(cont)
    N = pd.read_sql(f'select {NODE_COLS} from nodes where subnetwork_id = ?', con, params=(sid,))
    con.close()
    return N


def dlon_m(glon, x, y):
    """Longitude difference in metres (right across 180 degrees: the difference is wrapped into [-180, 180)). glon is a row,
    x and y a column."""
    return ((glon[None, :] - x[:, None] + 180.0) % 360.0 - 180.0) * M_LON * np.cos(np.radians(y[:, None]))


def topo_order(R, dist_out_max):
    """The order in which reaches are placed: SWORD topological order (upstream reaches first), the larger dist_out first
    within one batch; reaches left in a cycle follow by dist_out."""
    ids = set(dist_out_max.index)
    Rr = R.set_index('reach_id')
    ups = {r: [u for u in Rr.loc[r, [f'rch_id_up_{i}' for i in (1, 2, 3, 4)]] if u in ids and u != r] for r in ids}
    deg = {r: len(set(v)) for r, v in ups.items()}
    dns = {r: [] for r in ids}
    for r, v in ups.items():
        for u in set(v):
            dns[u].append(r)
    out, seen = [], set()
    ready = sorted([r for r in ids if deg[r] == 0], key=lambda r: -dist_out_max[r])
    import heapq
    h = [(-dist_out_max[r], r) for r in ready]; heapq.heapify(h)
    while len(out) < len(ids):
        if not h:
            r = min((r for r in ids if r not in seen), key=lambda r: (-dist_out_max[r], r))
            heapq.heappush(h, (-dist_out_max[r], r)); deg[r] = 0
        _, r = heapq.heappop(h)
        if r in seen:
            continue
        seen.add(r); out.append(r)
        for d in dns[r]:
            deg[d] -= 1
            if deg[d] == 0 and d not in seen:
                heapq.heappush(h, (-dist_out_max[d], d))
    return pd.Index(out)


def path_chains(paths, basins, T):
    """The pixel chain of each path, along the FlowNest flow (DataFrame with pixel_seq_id, row, col, reach_seq_id,
    path_seq_id, upstream_area_km2), and the position of the top pixel of each reach on its chain."""
    if len(paths) == 0:
        return {}, {}
    X = pq.read_table(IN.FN + 'flownest_merit90m_pixel_1km2.parquet',
                      filters=[('basin_id', 'in', sorted(set(basins))), ('path_seq_id', 'in', sorted(set(paths)))],
                      columns=['pixel_seq_id', 'row', 'col', 'reach_seq_id', 'path_seq_id', 'upstream_area_km2',
                               'upstream_channel_pixel_count']).to_pandas()
    dist = T.set_index('reach_seq_id').distance_to_basin_outlet_km
    X['d'] = dist.reindex(X.reach_seq_id).values
    X = X.sort_values(['path_seq_id', 'd', 'pixel_seq_id'], ascending=[True, False, False])
    out, top = {}, {}
    for p, g in X.groupby('path_seq_id', sort=False):
        g = g.reset_index(drop=True)
        dc = np.abs(np.diff(g.col.values)); dc = np.minimum(dc, 432000 - dc)
        step = np.maximum(np.abs(np.diff(g.row.values)), dc)
        assert (step == 1).all(), f'path {p}: the pixel chain is not continuous'
        assert (np.diff(g.upstream_area_km2.values) >= 0).all(), f'path {p}: the upstream area decreases downstream'
        out[p] = g
        f = g.drop_duplicates('reach_seq_id', keep='first')
        top.update(dict(zip(f.reach_seq_id.values, zip([p] * len(f), f.index.values))))
    return out, top


def dp_monotone(Dm):
    """Dm: n_node x n_pos distance matrix. Node positions only move forward along the path, with the least total distance.
    Returns the position index of each node and the total distance."""
    n, m = Dm.shape
    C = np.empty_like(Dm)
    C[0] = Dm[0]
    for i in range(1, n):
        C[i] = Dm[i] + np.minimum.accumulate(C[i - 1])
    j = np.empty(n, np.int64)
    j[-1] = int(np.argmin(C[-1]))
    for i in range(n - 1, 0, -1):
        j[i - 1] = int(np.argmin(C[i - 1, :j[i] + 1]))
    return j, float(C[-1, j[-1]])


def read_points(band, rows, cols):
    """Values of a raster at these points, read in 1024 x 1024 blocks."""
    v = np.full(len(rows), np.nan)
    key = (rows // 1024) * 100000 + cols // 1024
    for k in np.unique(key):
        s = np.flatnonzero(key == k)
        r0, r1, c0, c1 = rows[s].min(), rows[s].max() + 1, cols[s].min(), cols[s].max() + 1
        a = band.read(int(c0 % band.width), int(r0), int(c1 - c0), int(r1 - r0))
        v[s] = a[rows[s] - r0, cols[s] - c0]
    return v


def continue_from_neighbors(N, gi, order, D, Rr, res, part, orient, along, dist_m, upa, chains, top, T, P, gt, wrapped, band, w):
    """Nodes of reaches not matched to a path (rule: place as many as possible, but never wrongly; in doubt, leave it),
    continued only from SWORD neighbour reaches already placed.
      - Downstream neighbour placed (its most upstream node on pixel kv of path q, along the FlowNest flow): the candidate
        pixels are kv and above on q, or a tributary path that flows straight into q at or above kv (then q from the
        confluence down to kv).
      - Upstream neighbour placed (its most downstream node on pixel ku of q): the candidates are ku and below on q, then
        the path q flows into.
      Nodes are continued one by one outward from the end next to the neighbour, each beyond the previous one; from the
      first node that fails on, no node is placed.
      A node needs both: within half the river width + 180 m of the candidate pixel, and no other FlowNest channel more
      than 90 m closer (such as a river beyond a divide).
      Only long paths (> 8 km) are used, as in the reach step: in a global sample a quarter of the nodes continued onto
      short paths fell on gullies of 1-11 km2 beside SWORD rivers 40-2000 m wide. A node that only a short path could take
      stays unplaced (only_short_path_continues).
    Returns how each node was placed (matched_path / continued_from_neighbor / ''), from which reach, why it was not
    placed, and its distance to the nearest FlowNest channel."""
    n = len(N)
    placed_by = np.where(res['pixel_seq_id'] > 0, 'matched_path', '').astype(object)
    why = np.full(n, '', object)
    via = np.zeros(n, np.int64)
    near_any = np.full(n, np.nan)
    Tn = T.set_index('reach_seq_id')
    pdn = P.path_down
    trib = P[P.path_down.values != P.index.values].reset_index().groupby('path_down').path_seq_id.apply(set).to_dict()
    lon_of = lambda c: gt[0] + (c + 0.5) * gt[1]
    lat_of = lambda r: gt[3] + (r + 0.5) * gt[5]
    unm = [r for r in order if D.match_stage[r] <= 0]
    longp = lambda *qs: all(P.path_length_km[q] > reaches.MIN_PATH_KM for q in qs)

    def chain(q):
        if q not in chains:
            c, t = path_chains([q], [int(P.basin_id[q])], T)
            chains.update(c); top.update(t)
        return chains[q]

    def dist(x, y, G):
        glon, glat = lon_of(G.col.values), lat_of(G.row.values)
        if wrapped:
            glon = reaches.wrap_lon(glon)
        return np.hypot(dlon_m(glon, x, y), (glat[None, :] - y[:, None]) * M_LAT)

    changed = True
    while changed:
        changed = False
        for rid in unm:
            I = gi[rid]
            todo = placed_by[I] == ''
            if not todo.any():
                continue
            x, y = N.x.values[I], N.y.values[I]
            mg = MARGIN_KM * 1000; cy = np.cos(np.radians(y.mean()))
            c0 = int((x.min() - mg / (M_LON * cy) - gt[0]) / gt[1]); c1 = int((x.max() + mg / (M_LON * cy) - gt[0]) / gt[1]) + 1
            r0 = int((gt[3] - (y.max() + mg / M_LAT)) / -gt[5]); r1 = int((gt[3] - (y.min() - mg / M_LAT)) / -gt[5]) + 1
            A = reaches.read_window(band, c0, r0, c1 - c0, r1 - r0).astype(np.int64)
            yy, xx = np.nonzero(A > 0)
            if len(yy):
                W = pd.DataFrame({'row': r0 + yy, 'col': c0 + xx})
                near_any[I] = dist(x, y, W).min(1)
                win_paths = set(Tn.path_seq_id.reindex(A[yy, xx]).dropna().astype(np.int64))
            else:
                win_paths = set()
            opts = {'down': [], 'up': []}
            for nb in Rr.loc[rid, [f'rch_id_dn_{i}' for i in (1, 2, 3, 4)]]:
                if nb in gi:
                    k = gi[nb][0]
                    if placed_by[k] and along[k]:
                        q, kv = int(res['path_seq_id'][k]), int(res['chain_pos'][k])
                        segs = [([(q, np.arange(0, kv + 1))], longp(q))]
                        for s in sorted(trib.get(q, set()) & win_paths):
                            J = Tn.next_down_reach_seq_id[chain(s).reach_seq_id.iloc[-1]]
                            qq, kJ = top[J]; assert qq == q
                            if kJ <= kv:
                                segs.append(([(s, np.arange(len(chain(s)))), (q, np.arange(kJ, kv + 1))], longp(s, q)))
                        opts['down'] += [(int(nb), sg, lg) for sg, lg in segs]
            for nb in Rr.loc[rid, [f'rch_id_up_{i}' for i in (1, 2, 3, 4)]]:
                if nb in gi:
                    k = gi[nb][-1]
                    if placed_by[k] and along[k]:
                        q, ku = int(res['path_seq_id'][k]), int(res['chain_pos'][k])
                        segs = [([(q, np.arange(ku, len(chain(q))))], longp(q))]
                        if pdn[q] != q:
                            J = Tn.next_down_reach_seq_id[chain(q).reach_seq_id.iloc[-1]]
                            chain(int(pdn[q]))
                            q2, k0 = top[J]; assert q2 == pdn[q]
                            segs.append(([(q, np.arange(ku, len(chain(q)))), (q2, np.arange(k0, len(chain(q2))))], longp(q, q2)))
                        opts['up'] += [(int(nb), sg, lg) for sg, lg in segs]
            if not opts['down'] and not opts['up']:
                why[I[todo]] = 'no_placed_neighbor_reach'
                continue
            thr = w[I] / 2 + FAR_EXTRA_M
            dmin = np.full(len(I), np.inf)
            short_ok = np.zeros(len(I), bool)
            has_long = any(lg for sd in ('down', 'up') for _, _, lg in opts[sd])
            best = {}
            for side in ('down', 'up'):
                for nb, sg, lg in opts[side]:
                    G = pd.concat([chain(q).iloc[ix].assign(pos=ix) for q, ix in sg], ignore_index=True)
                    Dm = dist(x, y, G)
                    ok = (Dm <= thr[:, None]) & (Dm <= np.nan_to_num(near_any[I], nan=np.inf)[:, None] + 90)
                    if not lg:
                        short_ok |= ok.any(1)
                        continue
                    dmin = np.minimum(dmin, Dm.min(1))
                    acc = []
                    if side == 'down':
                        ub = len(G) - 1
                        for t in range(len(I) - 1, -1, -1):
                            if placed_by[I[t]]:
                                break
                            j = int(np.argmin(Dm[t, :ub + 1]))
                            if not ok[t, j]:
                                break
                            acc.append((t, j)); ub = j
                    else:
                        lb = 0
                        for t in range(len(I)):
                            if placed_by[I[t]]:
                                break
                            j = lb + int(np.argmin(Dm[t, lb:]))
                            if not ok[t, j]:
                                break
                            acc.append((t, j)); lb = j
                    key = (len(acc), -sum(Dm[t, j] for t, j in acc))
                    if acc and (side not in best or key > best[side][0]):
                        best[side] = (key, nb, G, Dm, acc)
            for side in ('down', 'up'):
                if side not in best:
                    continue
                _, nb, G, Dm, acc = best[side]
                for t, j in acc:
                    i = I[t]
                    if placed_by[i]:
                        continue
                    g = G.iloc[j]
                    for k in ('pixel_seq_id', 'row', 'col', 'reach_seq_id', 'path_seq_id'):
                        res[k][i] = g[k]
                    res['chain_pos'][i] = g.pos
                    part[i], orient[i], along[i] = '', 'along_flownest_flow', True
                    dist_m[i], upa[i] = Dm[t, j], g.upstream_area_km2
                    placed_by[i], via[i], why[i] = 'continued_from_neighbor', nb, ''
                    changed = True
            for t in np.flatnonzero(placed_by[I] == ''):
                i = I[t]
                na = near_any[i] if np.isfinite(near_any[i]) else np.inf
                far, closer = dmin[t] > thr[t], dmin[t] > na + 90
                why[i] = ('only_short_path_continues' if not has_long or ((far or closer) and short_ok[t]) else
                          'farther_than_half_width_plus_180m' if far else
                          'another_flownest_channel_closer' if closer else 'gap_between_this_node_and_neighbor')
    return placed_by, via, why, near_any


def main(sid, line_dir, out):
    """Place the nodes of system sid; writes the node table, the per-reach placement summary and the seams between
    neighbour reaches into out."""
    os.makedirs(out, exist_ok=True)
    p, R = reaches.load_system(sid)
    S = pd.read_csv(IN.FACE + 'sword_system_to_flownest_basin.csv', float_precision='round_trip', keep_default_na=False, na_values=['']).set_index('subnetwork_id')
    cont = S.continent[sid]
    D = pd.read_parquet(line_dir + f'sword_reach_to_flownest_path_{sid}.parquet').set_index('reach_id')
    Xd = pd.read_csv(line_dir + f'flow_direction_reach_{sid}.csv', float_precision='round_trip').set_index('reach_id')
    N = load_nodes(cont, sid)
    wrapped = R.attrs.get('wrapped', False)
    N['lon_orig'] = N.x.values
    if wrapped:
        N['x'] = reaches.wrap_lon(N.x.values)
    T, P = reaches.load_paths(np.r_[reaches.basins_of(p, R), p])
    pdn = P.path_down.to_dict()
    Dm_ = D[D.match_stage > 0]
    paths = sorted(set(Dm_.path_seq_id) | (set(Dm_.path2_seq_id) - {0}))
    chains, top = path_chains(paths, P.basin_id.reindex(paths).values, T)
    Tn = T.set_index('reach_seq_id')
    ds = IN.Grid(IN.FN + 'flownest_merit90m_reach_seq_id_1km2.tif'); gt = ds.gt; ds = None
    lon_of = lambda c: gt[0] + (c + 0.5) * gt[1]
    lat_of = lambda r: gt[3] + (r + 0.5) * gt[5]

    N = N.sort_values(['reach_id', 'dist_out', 'node_id'], ascending=[True, False, True], kind='stable').reset_index(drop=True)
    n = len(N)
    Dn0 = D.reindex(N.reach_id)
    w = np.where(N.width.values > 0, N.width.values, np.where(Dn0.width_m.values > 0, Dn0.width_m.values, 0.0))
    res = {k: np.zeros(n, np.int64) for k in ('pixel_seq_id', 'row', 'col', 'reach_seq_id', 'path_seq_id', 'chain_pos')}
    part = np.full(n, '', object)
    dist_m = np.full(n, np.nan); upa = np.full(n, np.nan)
    orient = np.full(n, '', object)
    along = np.zeros(n, bool)
    rrows = []
    end_at = {}
    moved = np.zeros(n, bool)
    near_m = np.full(n, np.nan)
    Rr = R.set_index('reach_id')
    gi = N.groupby('reach_id').indices
    dmax = N.groupby('reach_id').dist_out.max()
    order = topo_order(R, dmax)
    UPC, DNC = [f'rch_id_up_{i}' for i in (1, 2, 3, 4)], [f'rch_id_dn_{i}' for i in (1, 2, 3, 4)]
    head = {q: int(c.pixel_seq_id.values[0]) for q, c in chains.items()}
    area_bad = set(D.index[D.area_check.fillna('') == 'disagrees'])
    def head_of(r):
        dd = D.loc[r]
        return (head[int(dd.path_seq_id)] if dd.match_stage > 0 and dd.match_kind != 'two_paths_against_flow' and r not in area_bad
                else 0)
    up_heads = {r: {head_of(u) for u in Rr.loc[r, UPC] if u in gi and u != r} - {0} for r in gi}
    sib_heads = {r: {head_of(t) for dn in Rr.loc[r, DNC] if dn in gi and dn != r
                     for t in Rr.loc[dn, UPC] if t in gi and t != r and t != dn} - {0} for r in gi}
    def contains(pix, cnt, h):
        return (pix <= h) & (h < pix + cnt)
    jm_above = np.zeros(n, bool)
    jm_below = np.zeros(n, bool)
    jkeep = np.full(n, '', object)
    for rid in order:
        I = gi[rid]
        d = D.loc[rid]
        rec = dict(reach_id=rid, match_kind=d.match_kind, match_stage=int(d.match_stage), path_seq_id=int(d.path_seq_id),
                   path2_seq_id=int(d.path2_seq_id), nodes=len(I), orientation='', total_distance_forward_m=np.nan,
                   total_distance_reversed_m=np.nan, line_reach_direction=Xd.reach_direction.get(rid, ''))
        if d.match_stage <= 0:
            rrows.append(rec); continue
        a, b = int(d.path_seq_id), int(d.path2_seq_id)
        A = chains[a]
        if d.match_kind == 'two_paths':
            J = Tn.next_down_reach_seq_id[A.reach_seq_id.iloc[-1]]
            pb, k0 = top[J]; assert pb == b
            seg = [(A, 'a', np.arange(len(A))), (chains[b], 'b', np.arange(k0, len(chains[b])))]
        elif d.match_kind == 'two_paths_against_flow':
            B = chains[b]
            J = Tn.next_down_reach_seq_id[B.reach_seq_id.iloc[-1]]
            pa, k0 = top[J]; assert pa == a
            seg = [(A, 'a', np.arange(0, k0 + 1)), (B, 'b', np.arange(len(B) - 1, -1, -1))]
        else:
            assert b == 0
            seg = [(A, 'a', np.arange(len(A)))]
        G = pd.concat([s.iloc[ix].assign(part=pt, pos=ix) for s, pt, ix in seg], ignore_index=True)
        x, y = N.x.values[I], N.y.values[I]
        glon, glat = lon_of(G.col.values), lat_of(G.row.values)
        if wrapped:
            glon = reaches.wrap_lon(glon)
        mg = MARGIN_KM * 1000
        cy = np.cos(np.radians(y.mean()))
        keep = ((glon >= x.min() - mg / (M_LON * cy)) & (glon <= x.max() + mg / (M_LON * cy)) &
                (glat >= y.min() - mg / M_LAT) & (glat <= y.max() + mg / M_LAT))
        if not keep.any():
            keep[:] = True
        G = G[keep].reset_index(drop=True); glon, glat = glon[keep], glat[keep]
        Dm = np.hypot(dlon_m(glon, x, y), (glat[None, :] - y[:, None]) * M_LAT)
        jf, cf = dp_monotone(Dm)
        jr, cr = dp_monotone(Dm[:, ::-1]); jr = len(G) - 1 - jr
        rev = len(I) >= 2 and cr < REVERSE_GAIN * cf
        j = jr if rev else jf
        fwd = ~((G.part.values == 'b') & (d.match_kind == 'two_paths_against_flow'))
        if not rev:
            ban = np.zeros(len(G), bool)
            for up in Rr.loc[rid, [f'rch_id_up_{i}' for i in (1, 2, 3, 4)]]:
                if up in end_at:
                    pu, ku = end_at[up]
                    ban |= fwd & (G.path_seq_id.values == pu) & (G.pos.values < ku)
            if ban.any() and not ban.all() and ban[jf].any():
                Db = Dm.copy(); Db[:, ban] = 1e12
                j, _ = dp_monotone(Db)
                moved[I] = j != jf
            if rid in area_bad:
                jkeep[I] = 'area'
            elif d.match_kind != 'two_paths_against_flow' and (sib_heads[rid] or up_heads[rid]):
                pix, cnt = G.pixel_seq_id.values, G.upstream_channel_pixel_count.values
                CS = {h: contains(pix, cnt, h) for h in sib_heads[rid]}
                CU = {h: contains(pix, cnt, h) for h in up_heads[rid]}
                def straddle(jj):
                    return ({h for h, c in CS.items() if c[jj].any() and not c[jj[0]]},
                            {h for h, c in CU.items() if c[jj[-1]] and not c[jj].all()})
                aA, aB = straddle(j)
                if aA or aB:
                    ar = np.arange(len(I))
                    hit = np.zeros(len(I), bool)
                    for h in aA:
                        hit |= CS[h][j]
                    for h in aB:
                        hit |= ~CU[h][j]
                    tol = np.maximum(w[I] / 2 + FAR_EXTRA_M, Dm[ar, j])
                    bn = ban if not ban.all() else np.zeros(len(G), bool)
                    for _ in range(4):
                        b2 = bn.copy()
                        for h in aA:
                            b2 |= CS[h]
                        for h in aB:
                            b2 |= ~CU[h]
                        if b2.all():
                            jkeep[I[hit]] = 'no_pixel'
                            break
                        D2 = Dm.copy(); D2[:, b2] = 1e12
                        j2, _ = dp_monotone(D2)
                        assert not b2[j2].any()
                        nA, nB = straddle(j2)
                        if nA or nB:
                            aA |= nA; aB |= nB
                            continue
                        ch = j2 != j
                        if (Dm[ar, j2][ch] <= tol[ch] + 1e-6).all():
                            jm_below[I[ch & (j2 > j)]] = True; jm_above[I[ch & (j2 < j)]] = True
                            j = j2
                        else:
                            jkeep[I[hit]] = 'far'
                        break
                    else:
                        jkeep[I[hit]] = 'no_pixel'
            if fwd[j[-1]]:
                end_at[rid] = (int(G.path_seq_id.values[j[-1]]), int(G.pos.values[j[-1]]))
        near_m[I] = Dm.min(1)
        rec.update(orientation='reversed' if rev else 'along_flownest_flow', total_distance_forward_m=round(cf, 1),
                   total_distance_reversed_m=round(cr, 1), line_reach_direction=Xd.reach_direction.get(rid, ''))
        g = G.iloc[j]
        for k in ('pixel_seq_id', 'row', 'col', 'reach_seq_id', 'path_seq_id'):
            res[k][I] = g[k].values
        res['chain_pos'][I] = g.pos.values
        part[I] = g.part.values
        dist_m[I] = Dm[np.arange(len(I)), j]
        upa[I] = g.upstream_area_km2.values
        orient[I] = rec['orientation']
        along[I] = fwd[j] != rev
        rrows.append(rec)
    dsr =IN.Grid(IN.FN + 'flownest_merit90m_reach_seq_id_1km2.tif')
    placed_by, via, why, near_any = continue_from_neighbors(N, gi, order, D, Rr, res, part, orient, along, dist_m, upa, chains, top, T, P,
                                                            gt, wrapped, dsr, w)
    dsr = None
    resc = placed_by == 'continued_from_neighbor'
    near_m[resc] = dist_m[resc]
    for k, v in res.items():
        N[k] = v
    N['placed_by'], N['continued_from_reach_id'] = placed_by, via
    N['nearest_flownest_channel_m'] = np.round(near_any, 1)
    N['nearest_on_path_m'] = np.round(near_m, 1)
    N['part'], N['distance_m'], N['orientation'], N['along_flownest_flow'] = part, np.round(dist_m, 1), orient, along
    placed = N.pixel_seq_id.values > 0
    du = IN.Grid(IN.UPA)
    u0 = np.full(n, np.nan)
    u0[placed] = read_points(du, N.row.values[placed], N.col.values[placed])
    assert np.allclose(u0[placed], upa[placed], rtol=1e-6), 'the upstream area of the FlowNest pixel table differs from MERIT upa'
    N['merit_upa_km2'] = np.round(u0, 2)               # as node placement uses it (C rounds the pixel upa to 2 decimals)
    N['merit_upa_float32_km2'] = u0.astype(np.float32)  # the float32 value, used by the stage 4 area check and written in the release
    N['sword_facc_total_km2'] = reaches.sword_fill_to_nan(N.facc_total.values)   # SWORD facc unrounded, as in C
    N['upa_to_facc_ratio'] = N.merit_upa_float32_km2.astype(np.float64) / N.sword_facc_total_km2

    Dn = Dn0
    far = placed & (N.distance_m.values > w / 2 + FAR_EXTRA_M)
    pix_n = N[placed].groupby('pixel_seq_id').size()
    share = placed & (N.pixel_seq_id.map(pix_n).fillna(0).values > 1)
    N['nodes_on_this_pixel'] = np.where(placed, N.pixel_seq_id.map(pix_n).fillna(0), 0).astype(int)
    prev_u = N.groupby('reach_id').merit_upa_km2.shift(1).values
    upa_dec = placed & (N.merit_upa_km2.values < prev_u - 1e-6)
    last = N[placed].groupby('reach_id').tail(1).set_index('reach_id')
    first = N[placed].groupby('reach_id').head(1).set_index('reach_id')
    def downstream_paths(q):
        out = []
        while pdn.get(q, q) != q and len(out) < 1000:
            q = pdn[q]; out.append(q)
        return out
    outlet = T.sort_values(['distance_to_basin_outlet_km', 'reach_seq_id'], kind='stable').drop_duplicates('path_seq_id').set_index('path_seq_id').reach_seq_id
    def entry_pos(qup, qv):
        pv, kJ = top[Tn.next_down_reach_seq_id[outlet[qup]]]
        assert pv == qv
        return int(kJ)
    seam = []
    for r in last.index:
        for dn in Rr.loc[r, [f'rch_id_dn_{i}' for i in (1, 2, 3, 4)]]:
            if not dn or dn not in first.index:
                continue
            u, v = last.loc[r], first.loc[dn]
            pu_, pv_ = int(u.path_seq_id), int(v.path_seq_id)
            gap, npath = np.nan, np.nan
            if not (u.along_flownest_flow and v.along_flownest_flow):
                rel = 'not_checked_against_flow'
            elif pu_ == pv_:
                gap = int(v.chain_pos) - int(u.chain_pos)
                rel = 'same_path_ok' if gap >= 0 else 'same_path_backward'
            elif pv_ in downstream_paths(pu_):
                lst = downstream_paths(pu_); npath = lst.index(pv_)
                gap = int(v.chain_pos) - entry_pos(([pu_] + lst)[npath], pv_)
                rel = ('flows_into_above_merit_confluence' if gap < 0 else
                       'flows_into_ok' if npath == 0 else 'flows_into_via_other_paths')
            elif pu_ in downstream_paths(pv_):
                rel = 'junction_reversed'
            else:
                rel = 'not_connected'
            seam.append(dict(reach_id=r, dn_reach_id=int(dn), node_id=int(u.node_id), dn_node_id=int(v.node_id),
                             path_seq_id=pu_, dn_path_seq_id=pv_, seam=rel, pixels_between=gap, paths_between=npath,
                             upa_km2=u.merit_upa_km2, dn_upa_km2=v.merit_upa_km2,
                             gap_m=round(float(np.hypot(((v.x - u.x + 180) % 360 - 180) * M_LON * np.cos(np.radians(u.y)),
                                                        (v.y - u.y) * M_LAT)), 1)))
    SM = pd.DataFrame(seam, columns=['reach_id', 'dn_reach_id', 'node_id', 'dn_node_id', 'path_seq_id', 'dn_path_seq_id', 'seam',
                                     'pixels_between', 'paths_between', 'upa_km2', 'dn_upa_km2', 'gap_m'])
    SEAM_OK = ['same_path_ok', 'flows_into_ok', 'flows_into_via_other_paths', 'not_checked_against_flow']
    assert set(SM.seam) <= set(SEAM_OK) | {'same_path_backward', 'flows_into_above_merit_confluence', 'junction_reversed', 'not_connected'}
    pcnt = np.zeros(n, np.int64)
    for i in np.flatnonzero(placed):
        c = chains[int(N.path_seq_id.values[i])]
        k = int(N.chain_pos.values[i])
        assert c.pixel_seq_id.values[k] == N.pixel_seq_id.values[i]
        pcnt[i] = c.upstream_channel_pixel_count.values[k]
    alongp = placed & N.along_flownest_flow.values.astype(bool)
    below = np.zeros(n, bool)
    above = np.zeros(n, bool)
    for r, I in gi.items():
        Ia = I[alongp[I]]
        if len(Ia) == 0:
            continue
        pix, cnt = N.pixel_seq_id.values[Ia], pcnt[Ia]
        for h in sib_heads[r]:
            c = contains(pix, cnt, h)
            if c.any() and not c[0]:
                below[Ia[c]] = True
        for h in up_heads[r]:
            c = contains(pix, cnt, h)
            if c[-1] and not c.all():
                above[Ia[~c]] = True
    app = ((N.placed_by.values == 'matched_path') & (N.orientation.values == 'along_flownest_flow')
           & (Dn0.match_kind.values != 'two_paths_against_flow'))
    SA = SM[SM.seam == 'flows_into_above_merit_confluence']
    above_seam = N.node_id.isin(set(SA.dn_node_id)).values
    above |= above_seam
    used_u = set(SA.dn_node_id[[head_of(u) != 0 for u in SA.reach_id]])
    js = above_seam & (jkeep == '') & app
    jkeep[js] = np.where(N.node_id.isin(used_u).values[js], 'whole', 'nb')

    iss = [[] for _ in range(n)]
    def add(mask, issue, cause, action, why):
        for i in np.flatnonzero(mask):
            iss[i].append((issue, cause, action, why))
    reason = Dn.not_found_reason.fillna('').values
    nm = ~placed
    WHY = {'no_placed_neighbor_reach': 'no SWORD neighbour reach is placed to continue from, so any pixel would be a guess',
           'farther_than_half_width_plus_180m': 'the FlowNest channel that continues the placed neighbour reach is farther than half the '
                                                'river width + 180 m (e.g. node in the sea beyond the MERIT coastline, or in a lake bay)',
           'another_flownest_channel_closer': 'a FlowNest channel not connected to the placed neighbour reach (e.g. in another basin across '
                                              'a divide) is more than 90 m closer, so either pixel would be a guess',
           'gap_between_this_node_and_neighbor': 'a node between this one and the placed neighbour reach could not be placed, '
                                                 'and jumping over the gap would be a guess',
           'only_short_path_continues': 'only a path of 8 km or shorter continues the placed neighbour reach near this node, and short '
                                        'paths are not used (in tests they were side gullies of 1 to 11 km2 beside rivers 40 to 2000 m wide)'}
    assert set(why[nm]) <= set(WHY), set(why[nm]) - set(WHY)
    for r in np.unique(reason[nm]):
        for k, txt in WHY.items():
            add(nm & (reason == r) & (why == k), 'not_placed',
                f'line step: reach not matched to a path ({r or "unknown"}), continuing from neighbour reaches failed ({k})', 'not_placed', txt)
    for r in np.unique(reason[resc]):
        add(resc & (reason == r), 'reach_not_matched_to_path', f'line step: reach not matched to a path ({r or "unknown"})',
            'placed: continued from the placed SWORD neighbour reach (continued_from_reach_id) along the FlowNest flow',
            'not needed: placed')
    b_ag = placed & (N.part.values == 'b') & (Dn.match_kind.values == 'two_paths_against_flow')
    rv = placed & (N.orientation.values == 'reversed')
    add(rv & ~b_ag, 'reversed_vs_flownest_flow',
        'SWORD flow direction is opposite to the MERIT flow direction on this path (distributary, tidal or SWORD direction)', 'kept',
        'SWORD and MERIT are not edited, so the node is placed by geometry and flagged')
    add(~rv & b_ag, 'against_flownest_flow_part',
        'SWORD distributary drawn by MERIT as a tributary (MERIT has one flow direction per pixel)', 'kept',
        'SWORD and MERIT are not edited, so the node is placed by geometry and flagged')
    assert (along[placed] == ~(rv ^ b_ag)[placed]).all()
    jmv = jm_above | jm_below
    geo = far & ~moved & ~jmv & (N.distance_m.values - near_m <= 45)
    add(far & moved, 'far_from_pixel', 'moved below the end of the upstream SWORD reach, and the nearest allowed pixel is far', 'kept',
        'the node is not placed upstream of the upstream reach, so it stays on the nearest allowed pixel')
    add(far & jmv & ~moved, 'far_from_pixel', 'moved to the right side of the MERIT confluence, and the nearest allowed pixel is far '
        '(it was already this far before the move)', 'kept',
        'the node is kept on the right side of the MERIT confluence, so it stays on the nearest allowed pixel')
    add(geo, 'far_from_pixel', 'MERIT channel lies away from the SWORD centerline here (the nearest pixel of the matched path is also far)', 'kept',
        'nodes are only placed on the path matched in the line step, and moving them to another path would contradict the line step')
    add(far & ~moved & ~jmv & ~geo, 'far_from_pixel','keeping SWORD node order: the nearest pixel of the matched path is out of order '
        '(MERIT channel shorter than the SWORD centerline, e.g. a cut-off meander)', 'kept', 'node order along the river is kept (rule: positions never go back)')
    add(moved, 'moved_below_upstream_reach_end', 'the best pixels lay upstream of where the upstream SWORD reach ends on the same path',
        'moved: not placed upstream of the last node of the upstream reach', 'not needed: moved')
    add(share, 'shares_pixel', 'MERIT channel is shorter than the SWORD centerline here (meander cut-off or wide river), '
        'so two nodes 200 m apart fall on one 90 m pixel', 'kept', 'allowed by rule 2A')
    add(upa_dec, 'upa_decreases_downstream', 'nodes placed against the FlowNest flow direction', 'kept', 'see the reversed or against-flow flag')
    SEAM_TXT = {
        'same_path_backward': ('seam_backward_on_same_path', 'this node and the neighbour reach node are on one path but in the wrong order '
                               '(the downstream reach starts upstream of where the upstream reach ends)', 'kept',
                               'nodes stay on the pixels nearest the SWORD centerline, see the seam table'),
        'junction_reversed': ('seam_junction_reversed', 'SWORD says the upstream reach flows into the downstream reach, but in MERIT the '
                              'downstream reach path flows into the upstream reach path', 'kept', 'SWORD and MERIT are not edited, see the seam table'),
        'not_connected': ('seam_not_connected', 'this node and the neighbour reach node are on FlowNest paths not connected along the flow '
                          '(e.g. a SWORD side channel that MERIT does not draw)', 'kept', 'SWORD and MERIT are not edited, see the seam table')}
    for t, (iss_, cau, act, wh) in SEAM_TXT.items():
        q = SM.seam == t
        add(N.node_id.isin(set(SM.node_id[q]) | set(SM.dn_node_id[q])).values, iss_, cau, act, wh)
    add(jm_above, 'moved_above_merit_confluence', 'the nearest pixels lay at or below the MERIT confluence where another SWORD reach '
        'flowing into the same downstream reach joins, so the upa would include that reach', 'moved: placed above the MERIT confluence',
        'not needed: moved')
    add(jm_below, 'moved_below_merit_confluence', 'the nearest pixels lay above the MERIT confluence where an upstream SWORD reach '
        'joins, so the upa would not include that reach', 'moved: placed below the MERIT confluence', 'not needed: moved')
    WHYJ = {'far': 'moving it to the other side of the MERIT confluence would put a node farther than half the river width + 180 m '
                   'from its pixel (or farther than it is now)',
            'no_pixel': 'no pixel of the matched path within 5 km of the reach lies on the right side of the MERIT confluence(s) '
                        'and also not upstream of where the upstream SWORD reach ends',
            'area': 'the upstream area of the matched path differs more than 2 times from the SWORD facc_total (area_check disagrees '
                    'on the reach), so the path may be a side channel and the nodes are not moved along it',
            'whole': 'all nodes of this reach lie above that MERIT confluence, so SWORD and MERIT differ in topology here (not by a '
                     'few pixels) and the nodes are not moved',
            'nb': 'the upstream SWORD reach is not used by the confluence rule (it was continued from a neighbour reach, or its '
                  'matched path runs against the flow or has an upstream area more than 2 times different from the SWORD facc_total)',
            '': 'the confluence rule is applied only to reaches matched to one or two paths and placed along the FlowNest flow, '
                'not to reaches that run against the flow on part of their length or that were continued from a neighbour reach'}
    for k, txt in WHYJ.items():
        add(below & (jkeep == k), 'below_merit_confluence', 'the SWORD confluence is downstream of the MERIT confluence here, so this '
            'node lies below the MERIT confluence where another SWORD reach flowing into the same downstream reach joins, and its upa '
            'includes that reach', 'kept', txt)
        add(above & (jkeep == k), 'above_merit_confluence', 'the SWORD confluence is upstream of the MERIT confluence here, so this '
            'node lies above the MERIT confluence where an upstream SWORD reach joins, and its upa does not include that reach', 'kept', txt)
    for v in iss:
        for t in v:
            assert all(x and ';' not in x for x in t), t
    for c, k in zip(('issue', 'cause', 'action', 'why_not_fixed'), range(4)):
        N[c] = ['; '.join(t[k] for t in v) for v in iss]

    RR = pd.DataFrame(rrows).set_index('reach_id')
    g = N[placed].groupby('reach_id')
    RR['nodes_placed'] = g.size().reindex(RR.index).fillna(0).astype(int)
    RR['nodes_continued_from_neighbor'] = N[resc].groupby('reach_id').size().reindex(RR.index).fillna(0).astype(int)
    RR['upstream_end_pixel_seq_id'] = N.groupby('reach_id').pixel_seq_id.first().reindex(RR.index).fillna(0).astype(np.int64)
    RR['downstream_end_pixel_seq_id'] = N.groupby('reach_id').pixel_seq_id.last().reindex(RR.index).fillna(0).astype(np.int64)
    RR['median_distance_m'] = g.distance_m.median().reindex(RR.index).round(1)
    RR['max_distance_m'] = g.distance_m.max().reindex(RR.index).round(1)
    for c, mk in (('nodes_far_from_pixel', far), ('nodes_sharing_pixel', share), ('nodes_upa_decreases', upa_dec)):
        RR[c] = pd.Series(mk, index=N.reach_id).groupby(level=0).sum().reindex(RR.index).fillna(0).astype(int)
    RR = RR.reset_index()

    cols = ['node_id', 'reach_id', 'node_order', 'dist_out', 'lon_orig', 'y', 'width', 'max_width', 'lakeflag', 'n_chan_max', 'main_side',
            'pixel_seq_id', 'row', 'col', 'reach_seq_id', 'path_seq_id', 'placed_by', 'continued_from_reach_id', 'part', 'chain_pos', 'orientation',
            'along_flownest_flow', 'distance_m', 'nearest_on_path_m',
            'nearest_flownest_channel_m', 'nodes_on_this_pixel', 'merit_upa_km2', 'merit_upa_float32_km2', 'sword_facc_total_km2', 'upa_to_facc_ratio', 'issue', 'cause', 'action', 'why_not_fixed']
    N = N[cols].rename(columns={'dist_out': 'dist_out_m', 'width': 'width_m', 'max_width': 'max_width_m', 'lon_orig': 'lon', 'y': 'lat'})
    N.to_parquet(out + f'sword_node_to_flownest_pixel_{sid}.parquet', index=False)
    N.to_csv(out + f'sword_node_to_flownest_pixel_{sid}.csv', index=False)
    RR.to_csv(out + f'sword_reach_node_placement_{sid}.csv', index=False)
    SM.to_csv(out + f'seam_between_neighbor_reaches_{sid}.csv', index=False)
