"""Step 2 (one system at a time): every SWORD reach matched to one FlowNest river path, or two paths joined at a
confluence, inside its own basin. Sample points every 45 m or less along the SWORD line; the path that covers at least
80% of them within a window that grows with river width; checked by upstream area against SWORD facc_total. Neighbour
rules and rescue rules try the reaches left."""
import sqlite3, numpy as np, pandas as pd, pyogrio, shapely, pyarrow.parquet as pq

from . import inputs as IN


STEP_M = 45.0
MIN_PATH_KM = 8.0
COVER = 0.8
K_AUTO = 3
K_NEAR = 11
K_CAP = 40
PX_NS_M = 92.6
VETO_WIDTH_M = 100.0
RESCUE_MIN_REACH_KM = 0.65
SWORD_FILL = -9999.0
M_LON, M_LAT = 111320.0, 110574.0
AREA_TOL = 2.0
AG_MIN_COVER = 0.5
RESCUE = ['land_part_cover', 'partly_on_neighbor_path', 'within_water_surface', 'short_path_into_neighbor']
SEA = -2147483648
NB = [f'rch_id_{d}_{i}' for d in ('up', 'dn') for i in (1, 2, 3, 4)]


def open_sword_gpkg(cont):
    """Open one continent's SWORD GeoPackage read-only (sqlite read-only mode, so no -wal or -shm file is left beside it)."""
    return sqlite3.connect(f'file:{IN.gpkg_path(cont)}?mode=ro', uri=True)


def stored(v):
    """A ratio, share or cover as written in the tables: 6 significant digits (NaN, inf and 0 unchanged). The checks
    that decide on such a value round it themselves (area ratio on 3 decimals, shares and sea fractions on 4)."""
    v = np.asarray(v, float)
    out = v.copy()
    ok = np.isfinite(v) & (v != 0)
    out[ok] = [float('%.5e' % x) for x in v[ok]]
    return out


def cover_text(x):
    """A cover in a pair note: 6 significant digits, or more when 6 would put it on the other side of 80%."""
    for d in range(6, 18):
        t = '%.*g' % (d, x)
        if (float(t) >= COVER) == (x >= COVER):
            return t
    return t


def sword_fill_to_nan(v):
    """SWORD's missing value -9999 as NaN (float array)."""
    v = np.asarray(v, float)
    return np.where(v == SWORD_FILL, np.nan, v)


def sword_node_facc_total(conts=('AF', 'AS', 'EU', 'NA', 'OC', 'SA')):
    """SWORD v17c (build 0.0.15) facc_total of every node (km2, missing as NaN) -> DataFrame[node_id, reach_id,
    facc_total]. The line step reads the same column with the nodes of each system."""
    out = []
    for c in conts:
        con = open_sword_gpkg(c)
        T = pd.read_sql('select node_id, reach_id, facc_total from nodes', con)
        con.close()
        T['facc_total'] = sword_fill_to_nan(T.facc_total.values)
        out.append(T)
    T = pd.concat(out, ignore_index=True)
    assert T.node_id.is_unique
    return T


def ring(K):
    """The offsets at Chebyshev distance exactly K."""
    if K == 0:
        return [(0, 0)]
    return [(dr, dc) for dr in range(-K, K + 1) for dc in range(-K, K + 1) if max(abs(dr), abs(dc)) == K]


CACHE = {}
FACC_FLAGS = ('ok', 'suspect', 'same_river', 'braid_share', 'delta_arm', 'filled')


def use_continent_cache(cont):
    """For a run over many systems: read this continent's reaches and nodes into memory once; load_system then takes them
    from memory (one continent at a time)."""
    if CACHE.get('cont') == cont:
        return
    CACHE.clear()
    gp = IN.gpkg_path(cont)
    cols = ['reach_id', 'reach_length', 'width', 'lakeflag', 'type', 'main_side', 'facc_flag', 'river_name'] + NB
    CACHE['R'] = pyogrio.read_dataframe(gp, layer='reaches', columns=cols + ['subnetwork_id', 'dist_out', 'wse'])
    con = open_sword_gpkg(cont)
    CACHE['Nd'] = pd.read_sql('select subnetwork_id, node_id, reach_id, x, y, dist_out, facc_total from nodes', con)
    con.close()
    CACHE['cols'] = cols
    CACHE['cont'] = cont


def wrap_lon(x):
    """Longitude in [0, 360) for systems that cross 180 degrees."""
    return np.where(x < 0, x + 360.0, x)


def load_system(sid):
    """System sid: (its FlowNest basin p, its reaches with geometry, own basin and SWORD end points)."""
    S = pd.read_csv(IN.FACE + 'sword_system_to_flownest_basin.csv', float_precision='round_trip', keep_default_na=False, na_values=[''])
    row = S[S.subnetwork_id == sid].iloc[0]
    gp = IN.gpkg_path(row.continent)
    cols = ['reach_id', 'reach_length', 'width', 'lakeflag', 'type', 'main_side', 'facc_flag', 'river_name'] + NB
    cached = CACHE.get('cont') == row.continent
    if cached:
        R = CACHE['R'][CACHE['R'].subnetwork_id.values == sid][cols + ['geometry']].reset_index(drop=True)
    else:
        R = pyogrio.read_dataframe(gp, layer='reaches', columns=cols, where=f'subnetwork_id = {sid}')
    rb = pd.read_parquet(IN.FACE + 'sword_reach_to_flownest_basin.parquet', columns=['reach_id', 'basin_id', 'share'])
    assert R.facc_flag.isin(FACC_FLAGS).all(), set(R.facc_flag) - set(FACC_FLAGS)
    R = R.merge(rb.rename(columns={'basin_id': 'own_basin_id', 'share': 'own_basin_share'}), on='reach_id', how='left')
    if cached:
        Nd = CACHE['Nd'][CACHE['Nd'].subnetwork_id.values == sid][['node_id', 'reach_id', 'x', 'y', 'dist_out', 'facc_total']].reset_index(drop=True)
    else:
        con = open_sword_gpkg(row.continent)
        Nd = pd.read_sql('select node_id, reach_id, x, y, dist_out, facc_total from nodes where subnetwork_id = ?', con, params=(sid,))
        con.close()
    Nd['facc_total'] = sword_fill_to_nan(Nd.facc_total.values)
    Nd = Nd.sort_values(['dist_out', 'node_id'])
    R['sword_facc_total_km2'] = R.reach_id.map(Nd.groupby('reach_id').facc_total.median())
    R['node_facc_total'] = R.reach_id.map(Nd.iloc[::-1].groupby('reach_id', sort=False).facc_total.apply(lambda v: v.values.astype(float)))
    dn_end = Nd.drop_duplicates('reach_id', keep='first').set_index('reach_id')
    up_end = Nd.drop_duplicates('reach_id', keep='last').set_index('reach_id')
    R['up_x'], R['up_y'] = R.reach_id.map(up_end.x), R.reach_id.map(up_end.y)
    R['dn_x'], R['dn_y'] = R.reach_id.map(dn_end.x), R.reach_id.map(dn_end.y)
    R['n_nodes'] = R.reach_id.map(Nd.groupby('reach_id').size()).fillna(0).astype(int)
    b = shapely.bounds(R.geometry.values)
    R.attrs['wrapped'] = bool(np.nanmax(b[:, 2]) - np.nanmin(b[:, 0]) > 180)
    if R.attrs['wrapped']:
        R['geometry'] = shapely.transform(R.geometry.values, lambda xy: np.c_[wrap_lon(xy[:, 0]), xy[:, 1]])
        R['up_x'], R['dn_x'] = wrap_lon(R.up_x.values), wrap_lon(R.dn_x.values)
    return int(row.basin_id), R


def read_window(band, x0, y0, wd, h):
    """Read a window with the columns taken modulo the grid width (two pieces joined across 180 degrees); otherwise
    band.read(x0, y0, wd, h)."""
    W = band.width
    x0 %= W
    if x0 + wd <= W:
        return band.read(x0, y0, wd, h)
    k = W - x0
    return np.hstack([band.read(x0, y0, k, h), band.read(0, y0, wd - k, h)])


def basins_of(p, R):
    """The basin in which each reach is searched: the basin holding most of its length, or the system's basin p for reaches
    with none (mostly in the sea)."""
    return R.own_basin_id.fillna(p).astype(np.int64).values


def load_paths(basins):
    """The reach table of these basins (in reach_seq_id order) -> length and basin of each path, and the path it flows into."""
    basins = sorted({int(b) for b in np.atleast_1d(basins)})
    T = pq.read_table(IN.FN + 'flownest_merit90m_reach_1km2.parquet', filters=[('basin_id', 'in', basins)],
                      columns=['reach_seq_id', 'next_down_reach_seq_id', 'path_seq_id', 'path_level', 'upstream_area_km2',
                               'length_km', 'distance_to_basin_outlet_km', 'basin_id']).to_pandas()
    T = T.sort_values('reach_seq_id').reset_index(drop=True)
    g = T.groupby('basin_id').reach_seq_id.agg(['min', 'max', 'size'])
    assert len(g) == len(basins) and (g['min'] > 0).all() and (g['max'] - g['min'] + 1 == g['size']).all(), 'basin reaches not contiguous'
    assert T.groupby('path_seq_id').basin_id.nunique().max() == 1
    ix = lambda v: np.searchsorted(T.reach_seq_id.values, v)
    P = T.groupby('path_seq_id').agg(path_length_km=('length_km', 'sum'), path_level=('path_level', 'first'),
                                     path_upstream_area_km2=('upstream_area_km2', 'max'), basin_id=('basin_id', 'first'))
    last = T.loc[T.groupby('path_seq_id').distance_to_basin_outlet_km.idxmin()]
    nd = last.next_down_reach_seq_id.values
    P['path_down'] = pd.Series(np.where(nd == last.reach_seq_id.values, last.path_seq_id.values,
                                        T.path_seq_id.values[ix(nd)]), index=last.path_seq_id.values)
    return T, P


def reach_rows(A, T):
    """Channel pixels of the basins in T within a reach_seq_id window: returns (flat pixel index, row of T)."""
    seq = T.reach_seq_id.values
    nz = np.flatnonzero(A > 0)
    v = A.reshape(-1)[nz]
    j = np.minimum(np.searchsorted(seq, v), len(seq) - 1)
    ok = seq[j] == v
    return nz[ok], j[ok]


def path_raster(A, T):
    """reach_seq_id window -> path_seq_id window (only channel pixels of the basins in T have a value; 0 elsewhere)."""
    k, j = reach_rows(A, T)
    Pr = np.zeros(A.shape, np.int32)
    Pr.reshape(-1)[k] = T.path_seq_id.values[j]
    return Pr


def sample_lines(R):
    """Sample points along the SWORD reach lines, every 45 m or less: (lon, lat, length each point stands for, row of R)."""
    xy, gi = shapely.get_coordinates(R.geometry.values, return_index=True)
    ok = gi[:-1] == gi[1:]
    a, b, ri = xy[:-1][ok], xy[1:][ok], gi[:-1][ok]
    dlon, dlat = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    assert np.abs(dlon).max() < 180
    latm = np.radians((a[:, 1] + b[:, 1]) / 2)
    seg_m = np.hypot(dlon * M_LON * np.cos(latm), dlat * M_LAT)
    k = np.maximum(1, np.ceil(seg_m / STEP_M)).astype(np.int64)
    s = np.repeat(np.arange(len(k)), k)
    t = (np.arange(len(s)) - (np.cumsum(k) - k)[s] + 0.5) / k[s]
    return a[s, 0] + dlon[s] * t, a[s, 1] + dlat[s] * t, (seg_m / k / 1000.0)[s], ri[s]


def offsets_by_distance(K):
    """All offsets (dr, dc, Chebyshev distance) in a (2K+1)^2 window, nearest first by Euclidean distance (ties by dr, dc,
    so the order is fixed)."""
    o = [(dr, dc, max(abs(dr), abs(dc))) for dr in range(-K, K + 1) for dc in range(-K, K + 1)]
    return sorted(o, key=lambda z: (z[0] ** 2 + z[1] ** 2, z[0], z[1]))


def ring_pairs(Pr, rr, cc, Kpt):
    """For each sample point and each path: the smallest Chebyshev distance K (pixels); a point searches only up to its own
    limit Kpt."""
    keys, ks = [], []
    idx = np.arange(len(rr), dtype=np.int64)
    for K in range(int(Kpt.max()) + 1):
        sel = idx[Kpt >= K]
        r_, c_ = rr[sel], cc[sel]
        for dr, dc in ring(K):
            v = Pr[r_ + dr, c_ + dc]
            m = v > 0
            keys.append(sel[m] * (1 << 32) + v[m].astype(np.int64))
            ks.append(np.full(m.sum(), K, np.int16))
    keys, ks = np.concatenate(keys), np.concatenate(ks)
    o = np.lexsort((ks, keys))
    keys, ks = keys[o], ks[o]
    first = np.ones(len(keys), bool)    # also right when no path lies near any point
    first[1:] = keys[1:] != keys[:-1]
    keys, ks = keys[first], ks[first]
    return keys >> 32, keys & ((1 << 32) - 1), ks


def main(sid, out):
    """Match the reaches of system sid; writes sword_reach_to_flownest_path_<sid>.parquet and match_steps_<sid>.csv
    into out."""
    p, R = load_system(sid)
    n = len(R)
    ownb = basins_of(p, R)
    has_own = R.own_basin_id.notna().values
    T, P = load_paths(np.r_[ownb, p])
    pdn = P.path_down.to_dict()
    pbas = P.basin_id.to_dict()
    conn = lambda a, b: a == b or pdn.get(a) == b or pdn.get(b) == a
    lon, lat, w, ri = sample_lines(R)
    L = np.bincount(ri, weights=w, minlength=n)
    at = pd.Series(w).groupby(ri).cumsum().values - w / 2
    f0 = shapely.get_coordinates(shapely.get_point(R.geometry.values, 0))
    f1 = shapely.get_coordinates(shapely.get_point(R.geometry.values, -1))
    dd = lambda F, x, y: np.hypot((F[:, 0] - x) * M_LON * np.cos(np.radians(y)), (F[:, 1] - y) * M_LAT)
    ux, uy, vx, vy = (R[c].values for c in ('up_x', 'up_y', 'dn_x', 'dn_y'))
    starts_up = dd(f0, ux, uy) + dd(f1, vx, vy) <= dd(f1, ux, uy) + dd(f0, vx, vy)
    orient_ok = R.n_nodes.values >= 2
    pos_up = np.where(starts_up[ri], at, L[ri] - at)
    width =np.where(R.width.values > 0, R.width.values, 0.0)
    Klim = np.clip(np.ceil(width / 2 / PX_NS_M), K_NEAR, K_CAP).astype(int)
    M = int(Klim.max()) + 1
    ds = IN.Grid(IN.FN + 'flownest_merit90m_reach_seq_id_1km2.tif')
    gt = ds.gt
    col = np.floor((lon - gt[0]) / gt[1]).astype(np.int64)
    row = np.floor((lat - gt[3]) / gt[5]).astype(np.int64)
    y0, x0 = int(row.min() - M), int(col.min() - M)
    h, wd = int(row.max() - y0 + M + 1), int(col.max() - x0 + M + 1)
    A = read_window(ds, x0, y0, wd, h)
    rr, cc = row - y0, col - x0
    sea_pt = A[rr, cc] == SEA
    seaf = np.bincount(ri, weights=w * sea_pt, minlength=n) / L
    Pr = path_raster(A, T)
    del A
    du = IN.Grid(IN.UPA)
    U = read_window(du, x0, y0, wd, h)
    pt, path, K = ring_pairs(Pr, rr, cc, Klim[ri])
    D = pd.DataFrame({'pt': pt, 'path_seq_id': path, 'K': K, 'w': w[pt], 'ri': ri[pt]})
    D = D[P.basin_id.reindex(D.path_seq_id).values == ownb[D.ri.values]].reset_index(drop=True)
    D['long'] = P.path_length_km.reindex(D.path_seq_id).values > MIN_PATH_KM
    Dl = D[D.long]
    pts_of = dict(tuple(pd.Series(np.arange(len(ri))).groupby(ri)))

    def cover(Dx):
        c = Dx.groupby(['ri', 'path_seq_id']).agg(cov=('w', 'sum'), dK=('K', 'mean')).reset_index()
        c['cov'] /= L[c.ri.values]
        return c.sort_values(['ri', 'cov', 'dK', 'path_seq_id'], ascending=[True, False, True, True])

    pair_cost, rejected, against = {}, {}, {}

    def ordered_cover(g, t, q, r):
        """Tributary t flows into q, so going downstream the points should be on t first, then on q. A split point is chosen
        that maximises this in-order cover (points above it count for t, below it for q); the reverse order (q first, then
        t: SWORD flow opposite to FlowNest) is computed too. g is already cut to the current window. When the direction is
        unknown both return the cover of the two paths together (no veto)."""
        I = pts_of[r].values
        I = I[np.argsort(pos_up[I])]
        ht = np.isin(I, g.pt.values[g.path_seq_id.values == t])
        hq = np.isin(I, g.pt.values[g.path_seq_id.values == q])
        if not orient_ok[r]:
            u = float(w[I][ht | hq].sum() / L[r])
            return u, u
        ww = w[I]
        res = []
        for a_, b_ in ((ht, hq), (hq, ht)):
            ca = np.r_[0, np.cumsum(ww * a_)]
            cb = np.r_[0, np.cumsum(ww * b_)]
            res.append(float((ca + cb[-1] - cb).max() / L[r]))
        return tuple(res)

    def pairs(ps, g, r):
        """Pairs of paths joined at a confluence whose in-order cover is at least 80%: ((tributary t, the q it flows into), cover)."""
        o = []
        one = g.drop_duplicates(['pt', 'path_seq_id']).groupby('path_seq_id').w.sum() / L[r]
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                a, b = int(ps[i]), int(ps[j])
                if one.get(a, 0) >= COVER or one.get(b, 0) >= COVER:
                    continue
                if pdn.get(a) == b or pdn.get(b) == a:
                    t, q = (a, b) if pdn.get(a) == b else (b, a)
                    u = g.loc[g.path_seq_id.isin([a, b])].drop_duplicates('pt').w.sum() / L[r]
                    if u >= COVER:
                        fw, rv = ordered_cover(g, t, q, r)
                        if fw >= COVER:
                            pair_cost[(r, t, q)] = u
                            o.append(((t, q), fw))
                        else:
                            rejected.setdefault(r, {})[(t, q)] = fw
                            if rv >= COVER:
                                against.setdefault(r, {})[(q, t)] = rv
        return o

    stage, fK, it2 = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    kind, rel = np.full(n, 'not_found', object), np.full(n, '', object)
    pa, pb, p2 = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64)
    cv, c2 = np.full(n, np.nan), np.full(n, np.nan)
    best_cov = {}

    for K in range(1, K_AUTO + 1):
        c = cover(Dl[Dl.K <= K])
        top = c.drop_duplicates('ri').set_index('ri')
        best_cov[K] = top['cov'].reindex(range(n)).fillna(0).values
        t1 = top.reindex(np.flatnonzero(stage == 0))
        hit = t1.index[t1['cov'] >= COVER].values
        stage[hit], fK[hit], kind[hit] = 1, K, 'one_path'
        pa[hit] = t1.loc[hit, 'path_seq_id'].astype(np.int64).values
        cv[hit] = t1.loc[hit, 'cov'].values
        sec = c[c.ri.isin(hit)]
        sec = sec[sec.groupby('ri').cumcount() == 1]
        p2[sec.ri.values], c2[sec.ri.values] = sec.path_seq_id.values, sec['cov'].values
        todo = np.flatnonzero(stage == 0)
        cand = c[c.ri.isin(todo)]
        Dkg = dict(tuple(Dl[(Dl.K <= K) & Dl.ri.isin(cand.ri.unique())].groupby('ri')))
        for r, g in cand.groupby('ri'):
            o = pairs(g.path_seq_id.values, Dkg[r], r)
            if o:
                (a, b), u = max(o, key=lambda z: z[1])
                stage[r], fK[r], kind[r], pa[r], pb[r], cv[r] = 1, K, 'two_paths', a, b, u

    c = cover(Dl)
    best_cov['s2'] = c.drop_duplicates('ri').set_index('ri')['cov'].reindex(range(n)).fillna(0).values
    todo = np.flatnonzero(stage == 0)
    cand = c[c.ri.isin(todo)]
    Dg = dict(tuple(Dl[Dl.ri.isin(todo)].groupby('ri')))
    opts = {}
    for r, g in cand.groupby('ri'):
        o = [((int(a),), float(x)) for a, x in zip(g.path_seq_id.values, g['cov'].values) if x >= COVER]
        o += pairs(g.path_seq_id.values, Dg[r], r)
        if o:
            opts[r] = o
    rid = R.reach_id.values
    pos = pd.Series(np.arange(n), index=rid)
    lst = lambda i, cols: [int(pos[x]) for x in R.loc[i, cols].values if x and x in pos.index]
    nup = [lst(i, NB[:4]) for i in range(n)]
    ndn = [lst(i, NB[4:]) for i in range(n)]
    nbr = [nup[i] + ndn[i] for i in range(n)]
    n_up_raw = (R[NB[:4]].fillna(0).values > 0).sum(1)
    Dall = dict(tuple(Dl.groupby('ri')))
    up_end = lambda i: pa[i]
    dn_end = lambda i: pb[i] if pb[i] else pa[i]
    cur = lambda i: (int(pa[i]),) if not pb[i] else (int(pa[i]), int(pb[i]))
    switched, fix, swK = np.zeros(n, np.int64), np.full(n, '', object), np.zeros(n, int)
    win = lambda i: Klim[i] if stage[i] == 3 else (swK[i] if switched[i] else fK[i])
    sw_ratio = np.full((n, 2), np.nan)
    sw_no = np.full(n, '', object)

    def kgrid(r, ps, src=None):
        """For each sample point, the K of the nearest path in ps (Klim + 1 if none within the stage 2 window); src defaults
        to the long paths."""
        g = (Dall if src is None else src).get(r)
        k = pd.Series(Klim[r] + 1, index=pts_of[r].values)
        if g is not None:
            m = g[g.path_seq_id.isin(ps)].groupby('pt').K.min()
            k.loc[m.index] = m.values
        return k

    def need_K(r, ps):
        """The window K at which ps covers 80%; None if not reached within the stage 2 window."""
        k = kgrid(r, ps).sort_values(kind='stable')
        c = np.cumsum(w[k.index.values]) / L[r]
        j = np.searchsorted(c, COVER - 1e-9)
        return int(k.values[j]) if j < len(c) and k.values[j] <= Klim[r] else None

    def cover_in(r, ps, K):
        k = kgrid(r, ps)
        return float(w[k.index.values[k.values <= K]].sum() / L[r])

    def nearest_upa(sel, A_, B_, Kmax_):
        """For each sample point, the MERIT upa of the Euclidean-nearest pixel on an allowed path (A_ or B_) within a
        (2K+1) x (2K+1) window, the same landing as in the direction check; offsets are tried nearest first."""
        up = np.full(len(sel), np.nan, np.float32)
        s = np.arange(len(sel))
        for dr, dc, ch in offsets_by_distance(int(Kmax_.max()) if len(sel) else -1):
            if not len(s):
                break
            t = s[Kmax_[s] >= ch]
            if not len(t):
                continue
            v = Pr[rr[sel[t]] + dr, cc[sel[t]] + dc]
            m = (v == A_[t]) | ((B_[t] > 0) & (v == B_[t]))
            up[t[m]] = U[rr[sel[t[m]]] + dr, cc[sel[t[m]]] + dc]
            s = s[np.isnan(up[s])]
        return up

    rcache = {}

    def ratio(r, ps, K):
        """Area ratio of this reach matched to ps with window K (median of MERIT upa / SWORD facc_total)."""
        key = (r, ps, K)
        if key not in rcache:
            if not sf[r] > 0:
                rcache[key] = np.nan
            else:
                s = pts_of[r].values
                up = nearest_upa(s, np.full(len(s), ps[0]), np.full(len(s), ps[1] if len(ps) > 1 else 0), np.full(len(s), K))
                rcache[key] = float(np.nanmedian(up) / sf[r]) if np.isfinite(up).any() else np.nan
        return rcache[key]

    sf = R.sword_facc_total_km2.values
    it = 0

    def stage2_pass():
        nonlocal it
        tot = 0
        while True:
            it += 1
            new = {}
            for r, o in opts.items():
                if stage[r]:
                    continue
                NP = set()
                for j in nbr[r]:
                    if stage[j]:
                        NP |= {pa[j], pb[j]} - {0}
                if not NP:
                    continue
                best = None
                for ps, x in o:
                    if set(ps) & NP:
                        rk = 0
                    elif any(conn(a, b) for a in ps for b in NP):
                        rk = 1
                    else:
                        continue
                    key = (rk, len(ps), -x)
                    if best is None or key < best[0]:
                        best = (key, ps, x)
                if best:
                    new[r] = best
            if not new:
                it -= 1
                return tot
            for r, (key, ps, x) in new.items():
                stage[r], it2[r], cv[r] = 2, it, x
                kind[r] = 'one_path' if len(ps) == 1 else 'two_paths'
                pa[r], pb[r] = ps[0], (ps[1] if len(ps) > 1 else 0)
                rel[r] = 'same_path' if key[0] == 0 else 'connected_path'
                fK[r] = need_K(r, ps)
            tot += len(new)

    def switch_pass():
        """(a) Rematch: the path through the upstream and downstream neighbours (through_path), or on upstream from the only
        upstream reach of the downstream neighbour (upstream_continuation)."""
        ch = {}
        for r in np.flatnonzero(stage > 0):
            if switched[r]:
                continue
            C = set(cur(r))
            UE = {dn_end(u) for u in nup[r] if stage[u]}
            DE = {up_end(d) for d in ndn[r] if stage[d]}
            cands = []
            TH = UE & DE
            consistent = len(C) == 2 and (not UE or up_end(r) in UE) and (not DE or dn_end(r) in DE)
            if not (len(C) == 1 and C & TH) and not consistent:
                for q in sorted(TH):
                    ref = [u for u in nup[r] if stage[u] and dn_end(u) == q] + [d for d in ndn[r] if stage[d] and up_end(d) == q]
                    cands.append((int(q), 'through_path_switch', ref))
            for d in ndn[r]:
                if stage[d] and n_up_raw[d] == 1 and up_end(d) not in C:
                    cands.append((int(up_end(d)), 'upstream_continuation_switch', [d]))
            for q, why, refn in cands:
                K = need_K(r, (q,))
                if K is None:
                    continue
                rq, rc = ratio(r, (q,), K), ratio(r, cur(r), win(r))
                if why == 'through_path_switch' or K <= K_AUTO:
                    ch[r] = (q, why, K, rq, rc)
                    break
                sw_no[r] = f'{q}:{why[:7]}:needs_{2 * K + 1}x{2 * K + 1}'
        for r, (q, why, K, rq, rc) in ch.items():
            switched[r], fix[r], sw_ratio[r] = pa[r], why, (rc, rq)
            pa[r], pb[r], kind[r], swK[r] = q, 0, (kind[r] if kind[r] == 'between_neighbors' else 'one_path'), K
            cv[r] = cover_in(r, (q,), K)
        if ch:
            pass
        return len(ch)

    def between_pass():
        """(b) Fill: the lower end of the upstream neighbour and the upper end of the downstream neighbour are on one path."""
        new = {}
        for r in np.flatnonzero((stage == 0) & has_own):
            q = {dn_end(u) for u in nup[r] if stage[u]} & {up_end(d) for d in ndn[r] if stage[d]}
            if len(q) == 1 and pbas[next(iter(q))] == ownb[r]:
                new[r] = int(q.pop())
        for r, q in new.items():
            stage[r], kind[r], rel[r], pa[r], pb[r], fK[r] = 3, 'between_neighbors', 'between_neighbors', q, 0, Klim[r]
            cv[r] = cover_in(r, (q,), Klim[r])
        if new:
            pass
        return len(new)

    def against_pass():
        """(c) Two paths against the flow: an unmatched reach that, along the SWORD flow, is on a first and then on b, while
        in FlowNest b flows into a (mostly a SWORD side channel that MERIT, with one flow direction per pixel, draws as a
        tributary of the main river). Needs in-order cover of at least 80%, the lower end of an upstream neighbour on a, and
        no upper end of a downstream neighbour on a (not a detour back onto a)."""
        new = {}
        for r, d in against.items():
            if stage[r] or not has_own[r]:
                continue
            UE = {dn_end(u) for u in nup[r] if stage[u]}
            DE = {up_end(x) for x in ndn[r] if stage[x]}
            ok = [(ab, x) for ab, x in d.items() if ab[0] in UE and ab[0] not in DE]
            if ok:
                new[r] = max(ok, key=lambda z: z[1])
        for r, ((a, b), x) in new.items():
            stage[r], kind[r], rel[r], pa[r], pb[r], cv[r] = 3, 'two_paths_against_flow', 'against_flow', a, b, x
            fK[r] = need_K(r, (a, b)) or Klim[r]
        if new:
            pass
        return len(new)

    for loop in range(20):
        if stage2_pass() + switch_pass() + between_pass() + against_pass() == 0:
            break

    stage0 = stage.copy()
    any_nb0 = np.array([any(stage0[j] for j in nbr[i]) for i in range(n)])
    Kd = np.maximum(K_AUTO, np.ceil((width / 2 + 180) / PX_NS_M)).astype(int)
    land_cv, wfrac = np.full(n, np.nan), np.full(n, np.nan)
    lake = R.lakeflag.values == 1
    cands = np.flatnonzero((stage == 0) & has_own & (seaf < 0.5) & ~lake & ~(np.isin(np.arange(n), list(opts)) & any_nb0))
    Dc = dict(tuple(D[D.ri.isin(cands)].groupby('ri')))
    dw = IN.Grid(IN.WTH)
    bw = dw
    plen, pupa = P.path_length_km.to_dict(), P.path_upstream_area_km2.to_dict()
    jfail = [0]

    def covers(r, K, pts=None, long_only=False):
        """Cover of each path within window K, largest first (with pts, only these sample points and their length count)."""
        g = Dc.get(r)
        if g is None:
            return pd.Series(dtype=float)
        g = g[g.K.values <= K]
        if long_only:
            g = g[g.long.values]
        den = L[r]
        if pts is not None:
            g = g[np.isin(g.pt.values, pts)]
            den = w[pts].sum()
        return (g.groupby('path_seq_id').w.sum() / den).sort_values(ascending=False, kind='stable')

    def med_dist(r, X, pts=None):
        k = kgrid(r, (X,), Dc)
        return float((k if pts is None else k.loc[pts]).median())

    def evidence(r, X):
        """Neighbour reaches matched before filling: whether one is on the same path as X (support), whether one contradicts it."""
        U_ = [u for u in nup[r] if stage0[u]]
        D_ = [d for d in ndn[r] if stage0[d]]
        wit = any(dn_end(u) == X for u in U_) or any(up_end(d) == X for d in D_)
        ok = (all(dn_end(u) == X or pdn.get(dn_end(u)) == X for u in U_) and
              all(up_end(d) == X or pdn.get(X) == up_end(d) for d in D_))
        return wit, ok

    def nearest_rc(sel, X, K):
        """For each sample point, window row and column of the Euclidean-nearest pixel on path X within a (2K+1)^2 window
        (-1 if none)."""
        rw, cl = np.full(len(sel), -1), np.full(len(sel), -1)
        s = np.arange(len(sel))
        for dr, dc, ch in offsets_by_distance(int(K)):
            if not len(s):
                break
            m = Pr[rr[sel[s]] + dr, cc[sel[s]] + dc] == X
            rw[s[m]], cl[s[m]] = rr[sel[s[m]]] + dr, cc[sel[s[m]]] + dc
            s = s[~m]
        return rw, cl

    def water_ok(r, X, K, pts):
        """For SWORD width of at least 100 m: at least half of the length, taking the pixel of X nearest each sample point,
        must have MERIT water (wth not 0 and not -9999)."""
        if not width[r] >= VETO_WIDTH_M:
            return True, np.nan
        rw, cl = nearest_rc(pts, X, K)
        m = rw >= 0
        if not m.any():
            return False, 0.0
        ar, ac = rw[m] + y0, cl[m] + x0
        r0, c0 = int(ar.min()), int(ac.min())
        Wt = read_window(bw, c0, r0, int(ac.max()) - c0 + 1, int(ar.max()) - r0 + 1)
        v = Wt[ar - r0, ac - c0]
        f = float(w[pts[m]][(v != 0) & (v != -9999)].sum() / w[pts[m]].sum())
        return f >= 0.5, f

    def junction_dist(r, Y):
        """Distance (m) from the outlet pixel of short path Y (its pixel of largest MERIT upa, equal to the outlet area of Y)
        to the SWORD downstream end of this reach; inf if no outlet is found."""
        I = pts_of[r].values
        mg = int(np.ceil((width[r] / 2 + 1180) / PX_NS_M)) + 2
        r0, r1 = max(int(rr[I].min()) - mg, 0), min(int(rr[I].max()) + mg + 1, Pr.shape[0])
        c0, c1 = max(int(cc[I].min()) - mg, 0), min(int(cc[I].max()) + mg + 1, Pr.shape[1])
        uu = np.where(Pr[r0:r1, c0:c1] == Y, U[r0:r1, c0:c1], -1.0)
        k = int(np.argmax(uu))
        a, b = divmod(k, uu.shape[1])
        if not abs(uu.flat[k] - pupa[Y]) <= 1e-3 * pupa[Y] + 0.05:
            jfail[0] += 1
            return np.inf
        lon_ = gt[0] + (x0 + c0 + b + 0.5) * gt[1]
        lat_ = gt[3] + (y0 + r0 + a + 0.5) * gt[5]
        return float(np.hypot((lon_ - R.dn_x.values[r]) * M_LON * np.cos(np.radians(lat_)), (lat_ - R.dn_y.values[r]) * M_LAT))

    RULE, WHY = {}, {}
    for r in np.flatnonzero((stage == 0) & lake):
        WHY[r] = 'lake_not_rescued'

    def rescue_one(r):
        """Try rules (d), (e), (f), (g) in order: (rule, path, K, cover, land cover, water share) on success, else None;
        WHY[r] records where each rule stopped."""
        I = pts_of[r].values
        why = []
        if seaf[r] > 0:
            Io = I[np.argsort(pos_up[I])]
            sp = sea_pt[Io]
            j0 = int(np.argmax(sp))
            if not orient_ok[r]:
                why.append('d:direction_unknown')
            elif not (w[Io[j0:]][sp[j0:]].sum() >= 0.9 * w[Io[j0:]].sum() and (~sp).any()):
                why.append('d:sea_not_at_downstream_end')
            else:
                land = Io[~sp]
                for K, need_wit in ((K_AUTO, False), (int(Klim[r]), True)):
                    cl = covers(r, K, land, long_only=True)
                    if not len(cl) or cl.iloc[0] < COVER:
                        if need_wit:
                            why.append('d:land_cover<0.8')
                        continue
                    X = int(cl.index[0])
                    wit, ok = evidence(r, X)
                    g = ('d:other_path>=0.5' if (covers(r, K, land).drop(X) >= 0.5).any() else 'd:neighbor_conflict' if not ok else
                         'd:no_neighbor_on_path' if need_wit and not wit else 'd:far' if med_dist(r, X, land) > Kd[r] else '')
                    if not g:
                        okw, f = water_ok(r, X, K, land)
                        if okw:
                            return 'land_part_cover', X, K, float(covers(r, K, long_only=True).get(X, 0.0)), float(cl.iloc[0]), f
                        g = 'd:merit_dry'
                    why.append(g)
                    break
        for Kr in sorted({K_AUTO, int(min(Kd[r], Klim[r])), int(Klim[r])}):
            cl = covers(r, Kr, long_only=True)
            if len(cl) and cl.iloc[0] >= 0.5:
                break
        if not len(cl) or cl.iloc[0] < 0.5:
            why.append('e:cover<0.5')
        else:
            X = int(cl.index[0])
            wit, ok = evidence(r, X)
            g = ('e:no_matched_neighbor' if not any_nb0[r] else 'e:neighbor_conflict' if not ok else
                 'e:other_path>=0.3' if (covers(r, Kr).drop(X) >= 0.3).any() else '')
            if not g:
                okw, f = water_ok(r, X, Kr, I)
                if okw:
                    return 'partly_on_neighbor_path', X, Kr, float(cl.iloc[0]), np.nan, f
                g = 'e:merit_dry'
            why.append(g)
        if any_nb0[r]:
            why.append('f:has_matched_neighbor')
        elif not width[r] >= VETO_WIDTH_M:
            why.append('f:narrow')
        else:
            Kw = int(min(K_AUTO + np.ceil(width[r] / 2 / PX_NS_M), Klim[r]))
            cl = covers(r, Kw, long_only=True)
            if not len(cl) or cl.iloc[0] < COVER:
                why.append('f:cover<0.8')
            else:
                X = int(cl.index[0])
                g = 'f:other_path>=0.5' if (covers(r, Kw).drop(X) >= 0.5).any() else 'f:far' if med_dist(r, X) > Kd[r] else ''
                if not g:
                    okw, f = water_ok(r, X, Kw, I)
                    if okw:
                        return 'within_water_surface', X, Kw, float(cl.iloc[0]), np.nan, f
                    g = 'f:merit_dry'
                why.append(g)
        ca = covers(r, K_AUTO)
        if not len(ca) or ca.iloc[0] < COVER:
            why.append('g:cover<0.8')
        else:
            Y = int(ca.index[0])
            D_ = [d for d in ndn[r] if stage0[d]]
            g = ('g:long_path' if plen[Y] > MIN_PATH_KM else
                 'g:reach<0.65km' if R.reach_length.values[r] < RESCUE_MIN_REACH_KM * 1000 else
                 'g:other_path>=0.5' if (ca.drop(Y) >= 0.5).any() else
                 'g:has_upstream_reach' if len(nup[r]) else
                 'g:no_matched_downstream' if not D_ else
                 'g:not_into_downstream_path' if not all(pdn.get(Y) == up_end(d) for d in D_) else
                 'g:junction_far' if not junction_dist(r, Y) <= width[r] / 2 + 1180 else '')
            if not g:
                okw, f = water_ok(r, Y, K_AUTO, I)
                if okw:
                    return 'short_path_into_neighbor', Y, K_AUTO, float(ca.iloc[0]), np.nan, f
                g = 'g:merit_dry'
            why.append(g)
        WHY[r] = ';'.join(why)
        return None

    for r in cands:
        o = rescue_one(r)
        if o:
            RULE[r] = o
    for r, (rule, X, K, c_, lc, f) in RULE.items():
        stage[r], kind[r], rel[r], pa[r], pb[r], fK[r], cv[r] = 3, 'one_path', rule, X, 0, K, c_
        land_cv[r], wfrac[r] = lc, f
    dw = None

    nf = R.n_nodes.values
    FL = [a if isinstance(a, np.ndarray) else np.empty(0) for a in R.node_facc_total.values]
    assert all(len(a) == k for a, k in zip(FL, nf))
    flat = np.concatenate(FL + [np.full(1, np.nan)])
    off = np.r_[0, np.cumsum(nf)[:-1]]
    kpt = np.minimum((pos_up / L[ri] * nf[ri]).astype(np.int64), np.maximum(nf[ri] - 1, 0))
    fb_pt = np.where(nf[ri] > 0, flat[np.where(nf[ri] > 0, off[ri] + kpt, len(flat) - 1)], np.nan)
    fmed = R.sword_facc_total_km2.values
    agree = lambda x: (np.round(x, 3) >= 1 / AREA_TOL) & (np.round(x, 3) <= AREA_TOL)
    wK = np.array([win(i) for i in range(n)])
    mp = np.flatnonzero(stage[ri] > 0)
    cur_q = pd.Series(nearest_upa(mp, pa[ri[mp]], pb[ri[mp]], wK[ri[mp]]) / fb_pt[mp]).groupby(ri[mp]).median().reindex(range(n)).values
    nmax = lambda v: max([x for x in v if x == x], default=np.nan)
    fup = np.array([nmax([fmed[j] for j in nup[i]]) for i in range(n)])
    fdn = np.array([nmax([fmed[j] for j in ndn[i]]) for i in range(n)])
    facc_seq_ok = ~(fup > AREA_TOL * fmed) & ~(fdn < fmed / AREA_TOL)
    elig = (R.main_side.values == 0) & has_own & (seaf < 0.5) & (fmed > 0) & facc_seq_ok
    pool = np.flatnonzero(elig & ((stage == 0) | (np.isfinite(cur_q) & ~agree(cur_q))))
    E = D[np.isin(D.ri.values, pool)]
    cvp = E.groupby(['ri', 'path_seq_id']).w.sum().reset_index()
    cvp['cov'] = cvp.w / L[cvp.ri.values]
    E = E.merge(cvp.loc[cvp['cov'] >= AG_MIN_COVER, ['ri', 'path_seq_id', 'cov']], on=['ri', 'path_seq_id'])
    upE = nearest_upa(E.pt.values, E.path_seq_id.values.astype(np.int64), np.zeros(len(E), np.int64), Klim[E.ri.values])
    E = E.assign(upa=upE, q=upE / fb_pt[E.pt.values], rp=pos_up[E.pt.values] / L[E.ri.values])
    key = ['ri', 'path_seq_id']
    SQ = E.groupby(key).agg(cov=('cov', 'first'), q=('q', 'median'), rp0=('rp', 'min'), rp1=('rp', 'max'))
    SQ['upa_up'] = E[E.rp < 0.25].groupby(key).upa.median()
    SQ['upa_dn'] = E[E.rp > 0.75].groupby(key).upa.median()
    SQ = SQ.reset_index()
    AGC, AGWHY = {}, {}
    for r in pool:
        AGWHY[r] = 'no_path_cover>=0.5'
    for r, g in SQ.groupby('ri'):
        a = g[agree(g.q.values)]
        if not len(a):
            AGWHY[r] = 'no_path_with_same_area'
            continue
        if len(a) > 1:
            AGWHY[r] = 'two_paths_with_same_area'
            continue
        z = a.iloc[0]
        X = int(z.path_seq_id)
        AGWHY[r] = ('short_path' if plen[X] <= MIN_PATH_KM else
                    'same_as_current' if stage[r] and cur(r) == (X,) else
                    'direction_unknown' if not orient_ok[r] else
                    'path_not_along_whole_reach' if not (z.rp0 < 0.25 and z.rp1 > 0.75) else
                    'against_flow' if z.upa_up > 1.1 * z.upa_dn else '')
        if not AGWHY[r]:
            far_ = bool(not lake[r] and kgrid(r, (X,)).median() > Kd[r])
            AGC[r] = (X, float(z['cov']), float(z.q), far_)
            del AGWHY[r]
    AG = np.full(n, '', object)
    trusted = lambda j: stage[j] > 0 and (AG[j] != '' or not np.isfinite(cur_q[j]) or agree(cur_q[j]))
    witness = lambda j: stage[j] > 0 and (AG[j] != '' or (np.isfinite(cur_q[j]) and agree(cur_q[j])))
    SU, SD = [set(nup[r]) for r in range(n)], [set(ndn[r]) for r in range(n)]
    for r in range(n):
        for u in nup[r]:
            SD[u].add(r)
        for d in ndn[r]:
            SU[d].add(r)
    SA = [SU[r] & SD[r] for r in range(n)]
    for r in range(n):
        SU[r] -= SA[r]
        SD[r] -= SA[r]
    fit_up = lambda j, X: dn_end(j) == X or pdn.get(dn_end(j)) == X
    fit_dn = lambda j, X: up_end(j) == X or pdn.get(X) == up_end(j)
    fits = lambda r, X, keep: (all(fit_up(j, X) for j in SU[r] if keep(j)) and all(fit_dn(j, X) for j in SD[r] if keep(j))
                               and all(fit_up(j, X) or fit_dn(j, X) for j in SA[r] if keep(j)))
    nbrs = lambda r: SU[r] | SD[r] | SA[r]
    for rnd in range(len(AGC) + 1):
        ok = {}
        for r, (X, cv_, q_, far_) in AGC.items():
            if AG[r]:
                continue
            if not fits(r, X, trusted):
                AGWHY[r] = 'neighbor_conflict'
                continue
            if (cv_ < COVER or far_) and not any(witness(j) for j in nbrs(r)):
                AGWHY[r] = 'no_trusted_neighbor' if cv_ < COVER else 'far_without_witness'
                continue
            ok[r] = X
        bad = set()
        for r, X in ok.items():
            for u in SU[r]:
                if u in ok and not (ok[u] == X or pdn.get(ok[u]) == X):
                    bad |= {r, u}
            for a in SA[r]:
                if a in ok and not (ok[a] == X or pdn.get(ok[a]) == X or pdn.get(X) == ok[a]):
                    bad |= {r, a}
        for r in bad:
            AGWHY[r] = 'neighbor_switch_conflict'
            del ok[r]
        for r, X in ok.items():
            cv_, q_ = AGC[r][1], AGC[r][2]
            if stage[r]:
                switched[r] = next(p for p in cur(r) if p != X) if X in cur(r) else pa[r]
                fix[r] = ';'.join(z for z in (fix[r], 'area_guided_switch') if z)
                sw_ratio[r] = (cur_q[r], q_)
                AG[r], swK[r] = 'switch', Klim[r]
            else:
                AG[r] = 'rescue'
            stage[r], fK[r], it2[r] = 3, Klim[r], 0
            pa[r], pb[r], kind[r], rel[r], cv[r] = X, 0, 'one_path', 'area_guided', cv_
            AGWHY.pop(r, None)
        if not ok:
            break
    else:
        raise RuntimeError('same water did not converge')
    for r in np.flatnonzero(AG != ''):
        assert fits(r, pa[r], trusted), r
        if AGC[r][1] < COVER or AGC[r][3]:
            assert any(witness(j) for j in nbrs(r)), r

    nb_n, nb_ok = np.zeros(n, int), np.zeros(n, int)
    for i in np.flatnonzero(stage > 0):
        Pi = {pa[i], pb[i]} - {0}
        for j in nbr[i]:
            if stage[j]:
                nb_n[i] += 1
                nb_ok[i] += any(conn(a, b) for a in Pi for b in {pa[j], pb[j]} - {0})
    up_pt = np.full(len(rr), np.nan, np.float32)
    mp = np.flatnonzero(stage[ri] > 0)
    wK = np.array([win(i) for i in range(n)])
    up_pt[mp] = nearest_upa(mp, pa[ri[mp]], pb[ri[mp]], wK[ri[mp]])
    fn_area = pd.Series(up_pt.astype(np.float64)).groupby(ri).median().reindex(range(n)).values
    q_fin = pd.Series(up_pt / fb_pt).groupby(ri).median().reindex(range(n)).values
    jump = np.array([len(a) > 0 and a.max() > AREA_TOL * a.min() for a in FL])
    ms, ty, lk, ff = R.main_side.values, R['type'].values, R.lakeflag.values, R.facc_flag.values
    area_check = np.select([stage == 0, AG != '', ~np.isfinite(q_fin), agree(q_fin)], ['', 'used_in_matching', 'no_facc', 'agrees'], 'disagrees')
    area_note = np.array([';'.join(z for z, c in (('side_channel', ms[i] != 0), ('ghost', ty[i] == 6), ('lake', lk[i] == 1),
                                                  ('sword_facc_jump_in_reach', jump[i]), ('sword_facc_not_monotone', not facc_seq_ok[i]),
                                                  ('sword_facc_flag_' + ff[i], ff[i] != 'ok')) if c)
                          if area_check[i] == 'disagrees' else '' for i in range(n)], object)
    medK = np.full(n, np.nan)
    for i in np.flatnonzero(stage > 0):
        medK[i] = kgrid(i, cur(i), Dc if rel[i] == 'short_path_into_neighbor' else None).median()
    for r, d in rejected.items():
        if not (kind[r] == 'two_paths' and (pa[r], pb[r]) in d) and kind[r] != 'two_paths_against_flow':
            fix[r] = ';'.join(z for z in (fix[r], 'pair_rejected_order') if z)
    pcost = np.array([pair_cost.get((i, pa[i], pb[i]), np.nan) if kind[i] == 'two_paths' else np.nan for i in range(n)])
    rj = np.array([';'.join(f'{t}+{q}:{cover_text(x)}' for (t, q), x in rejected.get(i, {}).items()) for i in range(n)], object)
    ag = np.array([';'.join(f'{a}+{b}:{cover_text(x)}' for (a, b), x in against.get(i, {}).items()) for i in range(n)], object)
    short7 = cover(D[D.K <= K_AUTO]).drop_duplicates('ri').set_index('ri')['cov'].reindex(range(n)).fillna(0).values >= COVER
    has_opt = np.isin(np.arange(n), list(opts))
    any_nb = any_nb0
    best_any = np.max([best_cov[K] for K in range(1, K_AUTO + 1)] + [best_cov['s2']], axis=0)
    reason = np.select([stage > 0, seaf >= 0.5, ~has_own, has_opt & ~any_nb, has_opt, short7,
                        best_any >= 0.5],
                       ['', 'mostly_sea', 'not_in_any_basin', 'no_matched_neighbor', 'candidate_not_connected', 'only_short_path_nearby',
                        'partly_on_path'],
                       'far_from_path')
    sf = R.sword_facc_total_km2.values
    out_t = pd.DataFrame({
        'reach_id': rid, 'subnetwork_id': sid, 'basin_id': ownb, 'system_basin_id': p, 'reach_length_km': R.reach_length.values / 1000,
        'width_m': R.width.values, 'lakeflag': R.lakeflag.values, 'type': R['type'].values, 'main_side': R.main_side.values,
        'sword_facc_flag': R.facc_flag.values,
        'river_name': R.river_name.values, 'own_basin_id': R.own_basin_id.values, 'sea_fraction': seaf,
        'match_stage': stage, 'found_window': np.where(stage > 0, 2 * fK + 1, 0), 'stage2_iteration': it2,
        'stage2_search_window': 2 * Klim + 1, 'match_kind': kind, 'neighbor_relation': rel,
        'path_seq_id': pa, 'path2_seq_id': pb, 'cover': cv, 'runner_up_path_seq_id': p2, 'runner_up_cover': c2,
        'pair_unordered_cover': pcost, 'rejected_pairs': rj, 'against_flow_pairs': ag, 'fix_note': fix, 'switched_from_path_seq_id': switched,
        'area_ratio_before_switch': sw_ratio[:, 0], 'switch_not_done': sw_no, 'switch_window': np.where(switched > 0, 2 * swK + 1, 0),
        'land_cover': land_cv, 'merit_water_fraction': wfrac,
        'rescue_not_done': [WHY.get(i, '') if stage[i] == 0 else '' for i in range(n)], 'median_pixel_distance': medK,
        'neighbors_matched': nb_n, 'neighbors_connected': nb_ok,
        'merit_upa_on_path_km2': fn_area.round(1), 'sword_facc_total_km2': sf, 'area_ratio': q_fin,
        'area_check': area_check, 'area_note': area_note, 'area_guided': AG,
        'area_guided_not_done': [AGWHY.get(i, '') if AG[i] == '' and area_check[i] in ('', 'disagrees') else '' for i in range(n)],
        'not_found_reason': reason})
    for K in range(1, K_AUTO + 1):
        out_t[f'best_cover_{2*K+1}x{2*K+1}'] = best_cov[K]
    out_t['best_cover_stage2_window'] = best_cov['s2']
    out_t['path_length_km'] = P.path_length_km.reindex(out_t.path_seq_id).values.round(2)
    out_t['path_level'] = P.path_level.reindex(out_t.path_seq_id).values
    out_t.to_parquet(out + f'sword_reach_to_flownest_path_{sid}.parquet', index=False)
    out_t.to_csv(out + f'sword_reach_to_flownest_path_{sid}.csv', index=False)
    rows = []
    geo = AG == ''
    steps = [(f'{2*K+1}x{2*K+1}', geo & (stage == 1) & (fK == K)) for K in range(1, K_AUTO + 1)]
    steps += [(f'beyond_7x7_iteration_{i}', geo & (stage == 2) & (it2 == i)) for i in range(1, it2.max() + 1)]
    steps += [('between_neighbors', geo & (kind == 'between_neighbors')), ('against_flow', geo & (kind == 'two_paths_against_flow'))]
    steps += [(nm, geo & (rel == nm)) for nm in RESCUE]
    steps += [('area_guided_switch', AG == 'switch'), ('area_guided_rescue', AG == 'rescue')]
    cum = np.zeros(n, bool)
    for name, m in steps:
        cum |= m
        rows.append(dict(step=name, new_matched=int(m.sum()), new_one_path=int((m & (kind == 'one_path')).sum()),
                         new_two_paths=int((m & np.isin(kind, ['two_paths', 'two_paths_against_flow'])).sum()), new_length_km=round(float(L[m].sum()), 1),
                         cum_matched=int(cum.sum()), cum_matched_pct=round(100 * cum.sum() / n, 2),
                         cum_length_pct=round(100 * L[cum].sum() / L.sum(), 2)))
    W = pd.DataFrame(rows)
    W.to_csv(out + f'match_steps_{sid}.csv', index=False)
    m = stage > 0
