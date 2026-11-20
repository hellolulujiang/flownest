"""The FlowNest rule on a grid of D8 flow directions and upstream area.

Rule: a channel pixel is a pixel with upstream area >= threshold, both held as float32 as the C code does (and, with
outlets_are_channel_pixels, each given outlet: the release's basins of 0.05 to 1 km2 are one channel pixel each). From each
outlet, walk up the channel pixels depth first, entering the branch with the larger upstream area first, ties to the smaller grid
index. The order of the walk is the number, pixel_seq_id. A
hillslope pixel stores minus the number of the first channel pixel on its flow path. Everything else in this file is read off those
numbers, in the names of the v4 FlowNest release: pixel_seq_id, reach_seq_id, path_seq_id, channel_pixel_count,
upstream_channel_pixel_count, ... As in the release, every run of numbers is its first number and a count, and a basin outlet points to
itself. Upstream ids are range(pixel_seq_id, pixel_seq_id + upstream_channel_pixel_count).
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from numba import njit

ESRI_D8 = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}   # MERIT Hydro, HydroSHEDS


def _step_tables(codes):
    dr, dc = np.full(256, 9, np.int8), np.zeros(256, np.int8)
    for k, (r, c) in codes.items():
        dr[k], dc[k] = r, c
    return dr, dc


def _valid_pixels(d8, nodata):
    """Pixels with a flow direction: neither the declared nodata nor, on a float grid, NaN."""
    valid = d8 != nodata
    return valid & ~np.isnan(d8) if d8.dtype.kind == 'f' else valid


@njit(cache=True)
def _downstream(d8, valid, dr, dc):
    """Zero-based grid index of the next pixel, -1 where the flow leaves the grid, ends in a sink or enters nodata."""
    nrow, ncol = d8.shape
    ds = np.full(nrow * ncol, -1, np.int64)
    for r in range(nrow):
        for c in range(ncol):
            k = d8[r, c]
            if not valid[r, c] or k < 0 or k > 255:
                continue
            k = int(k)
            if dr[k] == 9:                              # a code outside the table: no step
                continue
            rr, cc = r + dr[k], c + dc[k]
            if 0 <= rr < nrow and 0 <= cc < ncol and valid[rr, cc]:
                ds[r * ncol + c] = rr * ncol + cc
    return ds


@njit(cache=True)
def _walk(ds, chan, area, roots):
    """Depth-first preorder over the channel pixels from each root (a channel pixel) in turn. Returns the channel pixels in number order."""
    n = ds.size
    start = np.zeros(n + 1, np.int64)
    idx = np.flatnonzero(chan)
    for p in idx:
        if ds[p] >= 0 and chan[ds[p]]:
            start[ds[p] + 1] += 1
    start = np.cumsum(start)
    fill = start[:-1].copy()
    kids = np.empty(start[-1], np.int64)
    for p in idx:                                       # idx ascends, so equal areas stay in grid order
        if ds[p] >= 0 and chan[ds[p]]:
            q = ds[p]
            i = fill[q]
            while i > start[q] and area[kids[i - 1]] < area[p]:
                kids[i] = kids[i - 1]
                i -= 1
            kids[i] = p
            fill[q] += 1
    order = np.empty(idx.size, np.int64)
    stack = np.empty(idx.size + 1, np.int64)
    done = np.zeros(n, np.bool_)
    k = 0
    for root in roots:
        sp = 1
        stack[0] = root
        while sp > 0:
            sp -= 1
            p = stack[sp]
            if done[p]:
                raise ValueError('an outlet is given twice, lies upstream of another outlet, or the flow directions contain a closed loop')
            done[p] = True
            order[k] = p
            k += 1
            for i in range(start[p + 1] - 1, start[p] - 1, -1):
                stack[sp] = kids[i]
                sp += 1
    return order[:k]


@njit(cache=True)
def _sign_hillslope(ds, num):
    """+number on channel pixels, -number on the hillslope pixels that reach them, 0 on the rest. Stops on a closed loop of flow directions."""
    v = num.copy()                                      # > 0 a number, < 0 minus the number reached, 0 not yet or no channel reached
    path = np.empty(ds.size, np.int64)
    seen = np.zeros(ds.size, np.bool_)
    for s in range(ds.size):
        if v[s] != 0 or seen[s]:
            continue
        m, p, end = 0, s, 0
        while True:
            seen[p] = True
            path[m] = p
            m += 1
            q = ds[p]
            if q < 0:
                break
            if num[q] > 0:
                end = num[q]
                break
            if v[q] != 0:
                end = v[q]
                break
            if seen[q]:
                found = False
                for i in range(m):
                    found = found or path[i] == q
                if found:
                    raise ValueError('the flow directions contain a closed loop')
                break
            p = q
        for i in range(m):
            v[path[i]] = end if end < 0 else -end
    return v


def _as_float32(threshold):
    """The threshold as the C code holds it: upa >= (float)A in float32, MERIT's upa being float32. A threshold not exact in float32
    (11.2 km2) is rounded so; number_grid and verify round the area to float32 too (a float64 area of 11.3 is held as float32(11.3)),
    so the float64 comparisons below are float32 ones and a float64 and a float32 area give the same numbering."""
    return float(np.float32(threshold))


@njit(cache=True)
def _reaches_and_paths(down, area, T):
    """Reach and path of every number at threshold T; 0 for numbers below it. The kept branch with the lowest number is the largest,
    and it carries the path on: a path goes up from the reach where it starts, always into that branch."""
    N = down.size
    kids = np.zeros(N + 1, np.int32)
    head = np.zeros(N + 1, np.int64)
    reach, path = np.zeros(N + 1, np.int64), np.zeros(N + 1, np.int64)
    for i in range(N):
        if area[i] >= T and down[i] > 0:
            kids[down[i]] += 1
            if head[down[i]] == 0:
                head[down[i]] = i + 1
    nr = nv = 0
    for i in range(N):
        if area[i] < T:
            continue
        n, d = i + 1, down[i]
        if d == 0 or kids[d] >= 2:
            nr += 1
            reach[n] = nr
        else:
            reach[n] = reach[d]
        if d == 0 or head[d] != n:
            nv += 1
            path[n] = nv
        else:
            path[n] = path[d]
    return reach, path


@njit(cache=True)
def _path_levels(next_down):
    """path_level of every reach (next_down[k] is the reach below reach k + 1, 0 at a basin outlet): 1 on the path that reaches the
    basin outlet; where a path starts, one more than the path it flows into; else that of the reach below (the row before)."""
    n = next_down.size
    level = np.zeros(n, np.int64)
    for k in range(n):
        d = next_down[k]
        if d == 0:
            level[k] = 1
        elif d == k:                                    # the reach below is the row before: the same path
            level[k] = level[k - 1]
        else:
            level[k] = level[d - 1] + 1
    return level


@dataclass
class Numbering:
    """A numbered grid; position i in each one-dimensional array is zero-based and belongs to pixel number first + i.

    raster: signed int32 grid, +p on a channel pixel, -p where the first channel reached is p, 0 elsewhere.
    grid_index: zero-based row-major grid index of each numbered pixel.
    down: one-based local number (1 .. N) of the next channel pixel, 0 at an outlet.
    upstream_channel_pixel_count: number of channel pixels upstream, including the pixel itself.
    area: upstream area of each numbered pixel, in the input area's units.
    ncol: number of grid columns; row and col are zero-based.
    first: number assigned to the first channel pixel (normally 1).
    threshold: base threshold in area's units, 0 if not given; not part of the fingerprint.
    """
    raster: np.ndarray
    grid_index: np.ndarray
    down: np.ndarray
    upstream_channel_pixel_count: np.ndarray
    area: np.ndarray
    ncol: int
    first: int
    threshold: float = 0.0

    @property
    def row(self):
        return self.grid_index // self.ncol

    @property
    def col(self):
        return self.grid_index % self.ncol

    def upstream(self, pixel_seq_id):
        """Channel pixel ids upstream, including this pixel, as a Python half-open range."""
        i = pixel_seq_id - self.first
        if not 0 <= i < self.grid_index.size:
            raise IndexError(f'{pixel_seq_id} is not a number of this numbering ({self.first} .. {self.first + self.grid_index.size - 1})')
        return range(pixel_seq_id, pixel_seq_id + int(self.upstream_channel_pixel_count[i]))

    def _area_on_the_network(self, threshold):
        """The area held against a threshold: at the numbering threshold every numbered pixel is on the network (an outlet numbered
        below it, with outlets_are_channel_pixels, too), so its area counts as the threshold there."""
        return np.maximum(self.area, _as_float32(threshold)) if _as_float32(threshold) == _as_float32(self.threshold) else self.area

    def reaches(self, threshold, first_reach=1, first_path=1):
        """Reach table at a threshold, in the columns of the FlowNest release and in its order. The columns that need the pixel size
        or the basins (length_km, distance_to_basin_outlet_km, unit_catchment_area_km2, basin_id, lon/lat, windows, crosswalks) are left
        to the release. Every run is its first number and a count: the pixels of a reach are downstream_end_pixel_seq_id and the
        channel_pixel_count - 1 after it, the channel pixels upstream of its downstream end upstream_channel_pixel_count from downstream_end_pixel_seq_id, the
        reaches upstream upstream_reach_count from reach_seq_id. A reach at a basin outlet points to itself.
        first_reach and first_path are the ids of
        the first reach and path (for a basin cut from the release, its ids there), as first is the number of the first pixel.
        upstream_area_km2 keeps the input area's units despite its published name. A threshold below the base raises ValueError."""
        if _as_float32(threshold) < _as_float32(self.threshold):      # in float32, as the C code compares
            raise ValueError(f'threshold {threshold} is below the numbering threshold {self.threshold}')
        reach, path = _reaches_and_paths(self.down, self._area_on_the_network(threshold), _as_float32(threshold))
        n = np.flatnonzero(reach)                       # number of each kept pixel, ascending
        r = reach[n]
        count = r.max() if r.size else 0
        a = np.full(count + 1, 2 ** 62, np.int64)       # the first number of a reach is its lowest
        np.minimum.at(a, r, n)
        b = np.zeros(count + 1, np.int64)
        np.maximum.at(b, r, n)
        ids = np.arange(1, count + 1)                   # local ids 1 .. count
        off, ro = self.first - 1, first_reach - 1
        dn = self.down[a[1:] - 1]
        upstream_count = self.upstream_channel_pixel_count[a[1:] - 1]
        next_down = reach[dn] * (dn > 0)
        reach_stop = np.searchsorted(a[1:], a[1:] + upstream_count)  # exclusive upstream reach boundary
        downstream_end = a[1:] + off
        return dict(reach_seq_id=ids + ro, downstream_end_pixel_seq_id=downstream_end, channel_pixel_count=b[1:] - a[1:] + 1,
                    upstream_channel_pixel_count=upstream_count,
                    next_down_reach_seq_id=np.where(next_down > 0, next_down + ro, ids + ro),
                    upstream_reach_count=reach_stop - ids + 1, path_seq_id=path[a[1:]] + first_path - 1, path_level=_path_levels(next_down),
                    upstream_area_km2=self.area[a[1:] - 1])

    def unit_catchments(self, threshold, first_reach=1):
        """Unit catchments as two half-open ranges of |v|, each stored as first/count.

        Returns reach_seq_id, first_1, count_1, first_2, count_2. The first range covers the reach's channel pixels; the second covers
        removed branches after the final kept branch's subtree. Zero count means empty. At the base threshold count_2 is zero.
        A threshold below the numbering's base threshold raises ValueError."""
        r = self.reaches(threshold, first_reach)
        off = self.first - 1
        a = r['downstream_end_pixel_seq_id'] - off
        b = a + r['channel_pixel_count'] - 1                 # the upstream end of the reach
        N = self.down.size
        kept = np.flatnonzero((self._area_on_the_network(threshold) >= _as_float32(threshold)) & (self.down > 0))
        final_branch = np.zeros(N + 1, np.int64)        # the final kept branch has the largest number
        np.maximum.at(final_branch, self.down[kept], kept + 1)
        u = final_branch[b]
        first_2 = np.where(u > 0, u + off + self.upstream_channel_pixel_count[np.maximum(u, 1) - 1], b + off + 1)
        stop = a + off + self.upstream_channel_pixel_count[a - 1]
        return dict(reach_seq_id=r['reach_seq_id'], first_1=a + off, count_1=r['channel_pixel_count'],
                    first_2=first_2, count_2=stop - first_2)

    def totals(self, values):
        """Total of a raster on the same grid over the catchment of every number, from one pass over the pixels. Index i of the
        result belongs to number first + i. The values are not masked: see upstream_totals."""
        from .stats import upstream_totals
        return upstream_totals(self.raster, self.upstream_channel_pixel_count, values, self.first)

    def means(self, values, pixel_area):
        """Area-weighted mean of a raster over the catchment of every number, and the catchment area, as the tuple (mean, area), from
        two passes over the pixels. Index i belongs to number first + i. The values are not masked: see upstream_totals."""
        from .stats import upstream_means
        return upstream_means(self.raster, self.upstream_channel_pixel_count, values, pixel_area, self.first)

    def fingerprint(self):
        """SHA-256 of the grid shape, the first number, the place, next pixel and upstream area of every number, and the signed raster
        (so a correction that sends a hillslope pixel to another channel pixel changes it too)."""
        h = hashlib.sha256()
        h.update(np.array([self.ncol, self.raster.shape[0], self.first], np.int64).tobytes())
        for a in (self.grid_index, self.down, self.area, self.raster):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


def upstream_pixels(d8, codes=ESRI_D8, nodata=255):
    """Upstream area in pixels, the pixel included, from the flow directions alone.

    nodata defaults to 255. For MERIT Hydro use nodata=247 (sea); 0 is a river mouth and 255 an inland sink, both outlets.
    Direction codes 1, 2, 4, 8, 16, 32, 64, 128 step E, SE, S, SW, W, NW, N, NE. NaN nodata is also accepted.
    """
    valid = _valid_pixels(d8, nodata)
    dr, dc = _step_tables(codes)
    ds = _downstream(d8, valid, dr, dc)
    return _accumulate(ds).reshape(d8.shape) * valid


@njit(cache=True)
def _accumulate(ds):
    n = ds.size
    a = np.ones(n)
    wait = np.zeros(n, np.int64)
    for p in range(n):
        if ds[p] >= 0:
            wait[ds[p]] += 1
    stack = np.empty(n, np.int64)
    k = 0
    for p in range(n):
        if wait[p] == 0:
            stack[k] = p
            k += 1
    while k > 0:
        k -= 1
        p = stack[k]
        q = ds[p]
        if q >= 0:
            a[q] += a[p]
            wait[q] -= 1
            if wait[q] == 0:
                stack[k] = q
                k += 1
    for p in range(n):
        if wait[p] > 0:                                 # never freed: the pixel is on a closed loop
            raise ValueError('the flow directions contain a closed loop')
    return a


def number_grid(d8, area, threshold=1.0, outlets=None, codes=ESRI_D8, nodata=255, first=1, outlets_are_channel_pixels=False):
    """Number a grid. d8 and area are 2-D arrays of the same shape. outlets are (row, col) pairs of channel pixels, numbered in the order
    given (largest first, for the rule), none upstream of another; or None for every outlet of the grid, largest first.
    outlets_are_channel_pixels=True makes each given outlet a channel pixel even below the threshold, as the release does for the
    basins of 0.05 to 1 km2: such an outlet is one pixel, one reach and one path at the numbering threshold, and in no network above.

    area and threshold must share units; reaches()['upstream_area_km2'] retains those units despite its published name.
    nodata defaults to 255. For MERIT Hydro use nodata=247 (sea); 0 is a river mouth and 255 an inland sink, both outlets.
    Direction codes 1, 2, 4, 8, 16, 32, 64, 128 step E, SE, S, SW, W, NW, N, NE. NaN nodata is also accepted.
    """
    if d8.shape != area.shape:
        raise ValueError('d8 and area must share a grid')
    if isinstance(first, bool) or int(first) != first or first < 1:
        raise ValueError(f'first is {first!r}: numbers start at 1 (0 means no number on the raster)')
    valid = _valid_pixels(d8, nodata)
    dr, dc = _step_tables(codes)
    ds = _downstream(d8, valid, dr, dc)
    a = area.astype(np.float32).astype(np.float64).ravel()   # held as float32, as the C code reads upa
    if np.isnan(a[valid.ravel()]).any():
        raise ValueError('area is NaN on a pixel with a flow direction')
    chan = valid.ravel() & (a >= _as_float32(threshold))
    if outlets is None:
        r = np.flatnonzero(chan & (ds < 0))
        r = r[np.lexsort((r, -a[r]))]
    else:
        rc = np.asarray(outlets, np.int64).reshape(-1, 2)
        if not ((rc >= 0).all() and (rc[:, 0] < d8.shape[0]).all() and (rc[:, 1] < d8.shape[1]).all()):
            raise ValueError('an outlet lies outside the grid')
        r = rc[:, 0] * d8.shape[1] + rc[:, 1]
        if outlets_are_channel_pixels:
            chan[r] |= valid.ravel()[r] & (a[r] > 0) & np.isfinite(a[r])
        if not chan[r].all():
            why = 'nodata, or upstream area not positive and finite' if outlets_are_channel_pixels else \
                'nodata, or upstream area below the threshold'
            raise ValueError(f'outlets {[tuple(int(x) for x in p) for p in rc[~chan[r]]]} are not channel pixels ({why})')
    fall = chan & (ds >= 0)
    if np.any(a[ds[fall]] < a[fall]):
        raise ValueError('upstream area falls along a flow path, so the channel pixels are not closed downstream')
    grid_index = _walk(ds, chan, a, r)
    N = grid_index.size
    if N + first - 1 > 2 ** 31 - 1:
        raise OverflowError(f'{N} channel pixels do not fit a signed 32-bit raster; raise the threshold or number each continent apart')
    num = np.zeros(ds.size, np.int64)
    num[grid_index] = np.arange(1, N + 1)
    down = np.zeros(N, np.int64)
    has = ds[grid_index] >= 0
    down[has] = num[ds[grid_index[has]]]                      # 0 where the flow leaves the numbered tree
    upstream_channel_pixel_count = _upstream_channel_pixel_counts(down)
    signed = _sign_hillslope(ds, num)
    into = ds[r] >= 0                                         # an outlet must not drain into another outlet's numbered land
    into[into] = signed[ds[r[into]]] != 0
    if into.any():
        raise ValueError(f'outlets {[divmod(int(p), d8.shape[1]) for p in r[into]]} drain into the land numbered from another '
                         f'outlet, so they are not outlets')
    off = first - 1
    raster = (np.sign(signed) * (np.abs(signed) + off)).astype(np.int32).reshape(d8.shape)
    return Numbering(raster, grid_index, down, upstream_channel_pixel_count, a[grid_index], d8.shape[1], first, threshold)


@njit(cache=True)
def _upstream_channel_pixel_counts(down):
    """One backward pass over the numbers: a pixel's subtree ends where its size says."""
    N = down.size
    size = np.ones(N + 1, np.int64)
    for n in range(N, 0, -1):
        if down[n - 1] > 0:
            size[down[n - 1]] += size[n]
    return size[1:]
