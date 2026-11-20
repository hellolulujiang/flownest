"""Checks of a numbering, each from the flow directions and not from the walk that made it."""
import numpy as np
from numba import njit


@njit(cache=True)
def _walk_hillslope_to_channel(ds, chan, number_of, pixels):
    """First channel pixel on the flow path of each pixel, followed one step at a time."""
    out = np.zeros(pixels.size, np.int64)
    for i in range(pixels.size):
        p = pixels[i]
        while p >= 0 and not chan[p]:
            p = ds[p]
        out[i] = number_of[p] if p >= 0 else 0
    return out


@njit(cache=True)
def _subtree_sizes(down):
    """Pixels in each subtree, leaves first, without using the order of the numbers."""
    N = down.size
    wait = np.zeros(N + 1, np.int64)
    for i in range(N):
        wait[down[i]] += 1
    size = np.ones(N + 1, np.int64)
    queue = np.empty(N, np.int64)
    k = 0
    for n in range(1, N + 1):
        if wait[n] == 0:
            queue[k] = n
            k += 1
    i = 0
    while i < k:
        n = queue[i]
        i += 1
        d = down[n - 1]
        if d > 0:
            size[d] += size[n]
            wait[d] -= 1
            if wait[d] == 0:
                queue[k] = d
                k += 1
    return size


def verify(n, d8, area, threshold, thresholds=(25,), sample=200000, seed=7, nodata=255, codes=None, outlets_are_channel_pixels=False):
    """Returns a dict of failed-item counts; every value must be 0.

    outlets_are_channel_pixels as given to number_grid: an outlet of the numbering below the threshold is then a channel pixel.

    Pass the same d8, area, threshold, codes and nodata that built the numbering n. Each check is made from the flow directions and
    the area: which pixels are numbered, where each one flows, the order of the branches at a confluence and the value of every
    pixel of the raster; none reads the walk that made the numbering. The outlets are taken in the order the numbering gives them."""
    from .core import _downstream, _step_tables, _valid_pixels, ESRI_D8, _reaches_and_paths, _as_float32
    dr, dc = _step_tables(ESRI_D8 if codes is None else codes)
    valid = _valid_pixels(d8, nodata)
    ds = _downstream(d8, valid, dr, dc)
    a = area.astype(np.float32).astype(np.float64).ravel()   # held as float32, as number_grid does
    chan = valid.ravel() & (a >= _as_float32(threshold))
    if outlets_are_channel_pixels and n.grid_index.size and (n.grid_index >= 0).all() and (n.grid_index < ds.size).all():
        outlet = n.grid_index[n.down == 0]
        chan[outlet] |= valid.ravel()[outlet] & (a[outlet] > 0) & np.isfinite(a[outlet])
    N = n.grid_index.size
    ids = np.arange(1, N + 1)
    bad = {}
    # the links must point at numbers and the pixels into the grid, or nothing below can be read
    bad['next_number_outside_the_numbers'] = int(np.sum((n.down < 0) | (n.down > N)))
    bad['pixel_outside_the_grid'] = int(np.sum((n.grid_index < 0) | (n.grid_index >= ds.size)))
    if bad['next_number_outside_the_numbers'] or bad['pixel_outside_the_grid']:
        return bad
    # the numbered pixels: channel pixels, each once, with their own area and their number in the raster
    bad['numbers_not_once_each'] = int(N - np.unique(n.grid_index).size)
    bad['number_not_a_channel_pixel'] = int(np.sum(~chan[n.grid_index]))
    bad['area_not_the_given_area'] = int(np.sum(n.area != a[n.grid_index]))
    v = n.raster.ravel().astype(np.int64)
    bad['raster_not_the_number'] = int(np.sum(v[n.grid_index] != ids + n.first - 1))
    numbered = np.zeros(ds.size, np.bool_)
    numbered[n.grid_index] = True
    bad['positive_on_a_pixel_not_numbered'] = int(np.sum((v > 0) & ~numbered))
    # every channel pixel's next pixel is the D8 step; the numbered pixels are closed upstream (a channel pixel that flows into a
    # numbered pixel is numbered) and an outlet flows into no numbered pixel
    q = ds[n.grid_index]
    bad['next_pixel_not_the_d8_step'] = int(np.sum((n.down > 0) & (q != n.grid_index[np.maximum(n.down, 1) - 1])))
    bad['outlet_flows_into_a_numbered_pixel'] = int(np.sum((n.down == 0) & (q >= 0) & numbered[np.maximum(q, 0)]))
    bad['channel_pixel_upstream_left_out'] = int(np.sum(chan & (ds >= 0) & numbered[np.maximum(ds, 0)] & ~numbered))
    # the subtree of every number is the range: count the pixels upstream of each from the links alone, leaves first
    size = _subtree_sizes(n.down)
    bad['subtree_not_the_range'] = int(np.sum(size[1:] != n.upstream_channel_pixel_count))
    # at every confluence the branches come in order of area, largest first, ties to the smaller grid index: the first is the next
    # number, each other one starts right after the subtree of the one before; the outlets follow each other the same way from 1
    d = n.down
    kids = np.flatnonzero(d > 0)                              # number - 1 of every pixel below a confluence or on a reach
    if kids.size:
        srt = kids[np.lexsort((n.grid_index[kids], -n.area[kids], d[kids]))]
        new = np.r_[True, d[srt][1:] != d[srt][:-1]]         # the largest branch at each pixel
        prev = np.r_[0, srt[:-1]] + 1                         # the number of the branch before
        off = srt + 1 != np.where(new, d[srt] + 1, prev + size[prev])
        bad['larger_branch_not_next'] = int(np.sum(off & new))
        bad['smaller_branch_not_after_the_larger'] = int(np.sum(off & ~new))
    else:
        bad['larger_branch_not_next'] = bad['smaller_branch_not_after_the_larger'] = 0
    roots = np.flatnonzero(d == 0) + 1
    bad['outlet_not_after_the_one_before'] = int(np.sum(roots != np.r_[1, roots[:-1] + size[roots[:-1]]])) if roots.size else 0
    # every other pixel: its value is that of its next pixel, minus it if that is numbered, and 0 where the flow ends. This holds
    # pixel by pixel, so it holds along every flow path; a sample is also followed step by step to its first channel pixel
    rest = np.flatnonzero(~numbered)
    t = ds[rest]
    vt = np.where(t >= 0, v[np.maximum(t, 0)], 0)
    bad['hillslope_pixel_not_the_value_of_its_next_pixel'] = int(np.sum(v[rest] != np.where(vt > 0, -vt, vt)))
    hillslope = np.flatnonzero(v < 0)
    if hillslope.size:
        s = np.random.default_rng(seed).choice(hillslope, min(sample, hillslope.size), replace=False)
        got = _walk_hillslope_to_channel(ds, v > 0, v, s)
        bad['hillslope_pixel_not_its_first_channel_pixel'] = int(np.sum(got != -v[s]))
    # every object at every threshold is the kept numbers of one range
    for T in (threshold,) + tuple(t for t in thresholds if _as_float32(t) > _as_float32(threshold)):   # in float32, as the C code
        T32 = _as_float32(T)
        on_area = np.maximum(n.area, T32) if T32 == _as_float32(threshold) else n.area   # every number is on the network at the numbering threshold
        reach, path = _reaches_and_paths(n.down, on_area, T32)
        on = on_area >= T32
        kept = np.cumsum(on)                                 # kept numbers up to n
        for name, ids_T in (('reach', reach), ('path', path)):
            nn = np.flatnonzero(ids_T)
            if nn.size == 0:                                  # no pixel reaches this threshold
                continue
            r = ids_T[nn]
            c = r.max()
            lo = np.full(c + 1, 2 ** 62, np.int64); np.minimum.at(lo, r, nn)
            hi = np.zeros(c + 1, np.int64); np.maximum.at(hi, r, nn)
            cnt = np.bincount(r, minlength=c + 1)
            bad[f'{name}_{T}_not_one_kept_range'] = int(np.sum(kept[hi[1:] - 1] - np.r_[0, kept][lo[1:] - 1] != cnt[1:]))
    bad['numbers_over_int32'] = int(N + n.first - 1 > 2 ** 31 - 1)
    return bad
