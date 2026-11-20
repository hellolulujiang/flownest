"""A made-up grid: the code runs and the properties hold. No number from this grid is a result."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from flownest import number_grid, verify, offset_table, translate, read_through


def toy():
    d8 = np.full((8, 8), 4, np.uint8)             # everything flows south
    d8[7, :] = 1                                  # the final row flows east
    d8[7, 7] = 0                                  # outlet
    d8[3, 2] = 1; d8[3, 3] = 1; d8[3, 4] = 2      # a tributary crosses to column 5
    area = accumulate(d8)
    return d8, area


def accumulate(d8):
    from flownest.core import _downstream, _step_tables, ESRI_D8
    dr, dc = _step_tables(ESRI_D8)
    ds = _downstream(d8, d8 != 255, dr, dc)
    a = np.ones(ds.size)
    wait = np.bincount(ds[ds >= 0], minlength=ds.size)
    stack = list(np.flatnonzero(wait == 0))
    while stack:
        p = stack.pop()
        q = ds[p]
        if q >= 0:
            a[q] += a[p]
            wait[q] -= 1
            if wait[q] == 0:
                stack.append(q)
    return a.reshape(d8.shape)


def test_numbers_and_properties():
    d8, area = toy()
    n = number_grid(d8, area, threshold=3)
    assert n.grid_index.size == (area >= 3).sum()
    assert sorted(n.raster[n.raster > 0]) == list(range(1, n.grid_index.size + 1))
    assert all(v == 0 for v in verify(n, d8, area, 3, thresholds=(6, 12)).values())


def test_upstream_range_is_the_subtree():
    d8, area = toy()
    n = number_grid(d8, area, threshold=1)        # every pixel is a channel pixel: the range counts every pixel upstream
    for p in range(1, n.grid_index.size + 1):
        upstream = n.upstream(p)
        assert len(upstream) == area.ravel()[n.grid_index[p - 1]]


def test_hillslope_pixels_point_at_their_channel():
    d8, area = toy()
    n = number_grid(d8, area, threshold=3)
    assert (n.raster != 0).all() and (n.raster < 0).any()


def test_a_loop_is_refused():
    d8, area = toy()
    d8[1, 1] = 64; d8[0, 1] = 4                   # two pixels point at each other
    try:
        number_grid(d8, area, threshold=99)
    except ValueError:
        return
    raise AssertionError('a closed loop was not refused')


def test_translation_between_two_numberings():
    d8, area = toy()
    a = number_grid(d8, area, threshold=2)
    d9 = d8.copy(); d9[2, 0] = 1                  # one flow direction moves
    b = number_grid(d9, accumulate(d9), threshold=2)
    t = offset_table(a, b)
    assert sum(count for _, count, _ in t) == a.grid_index.size
    assert offset_table(a, a) == [(a.first, a.grid_index.size, 0)]
    assert translate(t, a.first + a.grid_index.size) is None
    for p in range(1, a.grid_index.size + 1):
        v = read_through(a, b, p)
        new = translate(t, p)
        if new is None:
            assert v <= 0
        else:
            assert v == new



def test_one_pass_counts_the_pixels_upstream():
    d8, area = toy()
    n = number_grid(d8, area, threshold=3)
    got = n.totals(1.0)
    for i in range(n.grid_index.size):
        upstream = n.upstream(n.first + i)
        assert got[i] == ((np.abs(n.raster) >= upstream.start) & (np.abs(n.raster) < upstream.stop)).sum()


def test_totals_and_means_match_brute_force():
    d8, area = toy()
    rng = np.random.default_rng(3)
    vals, pix = rng.normal(size=d8.shape), rng.uniform(0.5, 1.5, d8.shape)
    n = number_grid(d8, area, threshold=3)
    tot = n.totals(vals)
    mean, a = n.means(vals, pix)
    v = np.abs(n.raster)
    for i in range(n.grid_index.size):
        upstream = n.upstream(n.first + i)
        m = (v >= upstream.start) & (v < upstream.stop)
        assert np.isclose(tot[i], vals[m].sum())
        assert np.isclose(a[i], pix[m].sum()) and np.isclose(mean[i], (vals * pix)[m].sum() / pix[m].sum())


def test_a_window_with_numbers_of_its_neighbours_is_safe():
    d8, area = toy()
    n = number_grid(d8, area, threshold=3, first=1000)
    r = n.raster.copy()
    r[0, 0] = -5                                   # a number outside the numbering: ignored, not written out of bounds
    from flownest import upstream_totals
    got = upstream_totals(r, n.upstream_channel_pixel_count, 1.0, n.first)
    assert got[0] == (np.abs(n.raster) >= n.first).sum() - 1


def test_unit_catchment_is_two_ranges():
    d8, area = toy()
    n = number_grid(d8, area, threshold=1)        # every pixel is a channel pixel at the base
    v = np.abs(n.raster).ravel()
    for T in (1, 3, 6, 12):
        reach, _ = __import__('flownest.core', fromlist=['_reaches_and_paths'])._reaches_and_paths(n.down, n.area, float(T))
        own = np.zeros(n.grid_index.size + 1, np.int64)  # reach of the first kept pixel at or below each number
        for i in range(1, n.grid_index.size + 1):
            k = i
            while k > 0 and n.area[k - 1] < T:
                k = n.down[k - 1]
            own[i] = reach[k] if k > 0 else 0
        u = n.unit_catchments(T)
        for j, rid in enumerate(u['reach_seq_id']):
            inside = ((v >= u['first_1'][j]) & (v < u['first_1'][j] + u['count_1'][j])) | ((v >= u['first_2'][j]) & (v < u['first_2'][j] + u['count_2'][j]))
            assert (inside == (own[v] == rid)).all()
        if T == 1:
            assert (u['count_2'] == 0).all()


def random_tree(size=40, seed=11):
    """Random D8 directions that all drain to the final pixel: each pixel steps to a neighbour one step closer to it."""
    from flownest import ESRI_D8
    rng = np.random.default_rng(seed)
    code = {v: k for k, v in ESRI_D8.items()}
    rr, cc = np.indices((size, size))
    dist = np.maximum(size - 1 - rr, size - 1 - cc)
    d8 = np.zeros((size, size), np.uint8)
    for r in range(size):
        for c in range(size):
            if dist[r, c] == 0:
                continue
            steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and 0 <= r + dr < size and 0 <= c + dc < size
                     and dist[r + dr, c + dc] == dist[r, c] - 1]
            d8[r, c] = code[steps[rng.integers(len(steps))]]
    return d8, accumulate(d8)


def test_unit_catchment_two_ranges_on_a_random_tree():
    from flownest.core import _reaches_and_paths
    d8, area = random_tree()
    n = number_grid(d8, area, threshold=4)
    assert all(x == 0 for x in verify(n, d8, area, 4, thresholds=(20, 80)).values())
    v = np.abs(n.raster).ravel()
    for T in (4, 20, 80):
        reach, _ = _reaches_and_paths(n.down, n.area, float(T))
        own = np.zeros(n.grid_index.size + 1, np.int64)
        for i in range(1, n.grid_index.size + 1):
            k = i
            while k > 0 and n.area[k - 1] < T:
                k = n.down[k - 1]
            own[i] = reach[k] if k > 0 else 0
        u = n.unit_catchments(T)
        two = 0
        for j, rid in enumerate(u['reach_seq_id']):
            inside = ((v >= u['first_1'][j]) & (v < u['first_1'][j] + u['count_1'][j])) | ((v >= u['first_2'][j]) & (v < u['first_2'][j] + u['count_2'][j]))
            assert (inside == (own[v] == rid)).all()
            two += u['count_2'][j] > 0 and u['first_2'][j] > u['first_1'][j] + u['count_1'][j]
        assert (T == 4) == (two == 0)              # at the base one range; above it some reaches need the second range


def test_upstream_pixels_matches_the_reference_accumulation():
    from flownest import upstream_pixels
    for d8 in (toy()[0], random_tree()[0]):
        assert (upstream_pixels(d8) == accumulate(d8)).all()


def test_reach_columns_match_brute_force():
    """Each column of reaches() against its definition, worked out from the links between reaches alone."""
    for (d8, area), base in ((toy(), 1), (toy(), 3), (random_tree(), 1), (random_tree(), 4), (random_tree(30, seed=5), 2)):
        n = number_grid(d8, area, threshold=base)
        upstream_counts = n.upstream_channel_pixel_count
        for T in sorted({base, 3 * base, 10 * base, 40 * base}):
            r = n.reaches(T)
            m = r['reach_seq_id'].size
            if m == 0:
                continue
            assert (r['reach_seq_id'] == np.arange(1, m + 1)).all()
            o = r['downstream_end_pixel_seq_id']
            u = o + r['channel_pixel_count'] - 1                             # the upstream end pixel
            nd = r['next_down_reach_seq_id']
            nd = np.where(nd == r['reach_seq_id'], 0, nd)                # a basin outlet points to itself; 0 below
            next_down_pixel = np.where(nd > 0, u[np.maximum(nd, 1) - 1], o)
            # the downstream reach holds the next pixel below this downstream end; only a basin-outlet reach points to itself
            for k in range(m):
                if nd[k]:
                    assert o[nd[k] - 1] <= next_down_pixel[k] <= u[nd[k] - 1]
                    assert next_down_pixel[k] == n.down[o[k] - 1]
                else:
                    assert next_down_pixel[k] == o[k] and n.down[o[k] - 1] == 0
            # upstream_channel_pixel_count: every numbered channel pixel upstream of the downstream end
            assert (r['upstream_channel_pixel_count'] == upstream_counts[o - 1]).all()
            # upstream_reach_count: the reaches upstream are the ones whose chain of reaches below passes through this one
            above = [set() for _ in range(m)]
            for k in range(m):
                j = k + 1
                while j:
                    above[j - 1].add(k + 1)
                    j = nd[j - 1]
            for k in range(m):
                assert above[k] == set(range(k + 1, k + 1 + r['upstream_reach_count'][k]))
                assert (r['upstream_reach_count'][k] == 1) == (not np.any(nd == k + 1))
            # a path goes on into the inflowing reach of the largest upstream area; path_level counts the paths down to the outlet
            for k in range(m):
                kids = np.flatnonzero(nd == k + 1)
                if kids.size:
                    head = kids[0]
                    assert r['upstream_area_km2'][head] == r['upstream_area_km2'][kids].max()
                    for j in kids:
                        assert (r['path_seq_id'][j] == r['path_seq_id'][k]) == (j == head)
            for k in range(m):
                level, j = 1, k
                while nd[j]:
                    level += r['path_seq_id'][nd[j] - 1] != r['path_seq_id'][j]
                    j = nd[j] - 1
                assert r['path_level'][k] == level
            assert r['path_seq_id'][0] == 1 and (np.diff(r['path_seq_id']) >= 0).all()


def test_first_reach_and_first_path_shift_the_ids():
    d8, area = random_tree()
    n = number_grid(d8, area, threshold=4)
    a, b = n.reaches(20), n.reaches(20, first_reach=101, first_path=51)
    for c in a:
        if c in ('reach_seq_id', 'next_down_reach_seq_id'):          # a basin outlet points to itself: shifted too
            assert (b[c] == a[c] + 100).all()
        elif c == 'path_seq_id':
            assert (b[c] == a[c] + 50).all()
        else:
            assert (b[c] == a[c]).all()


def test_edges_and_damaged_numberings():
    """A grid of one pixel, a grid with no channel pixel, outlets that cannot be numbered, a closed loop; and verify on numberings
    damaged after they were made."""
    import dataclasses
    import pytest
    from flownest import upstream_pixels
    one = number_grid(np.zeros((1, 1), np.uint8), np.ones((1, 1)), threshold=1)
    assert one.grid_index.size == 1 and all(x == 0 for x in verify(one, np.zeros((1, 1), np.uint8), np.ones((1, 1)), 1).values())
    d8, area = toy()
    none = number_grid(d8, area, threshold=1000)
    assert none.grid_index.size == 0 and all(x == 0 for x in verify(none, d8, area, 1000).values())
    assert offset_table(none, none) == [] and translate([], 5) is None
    with pytest.raises(ValueError):                # below the threshold
        number_grid(np.array([[1, 0]], np.uint8), np.array([[1., 2.]]), threshold=3, outlets=[(0, 1)])
    with pytest.raises(ValueError):                # (6, 7) is upstream of the outlet (7, 7)
        number_grid(d8, area, threshold=1, outlets=[(7, 7), (6, 7)])
    with pytest.raises(ValueError):                # outside the grid
        number_grid(d8, area, threshold=1, outlets=[(8, 0)])
    with pytest.raises(ValueError):                # two pixels flowing into each other
        upstream_pixels(np.array([[1, 16]], np.uint8))
    n = number_grid(d8, area, threshold=3)
    with pytest.raises(IndexError):
        n.upstream(n.grid_index.size + 1)
    t = offset_table(n, n)
    assert translate(t, 0) is None and translate(t, 1) == 1
    v = n.raster.copy().ravel()
    i = np.flatnonzero(v < 0)[0]
    v[i] = -1 if v[i] != -1 else -2                # one hillslope pixel sent to another channel pixel
    moved = dataclasses.replace(n, raster=v.reshape(n.raster.shape))
    for damaged, key in ((dataclasses.replace(n, raster=np.zeros_like(n.raster)), 'raster_not_the_number'),
                         (dataclasses.replace(n, area=np.zeros_like(n.area)), 'area_not_the_given_area'),
                         (moved, 'hillslope_pixel_not_the_value_of_its_next_pixel')):
        assert verify(damaged, d8, area, 3, thresholds=(6, 12))[key] > 0
    assert moved.fingerprint() != n.fingerprint()
    # links that are not numbers are refused before they are read: every other item of this wrong order would pass
    line, line_area = np.array([[1, 0]], np.uint8), np.array([[1., 2.]])
    two = number_grid(line, line_area, threshold=1)
    wrong_order = dataclasses.replace(two, raster=np.array([[1, 2]], two.raster.dtype), grid_index=np.array([0, 1]), area=np.array([1., 2.]),
                                      upstream_channel_pixel_count=np.array([1, 2]))
    for down in ([-1, -1], [0, 3]):
        assert verify(dataclasses.replace(wrong_order, down=np.array(down)), line, line_area, 1)['next_number_outside_the_numbers'] > 0
    assert verify(dataclasses.replace(two, grid_index=two.grid_index + 2), line, line_area, 1)['pixel_outside_the_grid'] > 0


def test_base_threshold_is_kept_and_finer_networks_are_refused():
    import dataclasses
    import pytest
    from flownest import Numbering
    d8, area = toy()
    n = number_grid(d8, area, threshold=3, first=101)
    assert n.threshold == 3
    for call in (n.reaches, n.unit_catchments):
        with pytest.raises(ValueError, match='below the numbering threshold'):
            call(2)
        assert call(3)['reach_seq_id'].size > 0
        assert call(6)['reach_seq_id'].size > 0
    assert dataclasses.replace(n, threshold=2).fingerprint() == n.fingerprint()
    direct = Numbering(n.raster, n.grid_index, n.down, n.upstream_channel_pixel_count, n.area, n.ncol, n.first)
    assert direct.fingerprint() == n.fingerprint()
    assert (n.upstream_channel_pixel_count == np.array([len(n.upstream(p)) for p in range(101, 101 + n.grid_index.size)])).all()


def test_nan_nodata_matches_an_integer_nodata():
    from flownest import upstream_pixels
    # A high area at nodata must never create a channel pixel; a flow into nodata ends at its edge.
    d8 = np.array([[1, 4, 247], [1, 0, 247], [1, 247, 255]], np.uint8)
    floating = d8.astype(np.float64)
    floating[d8 == 247] = np.nan
    area = upstream_pixels(d8, nodata=247)
    assert np.array_equal(upstream_pixels(floating, nodata=np.nan), area)
    area[d8 == 247] = 1000
    for threshold in (1, 2, 1001):
        want = number_grid(d8, area, threshold, nodata=247, first=47)
        got = number_grid(floating, area, threshold, nodata=np.nan, first=47)
        assert got.fingerprint() == want.fingerprint()
        assert not got.raster[np.isnan(floating)].any()
        assert all(x == 0 for x in verify(got, floating, area, threshold, nodata=np.nan).values())
    assert area[2, 2] == 1                           # MERIT's inland sink is valid with nodata=247
    assert upstream_pixels(d8)[2, 2] == 0           # the API default remains nodata=255


def test_explicit_outlet_on_a_loop_names_the_cause():
    import pytest
    with pytest.raises(ValueError, match='closed loop'):
        number_grid(np.array([[1, 16]], np.uint8), np.ones((1, 2)), outlets=[(0, 0)])


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)


def test_inputs_that_cannot_be_numbered():
    import pytest
    with pytest.raises(ValueError, match='NaN'):                # NaN area on a pixel with a direction
        number_grid(np.array([[1, 16]], np.uint8), np.array([[10.0, np.nan]]), threshold=1, outlets=[(0, 0)])
    for first in (0, -1, 1.5, True):                           # numbers start at 1; 0 means no number
        with pytest.raises(ValueError, match='first'):
            number_grid(np.array([[1, 16]], np.uint8), np.array([[10.0, 5.0]]), threshold=1, first=first)


def test_an_outlet_below_the_threshold_can_be_a_channel_pixel():
    """The release's rule for the basins of 0.05 to 1 km2: the outlet is one pixel, one reach and one path at the numbering threshold,
    in no network above it; the rest of the grid is numbered as before."""
    import pytest
    d8 = np.array([[4, 4, 0], [1, 0, 247]], np.uint8)        # (0, 0) -> (1, 0) -> (1, 1) outlet; (0, 1) -> (1, 1); (0, 2) outlet alone
    area = np.array([[1., 1., 0.5], [2., 4., np.nan]])
    with pytest.raises(ValueError):                            # without the option an outlet below the threshold is refused
        number_grid(d8, area, threshold=1, outlets=[(1, 1), (0, 2)], nodata=247)
    n = number_grid(d8, area, threshold=1, outlets=[(1, 1), (0, 2)], nodata=247, outlets_are_channel_pixels=True)
    plain = number_grid(d8, area, threshold=1, outlets=[(1, 1)], nodata=247)
    assert n.grid_index.size == plain.grid_index.size + 1 and (n.grid_index[:-1] == plain.grid_index).all()
    assert n.raster[0, 2] == n.grid_index.size and n.down[-1] == 0 and n.upstream_channel_pixel_count[-1] == 1
    assert all(x == 0 for x in verify(n, d8, area, 1, thresholds=(2,), nodata=247, outlets_are_channel_pixels=True).values())
    assert verify(n, d8, area, 1, thresholds=(2,), nodata=247)['number_not_a_channel_pixel'] == 1
    base, above = n.reaches(1), n.reaches(2)
    assert base['reach_seq_id'].size == plain.reaches(1)['reach_seq_id'].size + 1
    last = base['reach_seq_id'].size - 1
    assert base['downstream_end_pixel_seq_id'][last] == n.grid_index.size and base['next_down_reach_seq_id'][last] == last + 1
    assert base['path_seq_id'][last] == base['path_seq_id'].max() and base['upstream_area_km2'][last] == 0.5
    assert all((above[k] == plain.reaches(2)[k]).all() for k in above)
    u = n.unit_catchments(1)
    assert u['first_1'][last] == n.grid_index.size and u['count_1'][last] == 1 and u['count_2'][last] == 0


def test_an_outlet_below_the_threshold_that_drains_into_another_basin_is_refused():
    """An outlet made a channel pixel must still be an outlet: draining into another outlet's land, across hillslope pixels or straight
    into a channel pixel, it is refused (it would sit inside that basin's numbers)."""
    import pytest
    d8 = np.array([[1, 1, 0]], np.uint8)                      # (0, 0) -> (0, 1) -> (0, 2) outlet
    with pytest.raises(ValueError, match='drain into'):         # across the hillslope pixel (0, 1)
        number_grid(d8, np.array([[0.5, 0.8, 2.0]]), threshold=1, outlets=[(0, 2), (0, 0)], outlets_are_channel_pixels=True)
    with pytest.raises(ValueError):                            # straight into a channel pixel, listed after or before it
        number_grid(d8, np.array([[0.5, 1.5, 2.0]]), threshold=1, outlets=[(0, 2), (0, 0)], outlets_are_channel_pixels=True)
    with pytest.raises(ValueError):
        number_grid(d8, np.array([[0.5, 1.5, 2.0]]), threshold=1, outlets=[(0, 0), (0, 2)], outlets_are_channel_pixels=True)


def test_area_and_threshold_are_held_as_float32():
    """As the C code (upa >= (float)A, MERIT's upa float32): an area and a threshold not exact in float32 (11.2, 11.3 km2) are
    compared as float32, so a float64 and a float32 area give the same numbering, and verify and the derived thresholds agree."""
    d8 = np.array([[1, 1, 0]], np.uint8)          # east, east, outlet
    for v in (11.2, 11.3):
        for dtype in (np.float32, np.float64):
            one = number_grid(np.zeros((1, 1), np.uint8), np.array([[v]], dtype), threshold=v)
            assert one.grid_index.size == 1, (v, dtype)
            assert all(c == 0 for c in verify(one, np.zeros((1, 1), np.uint8), np.array([[v]], dtype), v).values()), (v, dtype)
            area = np.array([[1.0, 2.0, v]], dtype)
            n = number_grid(d8, area, threshold=1)
            assert n.reaches(v)['reach_seq_id'].size == 1, (v, dtype)      # the outlet pixel is a reach at A = its area
            assert n.unit_catchments(v)['reach_seq_id'].size == 1, (v, dtype)
            assert all(c == 0 for c in verify(n, d8, area, 1, thresholds=(v,)).values()), (v, dtype)
            # a numbering at v read at np.float32(v), the same threshold in float32, is not below it
            m = number_grid(d8, area, threshold=v)
            assert m.reaches(np.float32(v))['reach_seq_id'].size == m.reaches(v)['reach_seq_id'].size == 1, (v, dtype)
            assert all(c == 0 for c in verify(m, d8, area, v, thresholds=(np.float32(v),)).values()), (v, dtype)


def test_unit_windows_of_a_window_without_numbers_is_empty():
    from flownest.catchments import unit_windows, NODATA
    assert unit_windows(np.zeros((3, 4), np.int32)) == {}
    assert unit_windows(np.full((3, 4), NODATA, np.int32)) == {}
    assert unit_windows(np.array([[0, -2, 2], [1, -1, 0]], np.int32), 10, 20) == {1: (11, 12, 20, 22), 2: (10, 11, 21, 23)}
