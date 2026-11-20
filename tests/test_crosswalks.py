"""The crosswalk modules (stations, merit_basins, catchments) on the made-up release of test_release.py: each answer is checked
against the made-up numbering itself, by brute force, so no number from the C code is needed here. The global comparisons with the
release of the C code are scripts outside the tests (README, "How it was checked")."""
import os
import shutil
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pytest

pytest.importorskip('pyarrow')
pytest.importorskip('rasterio')
from flownest import Release, stations, merit_basins, catchments      # noqa: E402
from test_release import made_up, _tif, GT, NROW, NCOL, NODATA       # noqa: E402,F401  (made_up is the fixture)


def _lonlat(row, col):
    """the centre of a pixel"""
    return GT[0] + (col + .5) * GT[1], GT[3] + (row + .5) * GT[5]


HILLY_BASIN = 1        # its sources become hillslope pixels: at the base threshold of 1 pixel the made-up release has none


@pytest.fixture(scope='module')
def hilly(made_up, tmp_path_factory):
    """The made-up release with the sources of one basin turned into hillslope pixels, each -q, q the pixel below it (as a
    source is at a threshold above its area). Only the pixel raster changes."""
    n, counts = made_up['n'], made_up['upstream_counts']
    folder = tmp_path_factory.mktemp('hilly') / 'release'
    shutil.copytree(made_up['folder'], folder)
    ids = np.arange(1, n.grid_index.size + 1)
    leaf = (counts == 1) & (n.down != 0) & (made_up['basin_of'] == HILLY_BASIN)
    raster = made_up['raster'].copy()
    raster[n.row[leaf], n.col[leaf]] = -n.down[leaf]
    _tif(folder / 'flownest_merit90m_pixel_seq_id_1km2.tif', raster, NODATA)
    assert leaf.sum() >= 3
    return dict(folder=str(folder), raster=raster, leaf=ids[leaf])


def _upa(made_up):
    n = made_up['n']
    upa = np.full((NROW, NCOL), np.nan)
    upa[n.row, n.col] = n.area
    return upa


# ------------------------------------------------------------------------------------------------ catchments
def _brute_parent_and_count(pixel, end):
    parent, count = [], []
    for o in range(pixel.size):
        holding = [j for j in range(o) if pixel[j] <= pixel[o] < end[j]]
        parent.append(holding[-1] + 1 if holding else 0)
        count.append(sum(1 for j in range(o, pixel.size) if pixel[j] < end[o]))
    return np.array(parent), np.array(count)


def test_nest_and_first_downstream_by_brute_force(made_up, hilly):
    n, counts = made_up['n'], made_up['upstream_counts']
    rng = np.random.default_rng(7)
    pixel = np.sort(rng.choice(np.arange(1, n.grid_index.size + 1), 40))      # repeats: several objects on one pixel
    end = pixel + counts[pixel - 1]
    parent, count = catchments.nest(pixel, end)
    want_parent, want_count = _brute_parent_and_count(pixel, end)
    assert (parent == want_parent).all() and (count == want_count).all()
    every = np.arange(1, n.grid_index.size + 1)
    want = np.array([max([j + 1 for j in range(pixel.size) if pixel[j] <= q < end[j]], default=0) for q in every])
    assert (catchments.first_downstream(every, pixel, end, parent) == want).all()
    # the label raster: a hillslope pixel takes the label of the channel pixel it drains to, with its sign
    values = hilly['raster']
    labels = catchments.label_window(values, pixel, end, parent)
    assert (values < 0).any() and (labels < 0).any()
    land = (values != NODATA) & (values != 0)
    assert (labels[values == NODATA] == NODATA).all()
    assert (labels[land] == np.sign(values[land]) * want[np.abs(values[land]) - 1]).all()
    unit = catchments.unit_windows(labels)
    for s, (r0, r1, c0, c1) in unit.items():
        r, c = np.nonzero(np.abs(np.where(labels == NODATA, 0, labels)) == s)
        assert (r0, r1, c0, c1) == (r.min(), r.max() + 1, c.min(), c.max() + 1)


def test_crossing_ranges_and_unsorted_objects_are_refused():
    with pytest.raises(ValueError):
        catchments.nest([1, 3], [5, 7])
    with pytest.raises(ValueError):
        catchments.nest([3, 1], [4, 2])


def test_gauge_takes_the_nearer_area_down_on_a_tie():
    # one stem: nodes at pixels 1 (area 100) and 6 (area 50); a gauge at pixel 4 (area 70), [4, 10)
    node_pixel, node_area, node_id = np.array([1, 6, 6]), np.array([100., 50., 50.]), np.array([30, 20, 10])
    node_end = np.array([11, 10, 10])
    parent, _ = catchments.nest(node_pixel, node_end)
    side, seq, ratio = catchments.pair_gauges_with_nodes([4], [70.], [10], node_pixel, node_area, node_id, node_end, parent)
    assert side[0] == 'up' and seq[0] == 3 and ratio[0] == 50. / 70.         # |ln(50/70)| < |ln(100/70)|; node id 10 < 20
    side, seq, ratio = catchments.pair_gauges_with_nodes([4], [70.], [10], node_pixel, np.full(3, 100.), node_id, node_end, parent)
    assert side[0] == 'down' and seq[0] == 1                                  # the same area up and down: down
    side, seq, ratio = catchments.pair_gauges_with_nodes([20], [70.], [21], node_pixel, node_area, node_id, node_end, parent)
    assert side[0] == 'none' and seq[0] == 0 and np.isnan(ratio[0])


# ------------------------------------------------------------------------------------------------ stations
def test_stations_on_channel_hillslope_sea_and_off_the_grid(made_up, hilly):
    rel = Release(hilly['folder'])
    values, upa = hilly['raster'], _upa(made_up)
    rows, cols = np.nonzero(values > 0)
    hill_r, hill_c = np.nonzero((values < 0) & (values != NODATA))
    sea_r, sea_c = np.nonzero(values == NODATA)
    pts = [(rows[k], cols[k]) for k in range(0, rows.size, max(1, rows.size // 12))] + \
          [(hill_r[k], hill_c[k]) for k in range(0, hill_r.size, max(1, hill_r.size // 5))] + [(sea_r[0], sea_c[0])]
    lon = [_lonlat(r, c)[0] for r, c in pts] + [0.0]
    lat = [_lonlat(r, c)[1] for r, c in pts] + [GT[3] + 1.0]                     # the last one north of the grid
    m = len(lon)
    st = {'station_id': np.arange(100, 100 + m)[::-1].copy(), 'lat': np.array(lat), 'lon': np.array(lon),
          'reported_area_km2': np.full(m, np.nan), 'merit_area_km2': np.full(m, np.nan)}
    out = stations.place_stations(rel, st, made_up['folder'] + '/upa.tif')
    assert hill_r.size and (out['pixel_class'] == 'hillslope_pixel').sum() >= 1
    v = np.array([values[r, c] for r, c in pts] + [0])
    want_class = np.where(v > 0, 'channel_pixel', np.where(v == NODATA, 'no_land', 'hillslope_pixel'))
    want_class[-1] = 'outside_grid'
    assert list(out['pixel_class']) == list(want_class)
    drains = np.where(v > 0, v, np.where((v < 0) & (v != NODATA), -v, 0))
    drains[-1] = 0
    assert (out['drains_to_pixel_seq_id'] == drains).all()
    assert (out['pixel_seq_id'] == np.where(v > 0, v, 0)).all()
    assert (out['basin_id'] == np.where(drains > 0, made_up['basin_of'][np.maximum(drains, 1) - 1], 0)).all()
    # numbering: by pixel, a channel station before a hillslope one on the same pixel, then station id
    k = np.flatnonzero(drains > 0)
    on = v[k] > 0
    order = k[np.lexsort((st['station_id'][k], ~on, drains[k]))]
    assert (out['station_seq_id'][order] == np.arange(1, k.size + 1)).all()
    assert (out['station_seq_id'][drains == 0] == 0).all()
    # next down and upstream count, by brute force on the ranges of the channel stations
    counts = made_up['upstream_counts']
    for i in order:
        p = drains[i]
        holders = [j for j in order if v[j] > 0 and out['station_seq_id'][j] < out['station_seq_id'][i]
                   and drains[j] <= p < drains[j] + counts[drains[j] - 1]]
        want = out['station_seq_id'][holders[-1]] if holders else out['station_seq_id'][i]
        assert out['next_down_station_seq_id'][i] == want
        if v[i] > 0:
            inside = [j for j in order if out['station_seq_id'][j] >= out['station_seq_id'][i]
                      and drains[j] < p + counts[p - 1]]
            assert out['upstream_station_count'][i] == len(inside)
        else:
            assert out['upstream_station_count'][i] == 1
    ch = v > 0
    assert (out['upstream_area_km2'][ch] == upa[tuple(np.array(pts)[ch[:-1]].T)].astype(np.float32)).all()
    assert np.isnan(out['upstream_area_km2'][~ch]).all()


def test_a_station_moves_to_the_neighbour_carrying_its_merit_area(made_up, hilly):
    rel, n = Release(hilly['folder']), made_up['n']
    values, upa = hilly['raster'], _upa(made_up)
    p = hilly['leaf'][0]                                                      # a hillslope pixel; q, the pixel below, is next to it
    q = n.down[p - 1]
    (r, c), (qr, qc) = (n.row[p - 1], n.col[p - 1]), (n.row[q - 1], n.col[q - 1])
    target = upa[qr, qc]
    lon, lat = _lonlat(r, c)
    qlon, qlat = _lonlat(qr, qc)
    st = {'station_id': np.array([1, 2, 3, 4]), 'lat': np.array([lat, lat, qlat, qlat]), 'lon': np.array([lon, lon, qlon, qlon]),
          'reported_area_km2': np.full(4, np.nan), 'merit_area_km2': np.array([target, np.nan, target, target * 3 + 10])}
    out = stations.place_stations(rel, st, made_up['folder'] + '/upa.tif')
    assert values[r, c] == -q
    assert out['moved_to_neighbouring_pixel'][0] == 1 and out['pixel_class'][0] == 'channel_pixel'
    assert upa[out['row'][0], out['col'][0]] == target                       # q, or a neighbour of the same area
    assert out['moved_to_neighbouring_pixel'][1] == 0 and out['pixel_class'][1] == 'hillslope_pixel'   # no MERIT area: stays
    assert out['moved_to_neighbouring_pixel'][2] == 0 and out['pixel_seq_id'][2] == q                 # on a pixel carrying it
    # no neighbour carries an area so far off: it stays on q
    assert out['moved_to_neighbouring_pixel'][3] == 0 and out['pixel_seq_id'][3] == q


def test_longitude_wraps_on_a_grid_around_the_earth(made_up):
    rel = Release(made_up['folder'])
    r, c = stations.pixel_under(rel, [GT[3] - 1.0] * 3, [-180.0, 180.0, 179.9])
    assert (r == 0).all() and list(c) == [0, 0, NCOL - 1]


# ------------------------------------------------------------------------------------------------ MERIT-Basins
def _line(n, pixels, down_pixel=None):
    """the vertices of a line through the centres of pixel ids, from its downstream end up (after down_pixel when given)"""
    ids = ([down_pixel] if down_pixel else []) + list(pixels)
    return [_lonlat(n.row[p - 1], n.col[p - 1]) for p in ids]


def _reaches(lines, down):
    x, y, feature = [], [], []
    for k, line in enumerate(lines):
        for a, b in line:
            x.append(a), y.append(b), feature.append(k)
    m = len(lines)
    return {'comid': np.arange(1, m + 1), 'next_down_comid': np.array(down), 'upstream_area_km2': np.full(m, np.nan),
            'x': np.array(x, np.float64), 'y': np.array(y, np.float64), 'feature': np.array(feature, np.int64)}


@pytest.mark.parametrize('T', [1.0, 10.0])
def test_merit_basins_classes(made_up, hilly, T):
    rel = Release(hilly['folder'], made_up['coarser_thresholds'])
    n, reach_of, basin_of = made_up['n'], made_up['reach_of'], made_up['basin_of']
    values = hilly['raster']
    lookup = merit_basins.ReleaseReaches(rel, T)
    r = made_up['reaches'][T]
    # a reach at T of two or more pixels with a reach downstream of it, away from the hillslope pixels
    k = next(k for k in range(r['reach_seq_id'].size) if r['channel_pixel_count'][k] >= 2 and
             r['next_down_reach_seq_id'][k] != k + 1 and r['basin_id'][k] != HILLY_BASIN)
    first, count = int(r['downstream_end_pixel_seq_id'][k]), int(r['channel_pixel_count'][k])
    stem = list(range(first, first + count))
    below = int(n.down[first - 1])                                             # the pixel it drains to
    hill = np.argwhere((values < 0) & (values != NODATA))[0]
    sea = np.argwhere(values == NODATA)[0]
    lines = [_line(n, stem, below),                                            # 0 same_reach
             _line(n, stem[:-1], below),                                       # 1 same_downstream_end: shorter
             _line(n, stem[1:], first),                                        # 2 on_reach
             [_lonlat(*hill)],                                                 # 3 hillslope_pixel (no reach downstream)
             [_lonlat(*sea)],                                                  # 4 no_land
             [(x + 0.3 * GT[1], y) for x, y in _line(n, stem, below)],         # 5 off the pixel centres: outside_grid
             _line(n, stem[:1])]                                               # 6 a reach downstream but one vertex: outside_grid
    down = [9, 9, 9, 0, 0, 9, 9]
    expected = ['same_reach', 'same_downstream_end', 'on_reach', 'hillslope_pixel', 'no_land', 'outside_grid', 'outside_grid']
    if T > 1.0:                                                                # a channel pixel below T
        low = next(p for p in range(1, n.grid_index.size + 1)
                   if n.area[p - 1] < T and n.down[p - 1] != 0 and basin_of[p - 1] != HILLY_BASIN)
        lines.append(_line(n, [low], int(n.down[low - 1])))
        down.append(9)
        expected.append('channel_below_threshold')
    out = merit_basins.place_reaches(rel, lookup, _reaches(lines, down), made_up['folder'] + '/upa.tif')
    assert list(out['match_class']) == expected
    assert out['in_sequence_pixel_count'][0] == count == out['merit_basins_pixel_count'][0]
    for j, p in ((0, first), (1, first), (2, first + 1)):
        assert out['pixel_seq_id'][j] == p == out['drains_to_pixel_seq_id'][j]
        assert out['basin_id'][j] == basin_of[p - 1] and out[f'reach_seq_id_{rel.base}'][j] == reach_of[p - 1]
        assert out['upstream_area_km2'][j] == np.float32(n.area[p - 1])
    assert out['drains_to_pixel_seq_id'][3] == -values[tuple(hill)] and out['pixel_seq_id'][3] == 0
    assert out['merit_basins_downstream_end_row'][5] == -1 and out['merit_basins_pixel_count'][5] == 0
    if T > 1.0:
        assert (out['reach_seq_id_10km2'][:3] == k + 1).all()


def test_merit_basins_basin_and_reach_of_every_channel_pixel(made_up):
    """a one-vertex reach on each pixel of the made-up release (all channel at its base threshold): the pixel table's basin and
    base reach, also where the basin table's order is not the pixels' (GLOBAL_IDS) and for SMALL, which it leaves out"""
    rel = Release(made_up['folder'], made_up['coarser_thresholds'])
    n, size = made_up['n'], made_up['n'].grid_index.size
    pixels = np.r_[np.arange(size, 0, -1), 1, size]                           # out of order, with repeats
    lines = [[_lonlat(n.row[p - 1], n.col[p - 1])] for p in pixels]
    for T in (1.0, 10.0):
        out = merit_basins.place_reaches(rel, merit_basins.ReleaseReaches(rel, T), _reaches(lines, [0] * len(lines)),
                                         made_up['folder'] + '/upa.tif')
        assert (out['pixel_seq_id'] == pixels).all()
        assert (out['basin_id'] == made_up['basin_of'][pixels - 1]).all()
        assert (out[f'reach_seq_id_{rel.base}'] == made_up['reach_of'][pixels - 1]).all()
    assert set(made_up['basin_of'][pixels - 1]) == {1, 2, 3, 4}


def test_a_vertex_at_plus_180_is_the_pixel_at_minus_180(made_up):
    rel = Release(made_up['folder'])
    lon, lat = _lonlat(5, 0)
    r, c, on = merit_basins.pixel_of_vertex(rel, np.array([lon, lon + 360.0, lon + 0.1]), np.array([lat] * 3))
    assert list(on) == [True, True, False] and list(c[:2]) == [0, 0] and list(r[:2]) == [5, 5]


def test_read_merit_basins_from_a_file(tmp_path):
    gpd = pytest.importorskip('geopandas')
    shapely = pytest.importorskip('shapely')
    frame = gpd.GeoDataFrame({'COMID': [11, 12], 'NextDownID': [0, 11], 'uparea': [5.5, None]},
                             geometry=[shapely.LineString([(0.5, 0.5), (1.5, 0.5)]),
                                       shapely.LineString([(1.5, 0.5), (2.5, 0.5), (3.5, 1.5)])], crs='EPSG:4326')
    path = tmp_path / 'riv.gpkg'
    frame.to_file(path, layer='riv_pfaf_9_test', driver='GPKG')
    got = merit_basins.read_merit_basins(str(path))
    assert list(got['comid']) == [11, 12] and list(got['next_down_comid']) == [0, 11]
    assert got['upstream_area_km2'][0] == 5.5 and np.isnan(got['upstream_area_km2'][1])
    assert list(got['feature']) == [0, 0, 1, 1, 1] and list(got['x']) == [0.5, 1.5, 1.5, 2.5, 3.5]
