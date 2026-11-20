"""The release reader on a made-up release, and the package against a release made by the C code, when one is given.

The made-up release is three random trees and a one-pixel basin under the base threshold on a 40 x 120 'globe', numbered by this
package and written in the layout of the global release (row groups of 7 rows, so that every read crosses row groups). As in the
release of FlowNest 10.58, basin_id is a global number that is not the order of the pixels (GLOBAL_IDS): the basin table holds the
basins of the base threshold, basin_id 1 .. basin_file_count by row, and the FlowDivide level-3 groups are in metadata.json only.
No number from it is a result.

The second part numbers every basin of a release made by the C code again and compares it with that release column by column:

    FLOWNEST_TEST_RELEASE=<v4 release> FLOWNEST_TEST_D8=<dir.tif> FLOWNEST_TEST_UPA=<upa.tif> pytest tests/test_release.py

Set FLOWNEST_TEST_COARSER_THRESHOLDS to the v4 step 9 folder for comparisons with its coarser tables and crosswalk.

It is skipped without the three variables. It holds every window of a basin in memory, so give it the synthetic release of the C
tests (1/12 degree), not the 90 m one.
"""
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pytest

pa = pytest.importorskip('pyarrow')
pq = pytest.importorskip('pyarrow.parquet')
rasterio = pytest.importorskip('rasterio')
from flownest import number_grid, Release, read_window         # noqa: E402
from flownest.release import label_of, REACH_COLUMNS, COARSER_REACH_COLUMNS, LOCATION_COLUMNS  # noqa: E402
from test_small_grid import random_tree                       # noqa: E402

NROW, NCOL, NODATA = 40, 120, -2147483648
GT = [-180.0, 3.0, 0.0, 60.0, 0.0, -3.0]
THRESHOLDS = [1.0, 10.0, 500.0]                               # in pixels of the trees; at 500 the smallest tree has no reach
TREES = ((2, 5, 30, 1), (5, 50, 20, 2), (10, 90, 25, 3))      # row, column, size, seed
SMALL = (35, 10)                 # a river mouth draining 0.5 pixel: a basin under the base threshold, its outlet one channel pixel
GLOBAL_IDS = (2, 3, 1, 4)        # basin_id of the basins in the order of their pixels (the trees by area, then SMALL): not that order
LEVEL3 = (11, 11, 12, 12)        # their FlowDivide level-3 group, each group a run of the pixel numbers
BASIN_FILE_COUNT = 3             # the basin table holds basin_id 1 .. 3; basin 4 (SMALL) is in the pixel and reach tables only
BASIN_COLUMNS = ('basin_id', 'flowdivide_level1_id', 'flowdivide_level2_id', 'flowdivide_level3_id', 'outlet_pixel_seq_id',
                 'channel_pixel_count', 'first_reach_seq_id_1km2', 'reach_count_1km2', 'upstream_area_km2', 'is_endorheic',
                 'outlet_lon', 'outlet_lat', 'row_min', 'row_max', 'col_min', 'col_max')
TOPOLOGY = ['reach_seq_id', 'downstream_end_pixel_seq_id', 'channel_pixel_count', 'upstream_channel_pixel_count',
            'next_down_reach_seq_id', 'upstream_reach_count', 'path_seq_id', 'path_level']
WINDOWS = [p + s for p in ('unit_catchment_', 'upstream_catchment_') for s in ('row_min', 'row_max', 'col_min', 'col_max')]


def _tif(path, a, nodata):
    with rasterio.open(path, 'w', driver='GTiff', height=NROW, width=NCOL, count=1, dtype=a.dtype, nodata=nodata, crs='EPSG:4326',
                       transform=rasterio.Affine(GT[1], GT[2], GT[0], GT[4], GT[5], GT[3]),
                       tiled=True, blockxsize=16, blockysize=16) as f:    # tiles that do not fit the grid, as in the release
        f.write(a, 1)


def break_thresholds(next_down, upa):
    """Brute force: the upstream area at a basin outlet, else the smaller of the reach's own and the largest other donor's into the
    reach below. A basin outlet points to itself."""
    out = np.empty(next_down.size, np.float32)
    ids = np.arange(1, next_down.size + 1)
    for i, d in enumerate(next_down):
        if d == i + 1:
            out[i] = upa[i]
            continue
        others = np.flatnonzero((next_down == d) & (next_down != ids))
        others = others[others != i]
        assert others.size, 'a reach below with one donor'
        out[i] = min(upa[i], upa[others].max())
    return out


def _reference_lengths(n, reaches):
    """Independent scalar pixel-step sum on the release earth model, including the step into the downstream reach."""
    import math
    lengths = []
    for first, count in zip(reaches['downstream_end_pixel_seq_id'], reaches['channel_pixel_count']):
        total = 0.0
        for pixel in range(int(first), int(first + count)):
            below = n.down[pixel - 1]
            if below == 0:
                continue
            row, col = n.row[pixel - 1], n.col[pixel - 1]
            br, bc = n.row[below - 1], n.col[below - 1]
            lat1, lat2 = [math.radians(GT[3] + (r + .5) * GT[5]) for r in (row, br)]
            dlat, dlon = lat2 - lat1, math.radians(GT[1]) * (col != bc)
            mid, radius = (lat1 + lat2) / 2, 6371007.181
            h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            distance = radius * 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))
            e2 = 0.0066943799901413165
            w = math.sqrt(1 - e2 * math.sin(mid) ** 2)
            meridian, prime = 6378137 * (1 - e2) / w ** 3, 6378137 / w
            total += distance * math.hypot(prime * math.cos(mid) * dlon, meridian * dlat) / math.hypot(
                radius * math.cos(mid) * dlon, radius * dlat)
        lengths.append(float(f'{total:.3f}') / 1000)
    return np.array(lengths)


@pytest.fixture(scope='module')
def made_up(tmp_path_factory):
    from flownest.release import REACH_COLUMNS, LOCATION_COLUMNS, COARSER_REACH_COLUMNS
    folder = tmp_path_factory.mktemp('release')
    coarse_folder = tmp_path_factory.mktemp('coarser_thresholds')
    d8 = np.full((NROW, NCOL), 247, np.uint8)
    upa = np.full((NROW, NCOL), -9999, np.float32)
    for row, col, size, seed in TREES:
        tree, area = random_tree(size, seed)
        d8[row:row + size, col:col + size], upa[row:row + size, col:col + size] = tree, area
    d8[SMALL], upa[SMALL] = 0, 0.5
    trees = sorted(((row + size - 1, col + size - 1) for row, col, size, _ in TREES), key=lambda rc: -upa[rc])   # their outlets
    n = number_grid(d8, upa.astype(np.float64), threshold=THRESHOLDS[0], outlets=trees + [SMALL], nodata=247,
                    outlets_are_channel_pixels=True)
    ids = np.arange(1, n.grid_index.size + 1)
    upstream_counts = n.upstream_channel_pixel_count
    outlets = np.flatnonzero(n.down == 0) + 1
    assert outlets.size == len(GLOBAL_IDS) and outlets[-1] == ids[-1]
    global_ids, level3 = np.array(GLOBAL_IDS), np.array(LEVEL3)
    basin_of = global_ids[np.searchsorted(outlets, ids, side='right') - 1]
    files, coarse_files, reaches, locations = {}, {}, {}, {}

    def write(name, cols, coarse=False):
        target, manifest = (coarse_folder, coarse_files) if coarse else (folder, files)
        manifest[f'flownest_merit90m_{name}'] = {}
        table = pa.table(cols)
        if name.endswith('.geo.parquet'):
            geometry_type = 'MultiPolygon' if name.startswith('unit_catchment_') else 'LineString'
            geo = dict(version='1.1.0', primary_column='geometry', columns=dict(geometry=dict(encoding='WKB',
                       geometry_types=[geometry_type], covering=dict(bbox={key: ['bbox', key] for key in ('xmin', 'ymin', 'xmax', 'ymax')}))))
            table = table.replace_schema_metadata({b'geo': json.dumps(geo).encode()})
        pq.write_table(table, target / f'flownest_merit90m_{name}', row_group_size=7)

    def vector(name, r, coarse=False):
        size = r['reach_seq_id'].size
        bbox_type = pa.struct([(key, pa.float64()) for key in ('xmin', 'ymin', 'xmax', 'ymax')])
        write(name, dict(reach_seq_id=r['reach_seq_id'], geometry=pa.array([None] * size, type=pa.binary()),
                         bbox=pa.array([None] * size, type=bbox_type)), coarse)

    values = np.abs(n.raster).ravel()
    land = values != 0
    for threshold in THRESHOLDS:
        r = n.reaches(threshold)
        r['basin_id'] = basin_of[r['downstream_end_pixel_seq_id'] - 1]
        r['upstream_area_km2'] = r['upstream_area_km2'].astype(np.float32)
        q = _first_pixels_at(n, threshold)[values[land] - 1]
        r['unit_catchment_area_km2'] = np.bincount(np.searchsorted(r['downstream_end_pixel_seq_id'], q[q > 0], side='right') - 1,
                                                  minlength=r['reach_seq_id'].size).astype(np.float64)
        r['length_km'] = _reference_lengths(n, r)
        r['distance_to_basin_outlet_km'] = np.zeros(r['reach_seq_id'].size)
        for k, below in enumerate(r['next_down_reach_seq_id']):
            if below != k + 1:
                r['distance_to_basin_outlet_km'][k] = r['distance_to_basin_outlet_km'][below - 1] + r['length_km'][below - 1]
        location = dict(reach_seq_id=r['reach_seq_id'], downstream_end_lon=GT[0] + (n.col[r['downstream_end_pixel_seq_id'] - 1] + .5) * GT[1],
                        downstream_end_lat=GT[3] + (n.row[r['downstream_end_pixel_seq_id'] - 1] + .5) * GT[5])
        unit = n.unit_catchments(threshold)
        for name in WINDOWS:
            location[name] = []
        for k, first in enumerate(r['downstream_end_pixel_seq_id']):
            own = ((values >= unit['first_1'][k]) & (values < unit['first_1'][k] + unit['count_1'][k])) | \
                  ((values >= unit['first_2'][k]) & (values < unit['first_2'][k] + unit['count_2'][k]))
            upstream = (values >= first) & (values < first + r['upstream_channel_pixel_count'][k])
            for prefix, mask in (('unit_catchment_', own), ('upstream_catchment_', upstream)):
                row, col = np.flatnonzero(mask) // NCOL, np.flatnonzero(mask) % NCOL
                for suffix, value in zip(('row_min', 'row_max', 'col_min', 'col_max'), (row.min(), row.max() + 1, col.min(), col.max() + 1)):
                    location[prefix + suffix].append(value)
        locations[threshold] = {key: np.asarray(location[key]) for key in LOCATION_COLUMNS}
        reaches[threshold] = r
    base = reaches[THRESHOLDS[0]]
    reach_of = np.searchsorted(base['downstream_end_pixel_seq_id'], ids, side='right')
    base['break_threshold_km2'] = break_thresholds(base['next_down_reach_seq_id'], base['upstream_area_km2'])
    crosswalk = {'reach_seq_id_1km2': base['reach_seq_id']}
    for threshold, r in reaches.items():
        label = label_of(threshold)
        coarse = threshold != THRESHOLDS[0]
        if coarse:
            first = reach_of[r['downstream_end_pixel_seq_id'] - 1]
            r['downstream_end_reach_seq_id_1km2'] = first
            r['stem_reach_count_1km2'] = reach_of[r['downstream_end_pixel_seq_id'] + r['channel_pixel_count'] - 2] - first + 1
            q = _first_pixels_at(n, threshold)[base['downstream_end_pixel_seq_id'] - 1]
            crosswalk[f'reach_seq_id_{label}'] = np.where(q > 0, np.searchsorted(r['downstream_end_pixel_seq_id'], q, side='right'), 0)
        columns = COARSER_REACH_COLUMNS if coarse else REACH_COLUMNS
        write(f'reach_{label}.parquet', {key: r[key] for key in columns}, coarse)
        write(f'reach_location_{label}.parquet', locations[threshold], coarse)
        vector(f'reach_line_{label}.geo.parquet', r, coarse)
    vector('unit_catchment_1km2.geo.parquet', base)
    write('crosswalk_reach_1km2_to_coarser.parquet', crosswalk, True)
    write('pixel_1km2.parquet', dict(pixel_seq_id=ids, row=n.row, col=n.col, reach_seq_id=reach_of,
                                    path_seq_id=base['path_seq_id'][reach_of - 1], basin_id=basin_of,
                                    upstream_area_km2=n.area.astype(np.float32), upstream_channel_pixel_count=upstream_counts))
    boxes = {key: [] for key in ('row_min', 'row_max', 'col_min', 'col_max')}
    for first in outlets:
        pixels = np.flatnonzero((values >= first) & (values < first + upstream_counts[first - 1]))
        row, col = pixels // NCOL, pixels % NCOL
        for key, value in zip(boxes, (row.min(), row.max() + 1, col.min(), col.max() + 1)):
            boxes[key].append(value)
    boxes = {key: np.array(value) for key, value in boxes.items()}
    k = np.argsort(global_ids)[:BASIN_FILE_COUNT]             # the basin table: by basin_id, the basins of the base threshold
    o = outlets[k]
    basin = dict(basin_id=global_ids[k], flowdivide_level1_id=level3[k] // 10, flowdivide_level2_id=level3[k],
                 flowdivide_level3_id=level3[k], outlet_pixel_seq_id=o, channel_pixel_count=upstream_counts[o - 1],
                 first_reach_seq_id_1km2=reach_of[o - 1], reach_count_1km2=np.array([np.sum(base['basin_id'] == g) for g in global_ids[k]]),
                 upstream_area_km2=n.area[o - 1].astype(np.float32), is_endorheic=np.zeros(k.size, np.int8),
                 outlet_lon=GT[0] + (n.col[o - 1] + .5) * GT[1], outlet_lat=GT[3] + (n.row[o - 1] + .5) * GT[5],
                 **{key: value[k] for key, value in boxes.items()})
    assert tuple(basin) == BASIN_COLUMNS
    write('basin.parquet', basin)
    groups = []                                               # in metadata.json, keyed as the C code writes them
    for group in (11, 12):
        members = np.flatnonzero(level3 == group)             # a run of basins in the order of the pixels, SMALL too
        selected_reaches = np.flatnonzero(np.isin(base['basin_id'], global_ids[members]))
        paths = base['path_seq_id'][selected_reaches]
        first_pixel, count = outlets[members[0]], upstream_counts[outlets[members] - 1].sum()
        groups.append({key: int(value) for key, value in dict(
            flowdivide_level3_id=group, flowdivide_level1_id=group // 10, flowdivide_level2_id=group, basin_count=members.size,
            first_pixel_seq_id=first_pixel, channel_pixel_count=count, first_reach_seq_id_1km2=selected_reaches[0] + 1,
            reach_count_1km2=selected_reaches.size, first_path_seq_id_1km2=paths[0], path_count_1km2=np.unique(paths).size,
            row_min=boxes['row_min'][members].min(), row_max=boxes['row_max'][members].max(),
            col_min=boxes['col_min'][members].min(), col_max=boxes['col_max'][members].max(),
            unit_catchment_pixel_count=np.sum((values >= first_pixel) & (values < first_pixel + count)),
            polygon_part_count=selected_reaches.size).items()})
    raster = n.raster.copy()
    raster[d8 == 247] = NODATA
    reach_raster = np.where(n.raster > 0, 1, -1) * np.where(n.raster != 0, reach_of[np.maximum(np.abs(n.raster), 1) - 1], 0)
    reach_raster[d8 == 247] = NODATA
    for name, array in (('pixel_seq_id_1km2.tif', raster), ('reach_seq_id_1km2.tif', reach_raster)):
        _tif(folder / ('flownest_merit90m_' + name), array, NODATA)
        files['flownest_merit90m_' + name] = {}
    _tif(folder / 'dir.tif', d8, 247)
    _tif(folder / 'upa.tif', upa, -9999)
    metadata = dict(schema_version='4', numbering_version='0.2', grid=dict(geotransform=GT, nrow=NROW, ncol=NCOL, nodata=NODATA),
                    thresholds_km2=THRESHOLDS[:1], files=files, numbering_sha256=n.fingerprint(), basin_file_count=BASIN_FILE_COUNT,
                    basin_id=dict(basin_file_range=[1, BASIN_FILE_COUNT], numbered_range=[1, len(GLOBAL_IDS)]),
                    flowdivide_level3_groups=groups, earth_model='WGS84 ellipsoid (a=6378137 m, 1/f=298.257223563)')
    (folder / 'metadata.json').write_text(json.dumps(metadata))
    # as flownest.9.convert writes it: the coarser thresholds, and the numbering at its top and under "release"
    (coarse_folder / 'metadata.json').write_text(json.dumps(dict(schema_version='4', numbering_version='0.2', files=coarse_files,
                                                               thresholds_km2=THRESHOLDS[1:], numbering_sha256=n.fingerprint(),
                                                               release=dict(numbering_sha256=n.fingerprint()))))
    return dict(folder=str(folder), coarser_thresholds=str(coarse_folder), n=n, raster=raster, upstream_counts=upstream_counts,
                outlets=outlets, basin_of=basin_of, reaches=reaches, reach_of=reach_of, locations=locations,
                global_ids=global_ids, level3=level3)


def test_rows_cross_row_groups(made_up):
    rel = Release(made_up['folder'], made_up['coarser_thresholds'])
    N = made_up['n'].grid_index.size
    name = f'pixel_{rel.base}.parquet'
    for a, b in ((1, 1), (1, 7), (7, 8), (6, 15), (14, 14), (15, 40), (N - 3, N), (1, N)):
        t = rel.rows(name, a, b - a + 1)
        assert (t['pixel_seq_id'] == np.arange(a, b + 1)).all()
        assert (t['reach_seq_id'] == made_up['reach_of'][a - 1:b]).all()
    t = rel.pixels(9, 12, ['basin_id'])
    assert list(t) == ['basin_id'] and (t['basin_id'] == made_up['basin_of'][8:20]).all()
    for a, b in ((0, 3), (5, -1), (N, 2)):
        with pytest.raises(IndexError):
            rel.rows(name, a, b)
    with pytest.raises(KeyError):
        rel.file('reach_3km2.parquet')


@pytest.mark.parametrize('schema', [None, '3', 4])
def test_pre_v4_metadata_is_refused_before_reading_any_data(tmp_path, schema):
    metadata = {} if schema is None else {'schema_version': schema}
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=r'pre-v4 release.*reads only'):
        Release(tmp_path)


def test_crosswalk_package_is_refused_as_another_package_type(tmp_path):
    from flownest import __version__
    (tmp_path / 'metadata.json').write_text(json.dumps({'schema_version': '5'}))
    with pytest.raises(ValueError, match='unsupported package type') as refused:
        Release(tmp_path)
    assert f'flownest {__version__}' in str(refused.value) and 'pre-v4' not in str(refused.value)


def test_v4_layout_and_half_open_reads(made_up):
    rel = Release(made_up['folder'], made_up['coarser_thresholds'])
    assert rel.meta['schema_version'] == '4' and len(rel.meta['flowdivide_level3_groups']) == 2
    basins = rel.table('basin.parquet')
    assert tuple(basins) == BASIN_COLUMNS and list(basins['basin_id']) == [1, 2, 3]       # basin b on row b - 1
    assert rel.basin_file_count == BASIN_FILE_COUNT and rel.group(12)['basin_count'] == 2  # group 12 holds SMALL too
    with pytest.raises(KeyError):
        rel.group(99)
    assert list(rel.reaches(1, 1)) == list(REACH_COLUMNS)
    assert list(rel.rows('reach_location_1km2.parquet', 1, 1)) == list(LOCATION_COLUMNS)
    assert list(rel.reaches(1, 1, 10)) == list(COARSER_REACH_COLUMNS)
    assert list(rel.table('reach_line_1km2.geo.parquet')) == ['reach_seq_id', 'geometry', 'bbox']
    assert list(rel.table('unit_catchment_1km2.geo.parquet')) == ['reach_seq_id', 'geometry', 'bbox']
    size = made_up['n'].grid_index.size
    for first in (1, size + 1):
        assert all(value.size == 0 for value in rel.pixels(first, 0).values())
    with pytest.raises(TypeError):
        rel.pixels(1, 1.5)
    assert label_of(11.2) == '11p2km2'
    assert label_of(100) == '100km2'
    basin = rel.basin(1)
    assert np.array_equal(rel.window_of(basin, prefix=''), rel.window(basin['row_min'], basin['row_max'], basin['col_min'], basin['col_max']))
    location = {key: value[0] for key, value in rel.rows('reach_location_1km2.parquet', 1, 1).items()}
    assert rel.window_of(location).shape == (location['upstream_catchment_row_max'] - location['upstream_catchment_row_min'],
                                           location['upstream_catchment_col_max'] - location['upstream_catchment_col_min'])


@pytest.mark.parametrize('threshold', [10., 11.2, 500., 2000.])
def test_threshold_ids_and_empty_selections(made_up, threshold):
    rel, n = Release(made_up['folder']), made_up['n']
    expected = n.reaches(threshold)
    in_table = [k for k, g in enumerate(GLOBAL_IDS) if g <= BASIN_FILE_COUNT]     # in the order of the pixels, not of basin_id
    results = [rel.reaches_at(threshold, basin_id=GLOBAL_IDS[k]) for k in in_table]
    for key in TOPOLOGY:
        assert np.array_equal(np.concatenate([r[key] for r in results]), expected[key]), key
    for group in (11, 12):
        by_group = rel.reaches_at(threshold, flowdivide_level3_id=group)
        members = [i for i, k in enumerate(in_table) if LEVEL3[k] == group]
        for key in COARSER_REACH_COLUMNS:
            np.testing.assert_array_equal(by_group[key], np.concatenate([results[i][key] for i in members]))
    with pytest.raises(KeyError):                  # basin 4 is under the base threshold: not in the basin table
        rel.reaches_at(threshold, basin_id=4)
    with pytest.raises(ValueError):
        rel.reaches_at(threshold)
    with pytest.raises(KeyError):
        rel.reaches_at(threshold, flowdivide_level3_id=99)


def test_step9_folder_matches_derived_columns_and_crosswalk(made_up):
    stored = Release(made_up['folder'], made_up['coarser_thresholds'])
    derived = Release(made_up['folder'])
    for threshold in THRESHOLDS[1:]:
        label = label_of(threshold)
        crosswalk = stored.table('crosswalk_reach_1km2_to_coarser.parquet')
        for basin_id in range(1, BASIN_FILE_COUNT + 1):
            want = stored.reaches_at(threshold, basin_id=basin_id)
            got = derived.reaches_at(threshold, basin_id=basin_id)
            for key in COARSER_REACH_COLUMNS:
                np.testing.assert_allclose(got[key], want[key], rtol=0, atol=1e-9, err_msg=key)
            first, target = derived.reach_at_threshold(threshold, basin_id=basin_id)
            assert np.array_equal(target, crosswalk[f'reach_seq_id_{label}'][first - 1:first - 1 + target.size])


def test_reach_windows_reads_the_two_half_open_ranges(made_up):
    rel = Release(made_up['folder'])
    for basin_id in range(1, BASIN_FILE_COUNT + 1):
        n, window = rel.number_basin(basin_id, os.path.join(made_up['folder'], 'dir.tif'), os.path.join(made_up['folder'], 'upa.tif'))
        for threshold in THRESHOLDS:
            selection = rel.reach_range(basin_id, threshold)
            if selection is None:
                continue
            got = rel.reach_windows(n, window, threshold, selection[0])
            expected = made_up['locations'][threshold]
            for key in WINDOWS:
                assert np.array_equal(got[key], expected[key][selection[0] - 1:selection[0] - 1 + selection[1]]), key


def test_empty_v4_tables_support_empty_reads(tmp_path):
    table = pa.table({'reach_seq_id': pa.array([], type=pa.int64()), 'basin_id': pa.array([], type=pa.int64())})
    name = 'flownest_merit90m_reach_1km2.parquet'
    pq.write_table(table, tmp_path / name)
    (tmp_path / 'metadata.json').write_text(json.dumps(dict(schema_version='4', grid=dict(geotransform=GT, nrow=NROW,
        ncol=NCOL, nodata=NODATA), thresholds_km2=[1], files=[name], basin_file_count=0, basin_id=dict(basin_file_range=[1, 0]),
        flowdivide_level3_groups=[])))
    rel = Release(tmp_path)
    assert all(value.size == 0 for value in rel.rows('reach_1km2.parquet', 1, 0).values())
    assert all(value.size == 0 for value in rel.rows_at('reach_1km2.parquet', []).values())
    with pytest.raises(IndexError):
        rel.rows('reach_1km2.parquet', 1, 1)


def test_v4_release_numbered_before_10_58_is_refused(made_up, tmp_path):
    stored = json.loads(open(os.path.join(made_up['folder'], 'metadata.json')).read())
    for drop in ('basin_file_count', 'basin_id'):
        (tmp_path / 'metadata.json').write_text(json.dumps({key: value for key, value in stored.items() if key != drop}))
        with pytest.raises(ValueError, match=r'before FlowNest 10\.58.*does not read it'):
            Release(tmp_path)


def test_coarser_folder_rejects_different_numbering(made_up, tmp_path):
    stored = json.loads(open(os.path.join(made_up['coarser_thresholds'], 'metadata.json')).read())
    for key in ('numbering_sha256', 'release.numbering_sha256'):
        metadata = dict(stored, release=dict(stored['release']))
        if key == 'numbering_sha256':
            metadata['numbering_sha256'] = 'different'
        else:
            metadata['release']['numbering_sha256'] = 'different'
        (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match=f'{key} different is not the numbering_sha256'):
            Release(made_up['folder'], tmp_path)


def test_scattered_rows_and_values(made_up):
    rel = Release(made_up['folder'], made_up['coarser_thresholds'])
    N = made_up['n'].grid_index.size
    rng = np.random.default_rng(3)
    ids = rng.integers(1, N + 1, 60)                          # unsorted, with repeats, over many row groups
    t = rel.rows_at(f'pixel_{rel.base}.parquet', ids, ['reach_seq_id', 'basin_id'])
    assert list(t) == ['reach_seq_id', 'basin_id']
    assert (t['reach_seq_id'] == made_up['reach_of'][ids - 1]).all() and (t['basin_id'] == made_up['basin_of'][ids - 1]).all()
    t = rel.rows_at(f'pixel_{rel.base}.parquet', ids[:0])
    assert all(x.size == 0 for x in t.values()) and 'row' in t
    with pytest.raises(IndexError):
        rel.rows_at(f'pixel_{rel.base}.parquet', [1, N + 1])
    full = made_up['raster']
    rows, cols = rng.integers(0, NROW, 500), rng.integers(0, NCOL, 500)
    assert (rel.values_at(rows, cols) == full[rows, cols]).all()
    assert (rel.values_at(rows, cols - NCOL) == full[rows, cols]).all()     # columns west of 0 wrap
    assert rel.values_at([], []).size == 0
    with pytest.raises(ValueError):
        rel.values_at([NROW], [0])


def test_basins_and_their_reach_ranges(made_up):
    rel = Release(made_up['folder'], made_up['coarser_thresholds'])
    seen_none = False
    for o, g in zip(made_up['outlets'], made_up['global_ids']):
        if g > BASIN_FILE_COUNT:                              # SMALL: in the pixel and reach tables only
            with pytest.raises(KeyError):
                rel.basin(g)
            continue
        b = rel.basin(g)
        assert b['basin_id'] == g and b['outlet_pixel_seq_id'] == o and b['channel_pixel_count'] == made_up['upstream_counts'][o - 1]
        for T in THRESHOLDS:
            r = made_up['reaches'][T]
            mine = r['reach_seq_id'][r['basin_id'] == g]
            got = rel.reach_range(g, T)
            if mine.size:
                assert got == (mine[0], mine.size)
            else:
                assert got is None
                seen_none = True
    assert seen_none                                          # the made-up release has a basin under its largest threshold
    for g in (0, BASIN_FILE_COUNT + 2):
        with pytest.raises(KeyError):
            rel.basin(g)


def test_upstream_counts_and_basin_of(made_up):
    rel = Release(made_up['folder'])
    size, counts = made_up['n'].grid_index.size, made_up['upstream_counts']
    assert (rel.pixels(1, size, ['upstream_channel_pixel_count'])['upstream_channel_pixel_count'] == counts).all()
    for first, stop in ((5, 10), (30, 32), (size - 10, size + 1)):
        assert (rel.pixels(first, stop - first)['upstream_channel_pixel_count'] == counts[first - 1:stop - 1]).all()
    for pixel in (1, 2, 50, size):
        assert rel.upstream(pixel) == range(pixel, pixel + counts[pixel - 1])
    assert all(rel.basin_of(p) == made_up['basin_of'][p - 1] for p in range(1, size + 1, 13))
    assert rel.basin_of(made_up['outlets'][-1]) == 4                  # SMALL, the last pixel, from the pixel table
    with pytest.raises(KeyError):
        rel.basin_of(size + 1)


def test_windows_across_180_degrees(made_up):
    rel = Release(made_up['folder'], made_up['coarser_thresholds'])
    full = made_up['raster']
    assert (rel.window(3, 20, 40, 75) == full[3:20, 40:75]).all()
    assert (rel.window(3, 20, -7, 9) == np.hstack([full[3:20, -7:], full[3:20, :9]])).all()
    assert (rel.window(0, NROW, -1, NCOL - 1) == np.roll(full, 1, axis=1)).all()
    assert (rel.grid_columns(-3, 2) == [NCOL - 3, NCOL - 2, NCOL - 1, 0, 1]).all()
    path = rel.file('pixel_seq_id_1km2.tif')
    for w in ((0, 5, -NCOL, 3), (0, 5, -NCOL - 1, -1), (0, 5, -5, NCOL - 4), (5, 5, 0, 3), (0, NROW + 1, 0, 3)):
        with pytest.raises(ValueError):
            read_window(path, *w)
    # a window wholly west of column 0, as a unit window cut from a window across 180 degrees may be
    assert (read_window(path, 0, 5, -5, 0) == full[0:5, -5:]).all()
    assert (read_window(path, 2, 6, -NCOL, -NCOL + 4) == full[2:6, 0:4]).all()
    with pytest.raises(ValueError):
        read_window(path, 0, 5, 0, 5, shape=(NROW, NCOL + 1))
    with pytest.raises(ValueError):
        read_window(path, 0, 5, 0, 5, geotransform=[-179.0] + GT[1:])
    # the west edge 0.0004 pixel east and the east edge 0.0004 pixel west: every corner is within the thousandth of a pixel
    assert read_window(path, 0, 5, 0, 5, geotransform=[GT[0] + 4e-4 * GT[1], GT[1] * (1 - 8e-4 / NCOL)] + GT[2:]).shape == (5, 5)
    with pytest.raises(ValueError):                # very little per pixel, but 0.002 pixel at the east edge
        read_window(path, 0, 5, 0, 5, geotransform=[GT[0], GT[1] * (1 + 2e-3 / NCOL)] + GT[2:])
    assert rel.pixel_of_lonlat(-180 + 3 * 17 + 1.5, 60 - 3 * 4 - 1.5) == (4, 17)
    assert rel.pixel_of_lonlat(181.5, 58.5) == (0, 0)


def test_number_basin_gives_the_release_numbers(made_up):
    rel = Release(made_up['folder'], made_up['coarser_thresholds'])
    d8_tif, upa_tif = os.path.join(made_up['folder'], 'dir.tif'), os.path.join(made_up['folder'], 'upa.tif')
    for o, g in zip(made_up['outlets'], made_up['global_ids']):
        if g > BASIN_FILE_COUNT:
            continue
        n, w = rel.number_basin(g, d8_tif, upa_tif)
        v = rel.window(*w).astype(np.int64)
        mine = (v != NODATA) & (np.abs(v) >= o) & (np.abs(v) < o + made_up['upstream_counts'][o - 1])
        assert ((n.raster != 0) == mine).all() and (n.raster[mine] == v[mine]).all()
        for T in THRESHOLDS:
            rr = rel.reach_range(g, T)
            if rr is None:
                assert n.reaches(T)['reach_seq_id'].size == 0
                continue
            want = rel.reaches(*rr, T)
            got = n.reaches(T, first_reach=rr[0], first_path=int(want['path_seq_id'][0]))
            for c in TOPOLOGY:
                assert (got[c] == want[c]).all(), (g, T, c)


def test_pixel_of_lonlat_arrays_keep_the_scalar_edge_rule(made_up):
    rel = Release(made_up['folder'], made_up['coarser_thresholds'])
    lon = np.array([[-180., -177., 180., 181.5], [-180.00001, -177.00001, 540., -540.]])
    lat = np.array([[60.], [57.]])
    rows, cols = rel.pixel_of_lonlat(lon, lat)
    xx, yy = np.broadcast_arrays(lon, lat)
    assert rows.shape == cols.shape == lon.shape
    for i in np.ndindex(lon.shape):
        scalar = rel.pixel_of_lonlat(float(xx[i]), float(yy[i]))
        assert isinstance(scalar[0], int) and isinstance(scalar[1], int)
        assert (rows[i], cols[i]) == scalar
    assert rel.pixel_of_lonlat([], [])[0].size == 0
    for latitudes in ([-60.], [60., 60.0001]):
        with pytest.raises(ValueError, match='outside the grid'):
            rel.pixel_of_lonlat(0., latitudes)


def _first_pixels_at(n, threshold):
    """For every channel pixel (local id - 1), the local id of the first pixel at or below it on the network at the threshold, 0
    when its basin outlet is not on it: brute force down the pixels. At the base threshold every channel pixel is on the network,
    the outlet of a basin under it (SMALL) too."""
    on = (n.area >= threshold) | (threshold == THRESHOLDS[0])
    q = np.arange(1, n.down.size + 1)
    while True:
        move = ~on[q - 1] & (n.down[q - 1] > 0)
        if not move.any():
            return np.where(on[q - 1], q, 0)
        q = np.where(move, n.down[q - 1], q)


@pytest.mark.parametrize('T', THRESHOLDS)
def test_reaches_at_any_threshold_from_the_base_tables(made_up, T):
    """reaches_at(T) from the base tables alone gives the sequentially numbered reaches cut at T."""
    rel, n, reach_of = Release(made_up['folder'], made_up['coarser_thresholds']), made_up['n'], made_up['reach_of']
    want = n.reaches(T)
    first = np.r_[0, want['downstream_end_pixel_seq_id']]                       # coarse reach k + 1 starts at pixel first[k + 1]
    name = reach_of[want['downstream_end_pixel_seq_id'] - 1]                     # its downstream-end base reach
    # unit catchments at T in pixels, by brute force: every land pixel to the first pixel on the network below it
    v = np.abs(made_up['raster'].ravel())
    v = v[(v != 0) & (made_up['raster'].ravel() != NODATA)]
    q = _first_pixels_at(n, T)[v - 1]
    area = np.bincount(np.searchsorted(want['downstream_end_pixel_seq_id'], q[q > 0], side='right') - 1, minlength=name.size)
    from flownest.release import COARSER_REACH_COLUMNS
    got = {key: [] for key in COARSER_REACH_COLUMNS}
    # Derive from base tables alone, even though expected step 9 tables also exist in the fixture. By group, so SMALL, which has
    # no row in the basin table, is read too.
    rel = Release(made_up['folder'])
    for g in (11, 12):
        group = rel.reaches_at(T, flowdivide_level3_id=g)
        assert list(group) == list(COARSER_REACH_COLUMNS)
        for key in got:
            got[key].append(group[key])
    got = {key: np.concatenate(value) for key, value in got.items()}
    for g in (g for g in GLOBAL_IDS if g <= BASIN_FILE_COUNT):
        one, mine = rel.reaches_at(T, basin_id=g), got['basin_id'] == g
        for key in COARSER_REACH_COLUMNS:
            np.testing.assert_array_equal(one[key], got[key][mine], err_msg=str((g, key)))
    assert (got['downstream_end_reach_seq_id_1km2'] == name).all()
    end = want['downstream_end_pixel_seq_id'] + want['channel_pixel_count'] - 1
    assert (got['stem_reach_count_1km2'] == reach_of[end - 1] - name + 1).all()
    nd = want['next_down_reach_seq_id']
    assert (got['next_down_reach_seq_id'] == nd).all()
    for key in TOPOLOGY:
        assert (got[key] == want[key]).all(), key
    donors = want['next_down_reach_seq_id'][want['next_down_reach_seq_id'] != want['reach_seq_id']]
    assert ((got['upstream_reach_count'] == 1) == ~np.isin(want['reach_seq_id'], donors)).all()
    assert (got['basin_id'] == made_up['basin_of'][want['downstream_end_pixel_seq_id'] - 1]).all()
    assert (got['upstream_area_km2'] == want['upstream_area_km2'].astype(np.float32)).all()
    assert (got['unit_catchment_area_km2'] == area).all()
    for key in ('length_km', 'distance_to_basin_outlet_km'):
        np.testing.assert_allclose(got[key], made_up['reaches'][T][key], rtol=0, atol=1e-9)
    base = made_up['reaches'][THRESHOLDS[0]]
    covered = 0
    for g in (11, 12):                                        # the two groups: every base reach, once
        first, crosswalk = rel.reach_at_threshold(T, flowdivide_level3_id=g)
        assert first == covered + 1
        covered += crosswalk.size
        downstream_end = base['downstream_end_pixel_seq_id'][first - 1:first - 1 + crosswalk.size]
        pixel = _first_pixels_at(n, T)[downstream_end - 1]
        assert (crosswalk == np.where(pixel > 0, np.searchsorted(want['downstream_end_pixel_seq_id'], pixel, side='right'), 0)).all()
    assert covered == base['reach_seq_id'].size



@pytest.mark.parametrize('T', THRESHOLDS)
def test_window_at_any_threshold(made_up, T):
    """window_at(T): +o on the channel at T, -o on the pixels draining first to it, 0 where the basin has no network at T."""
    rel, n, reach_of = Release(made_up['folder'], made_up['coarser_thresholds']), made_up['n'], made_up['reach_of']
    want = n.reaches(T)
    name = want['reach_seq_id']
    raster = made_up['raster']
    sea, v = raster == NODATA, np.abs(raster)
    q = np.zeros(raster.shape, np.int64)
    land = ~sea & (v != 0)
    q[land] = _first_pixels_at(n, T)[v[land] - 1]
    o = np.zeros(raster.shape, np.int64)
    o[q > 0] = name[np.searchsorted(want['downstream_end_pixel_seq_id'], q[q > 0], side='right') - 1]
    sign = np.where((raster > 0) & (q == raster), 1, -1)
    expect = np.where(sea, NODATA, sign * o)
    for w in ((0, NROW, 0, NCOL), (3, 30, -20, 70)):
        got = rel.window_at(T, *w)
        cols = rel.grid_columns(w[2], w[3])
        assert (got == expect[w[0]:w[1]][:, cols]).all(), w
    # the channel at T, pixel by pixel: channel_pixel_count pixels a reach
    got = rel.window_at(T, 0, NROW, 0, NCOL)
    count = np.bincount(np.searchsorted(name, got[got > 0]), minlength=name.size)
    assert (count == want['channel_pixel_count']).all()


def test_any_threshold_is_refused_below_the_base(made_up):
    rel = Release(made_up['folder'], made_up['coarser_thresholds'])
    with pytest.raises(ValueError):
        rel.reaches_at(0.5, basin_id=1)
    with pytest.raises(ValueError):
        rel.reaches_at(10.0, basin_id=1, flowdivide_level3_id=11)


def test_a_threshold_equal_to_the_base_in_float32_is_the_base(made_up):
    """As the C code compares in float32: 1 + 1e-12 (float32 1.0) and np.float32(1) are the base threshold in every read."""
    rel = Release(made_up['folder'], made_up['coarser_thresholds'])
    base = THRESHOLDS[0]
    for T in (base + 1e-12, np.float32(base), np.float64(base)):
        for g in (11, 12):
            want, got = rel.reaches_at(base, flowdivide_level3_id=g), rel.reaches_at(T, flowdivide_level3_id=g)
            for key in want:
                np.testing.assert_array_equal(got[key], want[key], err_msg=str((T, g, key)))
            (f0, c0), (f1, c1) = rel.reach_at_threshold(base, flowdivide_level3_id=g), rel.reach_at_threshold(T, flowdivide_level3_id=g)
            assert f0 == f1 and (c0 == c1).all(), (T, g)
        for basin_id in (b for b in GLOBAL_IDS if b <= BASIN_FILE_COUNT):
            assert rel.reach_range(basin_id, T) == rel.reach_range(basin_id), (T, basin_id)
        assert (rel.window_at(T, 0, NROW, 0, NCOL) == rel.window_at(base, 0, NROW, 0, NCOL)).all(), T
    with pytest.raises(ValueError):
        rel.reaches_at(base - 1e-3, flowdivide_level3_id=11)


C_RELEASE = [os.environ.get(k) for k in ('FLOWNEST_TEST_RELEASE', 'FLOWNEST_TEST_D8', 'FLOWNEST_TEST_UPA')]
C_COARSER = os.environ.get('FLOWNEST_TEST_COARSER_THRESHOLDS')


@pytest.mark.skipif(not C_RELEASE[0], reason='set FLOWNEST_TEST_RELEASE (and FLOWNEST_TEST_COARSER_THRESHOLDS for step 9)')
def test_any_threshold_reproduces_the_examples_of_a_release_of_the_c_code():
    """Derived threshold tables and rasters against the independently C-made v4 step 9 tables and crosswalk."""
    expected = Release(C_RELEASE[0], C_COARSER)
    rel = Release(C_RELEASE[0])                  # derive, even when the expected step 9 folder is supplied
    basins = expected.table('basin.parquet')
    assert C_COARSER is None or len(expected.thresholds) > 1, 'the step 9 folder brought no threshold'
    # The package names these two columns for the base of the release, 1 km2; the synthetic world of the C tests has a base of
    # 1000 km2, and the C code names them for it.
    def as_c(column):
        return column.replace('_1km2', '_' + label_of(expected.thresholds[0])) if column.endswith('_1km2') else column
    for threshold in expected.thresholds[1:]:       # the main release's coarser ones and step 9's
        label = label_of(threshold)
        whole = expected.table(f'reach_{label}.parquet')
        crosswalk = expected.table('crosswalk_reach_1km2_to_coarser.parquet',
                                   ['reach_seq_id_1km2', f'reach_seq_id_{label}'])
        for i, basin_id in enumerate(basins['basin_id']):
            got = rel.reaches_at(threshold, basin_id=int(basin_id))
            selection = rel.reach_range(int(basin_id), threshold)
            if selection is None:
                assert got['basin_id'].size == 0, (basin_id, threshold)
                continue
            want = expected.reaches(*selection, threshold)
            for column in ('downstream_end_reach_seq_id_1km2', 'stem_reach_count_1km2', *TOPOLOGY, 'basin_id', 'upstream_area_km2'):
                assert (got[column] == want[as_c(column)]).all(), (basin_id, threshold, column)
            assert ((got['upstream_reach_count'] == 1) == (want['upstream_reach_count'] == 1)).all()
            first, target = rel.reach_at_threshold(threshold, basin_id=int(basin_id))
            k = np.bincount(np.searchsorted(got['reach_seq_id'], target[target > 0]), minlength=selection[1])
            area = want['unit_catchment_area_km2'].astype(np.float64)
            assert (np.abs(got['unit_catchment_area_km2'] - area) <= k * 5e-7 + 1e-9 * area).all(), (basin_id, threshold)
            for column in ('length_km', 'distance_to_basin_outlet_km'):
                np.testing.assert_allclose(got[column], want[column], rtol=0, atol=1e-6, err_msg=str((basin_id, threshold, column)))
            assert (target == crosswalk[f'reach_seq_id_{label}'][first - 1:first - 1 + target.size]).all(), (basin_id, threshold)
            if i % 9 == 0:
                window = [int(basins[key][i]) for key in ('row_min', 'row_max', 'col_min', 'col_max')]
                raster = rel.window_at(threshold, *window)
                ids = got['reach_seq_id']
                count = np.bincount(np.searchsorted(ids, raster[np.isin(raster, ids)]), minlength=ids.size)
                assert (count == got['channel_pixel_count']).all(), (basin_id, threshold)
        # by group, the basins under the base threshold too: together the whole table, row for row
        by_group = [rel.reaches_at(threshold, flowdivide_level3_id=g['flowdivide_level3_id']) for g in rel.groups]
        for column in ('reach_seq_id', 'downstream_end_reach_seq_id_1km2', 'stem_reach_count_1km2', *TOPOLOGY, 'basin_id',
                       'upstream_area_km2'):
            assert np.array_equal(np.concatenate([r[column] for r in by_group]), whole[as_c(column)]), (threshold, column)


@pytest.mark.skipif(not all(C_RELEASE), reason='set FLOWNEST_TEST_RELEASE, FLOWNEST_TEST_D8 and FLOWNEST_TEST_UPA')
def test_the_package_reproduces_a_release_of_the_c_code():
    rel = Release(C_RELEASE[0], C_COARSER)
    ids = rel.table('basin.parquet', ['basin_id'])['basin_id']
    for basin_id in ids:
        n, window = rel.number_basin(int(basin_id), C_RELEASE[1], C_RELEASE[2])
        raster = rel.window(*window).astype(np.int64)
        first, count = n.first, n.grid_index.size
        mine = (raster != rel.nodata) & (np.abs(raster) >= first) & (np.abs(raster) < first + count)
        assert ((n.raster != 0) == mine).all() and (n.raster[mine] == raster[mine]).all(), basin_id
        for threshold in rel.thresholds:              # with the step 9 folder these include its thresholds
            selection = rel.reach_range(int(basin_id), threshold)
            if selection is None:
                assert n.reaches(threshold)['reach_seq_id'].size == 0, (basin_id, threshold)
                continue
            want = rel.reaches(*selection, threshold)
            got = n.reaches(threshold, first_reach=selection[0], first_path=int(want['path_seq_id'][0]))
            for column in TOPOLOGY:
                assert (got[column] == want[column]).all(), (basin_id, threshold, column)
            assert (got['upstream_area_km2'].astype(np.float32) == want['upstream_area_km2']).all(), (basin_id, threshold)
            # Retain checks of the dropped columns by deriving them from the v4 table.
            below = rel.rows_at(f'reach_{label_of(threshold)}.parquet', want['next_down_reach_seq_id'],
                                ['downstream_end_pixel_seq_id', 'channel_pixel_count'])
            next_pixel = np.where(want['next_down_reach_seq_id'] == want['reach_seq_id'], want['downstream_end_pixel_seq_id'],
                                  below['downstream_end_pixel_seq_id'] + below['channel_pixel_count'] - 1)
            local = want['downstream_end_pixel_seq_id'] - n.first
            assert (next_pixel == np.where(n.down[local] > 0, n.down[local] + n.first - 1,
                                           want['downstream_end_pixel_seq_id'])).all(), (basin_id, threshold)
            donors = want['next_down_reach_seq_id'][want['next_down_reach_seq_id'] != want['reach_seq_id']]
            assert ((want['upstream_reach_count'] == 1) == ~np.isin(want['reach_seq_id'], donors)).all(), (basin_id, threshold)
            box = rel.reach_windows(n, window, threshold, first_reach=selection[0])
            location = rel.rows(f'reach_location_{label_of(threshold)}.parquet', *selection)
            for column in WINDOWS:
                assert (box[column] == location[column]).all(), (basin_id, threshold, column)
            unit = n.unit_catchments(threshold, first_reach=selection[0])
            base_ends = n.reaches(rel.thresholds[0])['downstream_end_pixel_seq_id']
            k = sum(np.searchsorted(base_ends, unit[f'first_{i}'] + unit[f'count_{i}']) -
                    np.searchsorted(base_ends, unit[f'first_{i}']) for i in (1, 2))
            area = want['unit_catchment_area_km2'].astype(np.float64)
            assert (np.abs(box['unit_catchment_area_km2'] - area) <= k * 5e-7 + 1e-9 * area).all(), (basin_id, threshold)


@pytest.mark.skipif(not C_RELEASE[0], reason='set FLOWNEST_TEST_RELEASE')
def test_the_base_threshold_read_through_reaches_at_is_the_base_table():
    """At the base every base reach is a reach, the one-pixel basins below the base (break_threshold_km2 < base) too."""
    rel = Release(C_RELEASE[0])
    base = rel.thresholds[0]
    basins = rel.table('basin.parquet')
    for i, basin_id in enumerate(basins['basin_id']):
        selection = rel.reach_range(int(basin_id))
        got = rel.reaches_at(base, basin_id=int(basin_id))
        if selection is None:
            assert got['basin_id'].size == 0, basin_id
            continue
        want = rel.reaches(*selection)
        for column in (*TOPOLOGY, 'basin_id', 'upstream_area_km2'):
            assert (got[column] == want[column]).all(), (basin_id, column)
        first, target = rel.reach_at_threshold(base, basin_id=int(basin_id))
        assert first == selection[0] and (target == np.arange(first, first + selection[1])).all(), basin_id
        window = [int(basins[key][i]) for key in ('row_min', 'row_max', 'col_min', 'col_max')]
        assert (rel.window_at(base, *window) == rel.window(*window, raster='reach')).all(), basin_id
    whole = rel.table(f'reach_{label_of(base)}.parquet')       # by group, the basins under the base threshold too
    by_group = [rel.reaches_at(base, flowdivide_level3_id=g['flowdivide_level3_id']) for g in rel.groups]
    for column in (*TOPOLOGY, 'basin_id', 'upstream_area_km2'):
        assert np.array_equal(np.concatenate([r[column] for r in by_group]), whole[column]), column
