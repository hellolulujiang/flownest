"""Gauging stations on a FlowNest release: the pixel under each station, its channel pixel, and the stations numbered so that the
stations upstream of a station are one run of numbers.

The rule is that of the release's GRDC crosswalk (`stations`):

1. The pixel under the station's coordinate (longitude wraps on a grid around the earth).
2. When the list gives the station's MERIT Hydro area, a station whose pixel is no channel pixel or whose upstream area misses that
   area by more than max(0.5 km2, 1 %) moves to the channel pixel of the 3 x 3 window around it whose upstream area is nearest to
   it (on a tie the nearer one, then the first in row order), when that pixel carries the area within the same margin; otherwise
   it stays where it is.
3. The pixel raster's value then says what the pixel is: + a channel pixel, - a hillslope pixel draining first to that channel
   pixel, 0 land of a basin under 0.05 km2, nodata no land.
4. The stations that drain to a channel pixel are numbered 1 .. n in the order of that pixel (a channel station before a hillslope
   one on the same pixel, then by station id). The stations upstream of a channel station, itself included, are then
   station_seq_id .. station_seq_id + upstream_station_count - 1, as the channel pixels upstream are one run.

Needs pyarrow and rasterio (pip install flownest[release]).
"""
import csv
import math
import numpy as np

AREA_MARGIN_KM2 = 0.5          # a pixel carries the station's MERIT area when it misses it by no more than
AREA_MARGIN_FRACTION = 0.01    # max(AREA_MARGIN_KM2, AREA_MARGIN_FRACTION * area)
NODATA = -2147483648

PIXEL_CLASSES = ('channel_pixel', 'hillslope_pixel', 'basin_below_min_area', 'no_land', 'outside_grid')

FORMS = {
    # Burek and Smilovic (2023) evaluation_csv/stationquality.csv: two title lines, then the names; their place on MERIT Hydro
    'burek2023': dict(titles=2, station_id='GRDC_No', lat='newlat', lon='newlon', reported_area_km2='GRDCarea',
                      merit_area_km2='area'),
    'plain': dict(titles=0, station_id='station_id', lat='lat', lon='lon', reported_area_km2='reported_area_km2',
                  merit_area_km2='merit_area_km2'),
}


def _positive_or_nan(text):
    try:
        value = float(text)
    except ValueError:
        return math.nan
    return value if value > 0 and math.isfinite(value) else math.nan


def read_station_list(path, form='burek2023'):
    """A station list as a dict of arrays: station_id, lat, lon, reported_area_km2, merit_area_km2 (NaN where not given or not
    above 0). When a column name comes again, its first column is read. In the burek2023 form a line with no GRDC_No is passed
    over (their file holds other, longer blocks)."""
    names = FORMS[form]
    out = {k: [] for k in ('station_id', 'lat', 'lon', 'reported_area_km2', 'merit_area_km2')}
    with open(path, newline='') as f:
        lines = csv.reader(f)
        for _ in range(names['titles']):
            next(lines)
        header = [h.strip() for h in next(lines)]
        column = {}
        for k, h in enumerate(header):
            column.setdefault(h, k)
        missing = [names[k] for k in ('station_id', 'lat', 'lon') if names[k] not in column]
        if missing:
            raise ValueError(f'{path}: no column {missing}')
        for record in lines:
            record = [v.strip() for v in record]
            if not any(record):
                continue
            if form == 'burek2023' and record[column[names['station_id']]] == '':
                continue
            out['station_id'].append(int(record[column[names['station_id']]]))
            out['lat'].append(float(record[column[names['lat']]]))
            out['lon'].append(float(record[column[names['lon']]]))
            for k in ('reported_area_km2', 'merit_area_km2'):
                out[k].append(_positive_or_nan(record[column[names[k]]]) if names[k] in column else math.nan)
    if not out['station_id']:
        raise ValueError(f'{path} holds no station')
    return {'station_id': np.array(out['station_id'], np.int64), 'lat': np.array(out['lat']), 'lon': np.array(out['lon']),
            'reported_area_km2': np.array(out['reported_area_km2']), 'merit_area_km2': np.array(out['merit_area_km2'])}


def pixel_under(rel, lat, lon):
    """Row and column of the pixel under each point, -1/-1 off the grid. On a grid around the earth longitude wraps, so -180 and
    +180 fall on the same pixel."""
    x0, dx, _, y0, _, dy = rel.geotransform
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    row = np.floor((lat - y0) / dy)
    col = np.floor((lon - x0) / dx)
    if abs(abs(rel.ncol * dx) - 360.0) < 1e-6 and dx > 0:
        east = (lon - x0) - 360.0 * np.floor((lon - x0) / 360.0)
        col = np.floor(east / dx)
        col = col - rel.ncol * np.floor(col / rel.ncol)
    on = np.isfinite(row) & np.isfinite(col) & (row >= 0) & (row < rel.nrow) & (col >= 0) & (col < rel.ncol)
    return np.where(on, row, -1).astype(np.int64), np.where(on, col, -1).astype(np.int64)


def _carries(area, merit_area):
    return np.abs(area - merit_area) <= np.maximum(AREA_MARGIN_KM2, AREA_MARGIN_FRACTION * merit_area)


def move_to_merit_area(rel, upa_tif, row, col, value, merit_area):
    """Step 2: the stations moved to the channel pixel carrying their MERIT area. Returns new row, col, value and moved (0/1)."""
    row, col, value = row.copy(), col.copy(), value.copy()
    moved = np.zeros(row.size, np.int64)
    look = (row >= 0) & np.isfinite(merit_area)
    here = np.flatnonzero(look & (value > 0))
    if here.size:
        ok = _carries(rel.values_at(row[here], col[here], upa_tif).astype(np.float64), merit_area[here])
        look[here[ok]] = False
    k = np.flatnonzero(look)
    if k.size == 0:
        return row, col, value, moved
    around_earth = abs(abs(rel.ncol * rel.geotransform[1]) - 360.0) < 1e-6
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]               # row order, as the C loop runs
    r = row[k][:, None] + np.array([s[0] for s in steps])
    c = col[k][:, None] + np.array([s[1] for s in steps])
    valid = (r >= 0) & (r < rel.nrow) & (around_earth | ((c >= 0) & (c < rel.ncol)))
    c = c % rel.ncol
    v = np.zeros(r.shape, np.int64)
    a = np.full(r.shape, np.inf)
    v[valid] = rel.values_at(r[valid], c[valid])
    channel = valid & (v > 0)
    a[channel] = rel.values_at(r[channel], c[channel], upa_tif).astype(np.float64)
    error = np.abs(a - merit_area[k][:, None])
    for j, i in enumerate(k):
        # the nearest area; on a tie the fewer steps, then the first in row order
        candidates = [(error[j, s], abs(dr) + abs(dc), s) for s, (dr, dc) in enumerate(steps) if channel[j, s]]
        if not candidates:
            continue
        e, _, s = min(candidates)
        if e <= max(AREA_MARGIN_KM2, AREA_MARGIN_FRACTION * merit_area[i]):
            moved[i] = int(r[j, s] != row[i] or c[j, s] != col[i])
            row[i], col[i], value[i] = r[j, s], c[j, s], v[j, s]
    return row, col, value, moved


def number_stations(drains_to, on_channel, station_id, basin_id, last):
    """Step 4. drains_to is the channel pixel each station drains to (0: none), last the last channel pixel upstream of a channel
    station (drains_to - 1 for a hillslope one). Returns station_seq_id, basin_station_seq_id, next_down_station_seq_id (itself
    when no station lies downstream) and upstream_station_count, all 0 for a station that drains to no channel pixel."""
    n_all = drains_to.size
    seq, basin_seq, next_down, upstream = (np.zeros(n_all, np.int64) for _ in range(4))
    k = np.flatnonzero(drains_to > 0)
    order = k[np.lexsort((k, station_id[k], ~on_channel[k], drains_to[k]))]
    seq[order] = np.arange(1, order.size + 1)
    new_basin = np.r_[True, basin_id[order][1:] != basin_id[order][:-1]]
    first_of_basin = np.maximum.accumulate(np.where(new_basin, np.arange(order.size), 0))
    basin_seq[order] = np.arange(order.size) - first_of_basin + 1
    stack = []                                   # channel stations whose upstream run may still hold the next station
    for i in order:
        while stack and last[stack[-1]] < drains_to[i]:
            stack.pop()
        next_down[i] = seq[stack[-1]] if stack else seq[i]
        if on_channel[i]:
            stack.append(i)
    sorted_pixels = drains_to[order]
    for j, i in enumerate(order):
        upstream[i] = (np.searchsorted(sorted_pixels, last[i], side='right') - j) if on_channel[i] else 1
    return seq, basin_seq, next_down, upstream


def place_stations(rel, stations, upa_tif):
    """Steps 1 to 4 for a station list (read_station_list) on a Release, with MERIT Hydro's upa raster. Returns the 19 columns of
    the release's station crosswalk, one row a station in list order."""
    n = stations['station_id'].size
    row, col = pixel_under(rel, stations['lat'], stations['lon'])
    on = row >= 0
    value = np.zeros(n, np.int64)
    value[on] = rel.values_at(row[on], col[on])
    row, col, value, moved = move_to_merit_area(rel, upa_tif, row, col, value, stations['merit_area_km2'])
    status = np.where(~on, 4, np.where(value > 0, 0, np.where(value == 0, 2, np.where(value == NODATA, 3, 1))))
    channel, hillslope = status == 0, status == 1
    pixel = np.where(channel, value, 0)
    drains_to = np.where(channel, value, np.where(hillslope, -value, 0))
    area = np.full(n, np.nan, np.float32)
    area[channel] = rel.values_at(row[channel], col[channel], upa_tif)
    basin = np.zeros(n, np.int64)
    reach = np.zeros(n, np.int64)
    path = np.zeros(n, np.int64)
    last = drains_to - 1
    d = np.flatnonzero(drains_to > 0)
    if d.size:
        rows = rel.rows_at(f'pixel_{rel.base}.parquet', drains_to[d], ['basin_id', 'reach_seq_id', 'path_seq_id',
                                                                         'upstream_channel_pixel_count'])
        basin[d] = rows['basin_id']
        c = channel[d]
        reach[d[c]] = rows['reach_seq_id'][c]
        path[d[c]] = rows['path_seq_id'][c]
        last[d[c]] = drains_to[d[c]] + rows['upstream_channel_pixel_count'][c] - 1
    seq, basin_seq, next_down, upstream = number_stations(drains_to, channel, stations['station_id'], basin, last)
    return {
        'station_seq_id': seq, 'basin_station_seq_id': basin_seq, 'grdc_station_id': stations['station_id'],
        'grdc_lat': stations['lat'], 'grdc_lon': stations['lon'], 'grdc_reported_area_km2': stations['reported_area_km2'],
        'row': row, 'col': col, 'moved_to_neighbouring_pixel': moved,
        'pixel_class': np.array([PIXEL_CLASSES[s] for s in status], dtype=object),
        'pixel_raster_value': np.where(on, value, 0), 'pixel_seq_id': pixel, 'drains_to_pixel_seq_id': drains_to,
        'upstream_area_km2': area, 'basin_id': basin, f'reach_seq_id_{rel.base}': reach, f'path_seq_id_{rel.base}': path,
        'next_down_station_seq_id': next_down, 'upstream_station_count': upstream,
    }
