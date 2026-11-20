"""MERIT-Basins reaches (Lin et al. 2019) on a FlowNest release: the release's pixel, reach and basin of each reach, with no
snapping, as both are drawn on the same MERIT Hydro pixels.

The rule is that of the release's MERIT-Basins crosswalk (`merit_basins`):

1. Every vertex of a reach's line is a MERIT Hydro pixel centre (within 1e-6 of a pixel), from its downstream end up. A reach with
   a reach downstream (NextDownID not 0) begins at the last pixel of that one, so its first vertex is passed over; its own pixels
   are the rest. A reach needs two vertices then, one at a basin outlet; else, or with a vertex off the pixel centres or off the
   grid, it is outside_grid.
2. Its downstream end is its first own vertex. On a channel pixel p of the release (raster value +p) the reach takes p's basin,
   base reach and reach at the threshold T asked for, and MERIT Hydro's upstream area there:
     same_reach               p begins a reach at T, the last vertex is that reach's last pixel, and every own vertex is the next
                              pixel number (p, p + 1, ...): the two reaches are the same run of pixels;
     same_downstream_end      p begins a reach at T, but the upstream end or the run differs;
     on_reach                 p lies inside a reach at T;
     channel_below_threshold  p is a channel pixel whose upstream area is below T.
   On other pixels: hillslope_pixel (-p: it drains first to channel pixel p), basin_below_min_area (0), no_land (nodata).

Needs pyogrio, geopandas (pyogrio.read_dataframe), shapely, pyarrow and rasterio.
"""
import numpy as np

MATCH_CLASSES = ('same_reach', 'same_downstream_end', 'on_reach', 'channel_below_threshold', 'hillslope_pixel',
                 'basin_below_min_area', 'outside_grid', 'no_land')
PIXEL_CENTRE_TOLERANCE = 1e-6
NODATA = -2147483648


def read_merit_basins(path):
    """The riv_pfaf layer of a MERIT-Basins file (.shp, or a .zip read inside), in its feature order: comid, next_down_comid,
    upstream_area_km2 (NaN where null), and the vertices (x, y) with the feature each belongs to; a feature that is no single
    line has no vertex."""
    import pyogrio
    import shapely
    source = f'/vsizip/{path}' if path.lower().endswith('.zip') else path
    layers = [name for name, _ in pyogrio.list_layers(source) if name.lower().startswith('riv_pfaf')]
    if len(layers) != 1:
        raise ValueError(f'{path} holds no one riv_pfaf layer')
    df = pyogrio.read_dataframe(source, layer=layers[0], columns=['COMID', 'NextDownID', 'uparea'])
    if df.crs is not None and not df.crs.is_geographic:
        raise ValueError(f'{path} is not in longitudes and latitudes')
    if df['COMID'].isna().any() or df['NextDownID'].isna().any():
        raise ValueError(f'{path}: a feature has no COMID or NextDownID')
    geometry = df.geometry.values
    is_line = np.asarray(shapely.get_type_id(geometry) == 1)                   # LineString
    xy, feature = shapely.get_coordinates(geometry[is_line], return_index=True)
    feature = np.flatnonzero(is_line)[feature]
    return {'comid': df['COMID'].to_numpy(np.int64), 'next_down_comid': df['NextDownID'].to_numpy(np.int64),
            'upstream_area_km2': df['uparea'].to_numpy(np.float64, na_value=np.nan), 'x': xy[:, 0], 'y': xy[:, 1],
            'feature': feature}


def pixel_of_vertex(rel, x, y):
    """Row and column of the pixel each vertex is the centre of, and whether it is one (within the tolerance, on the grid)."""
    x0, dx, _, y0, _, dy = rel.geotransform
    col = (x - x0) / dx - 0.5
    row = (y - y0) / dy - 0.5
    col_whole, row_whole = np.floor(col + 0.5), np.floor(row + 0.5)
    centre = np.isfinite(col) & np.isfinite(row) & (np.abs(col - col_whole) <= PIXEL_CENTRE_TOLERANCE) & \
        (np.abs(row - row_whole) <= PIXEL_CENTRE_TOLERANCE)
    c = np.where(centre, col_whole, 0).astype(np.int64)
    r = np.where(centre, row_whole, 0).astype(np.int64)
    if abs(abs(rel.ncol * dx) - 360.0) < 1e-6:
        c = c % rel.ncol                                                       # +180 and -180 are the same column
    on = centre & (r >= 0) & (r < rel.nrow) & (c >= 0) & (c < rel.ncol)
    return r, c, on


class ReleaseReaches:
    """What the rule looks up in a release (opened with its coarser thresholds when T is one of them): the base reach and basin
    of a channel pixel (the pixel table), the reaches at T and the base reach's reach at T."""

    def __init__(self, rel, threshold):
        from .release import label_of
        self.rel, self.threshold = rel, float(threshold)
        self.label = label_of(threshold)
        at_t = rel.table(f'reach_{self.label}.parquet', ['downstream_end_pixel_seq_id', 'channel_pixel_count'])
        self.outlet = at_t['downstream_end_pixel_seq_id'].astype(np.int64)
        self.last = self.outlet + at_t['channel_pixel_count'] - 1                  # a reach's channel pixels: [outlet, last]
        self.at_base = self.label == rel.base
        if not self.at_base:
            cross = rel.table(f'crosswalk_reach_{rel.base}_to_coarser.parquet', [f'reach_seq_id_{self.label}'])
            self.receiving = cross[f'reach_seq_id_{self.label}'].astype(np.int64)

    def base_reach_and_basin(self, pixel):
        """the base reach and the basin of each channel pixel, from the pixel table (the basin table is in basin_id order, not
        in pixel order, and leaves out the basins under the base threshold)"""
        rows = self.rel.rows_at(f'pixel_{self.rel.base}.parquet', pixel, ['reach_seq_id', 'basin_id'])
        return rows['reach_seq_id'].astype(np.int64), rows['basin_id'].astype(np.int64)

    def reach_at_t(self, reach0):
        return reach0 if self.at_base else self.receiving[reach0 - 1]


def place_reaches(rel, lookup, reaches, upa_tif):
    """Steps 1 and 2 for the reaches of read_merit_basins on a Release, with a ReleaseReaches at T and MERIT Hydro's upa raster.
    Returns the columns of the release's MERIT-Basins crosswalk at T, one row a reach in feature order."""
    n = reaches['comid'].size
    down = reaches['next_down_comid']
    feature = reaches['feature']
    points = np.bincount(feature, minlength=n)
    start = np.r_[0, np.cumsum(points)[:-1]]
    row, col, on = pixel_of_vertex(rel, reaches['x'], reaches['y'])
    # step 1: on the grid when every vertex is a pixel centre and there are enough of them
    off = np.bincount(feature[~on], minlength=n)                                  # vertices off the pixel centres or the grid
    on_grid = (off == 0) & (points >= np.where(down != 0, 2, 1))
    first_own = (down != 0).astype(np.int64)
    own = np.where(on_grid, points - first_own, 0)
    keep = on_grid[feature]
    value = np.zeros(feature.size, np.int64)
    value[keep] = rel.values_at(row[keep], col[keep])
    out = {k: np.zeros(n, np.int64) for k in ('outlet_row', 'outlet_col', 'outlet_value', 'end_value', 'channel', 'in_order',
                                               'pixel', 'drains_to', 'basin', 'reach0', 'reach_t')}
    out['outlet_row'][:] = -1
    out['outlet_col'][:] = -1
    area = np.full(n, np.nan, np.float32)
    match = np.full(n, MATCH_CLASSES.index('outside_grid'))
    g = np.flatnonzero(on_grid)
    at = start[g] + first_own[g]                                                   # each reach's downstream end vertex
    out['outlet_row'][g], out['outlet_col'][g], out['outlet_value'][g] = row[at], col[at], value[at]
    out['end_value'][g] = value[start[g] + points[g] - 1]
    # own vertices: the channel ones, and those that are the next pixel number from the downstream end
    position = np.arange(feature.size) - start[feature] - first_own[feature]     # 0 at the downstream end
    is_own = keep & (position >= 0)
    outlet_value = out['outlet_value'][feature]
    out['channel'] = np.bincount(feature[is_own & (value > 0)], minlength=n)
    out['in_order'] = np.bincount(feature[is_own & (outlet_value > 0) & (value == outlet_value + position)], minlength=n)
    # step 2
    v = out['outlet_value']
    c = np.flatnonzero(on_grid & (v > 0))
    p = v[c]
    out['pixel'][c] = out['drains_to'][c] = p
    area[c] = rel.values_at(out['outlet_row'][c], out['outlet_col'][c], upa_tif)
    reach0, out['basin'][c] = lookup.base_reach_and_basin(p)
    reach_t = lookup.reach_at_t(reach0)
    out['reach0'][c], out['reach_t'][c] = reach0, reach_t
    on_network = np.ones(c.size, bool) if lookup.at_base else area[c] >= np.float32(lookup.threshold)
    if (on_network & (reach_t <= 0)).any():
        raise ValueError('a pixel on the network at T lies in no reach of it')
    safe = np.maximum(reach_t, 1) - 1
    begins = lookup.outlet[safe] == p
    whole = begins & (lookup.last[safe] == out['end_value'][c]) & (out['in_order'][c] == own[c])
    match[c] = np.where(~on_network, 3, np.where(begins, np.where(whole, 0, 1), 2))
    h = np.flatnonzero(on_grid & (v < 0) & (v != NODATA))
    match[h] = 4
    out['drains_to'][h] = -v[h]
    match[np.flatnonzero(on_grid & (v == 0))] = 5
    match[np.flatnonzero(on_grid & (v == NODATA))] = 7
    table = {
        'merit_basins_comid': reaches['comid'], 'merit_basins_next_down_comid': down,
        'merit_basins_upstream_area_km2': reaches['upstream_area_km2'], 'merit_basins_vertex_count': points.astype(np.int64),
        'merit_basins_pixel_count': own, 'merit_basins_downstream_end_row': out['outlet_row'],
        'merit_basins_downstream_end_col': out['outlet_col'], 'merit_basins_downstream_end_pixel_raster_value': v,
        'merit_basins_upstream_end_pixel_raster_value': out['end_value'], 'on_channel_pixel_count': out['channel'],
        'in_sequence_pixel_count': out['in_order'], 'pixel_seq_id': out['pixel'], 'drains_to_pixel_seq_id': out['drains_to'],
        'upstream_area_km2': area, 'basin_id': out['basin'], f'reach_seq_id_{rel.base}': out['reach0'],
    }
    if not lookup.at_base:
        table[f'reach_seq_id_{lookup.label}'] = out['reach_t']
    table['match_class'] = np.array(MATCH_CLASSES, dtype=object)[match]
    return table
