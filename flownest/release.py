"""Read v4 releases in parts. Row k of each pixel/reach/location table holds id k + 1.

basin_id is the FullBasin global basin ID (FlowNest 10.58, numbering_version 0.2): the basin table lists the basins of at least
the base threshold, basin_id 1 .. basin_file_count without a gap, basin b on row b - 1; a smaller basin is in the pixel and
reach tables only. The FlowDivide Level-3 groups are in metadata.json (there is no group table).
All id runs are first/count. Windows use [row_min, row_max) and [col_min, col_max); a negative col_min wraps across 180 degrees.
Needs pyarrow and rasterio: pip install flownest[release]."""
import json
import operator
import os
import numpy as np
from .core import _as_float32

PREFIX = 'flownest_merit90m_'
REACH_COLUMNS = ('reach_seq_id', 'downstream_end_pixel_seq_id', 'channel_pixel_count', 'upstream_channel_pixel_count',
                 'next_down_reach_seq_id', 'upstream_reach_count', 'path_seq_id', 'path_level', 'basin_id', 'upstream_area_km2',
                 'unit_catchment_area_km2', 'length_km', 'distance_to_basin_outlet_km', 'break_threshold_km2')
COARSER_REACH_COLUMNS = REACH_COLUMNS[:-1] + ('downstream_end_reach_seq_id_1km2', 'stem_reach_count_1km2')
LOCATION_COLUMNS = ('reach_seq_id', 'downstream_end_lon', 'downstream_end_lat', 'unit_catchment_row_min',
                    'unit_catchment_row_max', 'unit_catchment_col_min', 'unit_catchment_col_max', 'upstream_catchment_row_min',
                    'upstream_catchment_row_max', 'upstream_catchment_col_min', 'upstream_catchment_col_max')


def label_of(threshold):
    """The name of a threshold in the file names, '1km2', '25km2', '11p2km2', ..."""
    return ('%.9gkm2' % float(threshold)).replace('.', 'p')


class Release:
    """A release folder. Tables come back as dicts of numpy arrays, one entry a column."""

    def __init__(self, folder, coarser_thresholds=None):
        from . import __version__
        self.folder = folder
        with open(os.path.join(folder, 'metadata.json')) as f:
            self.meta = json.load(f)
        self._check_schema(self.meta, folder)
        # basin_id: FullBasin's global ID (FlowNest 10.58); the basin table holds 1 .. basin_file_count, as the C reader checks
        ids = self.meta.get('basin_id')
        files_range = ids.get('basin_file_range') if isinstance(ids, dict) else None
        self.basin_file_count = self.meta.get('basin_file_count')
        if not (isinstance(self.basin_file_count, int) and files_range == [1, self.basin_file_count]):
            raise ValueError(f'{folder}: metadata.json has no basin_file_count with basin_id.basin_file_range [1, basin_file_count]; '
                             f'a release before FlowNest 10.58 numbers basin_id otherwise; flownest {__version__} does not read it')
        self.groups = self.meta['flowdivide_level3_groups']
        grid = self.meta['grid']
        self.geotransform, self.nrow, self.ncol, self.nodata = grid['geotransform'], grid['nrow'], grid['ncol'], grid['nodata']
        self.thresholds = [float(t) for t in self.meta['thresholds_km2']]
        self.base = label_of(self.thresholds[0])
        self._base32 = _as_float32(self.thresholds[0])   # the base as the C code holds it, in float32
        self._files = {os.path.basename(k): os.path.join(folder, k) for k in self.meta['files']}
        if coarser_thresholds is not None:
            with open(os.path.join(coarser_thresholds, 'metadata.json')) as f:
                meta = json.load(f)
            self._check_schema(meta, coarser_thresholds)
            # the coarser-threshold folder must name this release's numbering, at its top and under "release", as the C reader checks
            expected = self.meta.get('numbering_sha256')
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f'{folder}: metadata.json has no numbering_sha256, so no coarser-threshold package can be tied to it')
            for key, value in (('numbering_sha256', meta.get('numbering_sha256')),
                               ('release.numbering_sha256', meta.get('release', {}).get('numbering_sha256'))):
                if value != expected:
                    raise ValueError(f'{coarser_thresholds}: {key} {value} is not the numbering_sha256 {expected} of {folder}')
            if meta.get('numbering_version') != self.meta.get('numbering_version'):
                raise ValueError(f'{coarser_thresholds}: numbering_version {meta.get("numbering_version")} is not '
                                 f'{self.meta.get("numbering_version")} of {folder}')
            extra = [float(t) for t in meta['thresholds_km2']]
            if any(not _as_float32(t) > self._base32 for t in extra):
                raise ValueError(f'{coarser_thresholds}: thresholds {extra} are not all above the base threshold {self.thresholds[0]}')
            self.thresholds = [self.thresholds[0]] + sorted(set(self.thresholds[1:]) | set(extra))
            self._files.update({os.path.basename(k): os.path.join(coarser_thresholds, k) for k in meta['files']})
        self._opened, self._tables = {}, {}
        self._offsets = {}

    @staticmethod
    def _check_schema(meta, folder):
        from . import __version__
        version = meta.get('schema_version')
        if version != '4':
            if version is not None and str(version).isdigit() and int(version) > 4:
                raise ValueError(f'{folder}: schema_version {version} is not a FlowNest release but an unsupported package type '
                                 f'(a crosswalk package: read its tables with pandas or pyarrow); flownest {__version__} reads '
                                 'releases of schema_version "4".')
            raise ValueError(f'{folder}: pre-v4 release or unknown schema_version {version!r}; flownest {__version__} reads only '
                             'schema_version "4".')

    # ------------------------------------------------------------------------------------------------------- files and tables
    def file(self, name):
        """Path of a file of the release by its name without the prefix: 'basin.parquet', 'reach_25km2.parquet', ... (root, examples/ or the coarser-threshold package)."""
        if PREFIX + name not in self._files:
            raise KeyError(f'{PREFIX + name} is not a file of this release (metadata.json lists {sorted(self._files)})')
        return self._files[PREFIX + name]

    def _parquet(self, name):
        if name not in self._opened:
            import pyarrow.parquet as pq
            f = pq.ParquetFile(self.file(name))
            md = f.metadata
            ends = np.cumsum([md.row_group(g).num_rows for g in range(md.num_row_groups)])
            self._opened[name] = (f, ends)
        return self._opened[name]

    def table(self, name, columns=None):
        """A whole small table, such as 'basin.parquet' (one row a basin of at least the base threshold, basin b on row b - 1). Read once."""
        key = (name, None if columns is None else tuple(columns))
        if key not in self._tables:
            f, _ = self._parquet(name)
            t = f.read(columns=columns) if f.metadata.num_row_groups else f.schema_arrow.empty_table().select(
                columns or f.schema_arrow.names)                        # a table with no rows has no row group
            self._tables[key] = {c: t.column(c).to_numpy() for c in t.column_names}
        return self._tables[key]

    def rows(self, name, first, count, columns=None):
        """Rows with ids in [first, first + count), including zero-count reads."""
        first, count = operator.index(first), operator.index(count)
        f, ends = self._parquet(name)
        key = f.schema_arrow.names[0]                    # pixel_seq_id or reach_seq_id
        size = int(ends[-1]) if ends.size else 0
        stop = first + count
        if count < 0 or first < 1 or stop > size + 1:
            raise IndexError(f'ids [{first}, {stop}) are not rows of {name} ([1, {size + 1}))')
        want = None if columns is None else list(dict.fromkeys([key] + list(columns)))
        if count == 0:
            t = f.schema_arrow.empty_table().select(f.schema_arrow.names if want is None else want)
            return {c: t.column(c).to_numpy() for c in t.column_names if columns is None or c in columns}
        g0, g1 = np.searchsorted(ends, [first - 1, stop - 2], side='right')
        t = f.read_row_groups(list(range(g0, g1 + 1)), columns=want)
        start = int(ends[g0 - 1]) if g0 else 0
        t = t.slice(first - 1 - start, count)
        out = {c: t.column(c).to_numpy() for c in t.column_names}
        if not np.array_equal(out[key], np.arange(first, stop)):
            raise ValueError(f'{name}: row k does not hold id k + 1 around ids [{first}, {stop})')
        if columns is not None and key not in columns:
            del out[key]
        return out

    def rows_at(self, name, ids, columns=None):
        """Rows of scattered ids (the pixels of a set of gauges, say) of a table whose row k holds id k + 1, in the order of ids.
        Reads each row group that holds one of them once."""
        f, ends = self._parquet(name)
        key = f.schema_arrow.names[0]
        names = f.schema_arrow.names if columns is None else list(dict.fromkeys([key] + list(columns)))
        ids = np.asarray(ids, np.int64).ravel()
        size = int(ends[-1]) if ends.size else 0
        if ids.size and not 1 <= ids.min() <= ids.max() <= size:
            raise IndexError(f'ids {ids.min()} .. {ids.max()} are not all rows of {name} ([1, {size + 1}))')
        g = np.searchsorted(ends, ids - 1, side='right')
        order = np.argsort(g, kind='stable')
        groups, starts = np.unique(g[order], return_index=True)
        parts = {c: [f.schema_arrow.empty_table().column(c).to_numpy()] for c in names}
        for k, s, e in zip(groups, starts, np.r_[starts[1:], ids.size]):
            t = f.read_row_group(int(k), columns=names)
            at = ids[order[s:e]] - 1 - (int(ends[k - 1]) if k else 0)
            for c in names:
                parts[c].append(t.column(c).to_numpy()[at])
        back = np.empty(ids.size, np.int64)
        back[order] = np.arange(ids.size)
        out = {c: np.concatenate(parts[c])[back] for c in names}
        if not (out[key] == ids).all():
            raise ValueError(f'{name}: row k does not hold id k + 1')
        if columns is not None and key not in columns:
            del out[key]
        return out

    def reaches(self, first, count, threshold=None, columns=None):
        """Reach rows in [first, first + count) at a threshold (the base by default). Location columns are in reach_location_<T>."""
        label = self.base if threshold is None else label_of(threshold)
        return self.rows(f'reach_{label}.parquet', first, count, columns)

    def pixels(self, first, count, columns=None):
        """Channel pixel rows in [first, first + count)."""
        return self.rows(f'pixel_{self.base}.parquet', first, count, columns)

    # ------------------------------------------------------------------------------------------------------- objects
    def basin(self, basin_id):
        """One row of the basin table, as a dict of numbers: a basin of at least the base threshold (basin_id 1 .. basin_file_count)."""
        b = self.table('basin.parquet')
        k = int(basin_id) - 1
        if not 0 <= k < b['basin_id'].size or b['basin_id'][k] != basin_id:
            raise KeyError(f'basin {basin_id} is not row {k} of the basin table, which lists the basins of at least the base '
                           f'threshold (basin_id 1 .. {b["basin_id"].size}); a smaller basin is in the pixel and reach tables only')
        return {c: v[k].item() for c, v in b.items()}

    def reach_range(self, basin_id, threshold=None):
        """First/count of the reaches at a threshold of a basin of the basin table, or None for no network."""
        a = self.thresholds[0] if threshold is None else self._check_threshold(threshold)
        b = self.basin(basin_id)
        first, count = int(b[f'first_reach_seq_id_{self.base}']), int(b[f'reach_count_{self.base}'])
        if self._is_base(a):
            return first, count
        kept = self.reaches(first, count, columns=['break_threshold_km2'])['break_threshold_km2'] >= _as_float32(a)
        count = int(kept.sum())
        offset, _ = self._threshold_offsets(a, first)
        return (offset + 1, count) if count else None

    def basin_of(self, pixel_seq_id):
        """basin_id of a channel pixel, from the pixel table (any basin, also one under the base threshold)."""
        try:
            return int(self.pixels(operator.index(pixel_seq_id), 1, ['basin_id'])['basin_id'][0])
        except IndexError:
            raise KeyError(f'{pixel_seq_id} is not a channel pixel of this release') from None

    def group(self, flowdivide_level3_id):
        """One FlowDivide Level-3 group of metadata.json, as a dict: its basin count (all its basins, also those under the base
        threshold), its runs (first_pixel_seq_id/channel_pixel_count, first_reach_seq_id_<T>/reach_count_<T>,
        first_path_seq_id_<T>/path_count_<T>) and its window."""
        k = [g for g in self.groups if g['flowdivide_level3_id'] == flowdivide_level3_id]
        if len(k) != 1:
            raise KeyError(f'FlowDivide level-3 group {flowdivide_level3_id} is not in metadata.json flowdivide_level3_groups')
        return dict(k[0])

    def upstream(self, pixel_seq_id):
        """Channel pixel ids upstream, including this pixel, as a Python half-open range."""
        pixel_seq_id = operator.index(pixel_seq_id)
        count = self.pixels(pixel_seq_id, 1, ['upstream_channel_pixel_count'])['upstream_channel_pixel_count'][0]
        return range(pixel_seq_id, pixel_seq_id + int(count))

    # ------------------------------------------------------------------------------------------------------- any threshold
    def _base_selection(self, basin_id, flowdivide_level3_id):
        """First/count of base reaches in whole basins of one basin or FlowDivide level-3 group."""
        if (basin_id is None) == (flowdivide_level3_id is None):
            raise ValueError('give basin_id or flowdivide_level3_id, exactly one')
        if basin_id is not None:
            return self.reach_range(basin_id)
        group = self.group(flowdivide_level3_id)
        first = int(group[f'first_reach_seq_id_{self.base}'])
        count = int(group[f'reach_count_{self.base}'])
        return (first, count) if count else None

    def _break_at(self, brk, a):
        """break_threshold_km2 held against a: at the base every base reach is a reach, the one-pixel basins of 0.05 to 1 km2 (whose
        break_threshold_km2 is their area, below the base) too; above it, they are in no network."""
        return np.maximum(brk, a) if self._is_base(a) else brk

    def _is_base(self, a):
        """Whether a threshold is the base one, compared in float32 as the C code does (1 + 1e-12 is the base)."""
        return _as_float32(a) == self._base32

    def _check_threshold(self, threshold):
        a = float(threshold)
        if not np.isfinite(a) or _as_float32(a) < self._base32:
            raise ValueError(f'threshold {threshold} is below the base threshold {self.thresholds[0]} or is not finite: '
                             'a finer network needs a new numbering')
        return a

    def _threshold_offsets(self, threshold, first):
        """Numbers of retained reaches and paths before first; stream the two base columns, cache the result."""
        key = (_as_float32(threshold), first)
        if key not in self._offsets:
            if first == 1:
                self._offsets[key] = 0, 0
                return self._offsets[key]
            if self._is_base(threshold):
                path = self.reaches(first, 1, columns=['path_seq_id'])['path_seq_id'][0]
                self._offsets[key] = first - 1, int(path) - 1
                return self._offsets[key]
            reach_count = path_count = 0
            parquet, _ = self._parquet(f'reach_{self.base}.parquet')
            start = 1
            for batch in parquet.iter_batches(columns=['break_threshold_km2', 'next_down_reach_seq_id']):
                count = min(batch.num_rows, first - start)
                if count <= 0:
                    break
                kept = batch.column(0).to_numpy()[:count] >= _as_float32(threshold)
                down = batch.column(1).to_numpy()[:count]
                reach_count += int(kept.sum())
                path_count += int(np.sum(kept & (down != np.arange(start - 1, start + count - 1))))
                start += count
                if start == first:
                    break
            self._offsets[key] = reach_count, path_count
        return self._offsets[key]

    def reach_at_threshold(self, threshold, basin_id=None, flowdivide_level3_id=None):
        """Return (first base reach, reach_seq_id_<T> array), as the crosswalk table.

        Entry i is the T reach whose unit catchment contains base reach first + i; zero means no network in its basin.
        T reach ids are sequential globally, matching reach_<T>. Returns None for no base reaches."""
        a = self._check_threshold(threshold)
        selection = self._base_selection(basin_id, flowdivide_level3_id)
        if selection is None:
            return None
        a32 = _as_float32(a)                             # compared in float32, as the C code does
        r = self.reaches(*selection, columns=['next_down_reach_seq_id', 'break_threshold_km2'])
        target = _reach_at_threshold(_local_down(r['next_down_reach_seq_id'], selection[0]),
                                     self._break_at(r['break_threshold_km2'], a32), a32)
        starts = np.flatnonzero(target == np.arange(target.size))
        offset, _ = self._threshold_offsets(a, selection[0])
        return selection[0], np.where(target >= 0, np.searchsorted(starts, target) + offset + 1, 0)

    def reaches_at(self, threshold, basin_id=None, flowdivide_level3_id=None):
        """The 15 columns of reach_<T> of the coarser-threshold package, in approved order, for whole basins.

        Read the supplied coarser-threshold table when present; otherwise derive it from the base reach and pixel tables.
        Ids and path ids are sequential globally at T. Pixel counts always refer to the base channel pixels."""
        from .core import _path_levels
        a = self._check_threshold(threshold)
        selection = self._base_selection(basin_id, flowdivide_level3_id)
        keys = COARSER_REACH_COLUMNS
        if selection is None:
            return _empty_reaches()
        first = selection[0]
        offset, path_offset = self._threshold_offsets(a, first)
        r = self.reaches(*selection)
        down = _local_down(r['next_down_reach_seq_id'], first)
        a32 = _as_float32(a)                             # compared in float32, as the C code does
        brk, upa = self._break_at(r['break_threshold_km2'], a32), r['upstream_area_km2']
        target = _reach_at_threshold(down, brk, a32)
        starts = np.flatnonzero(target == np.arange(target.size))
        if starts.size == 0:
            return _empty_reaches()
        label = label_of(a)
        if label != self.base and PREFIX + f'reach_{label}.parquet' in self._files:
            return self.reaches(offset + 1, starts.size, a, list(keys))
        end = _upstream_end_reaches(starts, down, upa, brk, a32)
        lo = r['downstream_end_pixel_seq_id'][end].astype(np.int64)
        counts = r['channel_pixel_count'][end]
        ids = np.repeat(lo - np.r_[0, np.cumsum(counts)[:-1]], counts) + np.arange(counts.sum())
        pixel_area = self.rows_at(f'pixel_{self.base}.parquet', ids, ['upstream_area_km2'])['upstream_area_km2']
        on = (pixel_area >= a32) | self._is_base(a)                 # every base channel pixel is on the base network
        above = np.add.reduceat(on.astype(np.int64), np.r_[0, np.cumsum(counts)[:-1]])
        rank = np.full(target.size, -1, np.int64)
        rank[starts] = np.arange(starts.size)
        down_start = np.where(down[starts] >= 0, target[np.maximum(down[starts], 0)], -1)
        down_rank = np.where(down_start >= 0, rank[np.maximum(down_start, 0)], -1)
        area = np.bincount(rank[target[target >= 0]], weights=r['unit_catchment_area_km2'][target >= 0], minlength=starts.size)
        downstream_end = r['downstream_end_pixel_seq_id'][starts].astype(np.int64)
        count = lo + above - downstream_end
        next_down_pixel = _next_down_pixels(r)[starts]
        length = self._reach_lengths(downstream_end, count, next_down_pixel)
        distance = np.zeros(starts.size)
        for k, below in enumerate(down_rank):
            if below >= 0:
                distance[k] = distance[below] + length[below]
        path_starts = (down_rank < 0) | (down_rank != np.arange(starts.size) - 1)
        values = (np.arange(offset + 1, offset + starts.size + 1), downstream_end, count,
                  r['upstream_channel_pixel_count'][starts],
                  np.where(down_rank >= 0, down_rank, np.arange(starts.size)) + offset + 1,
                  np.searchsorted(starts, starts + r['upstream_reach_count'][starts]) - np.arange(starts.size),
                  np.cumsum(path_starts) + path_offset, _path_levels(down_rank + 1), r['basin_id'][starts], upa[starts], area,
                  length, distance, starts + first, end - starts + 1)
        return dict(zip(keys, values))

    def _reach_lengths(self, first, count, next_down_pixel):
        """Sum D8 pixel steps in id order, rounding metres to three decimals as the release does."""
        out = np.zeros(first.size)
        for k, (start, size, below) in enumerate(zip(first, count, next_down_pixel)):
            ids = np.r_[below, np.arange(start, start + size)]
            pixels = self.rows_at(f'pixel_{self.base}.parquet', ids, ['row', 'col'])
            rows, cols = pixels['row'], pixels['col']
            lengths = _step_lengths(rows[1:], rows[:-1], cols[1:] != cols[:-1], self.geotransform, self.nrow,
                                    self.meta.get('earth_model', 'WGS84 ellipsoid'))
            if below == start:                   # no step below a basin outlet
                lengths[0] = 0
            total = 0.0
            for value in lengths:
                total += float(value)
            out[k] = float(f'{total:.3f}') / 1000.0
        return out

    def window_at(self, threshold, row_min, row_max, col_min, col_max):
        """A window of the reach raster at any threshold A, made from the base reach raster: +o a channel pixel of the reach at A
        named by its sequential reach_seq_id at A, -o a pixel draining first to it, 0 land with no reach at A (as in the
        base raster, and the basins with no network at A), nodata kept. A base channel pixel is channel at A when its
        upstream_area_km2 (pixel table) >= A. A window across 180 degrees has a negative col_min, as in window."""
        a = self._check_threshold(threshold)
        v = self.window(row_min, row_max, col_min, col_max, raster='reach').astype(np.int64)
        out = np.zeros(v.shape, np.int64)
        land = (v != self.nodata) & (v != 0)
        out[v == self.nodata] = self.nodata
        if not land.any():
            return out
        r = np.abs(v[land])
        ur, inv = np.unique(r, return_inverse=True)
        if self._is_base(a):
            name_of = ur                                 # at the base every base reach is a reach, of every basin
        else:
            basins = np.unique(self.rows_at(f'reach_{self.base}.parquet', ur, ['basin_id'])['basin_id'])
            name_of = np.zeros(ur.size, np.int64)
            for basin_id in basins[basins <= self.basin_file_count]:   # a basin under the base threshold has no reach above it
                first, c = self.reach_at_threshold(a, basin_id=int(basin_id))
                k = (ur >= first) & (ur < first + c.size)
                name_of[k] = c[ur[k] - first]
        name = name_of[inv]
        channel = v[land] > 0
        p = self.window(row_min, row_max, col_min, col_max, raster='pixel').astype(np.int64)[land][channel]
        if not (p > 0).all():
            raise ValueError('a channel pixel of the reach raster is not a channel pixel of the pixel raster')
        up, pinv = np.unique(p, return_inverse=True)
        pix = f'pixel_{self.base}.parquet'
        if 'upstream_area_km2' not in self._parquet(pix)[0].schema_arrow.names:
            raise KeyError(f'{pix} is missing the required v4 upstream_area_km2 column')
        pa_ = self.rows_at(pix, up, ['upstream_area_km2'])['upstream_area_km2'].astype(np.float64)[pinv]
        sign = -np.ones(name.size, np.int64)
        sign[np.flatnonzero(channel)[(pa_ >= _as_float32(a)) | self._is_base(a)]] = 1   # at the base every channel pixel is +
        out[land] = sign * name
        return out

    # ------------------------------------------------------------------------------------------------------- rasters
    def grid_columns(self, col_min, col_max):
        """The grid column of each column of a window (a window across 180 degrees has a negative col_min)."""
        return np.arange(col_min, col_max) % self.ncol

    def _raster(self, raster):
        if raster in (None, 'pixel'):
            return self.file(f'pixel_seq_id_{self.base}.tif')
        if raster == 'reach':
            return self.file(f'reach_seq_id_{self.base}.tif')
        return raster

    def window(self, row_min, row_max, col_min, col_max, raster=None):
        """A window of the pixel raster (+p channel pixel, -p hillslope pixel draining first to p, 0 land of the basins under
        0.05 km2 (not numbered), -2147483648 (nodata) sea), of raster='reach' (the base reach raster), or of any raster file on the same grid (MERIT
        Hydro's dir or upa), across 180 degrees in its two parts joined."""
        return read_window(self._raster(raster), row_min, row_max, col_min, col_max, (self.nrow, self.ncol), self.geotransform)

    def values_at(self, rows, cols, raster=None):
        """The values of the pixel raster (or raster='reach', or a raster file on the same grid) at scattered pixels, in their
        order. Columns run 0 .. ncol - 1, or -ncol .. -1 for the western part of a window across 180 degrees. Reads each tile that
        holds one of them once."""
        import rasterio
        from rasterio.windows import Window
        rows = np.asarray(rows, np.int64).ravel()
        cols = np.asarray(cols, np.int64).ravel()
        if rows.shape != cols.shape:
            raise ValueError('rows and cols must have the same length')
        if cols.size and not -self.ncol <= cols.min() <= cols.max() < self.ncol:
            raise ValueError(f'columns {cols.min()} .. {cols.max()} are not all columns of the grid (-{self.ncol} .. {self.ncol - 1})')
        cols = cols % self.ncol
        if rows.size and not 0 <= rows.min() <= rows.max() < self.nrow:
            raise ValueError(f'rows {rows.min()} .. {rows.max()} are not all rows of the grid (0 .. {self.nrow - 1})')
        path = self._raster(raster)
        with rasterio.open(path) as src:
            _check_grid(src, path, (self.nrow, self.ncol), self.geotransform)
            bh, bw = src.block_shapes[0]
            out = np.zeros(rows.size, src.dtypes[0])
            key = rows // bh * ((self.ncol + bw - 1) // bw) + cols // bw
            order = np.argsort(key, kind='stable')
            _, starts = np.unique(key[order], return_index=True)
            for s, e in zip(starts, np.r_[starts[1:], rows.size]):
                m = order[s:e]
                r0, c0 = int(rows[m[0]]) // bh * bh, int(cols[m[0]]) // bw * bw
                a = src.read(1, window=Window(c0, r0, min(bw, self.ncol - c0), min(bh, self.nrow - r0)))
                out[m] = a[rows[m] - r0, cols[m] - c0]
        return out

    def window_of(self, row, prefix='upstream_catchment_', raster=None):
        """Window of a reach_location row (upstream_catchment_ or unit_catchment_); use prefix='' for a basin or group row."""
        return self.window(int(row[prefix + 'row_min']), int(row[prefix + 'row_max']), int(row[prefix + 'col_min']),
                           int(row[prefix + 'col_max']), raster)

    def pixel_of_lonlat(self, lon, lat):
        """Zero-based row and column for scalar points or broadcastable arrays; floor at pixel edges, longitude wraps."""
        x0, dx, _, y0, _, dy = self.geotransform
        if np.ndim(lon) == 0 and np.ndim(lat) == 0:
            row, col = int(np.floor((lat - y0) / dy)), int(np.floor((lon - x0) / dx)) % self.ncol
            if not 0 <= row < self.nrow:
                raise ValueError(f'latitude {lat} is outside the grid')
            return row, col
        lon, lat = np.broadcast_arrays(np.asarray(lon, np.float64), np.asarray(lat, np.float64))
        row = np.floor((lat - y0) / dy)
        if not ((row >= 0) & (row < self.nrow)).all():
            raise ValueError(f'latitude {lat} is outside the grid')
        if not np.isfinite(lon).all():
            raise ValueError('longitude must be finite')
        col = np.floor((lon - x0) / dx).astype(np.int64) % self.ncol
        return row.astype(np.int64), col

    # ------------------------------------------------------------------------------------------------------- the rule again
    def number_basin(self, basin_id, d8_tif, upa_tif):
        """Number one basin again from the flow directions and upstream area (the walk of this package, not the release's numbers);
        the release gives only the window, the outlet pixel and the first number. It works on the catchment window cut from d8_tif and upa_tif (MERIT Hydro's dir and upa on the release grid). The first
        number is the basin's outlet_pixel_seq_id, so where the two agree every number is the release's. Returns the Numbering and
        the window (row_min, row_max, col_min, col_max). The whole window is held in memory: fine for a basin of up to about
        10^5 km2 (10^7 pixels); not for the Amazon."""
        from .core import number_grid
        b = self.basin(basin_id)
        w = (b['row_min'], b['row_max'], b['col_min'], b['col_max'])
        d8 = self.window(*w, raster=d8_tif)
        upa = self.window(*w, raster=upa_tif).astype(np.float64)
        o = self.pixels(b['outlet_pixel_seq_id'], 1, ['row', 'col'])
        outlet = (int(o['row'][0]) - w[0], (int(o['col'][0]) - w[2]) % self.ncol)
        import rasterio
        with rasterio.open(d8_tif) as src:
            nodata = int(src.nodata) if src.nodata is not None else 247   # MERIT Hydro: 247 sea, 0 river mouth, 255 inland sink
        n = number_grid(d8, upa, threshold=self.thresholds[0], outlets=[outlet], nodata=nodata, first=b['outlet_pixel_seq_id'],
                        outlets_are_channel_pixels=True)          # the release's basins of 0.05 to 1 km2 are their outlet pixel
        if n.grid_index.size != b['channel_pixel_count']:
            raise ValueError(f'basin {basin_id}: {n.grid_index.size} channel pixels numbered, the release has '
                             f'{b["channel_pixel_count"]}')
        return n, w

    def reach_windows(self, n, window, threshold, first_reach=1):
        """The windows and unit-catchment areas of every reach of a Numbering cut on window (the two from number_basin), as the
        release stores them: unit_catchment_* from the pixels of its two ranges, upstream_catchment_* from pixels upstream of its
        downstream end, unit_catchment_area_km2 the WGS84 areas of the unit catchment's pixels summed."""
        from .stats import pixel_areas_km2
        row_min, _, col_min, _ = window
        x0, dx, _, y0, _, dy = self.geotransform
        u = n.unit_catchments(threshold, first_reach)
        v = np.abs(n.raster).astype(np.int64).ravel()
        order = np.argsort(v, kind='stable')
        vs = v[order]
        width = n.raster.shape[1]
        area_of_row = pixel_areas_km2(y0 + row_min * dy, -dy, dx, n.raster.shape[0])[:, 0]

        def pixels(first, count):
            return order[np.searchsorted(vs, first):np.searchsorted(vs, first + count)]

        def box(c):
            r, k = c // width + row_min, c % width + col_min
            b = [int(r.min()), int(r.max()) + 1, int(k.min()), int(k.max()) + 1]
            if b[3] <= 0:                                # wholly west of column 0: plain grid columns
                b[2], b[3] = b[2] + self.ncol, b[3] + self.ncol
            return b

        m = u['reach_seq_id'].size
        out = {c: np.zeros(m, np.int64) for c in ('unit_catchment_row_min', 'unit_catchment_row_max', 'unit_catchment_col_min',
                                                  'unit_catchment_col_max', 'upstream_catchment_row_min', 'upstream_catchment_row_max',
                                                  'upstream_catchment_col_min', 'upstream_catchment_col_max')}
        out['unit_catchment_area_km2'] = np.zeros(m)
        upstream_counts = n.upstream_channel_pixel_count
        for j in range(m):
            c = np.concatenate([pixels(u['first_1'][j], u['count_1'][j]), pixels(u['first_2'][j], u['count_2'][j])])
            for name, x in zip(('row_min', 'row_max', 'col_min', 'col_max'), box(c)):
                out['unit_catchment_' + name][j] = x
            out['unit_catchment_area_km2'][j] = area_of_row[c // width].sum()
            a = u['first_1'][j]
            for name, x in zip(('row_min', 'row_max', 'col_min', 'col_max'), box(pixels(a, upstream_counts[a - n.first]))):
                out['upstream_catchment_' + name][j] = x
        return dict(reach_seq_id=u['reach_seq_id'], **out)


def _empty_reaches():
    floats = {'upstream_area_km2', 'unit_catchment_area_km2', 'length_km', 'distance_to_basin_outlet_km'}
    return {key: np.zeros(0, np.float32 if key == 'upstream_area_km2' else np.float64 if key in floats else np.int64)
            for key in COARSER_REACH_COLUMNS}


def _next_down_pixels(reaches):
    """Derive the downstream pixel from the downstream reach, with basin-outlet self links."""
    ids = reaches['reach_seq_id']
    below = np.searchsorted(ids, reaches['next_down_reach_seq_id'])
    downstream_end = reaches['downstream_end_pixel_seq_id']
    return np.where(ids == reaches['next_down_reach_seq_id'], downstream_end,
                    downstream_end[below] + reaches['channel_pixel_count'][below] - 1)


def _step_lengths(rows, below_rows, changes_column, geotransform, nrow, earth_model):
    """Release row-step distances in metres: authalic great-circle arc, with its local WGS84 bearing correction."""
    _, dx, _, y0, _, dy = geotransform
    latitude = y0 + (rows + 0.5) * dy
    adjacent = np.clip(rows + np.sign(below_rows - rows), 0, nrow - 1)
    below_latitude = y0 + (adjacent + 0.5) * dy
    lat1, lat2 = np.radians(latitude), np.radians(below_latitude)
    dlat, dlon = np.radians(below_latitude - latitude), np.radians(dx) * changes_column
    radius = 6371007.181
    hav = np.minimum(1.0, np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    distance = radius * 2 * np.arctan2(np.sqrt(hav), np.sqrt(1 - hav))
    if earth_model.startswith('sphere'):
        return distance
    if not (earth_model.startswith('WGS84') or earth_model.startswith('MERIT')):
        raise ValueError(f'unsupported release earth_model: {earth_model}')
    midpoint = (lat1 + lat2) / 2
    e2 = 0.0066943799901413165
    w = np.sqrt(1 - e2 * np.sin(midpoint) ** 2)
    meridian, prime_vertical = 6378137 * (1 - e2) / w ** 3, 6378137 / w
    numerator = np.sqrt((prime_vertical * np.cos(midpoint) * dlon) ** 2 + (meridian * dlat) ** 2)
    denominator = np.sqrt((radius * np.cos(midpoint) * dlon) ** 2 + (radius * dlat) ** 2)
    return distance * np.divide(numerator, denominator, out=np.ones_like(distance), where=denominator > 0)


def _local_down(next_down_reach_seq_id, first):
    """The local index (id - first) of the reach below each reach of a run of ids from first, -1 at a basin outlet (which points
    to itself in the release)."""
    nd = np.asarray(next_down_reach_seq_id, np.int64)
    return np.where(nd == np.arange(first, first + nd.size), -1, nd - first)


def _reach_at_threshold(down, brk, a):
    """For base reaches in id order (down: the local index of the reach below, negative at a basin outlet, always below the reach),
    the local index of the reach at threshold a each belongs to or drains into (its downstream-end base reach), -1 when none: r itself
    when brk[r] >= a, else that of the reach below, none at a basin outlet. By pointer jumping, each step halving the paths left."""
    i = np.arange(down.size, dtype=np.int64)
    if (down >= i).any():
        raise ValueError('a reach drains into a reach with a larger id or into itself')
    p = np.where(brk >= a, i, np.where(down >= 0, down, -1))
    while True:
        q = np.where(p >= 0, p[np.maximum(p, 0)], -1)
        if (q == p).all():
            return p
        p = q


def _upstream_end_reaches(o, down, upa, brk, a):
    """For downstream-end base reaches o at threshold a, find the upstream-end base reach along the larger branch (the next id),
    while it drains into this reach, is on the network (upa >= a) and is not a downstream end at a (brk < a)."""
    n = down.size
    step = np.zeros(n, bool)
    step[:-1] = (down[1:] == np.arange(n - 1)) & (upa[1:] >= a) & (brk[1:] < a)
    stop = np.flatnonzero(~step)                         # the final reach never steps
    return stop[np.searchsorted(stop, o)]


def read_window(path, row_min, row_max, col_min, col_max, shape=None, geotransform=None):
    """Rows row_min .. row_max - 1 and columns col_min .. col_max - 1 of a raster of the whole globe; a negative col_min is a window
    across 180 degrees, read as columns col_min + ncol .. ncol - 1 and then 0 .. col_max - 1; a window wholly west of column 0
    (col_max <= 0, as a unit window cut from a window across 180 degrees may be) is the same columns plus ncol. shape and
    geotransform, when given, must be the raster's (a raster on another grid is refused)."""
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(path) as src:
        _check_grid(src, path, shape, geotransform)
        nrow, ncol = src.height, src.width
        if not (0 <= row_min < row_max <= nrow and -ncol <= col_min < col_max <= ncol and col_max - col_min <= ncol):
            raise ValueError(f'rows {row_min} .. {row_max}, columns {col_min} .. {col_max} are not a window of the grid')
        if col_max <= 0:
            col_min, col_max = col_min + ncol, col_max + ncol
        h = row_max - row_min
        if col_min >= 0:
            return src.read(1, window=Window(col_min, row_min, col_max - col_min, h))
        west = src.read(1, window=Window(col_min + ncol, row_min, -col_min, h))
        east = src.read(1, window=Window(0, row_min, col_max, h))
    return np.hstack([west, east])


def _check_grid(src, path, shape, geotransform):
    """Refuse a raster on another grid than the release's."""
    if shape is not None and (src.height, src.width) != tuple(shape):
        raise ValueError(f'{path} is {src.height} x {src.width}, not the grid {shape[0]} x {shape[1]}')
    if geotransform is not None:                         # each corner may be a thousandth of a pixel away, not more
        t = src.transform
        x, g = np.array([t.c, t.a, t.b, t.f, t.d, t.e]), np.asarray(geotransform, np.float64)
        e = x - g
        for col, row in ((0, 0), (src.width, 0), (0, src.height), (src.width, src.height)):
            if (abs(e[0] + e[1] * col + e[2] * row) > 1e-3 * abs(g[1]) or
                    abs(e[3] + e[4] * col + e[5] * row) > 1e-3 * abs(g[5])):
                raise ValueError(f'{path} has the geotransform {tuple(t)[:6]}, not the grid {geotransform}')
