"""A coarser view of a FlowNest number raster: factor x factor pixels make one cell (the 1/200 degree view of the
MERIT 90 m grid is factor 6).

The rule (the same as FlowDivide uses for its coarser views):
  - a pixel is land unless it is -2147483648; 0 is land with no object, +o and -o are object o;
  - a cell is land when at least a quarter of its pixels are land, otherwise it is -2147483648 in the view;
  - an object is a candidate of a land cell when it holds at least a quarter of the cell's land pixels;
  - the candidate holding the most pixels takes the cell, a tie to the smaller number;
  - a land cell no object takes is 0.

The window of an object is in the frame that keeps it narrowest: an object across 180 degrees has columns from below
0 (col_min < 0), as the windows of the release tables.
"""
import numpy as np
from numba import njit

NOT_LAND = -2147483648
MIN_LAND_PERCENT = 25
MIN_CANDIDATE_PERCENT = 25


@njit(cache=True)
def _fold_rows(fine, factor, object_count, coarse, cells, plain, shifted, crow0, totals):
    """Fold the fine rows (a whole number of cells high) into coarse rows from crow0 on.
    plain and shifted hold the windows [row_min, row_max, col_min, col_max]; totals is [land, taken, none]."""
    nr, nc = fine.shape
    cncol = nc // factor
    cell_pixels = factor * factor
    ids = np.empty(cell_pixels, np.int64)
    for i in range(nr // factor):
        crow = crow0 + i
        for ccol in range(cncol):
            land = 0
            n = 0
            for r in range(i * factor, (i + 1) * factor):
                for c in range(ccol * factor, (ccol + 1) * factor):
                    v = fine[r, c]
                    if v == NOT_LAND:
                        continue
                    land += 1
                    if v != 0:
                        ids[n] = v if v > 0 else -v
                        n += 1
            if land * 100 < cell_pixels * MIN_LAND_PERCENT:
                coarse[i, ccol] = NOT_LAND
                continue
            totals[0] += 1
            enough = (land * MIN_CANDIDATE_PERCENT + 99) // 100
            run = np.sort(ids[:n])
            winner = 0
            winner_pixels = 0
            k = 0
            while k < n:                            # ascending, so a tie stays with the smaller number
                j = k
                while j < n and run[j] == run[k]:
                    j += 1
                if j - k >= enough and j - k > winner_pixels:
                    winner = run[k]
                    winner_pixels = j - k
                k = j
            coarse[i, ccol] = winner
            if winner == 0:
                totals[2] += 1
                continue
            if winner > object_count:
                return winner                       # an object past the count
            totals[1] += 1
            o = winner - 1
            cells[o] += 1
            moved = ccol + cncol if ccol < cncol // 2 else ccol
            for w, col in ((plain, ccol), (shifted, moved)):
                if w[o, 1] == 0:
                    w[o, 0] = crow
                    w[o, 1] = crow + 1
                    w[o, 2] = col
                    w[o, 3] = col + 1
                else:
                    w[o, 0] = min(w[o, 0], crow)
                    w[o, 1] = max(w[o, 1], crow + 1)
                    w[o, 2] = min(w[o, 2], col)
                    w[o, 3] = max(w[o, 3], col + 1)
    return 0


class Fold:
    """What a fold gives. cells[o - 1] and window[o - 1] (row_min, row_max, col_min, col_max; half-open, all 0 when
    the object takes no cell) belong to object o."""

    def __init__(self, object_count, coarse_ncol):
        self.object_count = int(object_count)
        self.coarse_ncol = int(coarse_ncol)
        self.cells = np.zeros(self.object_count, np.int64)
        self._plain = np.zeros((self.object_count, 4), np.int64)
        self._shifted = np.zeros((self.object_count, 4), np.int64)
        self._totals = np.zeros(3, np.int64)
        self.window = None

    def _add(self, fine, factor, crow0):
        coarse = np.empty((fine.shape[0] // factor, fine.shape[1] // factor), np.int32)
        bad = _fold_rows(fine, factor, self.object_count, coarse, self.cells, self._plain, self._shifted, crow0,
                         self._totals)
        if bad:
            raise ValueError(f'object {bad} in the raster is past the {self.object_count} objects')
        return coarse

    def _finish(self):
        w = self._plain.copy()
        s = self._shifted
        narrower = (self.cells > 0) & (s[:, 3] - s[:, 2] < w[:, 3] - w[:, 2])
        w[narrower, 0:2] = s[narrower, 0:2]
        w[narrower, 2:4] = s[narrower, 2:4] - self.coarse_ncol
        self.window = w
        self.land_cells, self.cells_with_an_object, self.land_cells_with_no_object = (int(x) for x in self._totals)
        self.objects_with_cells = int(np.count_nonzero(self.cells))
        return self


def _check(shape, factor, object_count):
    if factor < 1 or shape[0] % factor or shape[1] % factor:
        raise ValueError(f'a raster of {shape[1]} x {shape[0]} cannot be folded by {factor}')
    if object_count < 0 or object_count > 2147483647:
        raise ValueError(f'{object_count} objects do not fit a 32-bit number')


def fold_array(fine, factor, object_count):
    """Fold a number raster held in memory. Returns (view, Fold)."""
    fine = np.ascontiguousarray(fine, np.int32)
    _check(fine.shape, factor, object_count)
    f = Fold(object_count, fine.shape[1] // factor)
    view = f._add(fine, factor, 0)
    return view, f._finish()


def fold_file(fine_path, factor, object_count, coarse_path=None, strip_cells=256):
    """Fold the GeoTIFF at fine_path, strip_cells coarse rows at a time, into a new tiled GeoTIFF at coarse_path (when
    given; nodata -2147483648). Returns the Fold. Needs rasterio."""
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(fine_path) as src:
        _check((src.height, src.width), factor, object_count)
        f = Fold(object_count, src.width // factor)
        dst = None
        if coarse_path is not None:
            profile = dict(driver='GTiff', width=src.width // factor, height=src.height // factor, count=1,
                           dtype='int32', nodata=NOT_LAND, crs=src.crs,
                           transform=src.transform * src.transform.scale(factor, factor),
                           tiled=True, blockxsize=512, blockysize=512, compress='deflate', predictor=2,
                           BIGTIFF='IF_SAFER')
            dst = rasterio.open(coarse_path, 'w', **profile)
        try:
            step = strip_cells * factor
            for row0 in range(0, src.height, step):
                rows = min(step, src.height - row0)
                fine = src.read(1, window=Window(0, row0, src.width, rows)).astype(np.int32, copy=False)
                view = f._add(np.ascontiguousarray(fine), factor, row0 // factor)
                if dst is not None:
                    dst.write(view, 1, window=Window(0, row0 // factor, view.shape[1], view.shape[0]))
        finally:
            if dst is not None:
                dst.close()
    return f._finish()
