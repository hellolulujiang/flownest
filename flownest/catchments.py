"""Catchments of objects placed on channel pixels (gauges, SWORD reaches, SWORD nodes), by the ranges of FlowNest.

An object on channel pixel p holds the channel pixels [p, p + U(p)), U(p) the upstream channel pixel count of p. Numbered in pixel
order, the ranges nest. So the first object downstream of a pixel is the innermost range holding it (on a pixel that holds several
objects, the last of them), and the unit catchment of an object is the land whose first object downstream it is. Its upstream
catchment is the union of the unit catchments of the objects in its range: objects [seq, seq + upstream_count).

This is the rule of the release's crosswalks_with_catchments package; here it runs on any set of pixels or any
window of the pixel raster, that package's rasters being the same answer for the whole grid.
"""
import numpy as np

NODATA = -2147483648


def nest(pixel, end):
    """For objects in seq order (pixel rising), each holding [pixel, end): the first object downstream of each one's range
    (parent, seq 1..n; 0 when none) and its upstream count (the objects in its range, itself included). Raises ValueError when
    two ranges cross."""
    pixel, end = np.asarray(pixel, np.int64), np.asarray(end, np.int64)
    n = pixel.size
    if n and (np.diff(pixel) < 0).any():
        raise ValueError('objects must come in pixel order')
    parent = np.zeros(n, np.int64)
    stack = []
    for o in range(n):
        while stack and end[stack[-1]] <= pixel[o]:
            stack.pop()
        if stack:
            if end[o] > end[stack[-1]]:
                raise ValueError(f'the range of object {o + 1} crosses that of object {stack[-1] + 1}')
            parent[o] = stack[-1] + 1
        stack.append(o)
    # the objects in a range are those that follow in seq order up to the first one past its end (ranges nest)
    upstream_count = np.searchsorted(pixel, end, side='left') - np.arange(n)
    return parent, upstream_count


def first_downstream(pixels, pixel, end, parent):
    """seq of the first object downstream of each channel pixel id (the innermost range holding it; on a pixel holding several
    objects, the last), 0 when none."""
    pixels = np.asarray(pixels, np.int64)
    label = np.searchsorted(pixel, pixels, side='right')          # the last object whose pixel is not past this one: seq
    open_ = label > 0
    while open_.any():
        k = np.flatnonzero(open_)
        outside = pixels[k] >= end[label[k] - 1]
        label[k[outside]] = parent[label[k[outside]] - 1]          # step out to the next range around it
        open_[k[~outside]] = False
        open_ &= label > 0
    return label


def label_window(values, pixel, end, parent):
    """A window of the release's pixel raster relabelled by the first object downstream: +s on a channel pixel, -s on a hillslope
    pixel draining first to a channel pixel whose first object is s, 0 where there is none (and land of the basins under the
    minimum area), nodata kept."""
    values = np.asarray(values)
    out = np.zeros(values.shape, np.int32)
    out[values == NODATA] = NODATA
    numbered = (values != 0) & (values != NODATA)
    label = first_downstream(np.abs(values[numbered].astype(np.int64)), pixel, end, parent)
    out[numbered] = np.sign(values[numbered]) * label
    return out


def unit_windows(labels, row0=0, col0=0):
    """For a relabelled window: {seq: (row_min, row_max, col_min, col_max)} of the pixels of each object's unit catchment, in grid
    rows and columns (half-open), the window's corner at (row0, col0)."""
    s = np.abs(labels.astype(np.int64))
    s[labels == NODATA] = 0
    rows, cols = np.nonzero(s)
    seq = s[rows, cols]
    order = np.argsort(seq, kind='stable')
    seq, rows, cols = seq[order], rows[order], cols[order]
    if seq.size == 0:  # no numbered pixel in the window
        return {}
    first = np.r_[0, np.flatnonzero(np.diff(seq)) + 1]
    return {int(seq[a]): (int(rows[a:b].min()) + row0, int(rows[a:b].max()) + 1 + row0, int(cols[a:b].min()) + col0,
                          int(cols[a:b].max()) + 1 + col0)
            for a, b in zip(first, np.r_[first[1:], seq.size])}


def upstream_window(seq, upstream_count, unit):
    """The window of an object's upstream catchment: the union of the unit windows of the objects in its range."""
    boxes = [unit[s] for s in range(seq, seq + upstream_count) if s in unit]
    if not boxes:
        return None
    b = np.array(boxes)
    return int(b[:, 0].min()), int(b[:, 1].max()), int(b[:, 2].min()), int(b[:, 3].max())


def pair_gauges_with_nodes(gauge_pixel, gauge_area, gauge_end, node_pixel, node_area, node_id, node_end, node_parent):
    """Each gauge with one placed node, chosen of two: down, the first node downstream (the gauge's pixel included); up, of the node
    pixels upstream (the gauge's own pixel left out) the one of largest upstream area, the lowest pixel on a tie. The one whose
    area is nearer the gauge's by |ln(area / gauge area)| is taken, down on a tie; on the pixel taken, the node of smallest id.
    Nodes come in seq order (pixel rising). Returns side ('down', 'up', 'none'), the node's seq (0 when none) and its area ratio."""
    gauge_pixel, gauge_area = np.asarray(gauge_pixel, np.int64), np.asarray(gauge_area, np.float64)
    node_pixel, node_area = np.asarray(node_pixel, np.int64), np.asarray(node_area, np.float64)
    down = first_downstream(gauge_pixel, node_pixel, node_end, node_parent)
    # the distinct node pixels and their area
    first_on_pixel = np.r_[0, np.flatnonzero(np.diff(node_pixel)) + 1]
    pixels, areas = node_pixel[first_on_pixel], node_area[first_on_pixel]
    side = np.full(gauge_pixel.size, 'none', dtype=object)
    seq = np.zeros(gauge_pixel.size, np.int64)
    ratio = np.full(gauge_pixel.size, np.nan)
    for g in range(gauge_pixel.size):
        a, b = np.searchsorted(pixels, [gauge_pixel[g] + 1, gauge_end[g]], side='left')
        up_pixel = pixels[a + int(np.argmax(areas[a:b]))] if b > a else 0      # argmax: the first, lowest pixel on a tie
        up_area = areas[a:b].max() if b > a else np.nan
        down_area = node_area[down[g] - 1] if down[g] else np.nan
        down_log = abs(np.log(down_area / gauge_area[g])) if down[g] else np.inf
        up_log = abs(np.log(up_area / gauge_area[g])) if up_pixel else np.inf
        if up_log < down_log:
            side[g], chosen, area = 'up', up_pixel, up_area
        elif down[g]:
            side[g], chosen, area = 'down', node_pixel[down[g] - 1], down_area
        else:
            continue
        on = np.arange(*np.searchsorted(node_pixel, [chosen, chosen + 1]))          # the nodes on the chosen pixel
        seq[g] = on[np.argmin(node_id[on])] + 1
        ratio[g] = area / gauge_area[g]
    return side, seq, ratio
