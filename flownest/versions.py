"""Moving between two numberings of the same grid.

A correction to the flow directions moves blocks of numbers. Old and new numbers then differ by a constant over long runs, so the
translation is a short table of runs rather than one row per pixel. A pixel of the new signed raster also answers for an old number
on its own: + means the place is still a channel, - gives the channel pixel the place now drains to.
"""
import numpy as np


def offset_table(old, new):
    """Runs of pixel_seq_id that move by one constant: first, count, offset; offset None means the pixels left the network."""
    if old.raster.shape != new.raster.shape:
        raise ValueError('the two numberings are not on the same grid')
    if old.grid_index.size == 0:
        return []
    size = old.ncol * old.raster.shape[0]
    place = np.zeros(size, np.int64)
    place[new.grid_index] = np.arange(new.first, new.first + new.grid_index.size)
    n_new = place[old.grid_index]
    o = np.arange(old.first, old.first + old.grid_index.size)
    gone = n_new == 0
    d = n_new - o
    cut = np.flatnonzero((d[1:] != d[:-1]) | (gone[1:] != gone[:-1])) + 1
    starts = np.r_[0, cut]
    stops = np.r_[cut, o.size]
    return [(int(o[a]), int(b - a), None if gone[a] else int(d[a])) for a, b in zip(starts, stops)]


def translate(table, number):
    """Old number to new number by the table; None if the pixel left the network or the number is not in the table."""
    if not table:
        return None
    lo = np.array([t[0] for t in table])
    i = np.searchsorted(lo, number, side='right') - 1
    if i < 0:
        return None
    first, count, off = table[i]
    if number >= first + count or off is None:
        return None
    return number + off


def read_through(old, new, number):
    """Old number to the value of the same pixel in the new raster: +p still a channel, -p now a hillslope pixel that drains to p."""
    i = number - old.first
    if not 0 <= i < old.grid_index.size:
        raise IndexError(f'{number} is not a number of the old numbering ({old.first} .. {old.first + old.grid_index.size - 1})')
    f = old.grid_index[i]
    return int(new.raster.ravel()[f])
