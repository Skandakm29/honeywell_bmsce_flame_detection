"""Blind-zone grid: per-cell blocked state, dwell timers in seconds, and blocked percentages per zone.

Every measurement is made on the UNION of all foreground, so overlapping objects are never counted twice.
Persistence is kept per cell, not per tracked object: a blob that splits, merges or changes track id does not
reset how long its cells have been blocked.
"""
import numpy as np

from .zones import rasterise


def cell_coverage(mask, cols, rows):
    """Share of each grid cell covered by a boolean pixel mask -> float [rows, cols] in 0..1."""
    h, w = mask.shape
    ys = np.linspace(0, h, rows + 1).round().astype(int)
    xs = np.linspace(0, w, cols + 1).round().astype(int)
    # Integral image: exact pixel counts for cells of any size, also when w or h is not divisible by the grid.
    integral = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    counts = (integral[np.ix_(ys[1:], xs[1:])] - integral[np.ix_(ys[:-1], xs[1:])]
              - integral[np.ix_(ys[1:], xs[:-1])] + integral[np.ix_(ys[:-1], xs[:-1])])
    area = np.outer(np.diff(ys), np.diff(xs))
    return counts / np.maximum(area, 1)


class ZoneGrid:
    """Zone and ignore polygons rasterised once for one analysis resolution."""
    def __init__(self, config, width, height):
        self.config, self.width, self.height = config, width, height
        self.cols, self.rows = config.grid
        ignore = np.zeros((height, width), bool)
        for polygon in config.ignore:
            ignore |= rasterise(polygon, width, height)
        self.ignore = ignore
        # weight[z] = share of each cell's pixels that belong to zone z (and are not ignored).
        self.weights = {z.name: cell_coverage(rasterise(z.polygon, width, height) & ~ignore, self.cols, self.rows)
                        for z in config.zones}
        empty = [name for name, w in self.weights.items() if w.sum() == 0]
        if empty:
            raise ValueError(f'zones {empty} cover no pixels outside the ignore polygons')

    def blocked_cells(self, foreground, min_cell_fraction=.5):
        """Cells where at least min_cell_fraction of the pixels are foreground (ignore polygons excluded)."""
        if foreground.shape != (self.height, self.width):
            raise ValueError(f'foreground mask must be {self.height}x{self.width}, got {foreground.shape}')
        return cell_coverage(foreground & ~self.ignore, self.cols, self.rows) >= min_cell_fraction

    def percent(self, cells):
        """Blocked share of every zone and of the whole frame, in percent, for a boolean cell array."""
        out = {name: float(100 * (w * cells).sum() / w.sum()) for name, w in self.weights.items()}
        out['_frame'] = float(100 * cells.mean())
        return out


class CellTimers:
    """How long each cell has been blocked, in seconds, tolerating detection gaps of up to gap_s seconds."""
    def __init__(self, shape, gap_s=1.5):
        if gap_s < 0:
            raise ValueError('gap_s must be >= 0')
        self.gap_s = gap_s
        self.since = np.full(shape, np.nan)      # time the current blocked spell started
        self.last = np.full(shape, np.nan)       # last time the cell was seen blocked

    def update(self, cells, t):
        """cells: boolean blocked flags now. Returns dwell seconds per cell (0 for cells that are free)."""
        expired = ~cells & ~(t - self.last <= self.gap_s)   # free now and the gap is too long (or never blocked)
        self.since[expired] = np.nan
        self.last[expired] = np.nan
        start = cells & np.isnan(self.since)
        self.since[start] = t
        self.last[cells] = t
        return np.where(np.isnan(self.since), 0., t - self.since)

    def reset(self):
        self.since[:] = np.nan
        self.last[:] = np.nan
