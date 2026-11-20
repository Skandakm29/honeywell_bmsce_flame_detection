"""Monitored-zone configuration: named polygons in normalised image coordinates (0..1).

Normalised coordinates keep one config valid at any camera or analysis resolution. Week 3 calibration can later
attach a floor projection to the same zones without changing this file format.

  {"grid": {"cols": 32, "rows": 24},
   "zones": [{"name": "battery rack", "polygon": [[0.1, 0.5], [0.6, 0.5], [0.6, 0.95], [0.1, 0.95]],
              "min_blocked_pct": 30, "persist_s": 10, "clear_s": 5}],
   "ignore": [[[0.8, 0.0], [1.0, 0.0], [1.0, 0.3], [0.8, 0.3]]]}

min_blocked_pct: zone alarm when at least this share of the zone is persistently blocked.
persist_s: a cell counts as persistently blocked once it has been blocked this long.
clear_s: the zone alarm ends after the persistent share stays below min_blocked_pct this long.
ignore: polygons whose pixels never count as blocked (doors, screens, windows).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

DEFAULT_GRID = (32, 24)  # cols, rows
MIN_AREA = 1e-4           # smallest polygon, as a share of the image (rounding can leave slivers)


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: tuple
    min_blocked_pct: float = 30.
    persist_s: float = 10.
    clear_s: float = 5.


@dataclass(frozen=True)
class ZoneConfig:
    zones: tuple
    ignore: tuple = ()
    grid: tuple = DEFAULT_GRID
    source: str = field(default='', compare=False)


def _polygon(points, where):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(f'{where}: a polygon needs at least 3 [x, y] points')
    if not np.isfinite(points).all() or points.min() < 0 or points.max() > 1:
        raise ValueError(f'{where}: polygon coordinates must be normalised to 0..1')
    if abs(cv2.contourArea(points.astype(np.float32))) < MIN_AREA:
        raise ValueError(f'{where}: polygon area is (almost) zero; it must cover at least {MIN_AREA:.2%} of the image')
    return tuple(map(tuple, points.tolist()))


def parse_zones(data, source=''):
    """Validate a zone config dict and return a ZoneConfig."""
    zones, names = [], set()
    for i, z in enumerate(data.get('zones', [])):
        name = str(z.get('name') or f'zone{i + 1}')
        if name in names:
            raise ValueError(f'duplicate zone name {name!r}')
        names.add(name)
        zone = Zone(name, _polygon(z.get('polygon'), name), float(z.get('min_blocked_pct', 30.)),
                    float(z.get('persist_s', 10.)), float(z.get('clear_s', 5.)))
        if not 0 < zone.min_blocked_pct <= 100:
            raise ValueError(f'{name}: min_blocked_pct must be in (0, 100]')
        if zone.persist_s < 0 or zone.clear_s < 0:
            raise ValueError(f'{name}: persist_s and clear_s must be >= 0')
        zones.append(zone)
    if not zones:
        raise ValueError('zone config needs at least one zone')
    grid = data.get('grid', {})
    grid = (int(grid.get('cols', DEFAULT_GRID[0])), int(grid.get('rows', DEFAULT_GRID[1])))
    if min(grid) < 1:
        raise ValueError('grid cols and rows must be >= 1')
    ignore = tuple(_polygon(p, f'ignore[{i}]') for i, p in enumerate(data.get('ignore', [])))
    return ZoneConfig(tuple(zones), ignore, grid, source)


def load_zones(path):
    path = Path(path)
    return parse_zones(json.loads(path.read_text(encoding='utf-8')), str(path))


def rasterise(polygon, width, height):
    """Boolean [height, width] mask of the pixels inside a normalised polygon."""
    mask = np.zeros((height, width), np.uint8)
    points = np.round(np.asarray(polygon) * [width, height]).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)
