"""Week 4 zone obstruction monitoring (image space; floor area once Week 3 calibration exists)."""
from .blindmap import BlindMap
from .confirm import ZoneAlarm
from .detector import ReferenceDetector
from .grid import CellTimers, ZoneGrid, cell_coverage
from .hidden import FireVisibility
from .ground import NullProjector
from .monitor import SceneMonitor
from .zones import Zone, ZoneConfig, load_zones, parse_zones, rasterise
