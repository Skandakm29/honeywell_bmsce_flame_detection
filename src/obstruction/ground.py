"""Interface for physical floor area of blocked cells, to be filled in by Week 3 calibration.

Until calibration exists, NullProjector reports None and nothing pretends to be metres. A Week 3 projector maps
each blocked cell's image corners onto the floor plane (homography, flat floor assumed) and sums the areas, so the
value means "floor area hidden from the camera inside the zone".
"""


class NullProjector:
    calibrated = False

    def blocked_area_m2(self, zone_name, cells, grid):
        return None
