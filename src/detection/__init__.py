"""Detection helpers that run around the flame model (no model weights here)."""
from .zoom import ZoomConfig, merge_boxes, crop_windows, select_candidates, detect_with_zoom
from .temporal import AlarmRule

__all__ = ['ZoomConfig', 'merge_boxes', 'crop_windows', 'select_candidates', 'detect_with_zoom', 'AlarmRule']
