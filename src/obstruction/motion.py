"""Camera movement against the learned empty scene (shift, rotation, zoom), for the zone monitor.

ORB features of the learned scene are matched to the live image and a similarity transform (shift, rotation,
scale) is fitted with RANSAC, so people and vehicles moving in the scene are ignored as outliers. If the camera has
moved by more than shift_px (analysis pixels, default 1.5 at 320 px wide, about 6 px at 1280), angle_deg or
scale_change, the learned scene no longer lines up and obstruction results would be misleading: the caller reports
VIEW CHANGED. When the camera returns (vibration that settles), the measurement drops below the limits again.
Too few matches (a plain or dark scene) means the movement is unknown, never "not moved" by assumption.

This only protects Week 4 from misleading alarms; tamper alarms and re-learning after a move belong to Week 5.
"""
import math

import cv2
import numpy as np


class CameraMotion:
    def __init__(self, shift_px=1.5, angle_deg=.5, scale_change=.01, check_s=.2, min_matches=15, features=600):
        self.shift_px, self.angle_deg, self.scale_change = shift_px, angle_deg, scale_change
        self.check_s, self.min_matches = check_s, min_matches
        self.orb = cv2.ORB_create(nfeatures=features, fastThreshold=10, edgeThreshold=15, patchSize=15)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.reset()

    def reset(self):
        self.reference = None
        self._last_check = None
        self.last = dict(moved=False, known=False, shift_px=None, angle_deg=None, scale=None, matches=0)

    def set_reference(self, grey):
        keypoints, descriptors = self.orb.detectAndCompute(grey, None)
        self.reference = (keypoints, descriptors)
        self._last_check = None

    def measure(self, grey):
        """Fitted movement of grey relative to the reference: dict(known, moved, shift_px, angle_deg, scale)."""
        ref_kp, ref_desc = self.reference
        kp, desc = self.orb.detectAndCompute(grey, None)
        if ref_desc is None or desc is None or len(ref_kp) < self.min_matches or len(kp) < self.min_matches:
            return dict(moved=False, known=False, shift_px=None, angle_deg=None, scale=None, matches=0)
        matches = self.matcher.match(ref_desc, desc)
        if len(matches) < self.min_matches:
            return dict(moved=False, known=False, shift_px=None, angle_deg=None, scale=None, matches=len(matches))
        src = np.float32([ref_kp[m.queryIdx].pt for m in matches])
        dst = np.float32([kp[m.trainIdx].pt for m in matches])
        matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.,
                                                      maxIters=500, confidence=.99)
        count = int(inliers.sum()) if inliers is not None else 0
        if matrix is None or count < self.min_matches:
            return dict(moved=False, known=False, shift_px=None, angle_deg=None, scale=None, matches=count)
        scale = math.hypot(matrix[0, 0], matrix[1, 0])
        angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
        # Movement of the image centre (shift + the rotation/scale part about the top-left origin).
        h, w = grey.shape[:2]
        centre = np.array([w / 2, h / 2, 1.])
        moved_centre = matrix @ centre
        shift = float(math.hypot(moved_centre[0] - w / 2, moved_centre[1] - h / 2))
        moved = shift > self.shift_px or abs(angle) > self.angle_deg or abs(scale - 1) > self.scale_change
        return dict(moved=moved, known=True, shift_px=shift, angle_deg=angle, scale=scale, matches=count)

    def update(self, grey, t):
        """grey: uint8 live image at the reference's size. Measures every check_s seconds; returns the latest."""
        if self.reference is None:
            return self.last
        if self._last_check is None or t - self._last_check >= self.check_s - 1e-9:
            self._last_check = t
            self.last = self.measure(grey)
        return self.last
