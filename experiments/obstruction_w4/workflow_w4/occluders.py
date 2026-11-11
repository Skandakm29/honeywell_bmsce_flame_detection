"""Synthetic camera obstructions for the `obstruction` class (week 4).

An obstruction is an object close to the lens that hides part of the view: a box, a steel sheet, a board, cloth,
a hand, a lens cover, or any defocused nearby object. Each one is described by a small JSON-able `spec`, so the
same obstruction can be re-rendered exactly (validation) or drawn at random (training).

  sample_spec(rng, cfg)        -> spec (geometry in 0..1 image coordinates, appearance parameters)
  hard_mask(spec, w, h)        -> uint8 0/1 mask of the hidden pixels (used for the label and the blind fraction)
  render(image, spec, donor)   -> image with the obstruction drawn on it (soft, defocused edge)
  label(spec, w, h)            -> (box xyxy in pixels, blind_fraction) or (None, 0.) if too small
  visible_flames(boxes, mask)  -> keep flame boxes that stay mostly visible

Only numpy and OpenCV. Textures are made at low resolution and resized, so rendering costs a few ms at 640 px.
"""
import math
import random

import cv2
import numpy as np

KINDS = ('cardboard', 'metal', 'plain', 'cloth', 'skin', 'dark', 'blurred_scene')
SHAPES = ('edge', 'floating', 'full')
TEXTURE_SIDE = 160          # textures are generated at this size and resized to the target


def _choice(rng, weights):
    names = list(weights)
    return rng.choices(names, [weights[n] for n in names])[0]


# ---------------------------------------------------------------- geometry
def _edge_polygon(rng):
    """Object entering from one side (or a corner). Canonical frame: enters from the left, then rotated."""
    depth = rng.uniform(.12, .95)                       # how far into the frame the object reaches
    if rng.random() < .6:
        y0, y1 = 0., 1.                                 # spans the whole side
    else:
        y0 = rng.uniform(0., .55)
        y1 = min(1., y0 + rng.uniform(.3, .9))
        if rng.random() < .5:                           # touch a corner
            y0, y1 = (0., y1) if rng.random() < .5 else (y0, 1.)
    slope = rng.uniform(-.35, .35)
    curve = rng.uniform(-.12, .12) if rng.random() < .5 else 0.
    wobble = .03 if rng.random() < .3 else .006         # irregular edge (hand, cloth) vs straight edge (box, sheet)
    points = []
    n = 9
    for i in range(n):
        y = y0 + (y1 - y0) * i / (n - 1)
        t = (y - y0) / max(1e-6, y1 - y0) - .5
        x = depth + slope * t + curve * (1 - 4 * t * t) + rng.uniform(-wobble, wobble)
        points.append([min(1., max(.02, x)), y])
    polygon = [[0., y0]] + points + [[0., y1]]
    side = rng.choice(('left', 'right', 'top', 'bottom'))
    mapping = {'left': lambda x, y: (x, y), 'right': lambda x, y: (1 - x, y),
               'top': lambda x, y: (y, x), 'bottom': lambda x, y: (y, 1 - x)}[side]
    return [list(mapping(x, y)) for x, y in polygon], side


def _floating_polygon(rng):
    """Object inside the view, not touching an edge (hanging board, hand, box on a shelf in front of the lens)."""
    cx, cy = rng.uniform(.25, .75), rng.uniform(.25, .75)
    w, h = rng.uniform(.18, .6), rng.uniform(.18, .6)
    angle = rng.uniform(-30, 30) * math.pi / 180
    if rng.random() < .65:                              # rectangle-like (box, board)
        corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        corners = [(x + rng.uniform(-.03, .03), y + rng.uniform(-.03, .03)) for x, y in corners]
    else:                                               # rounded (hand, bag, ball)
        corners = [(w / 2 * math.cos(a), h / 2 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    c, s = math.cos(angle), math.sin(angle)
    return [[cx + x * c - y * s, cy + x * s + y * c] for x, y in corners]


def sample_spec(rng, cfg=None):
    """Random obstruction. `rng` is a random.Random, so the result is reproducible from the seed."""
    cfg = cfg or {}
    shape = _choice(rng, cfg.get('shapes', {'edge': .6, 'floating': .25, 'full': .15}))
    kinds = dict(cfg.get('kinds', {'cardboard': 2, 'metal': 2, 'plain': 2, 'cloth': 1.5, 'skin': 1.5,
                                   'dark': 1, 'blurred_scene': 2}))
    if shape == 'full':
        kinds.pop('metal', None)                        # a sheet filling the whole view looks like 'plain'
    kind = _choice(rng, kinds)
    side = None
    if shape == 'edge':
        polygon, side = _edge_polygon(rng)
    elif shape == 'floating':
        polygon = _floating_polygon(rng)
    else:
        polygon = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
    return dict(kind=kind, shape=shape, side=side, polygon=[[round(x, 5), round(y, 5)] for x, y in polygon],
                feather=rng.uniform(.002, .02) if shape != 'full' else 0.,   # defocused edge, share of the diagonal
                defocus=rng.uniform(.004, .03),                               # texture blur, share of the diagonal
                brightness=rng.uniform(.55, 1.2),
                shade_angle=rng.uniform(0, 2 * math.pi), shade=rng.uniform(0., .35),
                hue=rng.random(), saturation=rng.random(), noise=rng.uniform(1.5, 5.),
                seed=rng.randrange(2 ** 31))


def hard_mask(spec, w, h):
    mask = np.zeros((h, w), np.uint8)
    # Pixel i covers [i, i+1); fillPoly works on pixel centres, so shift by half a pixel (x=0.5 of 300 px -> 150 px).
    points = np.array([[x * w - .5, y * h - .5] for x, y in spec['polygon']], np.float32)
    cv2.fillPoly(mask, [np.round(points * 8).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=3)
    return mask


def label(spec, w, h, min_fraction=.03):
    mask = hard_mask(spec, w, h)
    fraction = float(mask.mean())
    if fraction < min_fraction:
        return None, 0., mask
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], fraction, mask


def visible_flames(boxes, mask, hide_if_covered=.5):
    """Indices of flame boxes still visible: less than `hide_if_covered` of the box area is behind the obstruction."""
    keep = []
    h, w = mask.shape
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        a, b = int(max(0, math.floor(x1))), int(max(0, math.floor(y1)))
        c, d = int(min(w, math.ceil(x2))), int(min(h, math.ceil(y2)))
        if c <= a or d <= b or mask[b:d, a:c].mean() < hide_if_covered:
            keep.append(i)
    return keep


# ---------------------------------------------------------------- appearance
def _hsv_colour(h, s, v):
    pixel = np.uint8([[[int(h * 179), int(s * 255), int(v * 255)]]])
    return cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)


def _lowfreq(rng, shape, cells):
    small = rng.standard_normal((cells, cells)).astype(np.float32)
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def _texture(spec, tw, th, donor):
    """Base appearance at low resolution (th x tw x 3 float32, 0..255)."""
    rng = np.random.default_rng(spec['seed'])
    kind, shape = spec['kind'], (th, tw)
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    if kind == 'cardboard':
        base = _hsv_colour(rng.uniform(.06, .1), rng.uniform(.35, .6), rng.uniform(.45, .75))
        tex = base * (1 + .07 * _lowfreq(rng, shape, 6))[..., None]
        if rng.random() < .5:                                       # corrugation / fibre lines
            tex *= (1 + .03 * np.sin(xx * rng.uniform(.6, 1.4)))[..., None]
        if rng.random() < .4:                                       # packing tape band
            y = rng.integers(0, th)
            band = (np.abs(yy - y) < th * rng.uniform(.04, .1))
            tex[band] = tex[band] * .6 + 255 * .35
    elif kind == 'metal':
        grey = rng.uniform(.45, .85) * 255
        tex = np.full((th, tw, 3), grey, np.float32) * np.float32([1.04, 1., .97])
        brushed = cv2.blur(rng.standard_normal(shape).astype(np.float32), (25, 1))
        tex += (12 * brushed)[..., None]
        a = rng.uniform(0, math.pi)                                 # specular band
        d = (xx * math.cos(a) + yy * math.sin(a)) / max(tw, th)
        tex += (rng.uniform(25, 90) * np.exp(-((d - rng.uniform(.1, .9)) / rng.uniform(.05, .2)) ** 2))[..., None]
    elif kind == 'plain':
        tex = np.ones((th, tw, 3), np.float32) * _hsv_colour(spec['hue'], spec['saturation'] * .6, rng.uniform(.25, .95))
    elif kind == 'cloth':
        tex = np.ones((th, tw, 3), np.float32) * _hsv_colour(spec['hue'], .3 + .6 * spec['saturation'], rng.uniform(.25, .85))
        a = rng.uniform(0, math.pi)
        folds = np.sin((xx * math.cos(a) + yy * math.sin(a)) * rng.uniform(.05, .15) + .8 * _lowfreq(rng, shape, 4))
        tex *= (1 + .18 * folds)[..., None]
    elif kind == 'skin':
        tex = np.ones((th, tw, 3), np.float32) * _hsv_colour(rng.uniform(.02, .07), rng.uniform(.25, .55), rng.uniform(.35, .9))
        tex *= (1 + .08 * _lowfreq(rng, shape, 3))[..., None]
    elif kind == 'dark':
        tex = np.full((th, tw, 3), rng.uniform(2, 30), np.float32)
    elif kind == 'blurred_scene' and donor is not None:
        tex = cv2.resize(donor, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
        tex = cv2.GaussianBlur(tex, (0, 0), max(tw, th) * rng.uniform(.02, .06))
    else:                                                           # blurred_scene without a donor image
        tex = np.ones((th, tw, 3), np.float32) * _hsv_colour(spec['hue'], spec['saturation'] * .5, rng.uniform(.3, .8))
        tex *= (1 + .25 * _lowfreq(rng, shape, 3))[..., None]
    a = spec['shade_angle']                                         # lighting falls off across a near object
    ramp = ((xx / tw - .5) * math.cos(a) + (yy / th - .5) * math.sin(a))
    tex *= (spec['brightness'] * (1 + spec['shade'] * ramp))[..., None]
    sigma = spec['defocus'] * math.hypot(tw, th)
    if sigma > .3:
        tex = cv2.GaussianBlur(tex, (0, 0), sigma)
    return tex


def render(image, spec, donor=None):
    """Draw the obstruction on a BGR uint8 image (returned as a new array)."""
    h, w = image.shape[:2]
    scale = TEXTURE_SIDE / max(h, w)
    tw, th = max(8, round(w * scale)), max(8, round(h * scale))
    tex = cv2.resize(_texture(spec, tw, th, donor), (w, h), interpolation=cv2.INTER_LINEAR)
    small = hard_mask(spec, tw * 2, th * 2).astype(np.float32)      # soft edge computed at low resolution
    sigma = spec['feather'] * math.hypot(tw * 2, th * 2)
    if sigma > .3:
        small = cv2.GaussianBlur(small, (0, 0), sigma)
    alpha = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)   # ~0.5 on the labelled (hard) edge
    noise = np.random.default_rng(spec['seed'] + 1).standard_normal((h, w, 1)).astype(np.float32) * spec['noise']
    out = image.astype(np.float32) * (1 - alpha[..., None]) + (tex + noise) * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def spec_rng(seed):
    return random.Random(seed)
