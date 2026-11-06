"""Shared V10 helpers: paths, records, detector inference, CLIP, contact sheets."""
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
V9_SHA256 = '48d8e654ac1e3381e122a52cdfca100d1840e8cf6db0294ac882b52ab1c4a51f'
CLIP_MODEL = os.environ.get('V10_CLIP_MODEL', 'ViT-L-14')
CLIP_PRETRAINED = os.environ.get('V10_CLIP_PRETRAINED', 'laion2b_s32b_b82k')


def work_dir():
    path = Path(os.environ.get('V10_WORK', PACKAGE.parent / 'Honeywell-V10-work')).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False))
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def make_record(image, boxes=(), **extra):
    """Manifest record in the exact V9 format; extra keys (source, category, group) are ignored by training."""
    picture = cv2.imread(str(image))
    if picture is None:
        return None
    h, w = picture.shape[:2]
    clipped = []
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(0., x1), max(0., y1), min(float(w), x2), min(float(h), y2)
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            clipped.append([x1, y1, x2, y2])
    return dict(image=str(Path(image).resolve()), width=w, height=h, boxes=clipped,
                source_image_id=Path(image).stem, sha256=sha256(image), **extra)


def device():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_detector(checkpoint, expected_sha256=None):
    """Load an EMA checkpoint saved by train.py/train_v10.py into the TI YOLOv9 adapter."""
    import torch
    from model import Detector
    if expected_sha256 and sha256(checkpoint) != expected_sha256:
        raise RuntimeError(f'Checkpoint checksum mismatch: {checkpoint}')
    model = Detector(str(PACKAGE / 'tensorlab'))
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    return model.to(device()).eval()


def detect_records(model, records, size=640, batch=16, workers=4, floor=.2):
    """Per-record list of [x1, y1, x2, y2, score] in original pixels (same postprocess as evaluation)."""
    import torch
    from torch.utils.data import DataLoader
    from data import Flames, collate
    from metrics import postprocess
    dataset = Flames(records, size)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, collate_fn=collate, num_workers=workers)
    results = [None] * len(records)
    dev = next(model.parameters()).device
    with torch.inference_mode():
        for done, (tensors, _, indices, scales) in enumerate(loader):
            outputs = model(tensors.to(dev))
            for output, index, scale in zip(outputs, indices, scales):
                results[index] = postprocess(output, records[index], scale, floor)
            if done % 200 == 0:
                print(f'detect {min(len(records), (done+1)*batch)}/{len(records)}', flush=True)
    return results


def frame_tensor(frame, size=640):
    """BGR frame -> (1,3,size,size) tensor and scale, identical to data.Flames without augmentation."""
    import torch
    from data import letterbox
    image, _, scale = letterbox(frame, [], size)
    tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))).float() / 255.
    return tensor.unsqueeze(0), scale


def detect_frames(model, frames, size=640, floor=.25):
    import torch
    from metrics import postprocess
    tensors, scales = zip(*(frame_tensor(f, size) for f in frames))
    dev = next(model.parameters()).device
    with torch.inference_mode():
        outputs = model(torch.cat(tensors).to(dev))
    return [postprocess(o, dict(width=f.shape[1], height=f.shape[0]), s, floor)
            for o, f, s in zip(outputs, frames, scales)]


class Clip:
    """OpenCLIP wrapper (same model family as the week-2 embeddings). Set V10_FAKE_CLIP=1 only for CPU plumbing tests."""

    def __init__(self):
        self.fake = os.environ.get('V10_FAKE_CLIP') == '1'
        if self.fake:
            return
        import open_clip
        import torch
        self.torch = torch
        self.dev = device()
        self.model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=self.dev)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.mean = torch.tensor(open_clip.OPENAI_DATASET_MEAN, device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(open_clip.OPENAI_DATASET_STD, device=self.dev).view(1, 3, 1, 1)

    def images(self, pictures, batch=64):
        """BGR uint8 arrays (any size) -> L2-normalised float32 embeddings."""
        if self.fake:
            out = []
            for p in pictures:
                h = cv2.calcHist([cv2.resize(p, (64, 64))], [0, 1, 2], None, [4, 4, 4], [0, 256] * 3).ravel()
                out.append(h / (np.linalg.norm(h) + 1e-9))
            return np.asarray(out, dtype=np.float32)
        torch = self.torch
        chunks = []
        for start in range(0, len(pictures), batch):
            x = np.stack([cv2.resize(p, (224, 224), interpolation=cv2.INTER_AREA)[:, :, ::-1] for p in pictures[start:start+batch]])
            x = torch.from_numpy(x).to(self.dev).permute(0, 3, 1, 2).float().div_(255)
            with torch.no_grad(), torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.dev.type == 'cuda'):
                e = self.model.encode_image((x - self.mean) / self.std)
            chunks.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(chunks) if chunks else np.zeros((0, 768), np.float32)

    def texts(self, prompts):
        if self.fake:
            rng = np.random.default_rng(len(prompts))
            t = rng.normal(size=(len(prompts), 64)).astype(np.float32)
            return t / np.linalg.norm(t, axis=1, keepdims=True)
        torch = self.torch
        with torch.no_grad():
            t = self.model.encode_text(self.tokenizer(prompts).to(self.dev)).float()
        return torch.nn.functional.normalize(t, dim=-1).cpu().numpy()


FIRE_PROMPTS = ['a photo of fire', 'flames', 'a burning fire', 'a small flame', 'a candle flame',
                'a lighter flame', 'a fire burning in a room', 'a burning building', 'a campfire']
NUISANCE_PROMPTS = {
    'sunlight': ['sunlight through a window', 'a bright sun patch on the floor', 'sun glare in a camera'],
    'sunset': ['a sunset sky', 'an orange evening sky'],
    'lamp_bulb': ['a glowing lamp', 'a light bulb', 'a warm desk lamp'],
    'led_neon': ['orange LED lights', 'a neon sign', 'christmas lights'],
    'vehicle_lights': ['car headlights at night', 'red tail lights of a car'],
    'screen': ['a TV screen', 'a computer monitor', 'a phone screen'],
    'reflection': ['a reflection of light on metal', 'light reflected on a shiny floor'],
    'hi_vis_clothing': ['a person wearing a high visibility vest', 'orange clothing'],
    'orange_object': ['an orange object', 'a traffic cone', 'orange fruit'],
    'welding_sparks': ['welding sparks', 'grinding sparks', 'fireworks'],
    'heater': ['a glowing electric heater', 'red hot heater coils'],
    'other': ['an ordinary indoor scene', 'an ordinary outdoor scene', 'a person'],
}


class FireScorer:
    """Zero-shot P(fire) over fire prompts vs nuisance prompts; also returns the closest nuisance name."""

    def __init__(self, clip):
        self.clip = clip
        self.names = ['fire'] * len(FIRE_PROMPTS) + [k for k, v in NUISANCE_PROMPTS.items() for _ in v]
        self.text = clip.texts(FIRE_PROMPTS + [p for v in NUISANCE_PROMPTS.values() for p in v])

    def score(self, embeddings):
        logits = 100. * embeddings @ self.text.T
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        fire = p[:, :len(FIRE_PROMPTS)].sum(1)
        rest = p[:, len(FIRE_PROMPTS):].argmax(1) + len(FIRE_PROMPTS)
        return fire, [self.names[i] for i in rest]


def crop(image, box, expand=1.5, minimum=48):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box[:4]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(minimum, max(x2 - x1, y2 - y1) * expand) / 2
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, d = int(min(w, cx + half)), int(min(h, cy + half))
    return image[b:max(d, b+1), a:max(c, a+1)]


def contact_sheet(items, path, cols=6, tile=256):
    """items: (image_path or BGR array, box or None, caption). Box is drawn in red; crop expands around it."""
    rows = []
    for start in range(0, len(items), cols):
        row = []
        for source, box, caption in items[start:start+cols]:
            image = cv2.imread(str(source)) if not isinstance(source, np.ndarray) else source.copy()
            if image is None:
                image = np.zeros((tile, tile, 3), np.uint8)
            if box is not None:
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255),
                              max(2, image.shape[1] // 300))
                image = crop(image, box, expand=2.5, minimum=160)
            scale = tile / max(image.shape[:2])
            image = cv2.resize(image, (max(1, int(image.shape[1]*scale)), max(1, int(image.shape[0]*scale))))
            canvas = np.full((tile + 22, tile, 3), 255, np.uint8)
            canvas[:image.shape[0], :image.shape[1]] = image
            cv2.putText(canvas, str(caption)[:34], (3, tile + 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
            row.append(canvas)
        while len(row) < cols:
            row.append(np.full((tile + 22, tile, 3), 255, np.uint8))
        rows.append(np.hstack(row))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
