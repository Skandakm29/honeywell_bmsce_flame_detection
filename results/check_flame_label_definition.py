"""Read-only annotation/sample check; never changes the dataset or training."""
import collections
import hashlib
import io
import json
from pathlib import Path
import random
import zipfile
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
out = root/'results'
source = json.loads((root/'experiments/yolov9_flame/dataset_source.json').read_text())
report = {'paper': 'https://doi.org/10.1080/10095020.2024.2347922',
          'finding': 'Original authors explicitly label flame objects fire and smoke objects smoke.',
          'scope': 'Definition verification plus six reproducibly sampled training images; not an exhaustive annotation audit.',
          'splits': {}, 'sample_images': []}
with zipfile.ZipFile(root/'data/raw/dataset.zip') as archive:
    for split in ('train', 'val', 'test'):
        raw = archive.read(f'annotations/{split}.json')
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == source['annotation_sha256'][split]
        data = json.loads(raw)
        categories = {c['id']: c['name'] for c in data['categories']}
        counts = collections.Counter(categories[a['category_id']] for a in data['annotations'])
        report['splits'][split] = dict(categories=categories, boxes=counts, annotation_sha256=digest)
        if split != 'train':
            continue
        annotations = collections.defaultdict(list)
        for annotation in data['annotations']:
            annotations[annotation['image_id']].append(annotation)
        groups = collections.defaultdict(list)
        for image in data['images']:
            labels = frozenset(categories[a['category_id']] for a in annotations[image['id']])
            groups[labels].append(image)
        rng = random.Random(42)
        selected = []
        for labels in (frozenset(['fire']), frozenset(['fire', 'smoke']), frozenset(['smoke'])):
            selected.extend(rng.sample(groups[labels], 2))
        sheet = Image.new('RGB', (1280, 1080), '#181818')
        for index, image in enumerate(selected):
            picture = Image.open(io.BytesIO(archive.read('images/train/'+image['file_name']))).convert('RGB')
            draw = ImageDraw.Draw(picture)
            for annotation in annotations[image['id']]:
                x, y, w, h = annotation['bbox']
                label = categories[annotation['category_id']]
                color = '#ff3030' if label == 'fire' else '#00ffff'
                draw.rectangle((x, y, x+w, y+h), outline=color, width=max(2, picture.width//250))
            picture.thumbnail((630, 320))
            px, py = (index%2)*640, (index//2)*360
            sheet.paste(picture, (px+(640-picture.width)//2, py+30))
            ImageDraw.Draw(sheet).text((px+8, py+8), image['file_name'], fill='white')
            report['sample_images'].append(image['file_name'])
        sheet.save(out/'flame_label_samples.jpg', quality=92)
(out/'flame_label_definition_check.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
