"""Datasets for A²: the five robustness benchmarks and the two localization sets.

Downloads (see README "Data" for sizes and details):

  Waterbirds         hf download grodino/waterbirds --repo-type dataset \
                         --revision e9856c710d0da2e4029d116cdd9d5fce7cc2bc80 --local-dir data/waterbirds
  Spawrious          curl -L -o spawrious224.tar.gz \
                         'https://www.dropbox.com/s/hofkueo8qvaqlp3/spawrious224__entire_dataset.tar.gz?dl=1'
                     mkdir -p data/spawrious && tar -xzf spawrious224.tar.gz -C data/spawrious
  MetaShift          wget https://nlp.stanford.edu/data/gqa/images.zip && unzip images.zip -d data/gqa
  (Cat vs. Dog and Animals; images end up in data/gqa/images/<id>.jpg)
  Localization only  CUB-200-2011 (images.txt, bounding_boxes.txt) and ImageNet val images + box XMLs.

Benchmark rows come from manifests/<benchmark>_<split>.csv.gz: the exact images, labels and
worst-group ids behind the paper's numbers. Spawrious's own loader picks O2O training images in
unsorted directory order, so without the manifest a fresh copy would train on different images.
"""
import csv
import gzip
import io
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import warnings

import numpy as np
from PIL import Image, features
from torch.utils.data import Dataset

if not features.check_feature('libjpeg_turbo'):  # JPEG decoding differs slightly between libjpeg builds
    warnings.warn('Pillow is not built with libjpeg-turbo (the paper used it); install Pillow from pip for exact numbers.')

MANIFESTS = Path(__file__).resolve().parent / 'manifests'
BENCHMARKS = ['waterbirds', 'spawrious-o2o-hard', 'spawrious-m2m-hard', 'metashift-catdog', 'metashift-animals']
SIZE = 224


def _missing(path, what):
    return FileNotFoundError(f'{path} not found. See README "Data" for how to download {what}.')


def jpeg95(image):
    """Waterbirds images as the paper saw them: the parquet image saved as an RGB JPEG at quality 95."""
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, 'JPEG', quality=95)
    return Image.open(buffer).convert('RGB')


def waterbirds_images(root, split):
    """Encoded images of one Waterbirds split, in parquet row order."""
    import pyarrow.parquet as pq
    path = Path(root) / 'data' / f'{split}-00000-of-00001.parquet'
    if not path.is_file():
        raise _missing(path, 'Waterbirds')
    return pq.read_table(path, columns=['image']).column('image')


class Benchmark(Dataset):
    """One split of a benchmark in the paper's row order. Items are (image, label, group).

    root is the downloaded dataset: data/waterbirds, data/spawrious, or data/gqa/images (MetaShift).
    """

    def __init__(self, name, split, root, transform=None):
        if name not in BENCHMARKS:
            raise ValueError(f'unknown benchmark {name!r}; choose from {BENCHMARKS}')
        with gzip.open(MANIFESTS / f'{name}_{split}.csv.gz', 'rt', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.paths = [row['path'] for row in rows]
        self.labels = np.array([int(row['label']) for row in rows])
        self.groups = np.array([int(row['group']) for row in rows])
        self.name, self.root, self.transform = name, Path(root), transform
        self.parquet = waterbirds_images(root, split) if name == 'waterbirds' else None

    def __len__(self):
        return len(self.paths)

    def image(self, index):
        if self.parquet is not None:  # Waterbirds manifest paths are parquet row numbers
            return jpeg95(Image.open(io.BytesIO(self.parquet[int(self.paths[index])].as_py()['bytes'])))
        relative = self.paths[index] if self.name.startswith('spawrious') else f'{self.paths[index]}.jpg'
        path = self.root / relative
        if not path.is_file():
            raise _missing(path, self.name)
        return Image.open(path).convert('RGB')

    def __getitem__(self, index):
        image = self.image(index)
        return (self.transform(image) if self.transform else image), self.labels[index], self.groups[index]


class Localization(Dataset):
    """Waterbirds test (CUB boxes) or ImageNet val (XML boxes). Items are (image, 224x224 bool box mask).

    Box conventions follow the original evaluation: Waterbirds boxes are scaled to the 224x224 frame
    with floor/ceil; ImageNet boxes use Python round and are merged into their union.
    """

    def __init__(self, dataset, root, boxes, transform=None, limit=None):
        self.dataset, self.root, self.boxes, self.transform = dataset, Path(root), Path(boxes), transform
        if dataset == 'waterbirds':
            self.parquet = waterbirds_images(root, 'test')
            for name in ('images.txt', 'bounding_boxes.txt'):
                if not (self.boxes / name).is_file():
                    raise _missing(self.boxes / name, 'the CUB-200-2011 annotations')
            ids = dict(line.split()[::-1] for line in open(self.boxes / 'images.txt'))
            ids = {Path(path).name: image_id for path, image_id in ids.items()}
            cub = {line.split()[0]: tuple(map(float, line.split()[1:5])) for line in open(self.boxes / 'bounding_boxes.txt')}
            self.cub = [cub[ids[Path(item['path']).name]] for item in self.parquet.to_pylist()]
            n = len(self.cub)
        elif dataset == 'imagenet':
            n = 50000
        else:
            raise ValueError("dataset must be 'waterbirds' or 'imagenet'")
        self.n = min(n, limit or n)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        mask = np.zeros((SIZE, SIZE), dtype=bool)
        if self.dataset == 'waterbirds':
            image = jpeg95(Image.open(io.BytesIO(self.parquet[index].as_py()['bytes'])))
            (bx, by, bw, bh), (width, height) = self.cub[index], image.size
            sx, sy = SIZE / width, SIZE / height
            x, y, w, h = bx * sx, by * sy, bw * sx, bh * sy
            mask[max(0, math.floor(y)):min(SIZE, math.ceil(y + h)), max(0, math.floor(x)):min(SIZE, math.ceil(x + w))] = True
        else:
            stem = f'ILSVRC2012_val_{index + 1:08d}'
            path, xml = self.root / f'{stem}.JPEG', self.boxes / f'{stem}.xml'
            if not path.is_file() or not xml.is_file():
                raise _missing(path if not path.is_file() else xml, 'ImageNet val and its boxes')
            image = Image.open(path).convert('RGB')
            tree = ET.parse(xml).getroot()
            sx, sy = SIZE / int(tree.find('size/width').text), SIZE / int(tree.find('size/height').text)
            for box in tree.findall('object/bndbox'):
                x0, y0, x1, y1 = (int(float(box.find(k).text)) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
                if x1 > x0 and y1 > y0:
                    x0, y0, x1, y1 = round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy)
                    mask[max(0, y0):min(SIZE, y1), max(0, x0):min(SIZE, x1)] = True
        return (self.transform(image) if self.transform else image), mask
