"""A²: crop where a small ViT looks, embed the crops with a large ViT, and fit a linear probe.

  python a2.py --benchmark metashift-animals --data data/gqa/images      # ViT-S attention x ViT-G embedding
  python a2.py --benchmark waterbirds --data data/waterbirds \
      --attn dinov2-vits14 dinov2-vitb14 dinov2-vitl14 dinov2-vitg14       # where-to-look sweep
  python a2.py --benchmark waterbirds --data data/waterbirds --full-image  # full-image linear probe baseline
  python a2.py --plot                                                    # where-to-look figure from results/a2.csv

Benchmarks: waterbirds, spawrious-o2o-hard, spawrious-m2m-hard, metashift-catdog, metashift-animals.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

import models
from data import BENCHMARKS, Benchmark

FIELDS = ['benchmark', 'attn', 'emb', 'acc', 'worst_group']


def select_crops(attn, sizes=(64, 32)):
    """Greedily place non-overlapping squares (largest first), each covering the most attention.

    Every position is scored with an integral image; ties go to the first position in row-major order.
    """
    integral = np.zeros((attn.shape[0] + 1, attn.shape[1] + 1))
    integral[1:, 1:] = np.cumsum(np.cumsum(attn.astype(np.float64), 1), 0)
    crops = []
    for s in sizes:
        score = integral[s:, s:] - integral[:-s, s:] - integral[s:, :-s] + integral[:-s, :-s]
        y, x = np.ogrid[:score.shape[0], :score.shape[1]]
        for cx, cy, cs in crops:
            score[(x < cx + cs) & (x + s > cx) & (y < cy + cs) & (y + s > cy)] = -np.inf
        cy, cx = divmod(int(score.argmax()), score.shape[1])
        crops.append((cx, cy, s))
    return crops


class Views(Dataset):
    """The attention model's input and the 224x224 image that crops are cut from."""

    def __init__(self, benchmark, attention_transform):
        self.benchmark, self.transform = benchmark, attention_transform

    def __len__(self):
        return len(self.benchmark)

    def __getitem__(self, index):
        image = self.benchmark.image(index)
        return self.transform(image), np.array(image.resize((224, 224), Image.BICUBIC))


def a2_features(attention_model, embedding_model, benchmark, args):
    """Concatenated CLS embeddings of the attention-selected crops: [N, len(crops) * D]."""
    loader = DataLoader(Views(benchmark, models.transform(attention_model.name)), batch_size=args.batch_size,
                        num_workers=args.workers)
    to_tensor = models.normalize(embedding_model.name)
    features, boxes = [], []
    for inputs, bases in tqdm(loader, desc=f'{attention_model.name} x {embedding_model.name}'):
        crops = [select_crops(a, args.crops) for a in models.attention(attention_model, inputs.to(args.device))]
        views = torch.stack([to_tensor(Image.fromarray(base).crop((x, y, x + s, y + s)).resize((224, 224), Image.BILINEAR))
                             for base, image_crops in zip(bases.numpy(), crops) for x, y, s in image_crops])
        embeddings = np.concatenate([models.embed(embedding_model, v.to(args.device)) for v in views.split(args.batch_size)])
        features.append(embeddings.reshape(len(crops), -1))
        boxes += crops
    return np.concatenate(features), np.array(boxes)


def full_image_features(embedding_model, benchmark, args):
    benchmark.transform = models.transform(embedding_model.name, InterpolationMode.BICUBIC)
    loader = DataLoader(benchmark, batch_size=args.batch_size, num_workers=args.workers)
    features = [models.embed(embedding_model, x.to(args.device)) for x, _, _ in tqdm(loader, desc=f'full image x {embedding_model.name}')]
    benchmark.transform = None
    return np.concatenate(features), np.zeros((len(benchmark), 0, 3), dtype=int)


def fit_eval(train_x, train_y, test_x, test_y, test_groups):
    """Logistic regression on frozen features; accuracy and worst-group accuracy in percent."""
    correct = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42).fit(train_x, train_y).predict(test_x) == test_y
    groups = {int(g): 100 * correct[test_groups == g].mean() for g in np.unique(test_groups)}
    return 100 * correct.mean(), min(groups.values()), groups


def run(attention, embedding_model, splits, args):
    features, attention_model = {}, None
    for split, benchmark in splits.items():
        cache = Path(args.cache) / args.benchmark / f"{attention}__{args.emb}__{'-'.join(map(str, args.crops))}__{split}.npz"
        if cache.exists() and not args.limit:
            features[split] = np.load(cache)['features']
            continue
        if attention == 'full-image':
            x, boxes = full_image_features(embedding_model, benchmark, args)
        else:
            if attention_model is None:
                attention_model = embedding_model if attention == args.emb else models.load_model(attention, args.device, args.weights)
            x, boxes = a2_features(attention_model, embedding_model, benchmark, args)
        if not args.limit:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, features=x, boxes=boxes)
        features[split] = x
    return fit_eval(features['train'], splits['train'].labels, features['test'], splits['test'].labels, splits['test'].groups)


def save(path, row):
    """Add or replace one (benchmark, attn, emb) row."""
    path = Path(path)
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    key = lambda r: (r['benchmark'], r['attn'], r['emb'])
    rows = [r for r in rows if key(r) != key(row)] + [row]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, FIELDS, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def plot(paths, out, emb='dinov2-vitg14'):
    """Where to look: change in each benchmark's metric as the DINOv2 attention model grows (embedder fixed)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sizes = ['dinov2-vits14', 'dinov2-vitb14', 'dinov2-vitl14', 'dinov2-vitg14']
    params = [models.MODELS[s]['params'] for s in sizes]
    rows = {(r['benchmark'], r['attn']): r for p in paths for r in csv.DictReader(open(p)) if r['emb'] == emb}
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    curves = []
    for benchmark in BENCHMARKS:
        if not all((benchmark, s) in rows for s in sizes):
            continue
        metric = 'acc' if benchmark.startswith('spawrious') else 'worst_group'  # each benchmark's metric in the paper
        values = np.array([float(rows[benchmark, s][metric]) for s in sizes])
        curves.append(values - values[0])
        ax.plot(params, curves[-1], marker='o', linewidth=1.2, alpha=0.7, label=benchmark)
    if curves:
        ax.plot(params, np.mean(curves, axis=0), color='black', marker='o', linewidth=3, label='mean')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set(xscale='log', xlabel='Attention model parameters (DINOv2)', ylabel='Change vs. ViT-S attention (points)',
           title=f'Where to look (embedder: {emb})')
    ax.set_xticks(params, ['S', 'B', 'L', 'G'])
    ax.tick_params(axis='x', which='minor', length=0)
    ax.legend(frameon=False, fontsize=8)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=300)
    print(f'saved {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--benchmark', choices=BENCHMARKS)
    parser.add_argument('--data', help='data/waterbirds, data/spawrious, or data/gqa/images (MetaShift)')
    parser.add_argument('--attn', nargs='+', default=['dinov2-vits14'], help='attention model(s)')
    parser.add_argument('--emb', default='dinov2-vitg14', help='embedding model')
    parser.add_argument('--crops', type=int, nargs='+', default=[64, 32], help='crop sizes, largest first')
    parser.add_argument('--full-image', action='store_true', help='linear probe on the full image instead')
    parser.add_argument('--weights', help='folder of local checkpoints (optional; see README)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--limit', type=int, help='N evenly spaced rows per split (smoke test; not cached)')
    parser.add_argument('--cache', default='cache', help='feature cache folder')
    parser.add_argument('--out', default='results/a2.csv')
    parser.add_argument('--plot', action='store_true', help='draw the where-to-look figure instead')
    parser.add_argument('--csv', nargs='+', default=['results/a2.csv'], help='CSVs to plot')
    args = parser.parse_args()
    if args.plot:
        return plot(args.csv, 'figures/where_to_look.pdf', args.emb)
    if not (args.benchmark and args.data):
        parser.error('--benchmark and --data are required (or use --plot)')
    splits = {split: Benchmark(args.benchmark, split, args.data) for split in ('train', 'test')}
    if args.limit:
        for benchmark in splits.values():
            keep = np.linspace(0, len(benchmark) - 1, min(args.limit, len(benchmark))).astype(int)
            benchmark.paths = [benchmark.paths[i] for i in keep]
            benchmark.labels, benchmark.groups = benchmark.labels[keep], benchmark.groups[keep]
    embedding_model = models.load_model(args.emb, args.device, args.weights)
    for attention in ['full-image'] if args.full_image else args.attn:
        acc, worst, groups = run(attention, embedding_model, splits, args)
        save(args.out, dict(benchmark=args.benchmark, attn=attention, emb=args.emb, acc=f'{acc:.2f}', worst_group=f'{worst:.2f}'))
        print(f'{args.benchmark}  attn={attention}  emb={args.emb}:  accuracy {acc:.2f}  worst-group {worst:.2f}')
        print('  per group: ' + '  '.join(f'{g}:{a:.1f}' for g, a in groups.items()))
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
