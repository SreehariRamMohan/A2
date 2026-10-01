"""Figure 2: smaller self-supervised ViTs put more of their last-layer attention on the object.

For each image we take the last block's CLS->patch attention (mean over heads, nearest-upsampled to
224x224) and measure the fraction of attention mass inside the ground-truth box, and whether the
attention peak lands inside it.

  python localization.py --dataset waterbirds --data data/waterbirds --boxes data/CUB_200_2011 --models dinov2
  python localization.py --dataset imagenet --data data/imagenet/val --boxes data/imagenet/bboxes --models all
  python localization.py --plot                 # Figure 2 from results/localization.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from data import Localization

FIELDS = ['family', 'model', 'params_m', 'dataset', 'n_images', 'mass_mean', 'mass_std', 'hit_mean']
STYLE = {  # family: (color, marker, linestyle); the first six are Figure 2, the rest are appendix controls
    'DINOv1': ('#0891b2', 'v', '-'), 'DINOv2': ('#1f3a8a', 'o', '-'), 'DINOv3': ('#2f7d32', 'p', '-'),
    'MAE': ('#d97500', '^', '-'), 'iBOT': ('#7e3f9e', 'D', '-'), 'OpenCLIP': ('#b3262a', 's', '--'),
    'DINOv1 /8': ('#67c5d9', 'v', ':'), 'OpenAI CLIP': ('#e07b7e', 's', ':'), 'DeiT': ('#6b6b6b', 'x', ':'),
    'DeiT-III': ('#a3a3a3', 'P', ':'),
}


def mass_and_hit(attn, mask):
    total = float(attn.sum())
    if total < 1e-12:
        return 0.0, 0.0
    return float(attn[mask].sum()) / total, float(mask.flat[attn.argmax()])  # first peak on ties


def evaluate(name, args):
    model = models.load_model(name, args.device, args.weights)
    data = Localization(args.dataset, args.data, args.boxes, models.transform(name), args.limit)
    loader = DataLoader(data, batch_size=args.batch_size or models.batch_size(name), num_workers=args.workers)
    scores = []
    for images, masks in tqdm(loader, desc=name):
        attn = models.attention(model, images.to(args.device))
        scores += [mass_and_hit(a, m) for a, m in zip(attn, masks.numpy())]
    mass, hit = np.array(scores).T
    spec = models.MODELS[name]
    return dict(family=spec['family'], model=name, params_m=spec['params'], dataset=args.dataset, n_images=len(mass),
                mass_mean=f'{mass.mean():.6f}', mass_std=f'{mass.std():.6f}', hit_mean=f'{hit.mean():.6f}')


def save(path, row):
    """Add or replace one (dataset, model) row."""
    path = Path(path)
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    rows = [r for r in rows if (r['dataset'], r['model']) != (row['dataset'], row['model'])] + [row]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def plot(paths, metric, out, controls=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for p in paths for r in csv.DictReader(open(p))]
    families = list(STYLE) if controls else list(STYLE)[:6]
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4))
    for ax, (dataset, title) in zip(axes, [('waterbirds', 'Waterbirds'), ('imagenet', 'ImageNet val')]):
        for family in families:
            points = sorted((float(r['params_m']), 100 * float(r[f'{metric}_mean'])) for r in rows
                            if r['family'] == family and r['dataset'] == dataset)
            if points:
                color, marker, style = STYLE[family]
                ax.plot(*zip(*points), color=color, marker=marker, linestyle=style, linewidth=2.4, markersize=8,
                        markeredgecolor='white', markeredgewidth=0.8, label=family)
        ax.set(xscale='log', xlim=(17, 9000), ylim=(34, 86) if metric == 'mass' else (10, 100), title=title,
               xlabel='Attention Model Parameters (log scale)',
               ylabel='Attention Mass in GT Bbox (%)' if metric == 'mass' else 'Attention Hit Rate (%)')
        ax.set_xticks([21, 86, 300, 1100, 6716], ['21M', '86M', '300M', '1.1B', '6.7B'])
        ax.tick_params(axis='x', which='minor', length=0)
        ax.grid(alpha=0.25, linestyle='--', linewidth=0.7)
    axes[1].legend(loc='lower left', frameon=False, ncol=2, fontsize=9)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=300)
    print(f'saved {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['waterbirds', 'imagenet'])
    parser.add_argument('--data', help='data/waterbirds, or the flat ImageNet val image folder')
    parser.add_argument('--boxes', help='CUB_200_2011 folder (Waterbirds) or the ImageNet val XML folder')
    parser.add_argument('--models', nargs='+', default=['dinov2'], help="model names, family prefixes (e.g. dinov2), or 'all'")
    parser.add_argument('--weights', help='folder of local checkpoints (optional; see README)')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, help='first N images only (smoke test)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default='results/localization.csv')
    parser.add_argument('--plot', action='store_true', help='draw Figure 2 instead of evaluating')
    parser.add_argument('--csv', nargs='+', default=['results/localization.csv'], help='CSVs to plot')
    parser.add_argument('--controls', action='store_true', help='also plot the appendix control families')
    args = parser.parse_args()
    if args.plot:
        for metric in ('mass', 'hit'):
            plot(args.csv, metric, f"figures/fig2_{metric}{'_controls' if args.controls else ''}.pdf", args.controls)
        return
    if not (args.dataset and args.data and args.boxes):
        parser.error('--dataset, --data and --boxes are required (or use --plot)')
    names = [n for n in models.MODELS if 'all' in args.models or n in args.models or n.split('-')[0] in args.models]
    for name in names:
        row = evaluate(name, args)
        save(args.out, row)
        print(f"{name}: mass {100 * float(row['mass_mean']):.1f}%  hit {100 * float(row['hit_mean']):.1f}%  (n={row['n_images']})")
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
