"""Pretrained ViTs used in A²: load one by name, read its last-layer CLS attention, embed with its CLS token.

Weights are the official releases and download on first use (to $TORCH_HOME / $HF_HOME). DINOv3 is
gated: accept Meta's license on Hugging Face and run `hf auth login` first. To pre-fetch everything,
e.g. for compute nodes without internet:  python models.py --download all
"""
import argparse
import os
import types
from dataclasses import dataclass, field

os.environ.setdefault('XFORMERS_DISABLED', '1')  # DINOv2: plain PyTorch layers, as in the paper's runs

import torch
from torchvision import transforms as T

# Full float32 everywhere. PyTorch lets cuDNN use TF32 by default, which on A100/H100 perturbs the
# patch-embedding convolution by ~1e-3 and moves results by a few tenths of a point.
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

FB = 'https://dl.fbaipublicfiles.com'
IBOT = 'https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot'
DINOV2_HUB = os.environ.get('A2_DINOV2_HUB', 'facebookresearch/dinov2:b194f00db6136677fc8a4cc2ef2168f7699dfba2')
IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP_STATS = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def _spec(family, kind, source, weights, patch, params):
    return dict(family=family, kind=kind, source=source, weights=weights, patch=patch, params=params)


# name -> family label, loader kind, architecture / hub entry / HF repo / open_clip model, weights, patch, params (M)
MODELS = {
    'dinov1-vits16': _spec('DINOv1', 'dino', 'vit_small_patch16_224', f'{FB}/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth', 16, 21),
    'dinov1-vitb16': _spec('DINOv1', 'dino', 'vit_base_patch16_224', f'{FB}/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth', 16, 86),
    'dinov1-vits8': _spec('DINOv1 /8', 'dino', 'vit_small_patch8_224', f'{FB}/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth', 8, 21),
    'dinov1-vitb8': _spec('DINOv1 /8', 'dino', 'vit_base_patch8_224', f'{FB}/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth', 8, 86),
    **{f'dinov2-vit{s}14': _spec('DINOv2', 'dinov2', f'dinov2_vit{s}14_reg', f'{FB}/dinov2/dinov2_vit{s}14/dinov2_vit{s}14_reg4_pretrain.pth', 14, p)
       for s, p in [('s', 21), ('b', 86), ('l', 300), ('g', 1100)]},
    **{f'dinov3-vit{s}': _spec('DINOv3', 'dinov3', f'facebook/dinov3-vit{s}-pretrain-lvd1689m', None, 16, p)
       for s, p in [('s16', 21), ('s16plus', 29), ('b16', 86), ('l16', 300), ('h16plus', 840), ('7b16', 6716)]},
    'mae-vitb16': _spec('MAE', 'timm', 'vit_base_patch16_224.mae', None, 16, 86),
    'mae-vitl16': _spec('MAE', 'timm', 'vit_large_patch16_224.mae', None, 16, 307),
    'mae-vith14': _spec('MAE', 'timm', 'vit_huge_patch14_224.mae', None, 14, 632),
    **{f'ibot-vit{s}16': _spec('iBOT', 'dino', f'vit_{arch}_patch16_224', f'{IBOT}/vit{s}_16/checkpoint_teacher.pth', 16, p)
       for s, arch, p in [('s', 'small', 21), ('b', 'base', 85), ('l', 'large', 307)]},
    'clip-vitb16': _spec('OpenCLIP', 'clip', 'ViT-B-16', 'laion2b_s34b_b88k', 16, 86),
    'clip-vitl14': _spec('OpenCLIP', 'clip', 'ViT-L-14', 'laion2b_s32b_b82k', 14, 304),
    'clip-vith14': _spec('OpenCLIP', 'clip', 'ViT-H-14', 'laion2b_s32b_b79k', 14, 632),
    'clip-vitg14': _spec('OpenCLIP', 'clip', 'ViT-g-14', 'laion2b_s12b_b42k', 14, 1000),
    'clip-vitb16-oai': _spec('OpenAI CLIP', 'clip', 'ViT-B-16', 'openai', 16, 86),
    'clip-vitl14-oai': _spec('OpenAI CLIP', 'clip', 'ViT-L-14', 'openai', 14, 304),
    'deit-s16': _spec('DeiT', 'timm', 'deit_small_patch16_224.fb_in1k', None, 16, 22),
    **{f'deit3-{s}16': _spec('DeiT-III', 'timm', f'deit3_{arch}_patch16_224.fb_in1k', None, 16, p)
       for s, arch, p in [('s', 'small', 22), ('b', 'base', 87), ('l', 'large', 304)]},
}
BFLOAT16 = {'dinov3-vith16plus': 16, 'dinov3-vit7b16': 4}  # bfloat16 models and their batch sizes, as in the paper


@dataclass
class Backbone:
    name: str
    kind: str
    patch: int
    net: torch.nn.Module
    store: dict = field(default_factory=dict)


def batch_size(name):
    return BFLOAT16.get(name, 32)


def normalize(name):
    """PIL image -> normalized tensor, with the model's normalization statistics."""
    mean, std = CLIP_STATS if MODELS[name]['kind'] == 'clip' else IMAGENET_STATS
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def transform(name, resize=T.InterpolationMode.BILINEAR):
    """224x224 squash (no center crop), then the model's normalization."""
    return T.Compose([T.Resize((224, 224), interpolation=resize), normalize(name)])


def _state_dict(url, net, weights_dir):
    """Official checkpoint from weights_dir if present, else downloaded once to $TORCH_HOME."""
    file = url.split('/')[-2] + '_teacher.pth' if url.endswith('checkpoint_teacher.pth') else url.split('/')[-1]
    local = os.path.join(weights_dir, file) if weights_dir else None
    if local and os.path.isfile(local):
        state = torch.load(local, map_location='cpu', weights_only=False)
    else:
        state = torch.hub.load_state_dict_from_url(url, map_location='cpu', file_name=file, weights_only=False)
    state = {k.removeprefix('module.').removeprefix('backbone.'): v for k, v in state.get('state_dict', state).items()}
    missing = [k for k in net.state_dict() if k not in state]
    if missing:
        raise RuntimeError(f'{file} is missing weights such as {missing[:3]}')
    return {k: state[k] for k in net.state_dict()}


def _explicit_attention(self, x, *args, **kwargs):
    """DINOv2 attention exactly as in the paper's runs: softmax((q * scale) @ k^T) @ v, no fused kernel."""
    b, n, c = x.shape
    q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
    x = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1) @ v
    return self.proj(x.transpose(1, 2).reshape(b, n, c))


def load_model(name, device='cuda', weights_dir=None):
    """A frozen model whose forward pass records its last-block attention."""
    spec, store = MODELS[name], {}
    kind, source = spec['kind'], spec['source']
    if kind == 'dinov3':
        from transformers import AutoModel
        local = os.path.join(weights_dir, source.replace('/', '_')) if weights_dir else ''
        net = AutoModel.from_pretrained(local if os.path.isdir(local) else source, attn_implementation='eager',
                                        dtype=torch.bfloat16 if name in BFLOAT16 else torch.float32)
    elif kind == 'clip':
        import open_clip
        net = open_clip.create_model_and_transforms(source, pretrained=spec['weights'])[0].visual
        attention = net.transformer.resblocks[-1].attn  # nn.MultiheadAttention: ask it for per-head weights
        attention.register_forward_pre_hook(
            lambda m, args, kwargs: (args, {**kwargs, 'need_weights': True, 'average_attn_weights': False}), with_kwargs=True)
        attention.register_forward_hook(lambda m, args, out: store.update(attn=out[1]))
    else:
        if kind == 'dinov2':
            net = torch.hub.load(DINOV2_HUB, source, pretrained=False, trust_repo=True, skip_validation=True,
                                 source='local' if os.path.isdir(DINOV2_HUB) else 'github')
            # The paper built DINOv2 with the constructor defaults; the hub `_reg` entries override them.
            net.interpolate_antialias, net.interpolate_offset = False, 0.1
            for block in net.blocks:
                block.attn.forward = types.MethodType(_explicit_attention, block.attn)
        else:
            import timm
            local = os.path.join(weights_dir, source, 'model.safetensors') if weights_dir else ''
            overlay = dict(file=local) if os.path.isfile(local) else None  # offline copy of a timm checkpoint
            net = timm.create_model(source, pretrained=kind == 'timm', num_classes=0, pretrained_cfg_overlay=overlay)
            if kind == 'dino':  # the DINOv1 / iBOT reference code computes attention explicitly
                for block in net.blocks:
                    block.attn.fused_attn = False
        if spec['weights']:
            net.load_state_dict(_state_dict(spec['weights'], net, weights_dir))
        net.blocks[-1].attn.qkv.register_forward_hook(lambda m, args, out: store.update(qkv=out))
    return Backbone(name, kind, spec['patch'], net.to(device).eval().requires_grad_(False), store)


@torch.no_grad()
def attention(model, x):
    """Last-block CLS->patch attention averaged over heads, nearest-upsampled to the input: [B, 224, 224]."""
    grid = x.shape[-1] // model.patch
    model.store.clear()
    if model.kind == 'dinov3':
        attn = model.net(pixel_values=x.to(model.net.dtype), output_attentions=True).attentions[-1]
    else:
        model.net(x)
        attn = model.store.get('attn')
        if attn is None:  # recompute softmax(q k^T / sqrt(d)) from the hooked qkv projection
            qkv, block = model.store['qkv'], model.net.blocks[-1].attn
            b, n, _ = qkv.shape
            q, k, _ = qkv.reshape(b, n, 3, block.num_heads, -1).permute(2, 0, 3, 1, 4)
            if hasattr(block, 'q_norm'):
                q, k = block.q_norm(q), block.k_norm(k)
            attn = ((q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5).softmax(dim=-1)
    patches = attn[:, :, 0, -grid * grid:].float().cpu().numpy()  # patch tokens come after CLS and registers
    return patches.mean(1).reshape(-1, grid, grid).repeat(model.patch, 1).repeat(model.patch, 2)


@torch.no_grad()
def embed(model, x):
    """CLS embedding after the final norm (projected for CLIP): [B, D] float32."""
    if model.kind == 'dinov3':
        out = model.net(pixel_values=x.to(model.net.dtype)).pooler_output
    elif model.kind in ('dinov2', 'clip'):
        out = model.net(x)
    else:
        out = model.net.forward_features(x)[:, 0]
    return out.float().cpu().numpy()


def download(name):
    if MODELS[name]['kind'] == 'dinov3':
        from huggingface_hub import snapshot_download
        snapshot_download(MODELS[name]['source'])
    else:
        load_model(name, 'cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--download', nargs='+', metavar='NAME', help="model names, family prefixes (e.g. dinov2), or 'all'")
    args = parser.parse_args()
    if not args.download:
        for name, spec in MODELS.items():
            print(f"{name:20s} {spec['family']:12s} {spec['params']:>5}M  {spec['source']}")
    wanted = args.download or []
    unknown = [w for w in wanted if w != 'all' and not any(w in (n, n.split('-')[0]) for n in MODELS)]
    if unknown:
        parser.error(f'unknown model or family: {unknown}; run python models.py to list them')
    for name in [n for n in MODELS if 'all' in wanted or n in wanted or n.split('-')[0] in wanted]:
        print(f'downloading {name}', flush=True)
        download(name)
