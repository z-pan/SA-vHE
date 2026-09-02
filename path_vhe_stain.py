#!/usr/bin/env python3
"""Virtually stain each TPAF patch in the manifest, whole, at native resolution.

Stain the patch, crop afterwards -- never the other way round. The generator is not
translation-equivariant: its InstanceNorm statistics come from whatever it is handed
and its receptive field is cut off at the border, so a stained crop and the same
rectangle taken out of a stained patch are different images. path_vhe_prepare.py
recovered the rectangles for exactly this reason; this produces the patches they get
applied to.

Model and conventions are taken from the last run of the existing pipeline
(08_results_overlap_vHE_nuc_hi.log), not guessed:

    checkpoint  set37_train_1channel_saliency_A65_B220, epoch 80, net G_A
    input       1 channel, BGR2GRAY of the TPAF RGB, scaled x/127.5 - 1
    output      3 channels, (y + 1) / 2 * 255, RGB
    generator   unet_256, instance norm, no dropout
    window      204 px of the patch, upscaled 2.510x to 512 with PIL BICUBIC

That last line is the one that is easy to get wrong and hard to see afterwards. The
model was trained on 204 px native tiles blown up to 512, i.e. on tissue at an
effective 0.247 um/px, and it draws nuclei at the size it learned in *window* pixels.
Feed it 512 native pixels instead and every nucleus comes out about 2.5x too big --
the output still looks like plausible H&E, which is exactly why it survives inspection
of the image alone and only shows up when it is put beside the real thing at a matched
scale. The 204 is not assumed: it is the coordinate stride of the existing abutting
patch set in 02_og_TPAF_gray_patches, and 512/204 = 2.510.

Output is brought back to the patch's native resolution, the convention
stitch_and_compare.py uses, so the crop boxes in the manifest apply unchanged and no
resolution is invented that the acquisition did not have.

Windows overlap and are cross-faded rather than abutted. Abutting is what produces the
block grid: each window is inferred alone, so each carries its own edge effects and its
own normalisation statistics, and where two meet those two independent errors collide
with nothing in between. Levelling per tile does not fix it -- the mismatch is not
constant within a tile -- but overlapping and feathering removes the line to fade into.

This replaces make_overlap_patches.py + test.py + stitch_and_compare.py for this one
job. Same model, same arithmetic, but the windows never touch disk: for 139 patches
that is a few thousand files saved, and no chance of the tiling and the stitching
disagreeing about geometry because they are the same loop.

    python path_vhe_stain.py                          # plain grayscale input
    python path_vhe_stain.py --mask_dir <masks>       # nuclei-enhanced input as well
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time

import cv2
import numpy as np
import torch

CKPT = 'checkpoints/set37_train_1channel_saliency_A65_B220/80_net_G_A.pth'


def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    """cv2.imread returns None for a path with non-ASCII characters on Windows, and the
    source folder is named 拼接FOV."""
    im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if im is None:
        raise SystemExit(f'cannot decode {path}')
    return im


def imwrite_u(path, im):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, im)
    if not ok:
        raise SystemExit(f'cannot encode {path}')
    buf.tofile(path)


def load_generator(ckpt, device):
    sys.argv = sys.argv[:1]
    from models.networks import define_G
    G = define_G(1, 3, 64, 'unet_256', 'instance', True, 'normal', 0.02, [])
    sd = torch.load(ckpt, map_location='cpu')
    if hasattr(sd, '_metadata'):
        del sd._metadata
    G.load_state_dict(sd)
    return G.to(device).eval()


def enhance_nuclei(gray, mask):
    """The additive boost from make_overlap_patches.py, kept bit-identical.

    The uint8 addition wraps rather than saturating. That is how every previous result
    in this project was produced, so it is reproduced rather than corrected -- changing
    it here would silently make these patches incomparable with the earlier ones. The
    number of wrapped pixels is returned so it is at least visible.
    """
    soft = cv2.GaussianBlur(mask, (11, 11), 0)
    soft = np.where(soft > 158, soft - 158, 0).astype(np.uint8)
    over = int((gray.astype(np.int32) + soft.astype(np.int32) > 255).sum())
    return (gray + soft), over


def enhance_nuclei_v2(gray, mask, amp=97.0, feather=1.0, ring_um=0.0, ring_amp=0.0,
                      mpp=0.621):
    """Flat-topped nuclei enhancement, optionally suppressing the surround.

    What is wrong with the legacy version. It blurs the 0/255 mask with an 11x11
    Gaussian and keeps whatever exceeds 158. At 0.621 um/px that kernel spans 6.8 um --
    wider than a whole nucleus -- so on the sizes that actually occur here (real H&E
    equivalent diameter p25 3.14, p50 3.83, p75 4.92, p90 6.26 um) it does this:

        3 um    0% of the nucleus enhanced, mean boost 0.0
        4 um   17%, mean 1.4, CV 2.59
        6 um   56%, mean 29.1, CV 1.06
       10 um   78%, mean 51.2, CV 0.74

    The median nucleus is not enhanced at all, and what is enhanced is a gradient
    rather than a plateau -- which the generator renders as chromatin texture, and is
    why enhanced nuclei carry 2.4x the within-nucleus spread of real H&E. It also
    boosts large nuclei far more than small ones, which would bias any nuclear
    morphometry computed downstream.

    This version takes the mask itself with a 1 px feather: 100% coverage at every
    size, CV 0.16-0.24.

    ring_amp > 0 additionally darkens a ring_um band outside each nucleus. The gap to
    real H&E is not in the nuclei -- their haematoxylin is already within 4% -- it is
    that the surround carries 1.8x too much. The generator responds monotonically to
    input level, so taking the ring down takes its haematoxylin down with it.

    Saturating, not wrapping. The legacy function relies on uint8 overflow; that is
    reproduced in enhance_nuclei for continuity with earlier results, but there is no
    reason to carry it into a new variant.
    """
    m = (mask > 127).astype(np.float32)
    if feather > 0:
        m = np.clip(cv2.GaussianBlur(m, (0, 0), feather), 0, 1)
    add = amp * m
    if ring_amp > 0 and ring_um > 0:
        k = max(3, int(round(2 * ring_um / mpp)) | 1)
        grown = cv2.dilate((mask > 127).astype(np.uint8),
                           np.ones((k, k), np.uint8)).astype(np.float32)
        if feather > 0:
            grown = np.clip(cv2.GaussianBlur(grown, (0, 0), feather), 0, 1)
        add = add - ring_amp * np.clip(grown - m, 0, 1)
    return np.clip(gray.astype(np.float32) + add, 0, 255).astype(np.uint8), 0


def stain(G, gray, device, tile=204, patch=512, stride=102, feather=48, batch=8,
          min_signal=0):
    """Tile at native scale, upscale each window to the model's scale, infer, come back.

    Windows overlap and are cross-faded. Abutting them is what produces the block grid:
    each is inferred alone, so each carries its own edge effects and its own
    InstanceNorm statistics, and where two meet those independent errors collide with
    nothing in between. Per-tile levelling does not fix it -- the mismatch is not
    constant within a tile -- but overlapping and feathering removes the line.
    """
    from PIL import Image

    h, w = gray.shape[:2]
    ph, pw = max(h, tile), max(w, tile)
    pad = np.zeros((ph, pw), np.uint8)
    pad[:h, :w] = gray
    ys = list(range(0, max(1, ph - tile) + 1, stride))
    xs = list(range(0, max(1, pw - tile) + 1, stride))
    if ys[-1] != ph - tile:
        ys.append(ph - tile)
    if xs[-1] != pw - tile:
        xs.append(pw - tile)

    ramp = np.ones((tile, tile), np.float32)
    f = max(1, min(feather, tile // 2))
    lin = np.linspace(0, 1, f, dtype=np.float32)
    ramp[:f, :] *= lin[:, None]
    ramp[-f:, :] *= lin[::-1, None]
    ramp[:, :f] *= lin[None, :]
    ramp[:, -f:] *= lin[None, ::-1]
    ramp = ramp[..., None] + 1e-3

    acc = np.zeros((ph, pw, 3), np.float32)
    wsum = np.zeros((ph, pw, 1), np.float32)
    coords = [(y, x) for y in ys for x in xs]
    if min_signal > 0:
        # A stitched slide is mostly empty -- 63% of the 240703 mosaic is background.
        # Inferring there costs a third of the run and is worse than useless: the
        # generator has never seen an all-zero field and answers with invented tissue.
        # Windows with no signal are left at zero, which the weight sum then treats as
        # never written.
        keep = [(y, x) for y, x in coords
                if pad[y:y + tile, x:x + tile].max() > min_signal]
        if len(keep) < len(coords):
            print(f'    {len(coords) - len(keep)}/{len(coords)} windows are background, '
                  f'skipped', flush=True)
        coords = keep
    if not coords:
        return np.zeros((h, w, 3), np.uint8), 0
    with torch.no_grad():
        for i in range(0, len(coords), batch):
            chunk = coords[i:i + batch]
            # PIL BICUBIC, matching make_overlap_patches.py, which was checked against
            # the existing patches at zero difference. cv2's cubic is not the same
            # filter and would put the input slightly off the training distribution.
            up = np.stack([
                np.asarray(Image.fromarray(pad[y:y + tile, x:x + tile])
                           .resize((patch, patch), Image.BICUBIC))
                for y, x in chunk])
            t = torch.from_numpy(up).float().div_(127.5).sub_(1.0)
            t = t.unsqueeze(1).to(device)
            out = G(t).add_(1.0).mul_(127.5).clamp_(0, 255)
            out = out.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            for (y, x), o in zip(chunk, out):
                down = np.asarray(Image.fromarray(o).resize((tile, tile), Image.BICUBIC))
                acc[y:y + tile, x:x + tile] += down.astype(np.float32) * ramp
                wsum[y:y + tile, x:x + tile] += ramp
    full = (acc / np.maximum(wsum, 1e-6)).round().clip(0, 255).astype(np.uint8)
    return full[:h, :w], len(coords)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', default=None,
                    help='Stain one image file instead of the manifest. For a stitched '
                         'slide mosaic: the mosaic must be the same dtype and channel '
                         'order as the source FOVs, or the BGR2GRAY below lands on a '
                         'different input distribution than the 148-region run used.')
    ap.add_argument('--enh', choices=('legacy', 'flat', 'signed'), default='legacy',
                    help='How the nuclei mask modifies the input. legacy reproduces '
                         'make_overlap_patches.py bit for bit, including its uint8 '
                         'wrap and its failure to touch nuclei under ~5 um. flat gives '
                         'every nucleus the same plateau. signed also darkens a band '
                         'around each nucleus, which is where the remaining gap to '
                         'real H&E sits.')
    ap.add_argument('--enh_name', default='nuc_hi',
                    help='Variant directory name, so a new enhancement lands beside '
                         'the old one instead of overwriting it.')
    ap.add_argument('--enh_amp', type=float, default=97.0,
                    help='Plateau height. 97 matches the legacy maximum, so a flat-vs-'
                         'legacy comparison isolates the shape.')
    ap.add_argument('--enh_feather', type=float, default=1.0)
    ap.add_argument('--enh_ring_um', type=float, default=4.0)
    ap.add_argument('--enh_ring_amp', type=float, default=25.0)
    ap.add_argument('--mask', default=None,
                    help='Binary nuclei mask the same size as --image. With it the '
                         'input becomes the nuclei-enhanced variant (the proposed '
                         'arm); without it, plain grey (the UTOM baseline). Build it '
                         'with path_mosaic_masks.py, which uses the same BGR2GRAY the '
                         'generator is fed.')
    ap.add_argument('--white_bg', action='store_true', default=True,
                    help='Set everything outside the TPAF tissue mask to white. On by '
                         'default: H&E background is white, and leaving it black makes '
                         'every standard tissue detector treat the whole slide as '
                         'tissue.')
    ap.add_argument('--no_white_bg', dest='white_bg', action='store_false')
    ap.add_argument('--bg_close', type=int, default=21,
                    help='Closing kernel on the tissue mask, to keep small gaps inside '
                         'tissue from being painted white.')
    ap.add_argument('--bg_grow', type=int, default=9,
                    help='Dilate the tissue mask before painting, so the mask never '
                         'eats into stained tissue at its border.')
    ap.add_argument('--min_signal', type=int, default=6,
                    help='Skip inference windows whose native tile never exceeds this. '
                         'Set 0 to stain the background too.')
    ap.add_argument('--manifest',
                    default='results/path_screen/survey/_vhe/vhe_manifest.csv')
    ap.add_argument('--out', default='results/path_screen/survey/_vhe/stained')
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--mask_dir', default=None,
                    help='Nuclei masks named <stage_name>.png, from Cellpose-SAM. With '
                         'them the input is nuclei-enhanced, which is what the last run '
                         'of this pipeline used; without, it is plain grayscale. Both '
                         'are written when masks are given, because the second pass '
                         'costs seconds and settles by eye which input to use.')
    ap.add_argument('--tile', type=int, default=204,
                    help='Native pixels per window. The training convention, recovered '
                         'from the coordinate stride of 02_og_TPAF_gray_patches. '
                         'Changing it changes the magnification the model sees, which '
                         'changes the size of everything it draws.')
    ap.add_argument('--patch', type=int, default=512,
                    help='Model input size; --tile is upscaled to this.')
    ap.add_argument('--stride', type=int, default=102,
                    help='Step between windows in native px. Half the tile = 50%% '
                         'overlap, as in the last run of the old pipeline.')
    ap.add_argument('--feather', type=int, default=48)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='Stain only the first N patches.')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    rows = list(csv.DictReader(io.open(args.manifest, encoding='utf-8-sig')))
    patches = {}
    for r in rows:
        patches.setdefault(r['patch_path'], r)
    items = sorted(patches.items())
    if args.limit:
        items = items[:args.limit]

    os.makedirs(args.out, exist_ok=True)
    variants = ['gray'] + ([args.enh_name] if args.mask_dir else [])
    for v in variants:
        os.makedirs(os.path.join(args.out, v), exist_ok=True)

    if args.image:
        os.makedirs(args.out, exist_ok=True)
        G = load_generator(args.ckpt, args.device)
        src = imread_u(args.image)
        if src is None:
            raise SystemExit(f'cannot read {args.image}')
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) if src.ndim == 3 else src
        print(f'{args.ckpt}  on {args.device}')
        print(f'{os.path.basename(args.image)}  {gray.shape[1]}x{gray.shape[0]}  '
              f'tile {args.tile} -> {args.patch} ({args.patch / args.tile:.3f}x, '
              f'{args.tile * 0.621 / args.patch:.3f} um/px into the model), '
              f'stride {args.stride}', flush=True)
        t0 = time.time()
        if args.mask:
            m = imread_u(args.mask, cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise SystemExit(f'cannot read {args.mask}')
            if m.shape != gray.shape:
                raise SystemExit(f'mask {m.shape} != mosaic {gray.shape}')
            if args.enh == 'legacy':
                gray, over = enhance_nuclei(gray, m)
            else:
                gray, over = enhance_nuclei_v2(
                    gray, m, args.enh_amp, args.enh_feather,
                    args.enh_ring_um if args.enh == 'signed' else 0.0,
                    args.enh_ring_amp if args.enh == 'signed' else 0.0)
            print(f'nuclei enhancement: mask covers '
                  f'{100*float((m>0).mean()):.1f}% of the mosaic; {over} px wrapped '
                  f'past 255 ({100*over/gray.size:.3f}%) -- uint8 addition wraps, '
                  f'inherited from make_overlap_patches.py, see enhance_nuclei',
                  flush=True)
        im, nw = stain(G, gray, args.device, args.tile, args.patch, args.stride,
                       args.feather, args.batch, args.min_signal)
        if args.white_bg:
            # Skipped windows are left at zero and the feathered edges of their
            # neighbours fade into that zero, so an unstained slide comes out black
            # with a grey halo. Real H&E background is white, and every off-the-shelf
            # tool that finds tissue does it by looking for what is darker than the
            # slide -- give it black and it calls the whole slide tissue. Mask on the
            # TPAF signal, not on the vHE, because the vHE halo is exactly the thing
            # being removed.
            sig = cv2.GaussianBlur(gray, (0, 0), 3.0) > args.min_signal
            k = np.ones((args.bg_close, args.bg_close), np.uint8)
            sig = cv2.morphologyEx(sig.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
            sig = cv2.dilate(sig.astype(np.uint8),
                             np.ones((args.bg_grow, args.bg_grow), np.uint8)) > 0
            im[~sig] = 255
            print(f'background: {100*float((~sig).mean()):.0f}% of the mosaic set to '
                  f'white on the TPAF tissue mask', flush=True)
        name = os.path.splitext(os.path.basename(args.image))[0]
        out = os.path.join(args.out, name + '_vhe.png')
        imwrite_u(out, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        dt = time.time() - t0
        print(f'{nw} windows in {dt:.0f}s ({1000*dt/max(nw,1):.0f} ms/window)')
        print(f'-> {out}')
        return

    G = load_generator(args.ckpt, args.device)
    print(f'{args.ckpt}  on {args.device}')
    print(f'{len(items)} patches, tile {args.tile} px native -> {args.patch} px '
          f'({args.patch / args.tile:.3f}x, {args.tile * 0.621 / args.patch:.3f} '
          f'um/px into the model), stride {args.stride}, feather {args.feather}')
    print(f'variants: {", ".join(variants)}')
    print()

    t0 = time.time()
    n_win = 0
    wrapped = []
    for i, (path, r) in enumerate(items, 1):
        name = os.path.splitext(r['stage_name'])[0] or f'patch{i:03d}'
        gray = cv2.cvtColor(imread_u(path), cv2.COLOR_BGR2GRAY)
        inputs = {'gray': gray}
        if args.mask_dir:
            mp = os.path.join(args.mask_dir, name + '.png')
            if not os.path.exists(mp):
                print(f'  {name}: no mask, gray only')
            else:
                m = imread_u(mp, cv2.IMREAD_GRAYSCALE)
                if m.shape != gray.shape:
                    raise SystemExit(f'{name}: mask {m.shape} != patch {gray.shape}')
                if args.enh == 'legacy':
                    enh, over = enhance_nuclei(gray, m)
                else:
                    enh, over = enhance_nuclei_v2(
                        gray, m, args.enh_amp, args.enh_feather,
                        args.enh_ring_um if args.enh == 'signed' else 0.0,
                        args.enh_ring_amp if args.enh == 'signed' else 0.0)
                inputs[args.enh_name] = enh
                if over:
                    wrapped.append((name, over, gray.size))
        for v, src in inputs.items():
            im, nw = stain(G, src, args.device, args.tile, args.patch,
                           args.stride, args.feather, args.batch)
            n_win += nw
            imwrite_u(os.path.join(args.out, v, name + '.png'),
                      cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        if i % 20 == 0 or i == len(items):
            print(f'  {i}/{len(items)}  {n_win} windows  {time.time() - t0:.0f}s')

    print(f'\n{len(items)} patches, {n_win} windows, {time.time() - t0:.0f}s')
    if wrapped:
        print(f'{len(wrapped)} patches had pixels wrap past 255 in the nuclei boost '
              '(inherited behaviour, see enhance_nuclei):')
        for name, over, tot in sorted(wrapped, key=lambda t: -t[1])[:5]:
            print(f'  {name}: {over} px ({100 * over / tot:.2f}%)')
    for v in variants:
        print(f'  -> {os.path.join(args.out, v)}')


if __name__ == '__main__':
    main()
