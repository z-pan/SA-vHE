#!/usr/bin/env python3
"""UTOM inference with dither noise, to suppress the periodic grid artifact.

Why this exists
---------------
The vH (haematoxylin) generator paints a regular ~64 px lattice of fake nuclei.
It is not in the data: feeding the generator a perfectly *uniform* image (all
black, all grey, all white) still produces the lattice, with a 64 px
autocorrelation of 0.51-0.69.

Cause: ``netG=unet_256`` is run at 512 px, so its transposed-convolution decoder
emits a periodic response that each upsampling stage doubles (2->4->...->64), and
``norm=instance`` divides every feature map by its own standard deviation. On a
near-empty input that standard deviation is ~0, so padding effects and the
transposed-conv ripple get amplified into the dominant signal. This is why the
artifact is strongest exactly where there is least tissue (correlation between
tissue fraction and artifact strength: -0.39 over 112 patches).

Fix: give InstanceNorm some real variance to normalise against. A little Gaussian
dither on the input does exactly that, and collapses the 64 px autocorrelation.

Dither everywhere and the artifact goes away but so does some of the real signal,
because this generator is very sensitive to its input. Dither is therefore
weighted by local flatness (``--adaptive``, the default) so it lands where the
artifact lives and leaves textured tissue alone -- 20 emptiest vs 20 most
tissue-rich patches:

    scheme              empty artifact   empty r    tissue artifact   tissue r
    no dither                 0.219        0.547         0.030          0.491
    global   sigma=0.01       0.067        0.389         0.020          0.385
    global   sigma=0.03       0.016        0.234         0.003          0.210
    adaptive sigma=0.03       0.065        0.387         0.023          0.431

(``r`` = correlation between output and the TPAF input, i.e. how much of the real
structure survives.) Adaptive matches global sigma=0.01 on artifact suppression
while keeping more tissue signal. Use ``--global_dither`` to force uniform dither.

One side effect worth knowing: without dither the vH output has *negative*
nuclear contrast (nuclei -2.6 darker than their surroundings), which is backwards
for a haematoxylin channel. Dither flips it to the correct sign (+2.2 to +2.8).

IMPORTANT -- what dither does *not* fix
--------------------------------------
It removes the *periodicity*, not the artifact energy. The generator turns the
dither into speckle, so empty regions end up noisier than before. Over the 15
emptiest patches, measuring the output inside genuinely blank input areas:

    scheme              64px artifact   background std   background high-freq
    no dither                 0.230           16.1              43.8
    adaptive sigma=0.005      0.108           29.9             160.6
    adaptive sigma=0.01       0.086           32.9             223.9
    adaptive sigma=0.03       0.062           36.3             327.7

Every setting trades the grid for random noise; there is no sweet spot that gives
both. So use this when *periodic* structure is what hurts -- e.g. it would bias
FFT/texture analysis, or a downstream model keys on the lattice. For visual
cleanliness, masking the background out (as ``remove_background_for_vH_patches.py``
does) is the better tool, and the two can be combined.

The root fix is architectural (``resnet_9blocks``, or replacing
``ConvTranspose2d`` with ``Upsample``+``Conv``), which needs retraining.

Usage
-----
    python inference_with_noise.py \\
        --checkpoint checkpoints/set37_train_1channel_hematoxylin_512_A70_B220_lambda15/160_net_G_A.pth \\
        --input_dir  datasets/test_data/250711_slides/testA \\
        --output_dir results/vH_denoised \\
        --input_nc 1 --output_nc 1

Add ``--compare`` to also render the un-dithered version and print the artifact
score for both, so the improvement is measured rather than assumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.networks import define_G  # noqa: E402

_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


def artifact_score(img: np.ndarray, period: int = 64) -> float:
    """Autocorrelation of *img* at a lag of *period* px, averaged over both axes.

    ~0 means no periodic structure at that scale; the untreated vH output scores
    0.2-0.7 depending on how empty the input is.
    """
    x = img.astype(np.float64)
    x = x - x.mean()
    if x.std() < 1e-8:
        return 0.0
    ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(np.fft.fft2(x)) ** 2)))
    ac /= ac.max()
    c = ac.shape[0] // 2
    if c + period >= ac.shape[0]:
        return float("nan")
    return float((ac[c, c + period] + ac[c + period, c]) / 2)


def flatness_weight(x: torch.Tensor, win: int = 15, ref: float = 0.15) -> torch.Tensor:
    """Per-pixel weight that is 1 on flat input and 0 on textured input.

    The artifact only appears where the input carries little variance, so
    dithering everywhere is wasteful: it also perturbs genuine tissue and costs
    fidelity. Weighting the dither by local flatness keeps the noise where it is
    needed. Measured on 20 emptiest vs 20 most tissue-rich patches:

        scheme              empty artifact   empty r    tissue artifact   tissue r
        no dither                 0.219        0.547         0.030          0.491
        global   sigma=0.01       0.067        0.389         0.020          0.385
        global   sigma=0.03       0.016        0.234         0.003          0.210
        adaptive sigma=0.03       0.065        0.387         0.023          0.431

    Adaptive matches global sigma=0.01 on artifact suppression while keeping
    noticeably more of the tissue signal (r 0.431 vs 0.385, and 0.210 for the
    stronger global setting).
    """
    pad = win // 2
    mean = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(x, (pad,) * 4, mode="reflect"), win, 1)
    mean_sq = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(x * x, (pad,) * 4, mode="reflect"), win, 1)
    local_std = (mean_sq - mean * mean).clamp_min(0).sqrt()
    return 1.0 - (local_std / ref).clamp(0, 1)


def load_generator(args, device: torch.device):
    """Rebuild the UTOM generator and load the checkpoint weights."""
    net = define_G(args.input_nc, args.output_nc, args.ngf, args.netG,
                   args.norm, not args.no_dropout, "normal", 0.02, [])
    state = torch.load(args.checkpoint, map_location="cpu")
    if hasattr(state, "_metadata"):
        del state._metadata
    net.load_state_dict(state)
    return net.to(device).eval()


def to_tensor(path: Path, input_nc: int) -> torch.Tensor:
    """Load an image and normalise to [-1, 1], matching UTOM's new_transformA."""
    mode = "L" if input_nc == 1 else "RGB"
    arr = np.array(Image.open(path).convert(mode), dtype=np.float32) / 127.5 - 1.0
    if arr.ndim == 2:
        arr = arr[None]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)[None]


def to_image(out: torch.Tensor) -> np.ndarray:
    """Convert a generator output in [-1, 1] back to uint8."""
    arr = out[0].detach().cpu().numpy()
    arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return arr[0] if arr.shape[0] == 1 else arr.transpose(1, 2, 0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="UTOM inference with dither noise to suppress the periodic artifact.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Path to a *_net_G_A.pth file.")
    ap.add_argument("--input_dir", required=True, help="Directory of input images.")
    ap.add_argument("--output_dir", required=True, help="Where generated images are written.")
    ap.add_argument("--input_nc", type=int, default=1, help="1 for TPAF grayscale, 3 for RGB.")
    ap.add_argument("--output_nc", type=int, default=1,
                    help="1 for the vH/vE single-stain models, 3 for full H&E.")
    ap.add_argument("--noise_sigma", type=float, default=0.03,
                    help="Dither strength in [-1,1] space. 0.03 ~ 7.6/255 grey levels; "
                         "0 disables dithering.")
    ap.add_argument("--adaptive", action="store_true", default=True,
                    help="Weight the dither by local flatness (default). Suppresses the "
                         "artifact where it occurs while sparing textured tissue.")
    ap.add_argument("--global_dither", dest="adaptive", action="store_false",
                    help="Apply uniform dither everywhere instead of flatness-weighted.")
    ap.add_argument("--flat_win", type=int, default=15,
                    help="Window (px) for the local-variance estimate used by --adaptive.")
    ap.add_argument("--netG", default="unet_256", help="Generator architecture.")
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--norm", default="instance")
    ap.add_argument("--no_dropout", action="store_true", default=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0, help="Seed for the dither, for reproducibility.")
    ap.add_argument("--period", type=int, default=64,
                    help="Artifact period in px used by the reported score.")
    ap.add_argument("--compare", action="store_true",
                    help="Also run without dither and report both artifact scores.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    net = load_generator(args, device)

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in _EXT)
    if not files:
        sys.exit(f"No images found in {in_dir}")

    mode = "adaptive (flatness-weighted)" if args.adaptive else "global"
    print(f"device={device}  images={len(files)}  sigma={args.noise_sigma}  dither={mode}")
    scored, scored_plain = [], []
    for p in files:
        x = to_tensor(p, args.input_nc).to(device)
        with torch.no_grad():
            if args.noise_sigma > 0:
                dither = torch.randn_like(x) * args.noise_sigma
                if args.adaptive:
                    dither = dither * flatness_weight(x, args.flat_win)
                img = to_image(net(x + dither))
            else:
                img = to_image(net(x))
        Image.fromarray(img).save(out_dir / f"{p.stem}.png")
        g = img if img.ndim == 2 else img.mean(2)
        scored.append(artifact_score(g, args.period))

        if args.compare:
            with torch.no_grad():
                plain = to_image(net(x))
            cmp_dir = out_dir.parent / f"{out_dir.name}_nodither"
            cmp_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(plain).save(cmp_dir / f"{p.stem}.png")
            gp = plain if plain.ndim == 2 else plain.mean(2)
            scored_plain.append(artifact_score(gp, args.period))

    print(f"\nwrote {len(files)} images -> {out_dir}")
    print(f"\n{args.period}px artifact score (0 = no periodic structure)")
    if args.compare:
        print(f"  without dither : {np.mean(scored_plain):+.3f}   (max {np.max(scored_plain):+.3f})")
        print(f"  with dither    : {np.mean(scored):+.3f}   (max {np.max(scored):+.3f})")
        print(f"  -> reduced by {np.mean(scored_plain) - np.mean(scored):+.3f}")
        print(f"  un-dithered copies written -> {out_dir.parent / (out_dir.name + '_nodither')}")
    else:
        print(f"  mean {np.mean(scored):+.3f}   max {np.max(scored):+.3f}"
              f"   {'(clean)' if np.mean(scored) < 0.05 else '(still periodic)'}")


if __name__ == "__main__":
    main()
