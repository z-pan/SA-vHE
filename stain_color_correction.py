#!/usr/bin/env python3
"""Correct virtual-H&E colour by matching stain concentrations to real H&E.

The problem this solves
-----------------------
UTOM's virtual H&E has the right hue *balance* but sits on the violet side:
measured over 112 patches against registered real H&E,

    region      R-B (virtual)   R-B (real)   gap
    cytoplasm       -13.0          -2.1     -10.9   <- reads as purple
    nuclei          -34.7         -43.1      +8.4   <- not violet enough

Note the cytoplasm R-G is already correct (76.3 vs 76.1), so this is a narrow
defect: blue is a few counts too high and red a few too low. A global RGB or LAB
correction would fix the cytoplasm by dragging the nuclei with it, and the nuclei
are already too pale -- so the correction has to act on the two stains separately.

Method
------
Colour-deconvolve into haematoxylin / eosin / residual concentrations
(``skimage.color.rgb2hed``; round-trip error here is 0.42/255, i.e. negligible),
match each stain channel's mean and standard deviation to the real reference, and
reconstruct. Because haematoxylin carries the nuclei and eosin the cytoplasm,
each compartment is corrected by its own statistics.

Statistics are taken over tissue pixels only -- slide background is mostly white
and would otherwise dominate both distributions and flatten the correction.

This is a post-hoc fix that needs no retraining and is independent of the
generator, so it applies equally to any virtual-staining output.

Usage
-----
    python stain_color_correction.py \\
        --input_dir     datasets/test_data/250711_slides/05_results_vHE_patches_nuc_hi \\
        --reference_dir datasets/test_data/250711_slides/01_og_real_HE_patches \\
        --output_dir    results/vHE_color_corrected

Verify with the SA-CUT metric, which reports the same R-B gaps::

    python ../SA-CUT/scripts/eval_stain_color.py \\
        --virtual results/vHE_color_corrected \\
        --nuc-mask <mask dir> --real <real H&E dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.color import hed2rgb, rgb2hed

_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
_STAIN = ("haematoxylin", "eosin", "residual")


def _images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*")
                  if p.is_file() and p.suffix.lower() in _EXT)


def stain_stats(paths: list[Path], white_thresh: float, limit: int | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of each stain concentration over tissue pixels."""
    sums = np.zeros(3)
    sqs = np.zeros(3)
    n = 0
    for p in paths[:limit]:
        rgb = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        tissue = rgb.mean(2) < white_thresh
        if tissue.sum() < 100:
            continue
        hed = rgb2hed(rgb)[tissue]          # (N, 3) concentrations
        sums += hed.sum(0)
        sqs += (hed ** 2).sum(0)
        n += hed.shape[0]
    if n == 0:
        raise RuntimeError("No tissue pixels found — check --white-thresh.")
    mean = sums / n
    std = np.sqrt(np.maximum(sqs / n - mean ** 2, 1e-12))
    return mean, std


def correct(rgb: np.ndarray, src: tuple[np.ndarray, np.ndarray],
            ref: tuple[np.ndarray, np.ndarray], white_thresh: float,
            strength: float) -> np.ndarray:
    """Match this image's stain concentrations to the reference statistics."""
    src_mean, src_std = src
    ref_mean, ref_std = ref
    hed = rgb2hed(rgb)
    tissue = rgb.mean(2) < white_thresh

    scale = np.where(src_std > 1e-9, ref_std / src_std, 1.0)
    shifted = (hed - src_mean) * scale + ref_mean
    # strength<1 interpolates toward the correction, for a gentler adjustment.
    shifted = hed + strength * (shifted - hed)
    # Leave background untouched: it carries no stain and rescaling it only
    # introduces a colour cast on what should stay white.
    out = np.where(tissue[..., None], shifted, hed)
    return np.clip(hed2rgb(out), 0.0, 1.0)


def correct_masked(rgb: np.ndarray, mask: np.ndarray, white_thresh: float,
                   h_in: float, h_out: float, e_out: float,
                   blur: float) -> np.ndarray:
    """Scale haematoxylin differently inside and outside the nuclei.

    A single global stain scaling cannot fix both compartments: raising
    haematoxylin darkens the nuclei toward the reference but drags the cytoplasm
    further violet, because the generator has put haematoxylin where there should
    be none. Grid search over global (H, E) scalings bottomed out at a combined
    error of 16.8; splitting by the nuclear mask reaches 2.7.

        setting                          cytoplasm R-B   nuclei R-B
        uncorrected                          -13.0         -34.7
        global stat-matching                  -5.3         -28.8
        masked  H 1.5 in / 0.5 out            -2.8         -41.0
        real H&E reference                    -2.1         -43.1

    The mask is blurred first so the two scalings cross-fade instead of leaving a
    hard edge at the mask boundary.
    """
    weight = gaussian_filter(mask.astype(np.float32), blur)[..., None]
    hed = rgb2hed(rgb)
    h_scale = h_out + (h_in - h_out) * weight[..., 0]
    e_scale = e_out + (1.0 - e_out) * weight[..., 0]
    out = hed.copy()
    out[..., 0] *= h_scale
    out[..., 1] *= e_scale
    tissue = rgb.mean(2) < white_thresh
    out = np.where(tissue[..., None], out, hed)
    return np.clip(hed2rgb(out), 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Match virtual-H&E stain concentrations to a real H&E reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--input_dir", required=True, help="Virtual H&E images to correct.")
    ap.add_argument("--reference_dir", required=True, help="Real H&E images (the target).")
    ap.add_argument("--output_dir", required=True, help="Where corrected images are written.")
    ap.add_argument("--white-thresh", type=float, default=0.92,
                    help="Mean RGB in [0,1] above which a pixel counts as background.")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="0 = unchanged, 1 = full match. Lower it if the result "
                         "over-corrects.")
    ap.add_argument("--stats-limit", type=int, default=None,
                    help="Use only the first N images when estimating statistics.")
    ap.add_argument("--mask_dir", default=None,
                    help="Nuclear masks, matched by filename stem. Switches to the "
                         "masked correction, which fixes both compartments; without "
                         "it, global statistics matching fixes cytoplasm only.")
    ap.add_argument("--strip", default="",
                    help="Substring removed from input names to match masks.")
    ap.add_argument("--h_in", type=float, default=1.5, help="Haematoxylin scale in nuclei.")
    ap.add_argument("--h_out", type=float, default=0.5, help="Haematoxylin scale elsewhere.")
    ap.add_argument("--e_out", type=float, default=1.0, help="Eosin scale outside nuclei.")
    ap.add_argument("--mask_blur", type=float, default=2.0,
                    help="Gaussian sigma used to soften the mask boundary.")
    args = ap.parse_args()

    src_paths = _images(Path(args.input_dir))
    ref_paths = _images(Path(args.reference_dir))
    if not src_paths:
        sys.exit(f"No images in {args.input_dir}")
    if not ref_paths:
        sys.exit(f"No images in {args.reference_dir}")

    print(f"source {len(src_paths)} images | reference {len(ref_paths)} images")

    if args.mask_dir:
        masks = {p.stem: p for p in _images(Path(args.mask_dir))}
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"masked correction: H {args.h_in} in nuclei / {args.h_out} outside, "
              f"E {args.e_out} outside")
        done = skipped = 0
        for p in src_paths:
            stem = p.stem.replace(args.strip, "") if args.strip else p.stem
            mp = masks.get(stem)
            if mp is None:
                skipped += 1
                continue
            rgb = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            m = np.array(Image.open(mp).convert("L")) > 127
            fixed = correct_masked(rgb, m, args.white_thresh, args.h_in,
                                   args.h_out, args.e_out, args.mask_blur)
            Image.fromarray((fixed * 255).round().astype(np.uint8)).save(
                out_dir / f"{p.stem}.png")
            done += 1
        if skipped:
            print(f"  [warn] {skipped} images had no matching mask "
                  f"(indexed {len(masks)}; check --strip)", file=sys.stderr)
        print(f"\ncorrected {done} images -> {out_dir}")
        return

    src = stain_stats(src_paths, args.white_thresh, args.stats_limit)
    ref = stain_stats(ref_paths, args.white_thresh, args.stats_limit)

    print(f"\n{'stain':<14} {'src mean':>10} {'ref mean':>10} {'src std':>10} {'ref std':>10}")
    for i, name in enumerate(_STAIN):
        print(f"{name:<14} {src[0][i]:>10.4f} {ref[0][i]:>10.4f} "
              f"{src[1][i]:>10.4f} {ref[1][i]:>10.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in src_paths:
        rgb = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        fixed = correct(rgb, src, ref, args.white_thresh, args.strength)
        Image.fromarray((fixed * 255).round().astype(np.uint8)).save(
            out_dir / f"{p.stem}.png")

    print(f"\ncorrected {len(src_paths)} images -> {out_dir}")


if __name__ == "__main__":
    main()
