# SA-vHE

Virtual H&E staining of label-free two-photon autofluorescence (TPAF) images of ovarian
tissue, and the evaluation pipeline used to judge it.

The premise is that TPAF imaging needs no stain, no section consumed, and no turnaround
through a histology lab — but a pathologist and every tool in a diagnostic workflow
read H&E. Virtual staining is the bridge. Whether it is a usable bridge is an empirical
question about whether software and people built for real H&E accept the synthetic
kind, which is what most of the code here exists to measure.

Part of an ongoing PhD project on AI with nonlinear optical microscopy for ovarian
cancer diagnosis (Shanghai Jiao Tong University, Biomedical Engineering).

> **Status: research code, work in progress.** Results are not published here. This
> repository holds the pipeline and the model weights; it makes no claims about how the
> methods compare.

---

## What is here

### Virtual staining

Built on [UTOM](https://github.com/cabooster/UTOM) (unsupervised content-preserving
transformation for optical microscopy), which is itself a CycleGAN with a saliency
constraint separating tissue from background.

| | |
|---|---|
| `path_vhe_stain.py` | Whole-image staining: overlap-tile inference at the model's own scale, feathered. Also the nuclei-enhanced input variants. |
| `path_vhe_masks.py` | Cellpose-SAM nuclei masks on the TPAF side. |
| `path_vhe_correct.py`, `path_vhe_blend.py`, `path_vhe_tune3.py` | Stain colour correction: global statistics matching, per-compartment blending, and a three-compartment fit on CIELAB ΔE2000. |
| `path_vhe_prepare.py`, `path_vhe_collect.py`, `path_vhe_page.py` | Recovering crop geometry, pairing virtual against real H&E, and an interactive comparison page. |

### Slide assembly

| | |
|---|---|
| `pptx_stitch.py` | Assembles a native-resolution TPAF mosaic from a hand-made PowerPoint layout. The deck supplies the arrangement; the pixels come from the original TIFs, which are byte-identical to what PowerPoint embedded. Pairwise correlation then refines each position, and territories are assigned by nearest tile centre so overlapping tiles are never averaged. |
| `path_tpaf_stitch.py`, `path_tpaf_link.py` | Region-level stitching and linking TPAF fields to their H&E counterparts. |
| `path_mosaic_masks.py` | Cellpose-SAM over a whole mosaic, tiled. |

### Evaluation

| | |
|---|---|
| `path_band.py` | How far two *real* H&E slides sit from each other under the same estimator. Without it, a colour difference has no scale to be read on. |
| `path_downstream_seg.py` | Nuclei segmentation across every source — TPAF, real H&E, and each virtual variant — with the pixel scale converted per image. |
| `path_downstream_hv.py` | HoVer-Net (PanNuke) via TIAToolbox: instances plus a five-class nuclear type. |
| `path_downstream_stats.py` | Nuclear morphology per region, weighted toward what ovarian serous carcinoma grading actually uses (see below). |
| `colab/ch5_hovernet_colab.ipynb` | The HoVer-Net stage on Colab, where post-processing has enough CPU cores to be practical. |

### Region screening

`path_tiles.py`, `path_candidates.py`, `path_locate.py`, `path_structures.py`,
`path_report.py` — tiling a slide at a physical size, scoring tiles with a pathology
foundation model, and choosing regions worth comparing.

---

## Metrics follow the grading system, not convenience

Ovarian serous carcinoma is graded on a two-tier system (WHO 2020), and its operative
criterion for nuclear atypia is a **ratio**: grade 3 is ≥3:1 variation in nuclear size
across a field, grade 2 is under 3:1, grade 1 is uniform round-to-oval nuclei.
Alongside it sit hyperchromasia, irregular contours, crowding, and prominent nucleoli.

So `path_downstream_stats.py` reports size *variation* first and median size last:

| tier | measure | grading feature |
|---|---|---|
| 1 | `size_ratio` (p90/p10 of equivalent diameter) | the 3:1 rule |
| 1 | `solidity`, `circularity` | contour irregularity, pleomorphism |
| 1 | `density` | crowding, loss of polarity |
| 1 | `hema_od` | hyperchromasia |
| 2 | `hema_sd` | chromatin coarseness, nucleoli |
| 3 | `area_um2`, `eqdiam_um`, `ecc`, `nn_um` | only meaningful inside the ratio, or non-standard |

Nucleus-to-cytoplasm ratio is a grading feature and is absent: it needs cytoplasm
segmentation, which none of the segmentation models used here provide.

Two conventions run through the evaluation code:

- **Agreement, not correlation.** A stain that renders every nucleus 40% too large
  still correlates perfectly with the truth. Concordance (Lin's CCC) and Bland–Altman
  bias are reported; Pearson *r* appears only so the gap between the two is visible.
- **Measure the floor first.** Two halves of the same real H&E region differ from each
  other by some amount that owes nothing to virtual staining. A method's number means
  nothing until that floor is known, so `path_band.py` and the `real_HE_half` rows in
  the morphology output exist to establish it.

---

## Scale is the recurring trap

Five separate stages of this project produced two numbers that were each correct in
their own frame and were not the same quantity. They are documented in the scripts
because none of them raise an error:

| | |
|---|---|
| Generator input | The model reads 204 px native tiles upsampled to 512 (2.510×, 0.247 µm/px). Feeding 512 px native renders every nucleus 2.5× too large — and still produces a plausible-looking H&E. |
| Resampling filter | PIL `BICUBIC` and `cv2.INTER_CUBIC` are not the same filter. |
| Mosaic grey | `(R+G)/2` and `cv2.COLOR_BGR2GRAY` are different weightings of the same two channels. |
| Real H&E crops | Stored at 512×512 regardless of physical extent, so their scale runs 0.44–0.77 µm/px and varies region to region. A fixed pixel radius is a different physical distance in each. |
| HoVer-Net input | Trained at 0.25 µm/px. Given 0.621 it finds a fraction of the nuclei, silently, by an amount that depends on the image's own scale. |

---

## Environment

```
torch 2.7.1+cu118, cellpose 4.0.8          virtual staining, Cellpose-SAM
torch 2.1.0, tiatoolbox 1.5.1              HoVer-Net
```

Two environments because TIAToolbox pins an older torch. `path_downstream_hv.py` is the
only script that needs the second one.

On Windows, TIAToolbox's workers require an `if __name__ == '__main__':` guard; without
it a run deadlocks with an empty cache directory rather than failing.

## Model weights

Individual generator checkpoints are 208 MB, past GitHub's 100 MB file limit, so they
are published as **Release assets** rather than in the tree. `checkpoints/` is
gitignored. See the Releases page for what is available and which script expects which.

## Data

No image data is in this repository. The material is patient tissue from a clinical
collaboration and is not ours to distribute. Paths in the code point at a local layout
and are there to make the pipeline legible, not to be run as-is.

## Acknowledgements

[UTOM](https://github.com/cabooster/UTOM) ·
[CycleGAN / pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) ·
[Cellpose](https://github.com/MouseLand/cellpose) ·
[TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox) ·
[HoVer-Net](https://github.com/vqdang/hover_net)
