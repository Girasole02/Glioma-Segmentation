# BraTS glioma dataset notes

Reference material for the mpMRI volumes used by this project. Official BraTS preprocessing is applied **by the challenge organisers**, not re-implemented here.

## Modalities

Each case includes four co-registered, skull-stripped sequences:

- **T1 native (T1)** — anatomical reference (longitudinal relaxation).
- **T1 with gadolinium (T1Gd / T1ce)** — highlights active tumour where the blood–brain barrier is disrupted.
- **T2-weighted (T2)** — high water content (necrosis, cysts).
- **T2-FLAIR** — T2-like contrast with CSF suppression; emphasises edema and infiltration.

Volumes come from multiple scanners, sites, and protocols. Shared preprocessing includes:

- co-registration to a common anatomical template;
- interpolation to **1 mm³** isotropic resolution;
- **skull stripping** (non-brain tissue removed).

File layout follows BraTS naming, e.g. `BraTS-GLI-<id>-<timepoint>/` with suffixes such as `t1n`, `t1c`, `t2w`, `t2f`, `seg` (see `Config.modalities`). Timepoint `000` denotes the first pre-operative scan.

This work uses **pre-operative** glioma cases only.

## Labels

The `seg` volume stores integer labels. In this pipeline:

| ID | Acronym | Meaning |
| --- | --- | --- |
| 0 | Background | Healthy / non-labelled tissue |
| 1 | NETC | Non-enhancing tumour core (necrosis, non-enhancing core) |
| 2 | SNFH | Surrounding non-enhancing FLAIR hyperintensity |
| 3 | ET | Enhancing tumour |

Composite regions used for scoring:

- **Tumour core (TC)** = ET + NETC (typical surgical target).
- **Whole tumour (WT)** = ET + NETC + SNFH (full disease extent including edema).

Some BraTS task descriptions also mention **RC** (resection cavities) in post-treatment settings; that label is **not** used in this pre-operative setup.

## Evaluation metrics (challenge vs this repo)

BraTS reports lesion-wise overlap and boundary metrics so large lesions do not dominate small ones:

- **Dice / DSC** — voxel overlap, ignoring true-negative background.
- **NSD** (normalised surface distance) — boundary overlap in some BraTS tracks.

This codebase reports **Dice** and **HD95** (95th-percentile Hausdorff distance) on ET, WT, and TC, matching the thesis experimental protocol.
