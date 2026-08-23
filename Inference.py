"""
Sliding-window inference for 3D glioma segmentation.

Matches the thesis protocol and the validation loop in Training.py:
- 3D U-Net, 4 MRI channels (T1, T1ce, T2, FLAIR)
- z-score normalisation on brain voxels only
- sliding window of 128 x 128 x 128 with 50 percent overlap
- trilinear resampling of logits onto the original grid
- voxel-wise argmax over 4 classes (background, NETC, SNFH, ET)
- optional Dice / HD95 on BraTS composite regions ET, WT, TC
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from Config import Config
from Model import UNet3D
from Utils import normalizzazione_zscore
from Visualizzazioni import visualizza_modalita_e_segmentazione


ROI_SIZE = (128, 128, 128)
OVERLAP = 0.5
CHANNEL_ORDER = ("t1", "t1ce", "t2", "flair")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 3D U-Net inference on a BraTS case (thesis sliding-window protocol)."
    )
    parser.add_argument(
        "--case_dir",
        type=str,
        required=True,
        help="Patient folder containing NIfTI files named {folder}-{suffix}.nii.gz",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model_brats.pth",
        help="Checkpoint written by training (default: best_model_brats.pth)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for the predicted mask and overlay (default: Config.save_dir/inference)",
    )
    parser.add_argument(
        "--roi_size",
        type=int,
        nargs=3,
        default=list(ROI_SIZE),
        metavar=("D", "H", "W"),
        help="Sliding-window size (default: 128 128 128)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=OVERLAP,
        help="Window overlap in [0, 1) (default: 0.5)",
    )
    parser.add_argument(
        "--no_seg",
        action="store_true",
        help="Do not load a ground-truth mask even if one is present",
    )
    return parser.parse_args()


def nifti_path(case_dir: Path, suffix: str) -> Path:
    return case_dir / f"{case_dir.name}-{suffix}.nii.gz"


def load_case(case_dir: Path, load_seg: bool):
    """Load four modalities (and optional seg) with the same preprocessing as Dataset.py."""
    images = {}
    sitk_ref = None
    keys = CHANNEL_ORDER + (("seg",) if load_seg else ())
    for key in keys:
        suffix = Config.modalities[key]
        path = nifti_path(case_dir, suffix)
        if not path.exists():
            if key == "seg":
                continue
            raise FileNotFoundError(f"Missing volume: {path}")
        sitk_img = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(sitk_img)
        array = np.expand_dims(array, axis=0)
        if key == "seg":
            images[key] = array.astype(np.uint8)
        else:
            images[key] = normalizzazione_zscore(array)
            if sitk_ref is None:
                sitk_ref = sitk_img

    if sitk_ref is None:
        raise RuntimeError("Could not read a reference MRI volume.")

    input_tensor = torch.cat(
        [torch.from_numpy(images[key]).float() for key in CHANNEL_ORDER],
        dim=0,
    )
    seg_tensor = torch.from_numpy(images["seg"]).long() if "seg" in images else None

    parts = case_dir.name.split("-")
    subject_id = parts[2] if len(parts) > 2 else case_dir.name
    timepoint = parts[3] if len(parts) > 3 else "000"

    return {
        "input": input_tensor,
        "seg": seg_tensor,
        "sitk_ref": sitk_ref,
        "subject_id": subject_id,
        "timepoint": timepoint,
    }


def load_model(checkpoint_path: str, device: torch.device) -> UNet3D:
    model = UNet3D(Config.in_channels, Config.out_channels, Config.dropout_rate).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def predict_volume(model, volume: torch.Tensor, device, roi_size, overlap) -> np.ndarray:
    inputs = volume.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs,
            roi_size=tuple(roi_size),
            sw_batch_size=1,
            predictor=model,
            overlap=overlap,
        )
        spatial = volume.shape[1:]
        if tuple(logits.shape[2:]) != tuple(spatial):
            logits = F.interpolate(logits, size=spatial, mode="trilinear", align_corners=False)
        pred = torch.argmax(logits, dim=1).squeeze(0)
    return pred.cpu().numpy().astype(np.uint8)


def region_metrics(pred: np.ndarray, target: np.ndarray, device: torch.device) -> dict:
    pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).to(device)
    target_t = torch.from_numpy(target.astype(np.int64)).unsqueeze(0).unsqueeze(0).to(device)

    dice = DiceMetric(include_background=False, reduction="mean")
    hd95 = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    whole = torch.tensor([1, 2, 3], device=device)
    core = torch.tensor([1, 3], device=device)

    maps = {
        "ET": ((pred_t == 3).float(), (target_t == 3).float()),
        "WT": (torch.isin(pred_t, whole).float(), torch.isin(target_t, whole).float()),
        "TC": (torch.isin(pred_t, core).float(), torch.isin(target_t, core).float()),
    }

    results = {}
    for name, (pred_mask, target_mask) in maps.items():
        dice.reset()
        hd95.reset()
        dice(y_pred=pred_mask, y=target_mask)
        hd95(y_pred=pred_mask, y=target_mask)
        results[f"dice_{name}"] = float(torch.nan_to_num(dice.aggregate(), nan=0.0).item())
        results[f"hd95_{name}"] = float(torch.nan_to_num(hd95.aggregate(), nan=0.0).item())
    return results


def save_nifti(pred: np.ndarray, reference: sitk.Image, path: Path) -> None:
    out = sitk.GetImageFromArray(pred)
    out.CopyInformation(reference)
    sitk.WriteImage(out, str(path), useCompression=True)


def save_overlay(sample: dict, pred: np.ndarray, output_dir: Path) -> None:
    vis_sample = {
        "input": sample["input"],
        "seg": sample["seg"] if sample["seg"] is not None else torch.from_numpy(pred).long(),
        "subject_id": sample["subject_id"],
        "timepoint": sample["timepoint"],
    }
    visualizza_modalita_e_segmentazione(str(output_dir), vis_sample, pred_seg=pred)


def main():
    args = parse_args()
    case_dir = Path(args.case_dir).expanduser().resolve()
    if not case_dir.is_dir():
        raise NotADirectoryError(case_dir)

    output_dir = Path(args.output_dir) if args.output_dir else Path(Config.save_dir) / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = Config.device
    print(f"Device: {device}")
    print(f"Case: {case_dir}")
    print(f"Window: {tuple(args.roi_size)}  overlap={args.overlap}")

    sample = load_case(case_dir, load_seg=not args.no_seg)
    model = load_model(args.checkpoint, device)
    pred = predict_volume(model, sample["input"], device, args.roi_size, args.overlap)

    pred_path = output_dir / f"{case_dir.name}_pred.nii.gz"
    save_nifti(pred, sample["sitk_ref"], pred_path)
    save_overlay(sample, pred, output_dir)
    print(f"Saved mask: {pred_path}")

    if sample["seg"] is not None:
        gt = sample["seg"].numpy()
        if gt.ndim == 4:
            gt = gt[0]
        metrics = region_metrics(pred, gt, device)
        metrics_path = output_dir / f"{case_dir.name}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("Metrics vs ground truth:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        print(f"Saved metrics: {metrics_path}")
    else:
        print("No ground-truth mask loaded; skipped Dice / HD95.")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
