"""Evaluate a checkpoint on the test tiles and optionally save the predicted masks.

Example:
    python test.py --checkpoint runs/full/best.pt --data-root data/82lb --out-dir results/full

Metrics are computed from the confusion matrix pooled over all tiles and averaged over
the two classes (background, terrace), as reported in the paper (mIoU, F1, recall,
precision); overall accuracy and terrace-class values are printed as well.  The model
architecture and DEM normalisation statistics are restored from the checkpoint.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from vepmdtnet.checkpoint import load_model, logits_to_prediction, model_inputs
from vepmdtnet.data import TerraceDataset, load_terrain_statistics


def compute_metrics(cm: np.ndarray) -> Dict[str, object]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    iou = tp / (tp + fp + fn + 1.0e-6)
    precision = tp / (tp + fp + 1.0e-6)
    recall = tp / (tp + fn + 1.0e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1.0e-6)
    return {
        "mIoU": float(iou.mean()),
        "F1": float(f1.mean()),
        "Recall": float(recall.mean()),
        "Precision": float(precision.mean()),
        "OA": float(np.trace(cm) / (cm.sum() + 1.0e-6)),
        "per_class": {
            "IoU": iou.tolist(), "F1": f1.tolist(),
            "Recall": recall.tolist(), "Precision": precision.tolist(),
        },
        "confusion_matrix": cm.tolist(),
    }


def save_mask_geotiff(output_path: Path, mask: np.ndarray, reference_path: Path) -> None:
    from osgeo import gdal

    reference = gdal.Open(str(reference_path), gdal.GA_ReadOnly)
    if reference is None:
        raise RuntimeError(f"Cannot open reference raster: {reference_path}")
    width, height = reference.RasterXSize, reference.RasterYSize
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = gdal.GetDriverByName("GTiff").Create(
        str(output_path), width, height, 1, gdal.GDT_Byte, options=["TILED=YES", "COMPRESS=LZW"])
    if output is None:
        raise RuntimeError(f"Cannot create {output_path}")
    output.SetGeoTransform(reference.GetGeoTransform())
    output.SetProjection(reference.GetProjection())
    output.GetRasterBand(1).WriteArray(mask.astype(np.uint8))
    output.FlushCache()
    output = None
    reference = None


def visual_row(image: torch.Tensor, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """RGB (or stretched DEM) | GT | Pred, as a BGR uint8 image."""
    array = image.detach().cpu().numpy()
    if array.shape[0] >= 3:
        base = np.clip(np.transpose(array[:3], (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
        base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    else:
        band = array[0]
        lo, hi = np.percentile(band, [2, 98])
        band = np.clip((band - lo) / (hi - lo + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        base = cv2.cvtColor(band, cv2.COLOR_GRAY2BGR)
    gt_vis = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    pred_vis = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    cv2.putText(gt_vis, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(pred_vis, "Pred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return np.hstack((base, gt_vis, pred_vis))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="write metrics.json here (and masks/visuals if requested)")
    parser.add_argument("--save-masks", action="store_true", help="GeoTIFF masks (0/255)")
    parser.add_argument("--save-visuals", action="store_true", help="image | GT | Pred PNGs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dem-stats", type=Path, default=None,
                        help="override the DEM statistics stored in the checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config, modality, meta = load_model(args.checkpoint, device)
    print(f"Modality: {modality}; preset: {meta.get('preset', 'n/a')}; epoch: {meta.get('epoch')}")

    statistics = None
    if modality != "rgb":
        if args.dem_stats is not None:
            statistics = load_terrain_statistics(args.dem_stats)
        elif isinstance(meta.get("terrain_statistics"), dict):
            statistics = meta["terrain_statistics"]
        else:
            statistics = load_terrain_statistics(args.data_root / "dem_stats.json")

    dataset = TerraceDataset(
        dataset_root=args.data_root, split=args.split, terrain_statistics=statistics,
        modality=modality, input_size=tuple(config.input_size),
        image_channels=config.image_channels, image_scale=255.0, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    index_by_name = {name: i for i, name in enumerate(dataset.names)}
    if (args.save_masks or args.save_visuals) and args.out_dir is None:
        parser.error("--save-masks / --save-visuals need --out-dir")
    mask_dir = args.out_dir / "masks" if args.out_dir else None
    vis_dir = args.out_dir / "visuals" if args.out_dir else None
    if args.save_visuals:
        vis_dir.mkdir(parents=True, exist_ok=True)

    confusion = np.zeros((2, 2), dtype=np.int64)
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Testing ({len(dataset)} tiles)", unit="batch"):
            inputs = model_inputs(batch, modality, device)
            with autocast(enabled=amp_enabled):
                logits = model(*inputs)
            preds = logits_to_prediction(logits).cpu().numpy()
            gts = batch["mask"][:, 0].long().numpy()
            confusion += np.bincount(gts.reshape(-1) * 2 + preds.reshape(-1),
                                     minlength=4).reshape(2, 2)
            for k, name in enumerate(batch["name"]):
                pred8 = (preds[k] * 255).astype(np.uint8)
                if args.save_masks:
                    save_mask_geotiff(mask_dir / f"{name}.tif", pred8,
                                      dataset.reference_path(index_by_name[name]))
                if args.save_visuals:
                    shown = batch["image"][k] if "image" in batch else batch["terrain"][k]
                    row = visual_row(shown, (gts[k] * 255).astype(np.uint8), pred8)
                    cv2.imwrite(str(vis_dir / f"vis_{name}.png"), row)

    metrics = compute_metrics(confusion)
    print("\n" + "=" * 52)
    print(f"{'Metric':<12}{'Mean':>10}{'Background':>14}{'Terrace':>12}")
    print("-" * 52)
    for key, per_class in (("mIoU", "IoU"), ("F1", "F1"), ("Recall", "Recall"),
                           ("Precision", "Precision")):
        values = metrics["per_class"][per_class]
        print(f"{key:<12}{metrics[key]:>10.4f}{values[0]:>14.4f}{values[1]:>12.4f}")
    print("-" * 52)
    print(f"{'OA':<12}{metrics['OA']:>10.4f}")
    print("=" * 52)

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        report = {"checkpoint": str(args.checkpoint), "modality": modality,
                  "split": args.split, "n_tiles": len(dataset), **metrics}
        (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Results written to {args.out_dir}")


if __name__ == "__main__":
    main()
