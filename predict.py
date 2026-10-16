"""Sliding-window terrace prediction for a whole scene (co-registered RGB and DEM GeoTIFFs).

Example:
    python predict.py --checkpoint runs/full/best.pt --image scene_rgb.tif --dem scene_dem.tif \
        --output results/scene_mask.tif --probability results/scene_prob.tif

The scene is cut into 256x256 windows with a stride of 128 px; overlapping foreground
probabilities are blended with a Hann window.  RGB uses the first three bands divided by
255; the DEM is clipped and z-scored with the training statistics stored in the
checkpoint.  RGB-only / DEM-only checkpoints need only --image / --dem.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.cuda.amp import autocast
from tqdm import tqdm

from vepmdtnet.checkpoint import load_model
from vepmdtnet.data import _read_image, _read_single, load_terrain_statistics, normalise_dem


def window_positions(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def blend_weight(height: int, width: int) -> np.ndarray:
    y = np.hanning(height) if height > 1 else np.ones(1, dtype=np.float32)
    x = np.hanning(width) if width > 1 else np.ones(1, dtype=np.float32)
    return np.maximum(np.outer(y, x).astype(np.float32), 0.05)


def pad_chw(array: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    pad_h = max(0, target_h - array.shape[-2])
    pad_w = max(0, target_w - array.shape[-1])
    if pad_h == 0 and pad_w == 0:
        return array
    return np.pad(array, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")


def foreground_probability(logits: Tensor) -> Tensor:
    if logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    raise ValueError("Binary segmentation expects one or two output channels")


def predict_scene(
    model: torch.nn.Module,
    arrays: Sequence[np.ndarray],
    tile_size: Tuple[int, int],
    stride: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Blend tile probabilities over a scene; ``arrays`` are the CHW model inputs."""
    tile_h, tile_w = tile_size
    height, width = arrays[0].shape[-2:]
    if stride <= 0 or stride > min(tile_h, tile_w):
        raise ValueError(f"stride must be in [1, {min(tile_h, tile_w)}]")
    tops = window_positions(height, tile_h, stride)
    lefts = window_positions(width, tile_w, stride)
    weight = blend_weight(tile_h, tile_w)
    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    amp_enabled = device.type == "cuda"
    patches: List[List[np.ndarray]] = [[] for _ in arrays]
    locations: List[Tuple[int, int, int, int]] = []

    def flush() -> None:
        if not locations:
            return
        tensors = [torch.from_numpy(np.stack(p)).to(device, dtype=torch.float32) for p in patches]
        with autocast(enabled=amp_enabled):
            probabilities = foreground_probability(model(*tensors)).float().cpu().numpy()
        for index, (top, left, patch_h, patch_w) in enumerate(locations):
            local = weight[:patch_h, :patch_w]
            probability_sum[top:top + patch_h, left:left + patch_w] += (
                probabilities[index, :patch_h, :patch_w] * local)
            weight_sum[top:top + patch_h, left:left + patch_w] += local
        for p in patches:
            p.clear()
        locations.clear()

    with torch.inference_mode(), tqdm(total=len(tops) * len(lefts), desc="Predicting", unit="tile") as bar:
        for top in tops:
            for left in lefts:
                window = (slice(None), slice(top, min(top + tile_h, height)),
                          slice(left, min(left + tile_w, width)))
                for p, array in zip(patches, arrays):
                    p.append(pad_chw(array[window], tile_h, tile_w))
                patch_h, patch_w = arrays[0][window].shape[-2:]
                locations.append((top, left, patch_h, patch_w))
                if len(locations) >= batch_size:
                    bar.update(len(locations))
                    flush()
        bar.update(len(locations))
        flush()
    return probability_sum / np.maximum(weight_sum, 1.0e-6)


def open_raster(path: Path):
    from osgeo import gdal

    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    return dataset


def check_alignment(image_path: Path, dem_path: Path) -> None:
    image_ds, dem_ds = open_raster(image_path), open_raster(dem_path)
    if (image_ds.RasterXSize, image_ds.RasterYSize) != (dem_ds.RasterXSize, dem_ds.RasterYSize):
        raise ValueError("RGB and DEM raster dimensions do not match")
    a, b = image_ds.GetGeoTransform(), dem_ds.GetGeoTransform()
    if a and b and not np.allclose(a, b, rtol=0.0, atol=1.0e-8):
        raise ValueError("RGB and DEM geotransforms do not match")
    pa, pb = image_ds.GetProjection(), dem_ds.GetProjection()
    if pa and pb and pa != pb:
        raise ValueError("RGB and DEM coordinate systems do not match")


def write_geotiff(path: Path, array: np.ndarray, reference_path: Path, gdal_type: int) -> None:
    from osgeo import gdal

    reference = open_raster(reference_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = array.shape
    output = gdal.GetDriverByName("GTiff").Create(
        str(path), width, height, 1, gdal_type,
        options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"])
    if output is None:
        raise RuntimeError(f"Cannot create output raster: {path}")
    output.SetGeoTransform(reference.GetGeoTransform())
    output.SetProjection(reference.GetProjection())
    output.GetRasterBand(1).WriteArray(array)
    output.FlushCache()
    output = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None, help="RGB GeoTIFF (first 3 bands used)")
    parser.add_argument("--dem", type=Path, default=None, help="DEM GeoTIFF on the same grid")
    parser.add_argument("--output", type=Path, required=True, help="output mask GeoTIFF (0/255)")
    parser.add_argument("--probability", type=Path, default=None,
                        help="optional float32 foreground-probability GeoTIFF")
    parser.add_argument("--dem-stats", type=Path, default=None,
                        help="override the DEM statistics stored in the checkpoint")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    model, config, modality, meta = load_model(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} (modality={modality}, preset={meta.get('preset', 'n/a')})")
    need_rgb, need_dem = modality in ("rgbdem", "rgb"), modality in ("rgbdem", "dem")
    if need_rgb and args.image is None:
        parser.error("this checkpoint needs --image")
    if need_dem and args.dem is None:
        parser.error("this checkpoint needs --dem")

    arrays = []
    if need_rgb:
        arrays.append(_read_image(args.image, config.image_channels, 255.0))
    if need_dem:
        statistics: Optional[dict] = meta.get("terrain_statistics")
        if args.dem_stats is not None:
            statistics = load_terrain_statistics(args.dem_stats)
        if not isinstance(statistics, dict):
            parser.error("checkpoint has no DEM statistics; pass --dem-stats dem_stats.json")
        dem_raw, dem_nodata = _read_single(args.dem)
        arrays.append(normalise_dem(dem_raw, dem_nodata, statistics)[None])
    if need_rgb and need_dem:
        check_alignment(args.image, args.dem)
        if arrays[0].shape[-2:] != arrays[1].shape[-2:]:
            raise ValueError("RGB and DEM array dimensions do not match")
    reference = args.image if need_rgb else args.dem

    probability = predict_scene(model, arrays, (args.tile, args.tile), args.stride,
                                args.batch_size, device)
    from osgeo import gdal

    write_geotiff(args.output, (probability >= args.threshold).astype(np.uint8) * 255,
                  reference, gdal.GDT_Byte)
    print(f"Saved mask: {args.output}")
    if args.probability is not None:
        write_geotiff(args.probability, probability.astype(np.float32), reference, gdal.GDT_Float32)
        print(f"Saved foreground probability: {args.probability}")


if __name__ == "__main__":
    main()
