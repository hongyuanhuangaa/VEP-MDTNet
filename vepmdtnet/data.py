"""Tile dataset and raster helpers shared by training, testing and prediction.

Expected layout (one sub-folder per split, matching file stems across folders):

    <root>/dem_stats.json            (optional; computed from the train split if missing)
    <root>/<split>/image/<n>.tif     RGB(+NIR) tile; the first 3 bands are used, scaled by 1/255
    <root>/<split>/dem/<n>.tif       DEM tile (float, nodata allowed)
    <root>/<split>/label/<n>.tif     terrace mask (0/1 or 0/255)
    <root>/<split>/boundary/<n>.tif  optional terrace boundary mask (read only with with_boundary=True)

The DEM is clipped to the 1st-99th percentile of the training split and z-scored with the
training mean/std; nodata pixels are set to 0 after normalisation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from osgeo import gdal
from torch.utils.data import Dataset
from tqdm import tqdm


SUPPORTED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
TERRAIN_FACTOR_NAMES: Tuple[str, ...] = ("dem",)
CONTINUOUS_FACTORS: Tuple[str, ...] = ("dem",)
MODALITIES = ("rgbdem", "rgb", "dem")


def _file_map(folder: Path) -> Dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Required dataset folder does not exist: {folder}")
    result: Dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            if path.stem in result:
                raise ValueError(f"Duplicate sample stem '{path.stem}' in {folder}")
            result[path.stem] = path
    if not result:
        raise FileNotFoundError(f"No supported raster files in {folder}")
    return result


def sort_stems(stems: Sequence[str]) -> List[str]:
    """Numeric stems first in numeric order, then the rest alphabetically."""
    return sorted(stems, key=lambda value: (int(value) if value.isdigit() else 10**12, value))


def _read_gdal(path: Path) -> Tuple[np.ndarray, Optional[float]]:
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    array = dataset.ReadAsArray()
    nodata = dataset.GetRasterBand(1).GetNoDataValue()
    dataset = None
    if array is None:
        raise RuntimeError(f"Cannot read raster: {path}")
    return np.asarray(array), nodata


def _read_image(path: Path, channels: int, scale: float) -> np.ndarray:
    array, _ = _read_gdal(path)
    if array.ndim == 2:
        array = array[None]
    if array.shape[0] < channels:
        raise ValueError(f"{path} has {array.shape[0]} bands, expected {channels}")
    image = array[:channels].astype(np.float32)
    if scale > 0:
        image /= scale
    return np.ascontiguousarray(image)


def _read_single(path: Path) -> Tuple[np.ndarray, Optional[float]]:
    array, nodata = _read_gdal(path)
    if array.ndim == 3:
        array = array[0]
    return np.asarray(array), nodata


def _valid_mask(array: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    valid = (
        np.isfinite(array)
        & (array > -1.0e20)
        & (~np.isclose(array, -9999.0))
    )
    if nodata is not None:
        if np.isnan(float(nodata)):
            valid &= ~np.isnan(array)
        else:
            valid &= ~np.isclose(array, nodata)
    return valid


def _binary_target(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array, nan=0.0)
    maximum = float(array.max()) if array.size else 0.0
    threshold = 0.5 if maximum <= 1.0 else 127.0
    return (array > threshold).astype(np.float32)


def compute_terrain_statistics(
    dataset_root: Path,
    split: str = "train",
    stride: int = 8,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> Dict[str, object]:
    """Compute robust train-only statistics for the DEM input."""

    bands: List[Dict[str, object]] = []
    for factor in CONTINUOUS_FACTORS:
        files = _file_map(Path(dataset_root) / split / factor)
        samples: List[np.ndarray] = []
        for path in tqdm(files.values(), desc=f"{factor} statistics", unit="tile"):
            array, nodata = _read_single(path)
            array = array.astype(np.float32)[::stride, ::stride]
            valid = _valid_mask(array, nodata)
            if np.any(valid):
                samples.append(array[valid])
        if not samples:
            raise RuntimeError(f"No valid pixels were found for {factor}")
        values = np.concatenate(samples)
        clip_min, clip_max = np.percentile(
            values, [lower_percentile, upper_percentile]
        )
        clipped = np.clip(values, clip_min, clip_max)
        bands.append(
            {
                "name": factor,
                "clip_min": float(clip_min),
                "clip_max": float(clip_max),
                "mean": float(clipped.mean()),
                "std": float(max(clipped.std(), 1.0e-6)),
            }
        )

    return {
        "factor_names": list(TERRAIN_FACTOR_NAMES),
        "source_split": split,
        "sample_stride": stride,
        "percentiles": [lower_percentile, upper_percentile],
        "bands": bands,
    }


def save_terrain_statistics(statistics: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_terrain_statistics(path: Path) -> Dict[str, object]:
    statistics = json.loads(Path(path).read_text(encoding="utf-8"))
    names = tuple(str(item["name"]) for item in statistics.get("bands", []))
    if names != TERRAIN_FACTOR_NAMES:
        raise ValueError(f"Unexpected terrain statistics order in {path}: {names}")
    return statistics


def load_or_compute_terrain_statistics(dataset_root: Path) -> Dict[str, object]:
    """Read <root>/dem_stats.json, computing it from the train split when absent."""
    path = Path(dataset_root) / "dem_stats.json"
    if path.exists():
        return load_terrain_statistics(path)
    print("Computing DEM statistics from the training split...")
    statistics = compute_terrain_statistics(dataset_root=dataset_root, split="train")
    save_terrain_statistics(statistics, path)
    return statistics


def dem_band_statistics(statistics: Dict[str, object]) -> Dict[str, object]:
    bands = statistics.get("bands")
    if not isinstance(bands, list):
        raise ValueError("terrain statistics must contain a bands list")
    names = tuple(str(item.get("name")) for item in bands)
    if names != TERRAIN_FACTOR_NAMES:
        raise ValueError(f"Terrain channel order must be {TERRAIN_FACTOR_NAMES}, got {names}")
    return bands[0]


def normalise_dem(
    array: np.ndarray,
    nodata: Optional[float],
    statistics: Dict[str, object],
) -> np.ndarray:
    """Percentile clip + z-score with the training statistics; invalid pixels -> 0."""
    band = dem_band_statistics(statistics)
    mean = float(band["mean"])
    std = max(float(band["std"]), 1.0e-6)
    valid = _valid_mask(array, nodata)
    result = np.where(valid, array.astype(np.float32), mean)
    result = np.clip(result, float(band["clip_min"]), float(band["clip_max"]))
    result = (result - mean) / std
    result[~valid] = 0.0
    return np.ascontiguousarray(result, dtype=np.float32)


class TerraceDataset(Dataset):
    """Paired RGB / DEM / label (/ boundary) tiles of one split.

    ``modality`` selects the inputs that are read: ``"rgbdem"`` (both), ``"rgb"`` or
    ``"dem"``.  Items are dicts with ``image`` (3xHxW, RGB modalities), ``terrain``
    (1xHxW, DEM modalities), ``mask`` (1xHxW), optionally ``boundary`` and ``name``.
    The boundary is never a model input; it is only read for boundary-band evaluation.
    ``augment`` applies random horizontal/vertical flips and 90-degree rotations.
    """

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        terrain_statistics: Optional[Dict[str, object]] = None,
        modality: str = "rgbdem",
        input_size: Tuple[int, int] = (256, 256),
        image_channels: int = 3,
        image_scale: float = 255.0,
        with_boundary: bool = False,
        augment: bool = False,
        image_dir: str = "image",
        dem_dir: str = "dem",
        label_dir: str = "label",
        boundary_dir: str = "boundary",
    ) -> None:
        super().__init__()
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}")
        self.uses_rgb = modality in ("rgbdem", "rgb")
        self.uses_dem = modality in ("rgbdem", "dem")
        base = Path(dataset_root) / split
        folders = {"label": label_dir}
        if self.uses_rgb:
            folders["image"] = image_dir
        if self.uses_dem:
            folders["dem"] = dem_dir
        if with_boundary:
            folders["boundary"] = boundary_dir
        maps = {kind: _file_map(base / folder) for kind, folder in folders.items()}
        reference_kind = "image" if self.uses_rgb else "dem"
        reference = set(maps[reference_kind])
        for name, mapping in maps.items():
            if set(mapping) != reference:
                missing = sorted(reference - set(mapping))[:10]
                extra = sorted(set(mapping) - reference)[:10]
                raise ValueError(
                    f"Pair mismatch for {split}/{name}: missing={missing}, extra={extra}"
                )
        ordered_stems = sort_stems(reference)
        self.samples = [
            {kind: mapping[stem] for kind, mapping in maps.items()}
            for stem in ordered_stems
        ]
        self.names = ordered_stems
        self.reference_kind = reference_kind
        self.modality = modality
        self.input_size = input_size
        self.image_channels = image_channels
        self.image_scale = image_scale
        self.with_boundary = with_boundary
        self.augment = augment
        self.statistics = None
        if self.uses_dem:
            if terrain_statistics is None:
                raise ValueError("DEM input requires terrain statistics")
            dem_band_statistics(terrain_statistics)
            self.statistics = terrain_statistics

    def __len__(self) -> int:
        return len(self.samples)

    def reference_path(self, index: int) -> Path:
        """Raster whose georeference is copied to predicted masks."""
        return Path(self.samples[index][self.reference_kind])

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        height, width = self.input_size
        arrays: Dict[str, np.ndarray] = {}
        if self.uses_rgb:
            image = _read_image(sample["image"], self.image_channels, self.image_scale)
            arrays["image"] = np.stack(
                [
                    cv2.resize(band, (width, height), interpolation=cv2.INTER_LINEAR)
                    for band in image
                ]
            )
        if self.uses_dem:
            dem_raw, dem_nodata = _read_single(sample["dem"])
            dem = normalise_dem(dem_raw, dem_nodata, self.statistics)
            arrays["terrain"] = cv2.resize(
                dem, (width, height), interpolation=cv2.INTER_LINEAR
            )[None]
        label, _ = _read_single(sample["label"])
        arrays["mask"] = cv2.resize(
            _binary_target(label), (width, height), interpolation=cv2.INTER_NEAREST
        )[None]
        if self.with_boundary:
            boundary, _ = _read_single(sample["boundary"])
            arrays["boundary"] = cv2.resize(
                _binary_target(boundary), (width, height), interpolation=cv2.INTER_NEAREST
            )[None]

        if self.augment:
            if random.random() < 0.5:
                arrays = {key: np.flip(value, axis=2) for key, value in arrays.items()}
            if random.random() < 0.5:
                arrays = {key: np.flip(value, axis=1) for key, value in arrays.items()}
            if height == width:
                rotations = random.randint(0, 3)
                arrays = {
                    key: np.rot90(value, rotations, axes=(1, 2))
                    for key, value in arrays.items()
                }

        item: Dict[str, object] = {
            key: torch.from_numpy(np.ascontiguousarray(value)).float()
            for key, value in arrays.items()
        }
        item["name"] = self.names[index]
        return item
