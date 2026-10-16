# VEP-MDTNet

PyTorch implementation of **VEP-MDTNet: a deep learning model for terrace extraction in the mountainous and
hilly region of western Liaoning, China**.

VEP-MDTNet segments terraces from co-registered high-resolution RGB imagery and a DEM. It is a U-Net-style
network with four parts:

**Two independent ResNet-50 encoders** extract RGB and DEM features. The RGB encoder uses deformable
convolutions (DCN) in stages 3–5.

**Virtual-branch fusion.** A virtual branch fuses the two modalities stage by stage. At stage $`n \geq 2`$:

```math
V_n=\frac{\phi_n^{R}(R_n)+\phi_n^{D}(D_n)}{2}+\mathcal{F}_n\left(\mathrm{Concat}\left[\phi_n^{R}(R_n),\ \phi_n^{D}(D_n),\ B_n\left(\phi_n^{V_{n-1}}(V_{n-1})\right)\right]\right)
```

$`R_n`$ and $`D_n`$ are the RGB and DEM encoder features. $`\phi`$ are 1×1 convolutions with group
normalization (GN) and SiLU, and $`B_n`$ is bilinear interpolation. $`\mathcal{F}_n`$ is a residual
mapping of two 3×3 convolutions, each followed by GN, with SiLU after the first. Stage 1 has no
$`V_{n-1}`$ term.

**E-P Skip Gate.** For each skip connection, the edge guidance response $`E_n`$ and the patch region
response $`P_n`$ are predicted from the virtual feature $`V_n`$. Together with the upsampled decoder
feature $`U_n`$, they give the gating coefficient $`\eta_n`$, which filters the skip feature:

```math
\begin{aligned}
E_n &= \sigma\left(\mathrm{Conv}_{1\times1}\left[\mathrm{SiLU}\left(\mathrm{GN}\left(\mathrm{Conv}_{3\times3}(V_n)\right)\right)\right]\right), \qquad
P_n = \sigma\left(\mathrm{Conv}_{1\times1}(V_n)\right),\\
\eta_n &= \sigma\left(\mathrm{Conv}_{1\times1}\left(\mathrm{Concat}[V_n,U_n,E_n,P_n]\right)\right), \qquad
\tilde{V}_n = \eta_n \odot V_n,\\
F_n^{\mathrm{skip}} &= \mathrm{Conv}_{3\times3}\left(\mathrm{Conv}_{3\times3}\left(\mathrm{Concat}[U_n,\tilde{V}_n,E_n,P_n]\right)\right)
\end{aligned}
```

**MDT (multi-scale deformable Transformer).** The MDT is built from multi-scale deformable attention
(MSDA). The MDT encoder (4 layers) relates $`V_1`$, $`V_3`$ and $`V_5`$. The MDT decoder (2 layers) uses
the encoded $`V_3'`$ as queries and produces $`F_{\mathrm{MDT}}`$, which is fused into decoder stage
$`D_3`$.

```
RGB ─ ResNet-50 (DCN in stages 3-5) ─┐
                                     ├─ virtual branch V1..V5 ─┬─ E-P gated skips ─┐
DEM ─ ResNet-50 ─────────────────────┘                        └─ MDT(V1,V3,V5) ──→ U-Net decoder → mask
```

## Installation

The code was developed with Python 3.8, PyTorch 2.4.1 / torchvision 0.19.1 (CUDA 12.1) and GDAL 3.0. Any
PyTorch ≥ 1.12 with torchvision ≥ 0.13 should work. No custom CUDA operators are needed: DCN uses
`torchvision.ops.deform_conv2d`, and MSDA is written in plain PyTorch.

```bash
conda create -n vepmdt python=3.8 gdal -c conda-forge
conda activate vepmdt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Dataset

The terrace dataset used in the paper is available from the
[**dataset-v1.0 release**](https://github.com/hongyuanhuangaa/VEP-MDTNet/releases/tag/dataset-v1.0). It covers the
mountainous and hilly region of western Liaoning (Jianping County), China, and contains **1,720 terraces**
with manually interpreted terrace masks and **terrace boundaries**. It also includes the DEM, slope and
contour layers.

The archive `82lb.zip` (4.59 GB) is split into three parts because GitHub limits each release file to 2 GB.
Download all three parts and join them (or open `82lb.zip.001` with 7-Zip). Then check the result against
`SHA256SUMS.txt`:

```bash
cat 82lb.zip.001 82lb.zip.002 82lb.zip.003 > 82lb.zip        # Windows: copy /b 82lb.zip.001+82lb.zip.002+82lb.zip.003 82lb.zip
unzip 82lb.zip -d data/                                       # the examples below use data/82lb
```

```
82lb/
├── train/  (5946 tiles)   image/ dem/ label/ boundary/ contour/ slope/
├── val/    (1701 tiles)   image/ dem/ label/ boundary/ contour/ slope/
├── test/   (873 tiles)    image/ dem/ label/ boundary/ contour/ slope/
└── source/                full-area rasters: boundary.tif, contour.tif, dem_aligned.tif, slope.tif
```

| layer | content |
|---|---|
| `image` | GF-6 imagery, 2 m, spring 2024, 4 bands uint8. The first three bands (RGB) are used, divided by 255 |
| `dem` | 5 m DEM resampled to the image grid, float32, elevation in m. Clipped to the 1st–99th percentile of the training split and z-scored |
| `label` | terrace mask, uint8 (0 = background, 255 = terrace) |
| `boundary` | terrace boundary mask, uint8 (0 / 255). Used only for boundary-band evaluation, never as model input |
| `contour` | 5 m contour lines, float32 (contour elevation; −9999 elsewhere). Not used by the model |
| `slope` | slope in degrees, float32. Not used by the model |

How the tiles were made:
- Tiles are 256 × 256 GeoTIFFs, cut with a sliding window with 20 % overlap. Files are matched by name across
  folders.
- The study area was divided into 10 non-overlapping spatial subregions, assigned to train / val / test in a
  7:2:1 ratio.
- Each location appears three times in every split: as the original tile and as its horizontally and
  vertically flipped copies.
- `dem_stats.json` (the DEM statistics of the training split) is created automatically in the dataset folder
  on the first run.

Terrace labels were visually interpreted in ArcGIS 10.7. The dataset is provided for research purposes;
please cite the paper if you use it. The imagery is derived from GF-6 data and remains subject to the terms
of its original provider.

## Usage

### Training

```bash
python train.py --preset full --data-root data/82lb --out-dir runs
```

Defaults reproduce the paper's protocol:

| Setting | Value |
|---|---|
| Optimizer | AdamW (β = 0.95/0.999, weight decay 5e-4) |
| Learning rate | 1e-4. Linear warm-up from 0.1× over 10 epochs, then cosine decay to 0.005× |
| Batch | 8 × 4 gradient-accumulation steps (effective 32) |
| Precision | Mixed precision on CUDA, gradient clipping at 1.0 |
| Loss | 0.3 BCE + 0.7 Dice |
| Schedule | 100 epochs, no early stopping. The checkpoint with the best validation mIoU is kept |
| Augmentation | Random flips and 90° rotations, plus a synchronized RGB/DEM/mask CutMix. Its probability adapts to the validation-mIoU trend within [0.1, 0.4] |

Each run writes `runs/<preset>/best.pt`, `train_log.txt`, `history.json` and `curves.png`.

Every model variant of the paper is a preset (`--preset`):

| preset | paper | fusion | DCN | MDT | E | P | gate |
|---|---|---|:-:|:-:|:-:|:-:|:-:|
| `full` | VEP-MDTNet (Full) | virtual branch | ✓ | ✓ | ✓ | ✓ | ✓ |
| `base` | Base | concatenation | | | | | |
| `wo_vb` | w/o Virtual Branch = average fusion | average | ✓ | ✓ | ✓ | ✓ | ✓ |
| `concat` | concatenation fusion | concatenation | ✓ | ✓ | ✓ | ✓ | ✓ |
| `wo_mdt` | w/o MDT | virtual branch | ✓ | | ✓ | ✓ | ✓ |
| `wo_ep` | w/o E-P Skip Gate | virtual branch | ✓ | ✓ | | | |
| `wo_e` / `wo_p` | w/o E / w/o P | virtual branch | ✓ | ✓ | –/✓ | ✓/– | ✓ |
| `wo_dcn` | w/o DCN | virtual branch | | ✓ | ✓ | ✓ | ✓ |
| `rgb_only` / `dem_only` | single-modality input | – (one encoder) | ✓/– | ✓ | ✓ | | ✓ |

When a module is removed it falls back as follows:

- E / P → zero maps
- E-P Skip Gate → an ordinary U-Net skip connection
- MDT → no F_MDT (the decoder still starts from the projected V5)
- DCN → the plain ResNet 3×3 convolution

### Testing

```bash
python test.py --checkpoint runs/full/best.pt --data-root data/82lb --out-dir results/full --save-masks
```

The architecture and the DEM statistics are read from the checkpoint. The script reports mIoU, F1, recall
and precision as class means over background and terrace from the confusion matrix pooled over all test tiles
(the definition used in the paper), together with OA and per-class values.

### Whole-scene prediction

```bash
python predict.py --checkpoint runs/full/best.pt --image scene_rgb.tif --dem scene_dem.tif \
    --output results/scene_mask.tif --probability results/scene_prob.tif
```

The scene is processed in 256 × 256 windows with a stride of 128 and Hann-window blending. RGB and DEM must
share the same grid. RGB-only and DEM-only checkpoints need only `--image` or only `--dem`.

### Using the model in your own code

```python
import torch
from vepmdtnet import build_model
from vepmdtnet.presets import make_config

config, modality = make_config("full", pretrained=False)   # any preset listed above
model = build_model(config).eval()                         # rgbdem presets
logits = model(torch.rand(1, 3, 256, 256), torch.randn(1, 1, 256, 256))   # (1, 2, 256, 256)
```

## Repository structure

```
vepmdtnet/          model package
  config.py           VEPMDTNetConfig: all architecture switches
  presets.py          the paper's model variants
  model.py            VEPMDTNet
  single_modality.py  RGB-only / DEM-only variants
  encoder.py          DualEncoder: RGB and DEM ResNet-50 encoders + virtual branch
  virtual_branch.py   VirtualBranchStage (phi_R, phi_D, phi_V, F_n)
  dcn.py              DCNv2 (torchvision deform_conv2d)
  ep_skip_gate.py     EdgeGuidanceHead (E), PatchRegionHead (P), EPSkipGate (eta)
  decoder.py          decoder D4..D1, F_MDT fused at D3
  mdt.py              MDT
  msda.py             MSDA and the MDT encoder / decoder layers
  losses.py           0.3 BCE + 0.7 Dice
  data.py             tile dataset and DEM normalization
  checkpoint.py       loading with the stored architecture
  complexity.py       parameter / FLOP count
train.py  test.py  predict.py
```

Checkpoints written with the development version of the code (package `vf_emrt`, module names such as
`bridge` / `guidance`) are converted to the current names automatically by `vepmdtnet.checkpoint`.

## Citation

If you use this code, please cite the paper. The reference will be added after publication.

## License

The code is released under the [MIT License](LICENSE).
