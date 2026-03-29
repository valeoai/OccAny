<div align="center">

# OccAny: Generalized Unconstrained Urban 3D Occupancy

### CVPR 2026

<div>
  <a href="https://anhquancao.github.io/">Anh-Quan Cao</a>&nbsp;&nbsp;
  <a href="https://tuanhungvu.github.io/">Tuan-Hung Vu</a>
  <br>
  Valeo.ai, Paris, France
</div>

<br>

[![Project page](https://img.shields.io/badge/Project_Page-OccAny-darkgreen?style=flat-square)](https://valeoai.github.io/OccAny/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.23502-b31b1b.svg?style=flat-square)](http://arxiv.org/abs/2603.23502)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Model_Checkpoints-yellow?style=flat-square)](https://huggingface.co/anhquancao/OccAny/tree/main/checkpoints)

**TL;DR:** A unified framework for generalized unconstrained urban 3D occupancy prediction.
<!-- <p align="center">
  <img src="/assets/teaser.jpg" alt="Pipeline">
</p> -->

</div>


https://github.com/user-attachments/assets/a3abde5b-c170-4b5d-818c-262b0df2546f


---

OccAny provides demo inputs, pretrained checkpoints, inference scripts, evaluation utilities, training dataset preparation and training scripts, and visualization tools for urban 3D occupancy under unconstrained camera inputs. This public release includes two model variants:

- **OccAny**, based on Must3R + SAM2
- **OccAny+**, based on Depth Anything 3 + SAM3

The repository also includes sample RGB scenes in `demo_data/input`, pretrained weights in `checkpoints/`, and viewers for both point-cloud and voxel-grid outputs.

## 📑 Table of Contents

- [🔧 Installation](#-installation)
- [📦 Checkpoints](#-checkpoints)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Key Inference Flags](#%EF%B8%8F-key-inference-flags)
- [👁️ Visualization](#%EF%B8%8F-visualization)
- [📊 Evaluation](#-evaluation)
- [🚗 Ego Trajectory Evaluation](#-ego-trajectory-evaluation)
- [🏋️ Training](#%EF%B8%8F-training)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)
- [📝 Citation](#-citation)

## 🔧 Installation

The commands below create the environment used for the public release and keep all required third-party dependencies local to this repository.

### Clone the Repository

```bash
git clone https://github.com/valeoai/OccAny.git
cd OccAny
```

### Create a Python Environment

```bash
conda create -n occany python=3.12 -y
conda activate occany
python -m pip install --upgrade pip setuptools wheel ninja
```

### Install PyTorch and CUDA

```bash
conda install -c nvidia cuda-toolkit=12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install xformers==0.0.29.post2
```

### Install Shared Python Dependencies

```bash
pip install -r requirements.txt
```

### Install `torch-scatter`

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
pip install torch-scatter --no-cache-dir --no-build-isolation
```

### Vendored Third-Party Code

OccAny relies on the vendored copies bundled in `third_party/`:

- `third_party/croco` for `croco`
- `third_party/dust3r` for `dust3r`
- `third_party/Grounded-SAM-2` for Grounded-SAM-2, `sam2`, and `groundingdino`
- `third_party/sam3` for SAM3
- `third_party/Depth-Anything-3` for Depth Anything 3

`inference.py` already prepends these paths automatically at runtime. If you want to import the vendored packages in a shell, notebook, or standalone sanity check, export them explicitly:

```bash
export PYTHONPATH="$PWD/third_party:$PWD/third_party/dust3r:$PWD/third_party/croco/models/curope:$PWD/third_party/Grounded-SAM-2:$PWD/third_party/Grounded-SAM-2/grounding_dino:$PWD/third_party/sam3:$PWD/third_party/Depth-Anything-3/src:$PYTHONPATH"
```

Avoid adding `third_party/sam2` on top of this unless you explicitly need the standalone SAM2 copy, because it exposes the same top-level module name as `third_party/Grounded-SAM-2`.

### Compile CroCo's `curope` Extension

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

cd third_party/croco/models/curope
python setup.py install
cd ../../../..
```

This builds a `curope*.so` file next to the sources. The `PYTHONPATH` export above includes that directory so `models.curope` can resolve it at runtime.

The vendored `third_party/croco/models/curope/setup.py` currently targets SM 70, 80, and 90. If your GPU uses a different compute capability, update `all_cuda_archs` there before rebuilding.

### Sanity Check (Optional)

```bash
python - <<'PY'
import sys
from pathlib import Path

repo_root = Path.cwd()
for path in reversed([
    repo_root / "third_party",
    repo_root / "third_party" / "dust3r",
    repo_root / "third_party" / "croco" / "models" / "curope",
    repo_root / "third_party" / "Grounded-SAM-2",
    repo_root / "third_party" / "Grounded-SAM-2" / "grounding_dino",
    repo_root / "third_party" / "sam3",
    repo_root / "third_party" / "Depth-Anything-3" / "src",
]):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

import torch
import sam2
import sam3
import groundingdino
import depth_anything_3
import dust3r.utils.path_to_croco  # noqa: F401
from croco.models.pos_embed import RoPE1D

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("RoPE1D backend:", RoPE1D.__name__)
print("third-party imports: ok")
PY
```

## 📦 Checkpoints

Model checkpoints are hosted on Hugging Face:

- [anhquancao/OccAny/tree/main/checkpoints](https://huggingface.co/anhquancao/OccAny/tree/main/checkpoints)

Download checkpoints with:

```bash
cd OccAny
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='anhquancao/OccAny', repo_type='model', local_dir='.', allow_patterns='checkpoints/*')"
```

Expected files under `checkpoints/`:

- `occany_plus_gen.pth`
- `occany_plus_recon.pth`
- `occany.pth`
- `occany_recon.pth`
- `occany_plus_recon_1B.pth`
- `groundingdino_swinb_cogcoor.pth`
- `sam2.1_hiera_large.pt`

## 🚀 Quick Start

After installation and checkpoint download, you can run the demo commands below from the repo root as-is. By default:

- RGB inputs are read from `./demo_data/input`
- Outputs are written to `./demo_data/output`
- The repo already includes sample scenes such as `kitti_08_1390` and `nuscenes_scene-0039`

### Inference Recipes

The following presets reproduce the default demo pipeline for each released model variant.

#### OccAny+ (Depth Anything 3 + SAM3)

```bash
python inference.py \
  --batch_gen_view 2 \
  --view_batch_size 2 \
  --semantic distill@SAM3 \
  --compute_segmentation_masks \
  --gen \
  -rot 30 \
  -vpi 2 \
  -fwd 5 \
  --seed_translation_distance 2 \
  --recon_conf_thres 2.0 \
  --gen_conf_thres 6.0 \
  --apply_majority_pooling \
  --model occany_da3
```

#### OccAny (Must3R + SAM2)

```bash
python inference.py \
  --batch_gen_view 2 \
  --view_batch_size 2 \
  --semantic distill@SAM2_large \
  --compute_segmentation_masks \
  --gen \
  -rot 30 \
  -vpi 2 \
  -fwd 5 \
  --seed_translation_distance 2 \
  --recon_conf_thres 2.0 \
  --gen_conf_thres 2.0 \
  --apply_majority_pooling \
  --model occany_must3r
```


Each processed scene is written under `./demo_data/output/<frame_id>_<model>/`. Common artifacts include:

- `pts3d_render.npy` for reconstruction-view point clouds and metadata
- `pts3d_render_gen.npy` for generated-view point clouds and metadata when `--gen` is enabled
- `pts3d_render_recon_gen.npy` for the merged point-cloud bundle
- `voxel_predictions.pkl` for voxelized occupancy predictions, camera metadata, and visualization inputs

The `pts3d_*.npy` files are what `vis_viser.py` reads, while `voxel_predictions.pkl` is what `vis_voxel.py` and `compute_metrics_from_saved_voxels.py` consume.

`inference.py` currently uses an urban voxel grid tuned for the included demo scenes:

- `voxel_size = 0.4`
- `occ_size = [200, 200, 24]`
- `voxel_origin = [-40.0, -40.0, -3.6]`

This `200 x 200 x 24` grid is only the default for direct demo inference. The evaluation pipeline uses its own dataset-specific layouts, so only edit these constants when you want the standalone demo outputs to follow another convention. Two common evaluation presets are:

**KITTI**

```python
voxel_size = 0.2
occ_size = [256, 256, 32]
voxel_origin = np.array([0.0, -25.6, -2.0], dtype=np.float32)
```

**nuScenes**

```python
voxel_size = 0.4
occ_size = [200, 200, 16]
voxel_origin = np.array([-40.0, -40.0, -1.0], dtype=np.float32)
```


## ⚙️ Key Inference Flags

The most commonly adjusted flags fall into three groups: common flags, semantic flags, and generation-specific flags. If you only want reconstruction output, omit `--gen` and any flag whose scope below is `Generation` or `Generation + semantic`.

<details>
<summary>List of inference flags</summary>

| Flag | Scope | Description |
| --- | --- | --- |
| `--model` | Common | Select the inference backbone: `occany_da3` or `occany_must3r` |
| `--input_dir` | Common | Directory containing RGB demo scene folders |
| `--output_dir` | Common | Directory where outputs are written |
| `--gen` | Common toggle | Enable novel-view generation before voxel fusion |
| `-vpi`, `--views_per_interval` | Generation | Number of generated views sampled per reconstruction view |
| `-fwd`, `--gen_forward_novel_poses_dist` | Generation | Forward offset for generated views, in meters |
| `-rot`, `--gen_rotate_novel_poses_angle` | Generation | Left/right yaw rotation applied to generated views, in degrees |
| `--num_seed_rotations` | Generation | Number of additional seed rotations used when initializing generated poses |
| `--seed_rotation_angle` | Generation | Angular spacing between seed rotations, in degrees |
| `--seed_translation_distance` | Generation | Lateral translation paired with each seed rotation, in meters |
| `--batch_gen_view` | Generation | Number of generated views processed in parallel |
| `--semantic` | Semantic | Enable semantic inference with a SAM2 or SAM3 variant |
| `--compute_segmentation_masks` | Semantic | Save segmentation masks during semantic inference |
| `--view_batch_size` | Semantic | Number of views processed together during semantic inference |
| `--recon_conf_thres` | Reconstruction | Confidence threshold used when voxelizing reconstructed points |
| `--gen_conf_thres` | Generation | Confidence threshold used when voxelizing generated points |
| `--no_semantic_from_rotated_views` | Generation + semantic | Ignore semantics from rotated generated views |
| `--only_semantic_from_recon_view` | Generation + semantic | Use semantics only from reconstruction views, even when generated views are present |
| `--gen_semantic_from_distill_sam3` | Generation + semantic | For `pretrained@SAM3`, infer generated-view semantics from distilled SAM3 features when available |
| `--apply_majority_pooling` | Post-processing | Apply 3x3x3 majority pooling to the fused voxel grid |

</details>


## 👁️ Visualization

### Point-Cloud Viewer (`viser`)

Use `vis_viser.py` to inspect the saved `pts3d_*.npy` point-cloud outputs interactively:

```bash
python vis_viser.py --input_folder ./demo_data/output
```

You can point `--input_folder` either to the output root or directly to a single scene folder. In the viewer, the common dropdown options are:

- `render` for reconstruction output
- `render_gen` for generated-view output
- `render_recon_gen` for the combined output

### Voxel Renderer (`mayavi`)

`vis_voxel.py` renders voxel predictions to image files. Install `mayavi` separately if you want to use this path:

```bash
pip install mayavi
python vis_voxel.py --input_root ./demo_data/output --dataset nuscenes
```

Helpful notes:

- The script writes rendered images to `./demo_data/output_vis` by default
- If the requested `--prediction_key` is missing, it automatically falls back to the best available `render*` grid
- Use `--dataset kitti` for KITTI-style scenes and `--dataset nuscenes` for nuScenes-style surround-view scenes
- Add `--save_input_images` if you also want stacked input RGB images next to the voxel render


## 📊 Evaluation

This section covers the end-to-end evaluation workflow for KITTI and nuScenes using the provided shell and SLURM wrappers.


### Download Evaluation Datasets
#### KITTI

Download the following assets:

- The **Semantic Scene Completion dataset v1.1** (SemanticKITTI voxel data, 700 MB) from the [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html#download)
- The **KITTI Odometry Benchmark calibration data** (calibration files, 1 MB) and **RGB images** (color, 65 GB) from the [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

The dataset folder at **/path/to/kitti** should have the following structure:

```
└── /path/to/kitti/
  └── dataset
    ├── poses
    └── sequences
```

#### nuScenes

1. Download nuScenes with the following script. By default, it downloads to `$PROJECT/data/nuscenes`.

```bash
export PROJECT=$PWD
mkdir -p $PROJECT/data/nuscenes
python dataset_setup/nuscenes/download.py --download_dir $PROJECT/data/nuscenes --output_dir $PROJECT/data/nuscenes --download_workers 16
```

2. Install the `nuscenes-devkit`:

```bash
pip install nuscenes-devkit --no-cache-dir
```

3. Download the voxel ground truth from [Occ3D-nuScenes](https://github.com/Tsinghua-MARS-Lab/Occ3D), including the following files:

```bash
annotations.json
gts.tar.gz
imgs.tar.gz
```

4. Extract them under `$PROJECT/data/nuscenes`. You should then have the following structure:

```
$PROJECT/data/nuscenes/
├── annotations.json
├── can_bus/
├── gts/
├── imgs/
├── maps/
├── samples/
├── sweeps/
├── v1.0-test/
└── v1.0-trainval/
```


### Prepare Output and Dataset Roots

1. Set `PROJECT` and `SCRATCH`, then create the evaluation directories:

   ```bash
   export PROJECT=$PWD
   export SCRATCH=$PWD/eval_output
   mkdir -p \
     "$SCRATCH/ssc_voxel_pred" \
     "$SCRATCH/ssc_output" \
     "$SCRATCH/data/kitti_processed" \
     "$SCRATCH/data/nuscenes_processed"
   ```

2. If your datasets are not under `$PROJECT/data/kitti` and `$PROJECT/data/nuscenes`, override the roots:

   ```bash
   export KITTI_ROOT=/path/to/kitti
   export NUSCENES_ROOT=/path/to/occ3d_nuscenes
   ```

3. Build the vendored Grounded-SAM-2 / GroundingDINO extension:

   ```bash
   pip install -e third_party/Grounded-SAM-2
   # REQUIRE GCC > 9
   pip install --no-build-isolation -e third_party/Grounded-SAM-2/grounding_dino
   ```

   If `groundingdino/_C` fails to load (for example, `NameError: name '_C' is not defined` in `ms_deform_attn.py`), rerun this step in your `occany` environment.

4. Pre-extract the GroundingDINO boxes once:

   ```bash
   python extract_gdino_boxes_kitti.py --image_size 1216 --box_threshold 0.05 --text_threshold 0.05
   python extract_gdino_boxes_nuscenes.py --image_size 1328 --box_threshold 0.05 --text_threshold 0.05
   ```

   Cached boxes are written to:

   ```text
   $SCRATCH/data/kitti_processed/resized_1216_box5_text5_DINOB/<sequence>_<frame_id>/boxes.npz
   $SCRATCH/data/nuscenes_processed/resized_1328_box5_text5_DINOB/<scene_name>/<frame_token>_<camera_name>/boxes.npz
   ```

   `sh/eval_occany.sh` already uses these cache folders, so later evaluation runs can reuse the detections.


`sh/eval_occany.sh` writes voxel predictions under `$SCRATCH/ssc_voxel_pred/<preset-output-dir>/...` and sampled visualization artifacts under `$SCRATCH/ssc_output/<preset-output-dir>/...`.


### Local Shell Workflow

> [!CAUTION]
> Evaluation can take a very long time on a single process because some extraction presets generate up to 180 novel views. We therefore provide the SLURM commands in the [SLURM Workflow](#slurm-workflow) section, which run 20 processes in parallel for occupancy extraction. We have only tested the SLURM path but the local shell should output the same results.

> [!NOTE]
> To maximize performance, some presets sample novel views densely, generating roughly 150–180 views. You can reduce runtime by lowering `-vpi` (views per reconstruction view). In general, the total number of novel views is `n_recon × vpi × (3 if rot > 0 else 1)`.

Evaluation is a two-step workflow:

1. Run `extract_output_occany.py` through `sh/eval_occany.sh` (or `slurm/eval_occany.slurm`) to save voxel predictions.
2. Run `compute_metrics_from_saved_voxels.py` through `sh/compute_metric.sh` (or `slurm/compute_metric.slurm`) to compute SSC metrics from the saved `voxel_predictions.pkl` files.

Run both commands from the repo root. Each block below mirrors the corresponding SLURM example without the `sbatch` wrapper.

**Command template:**

```bash
EXP_LIST=<exp_list> EXP_ID=<id> bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=<mode> EXP_LIST=<metric_exp_list> EXP_ID=<id> bash sh/compute_metric.sh
```

#### OccAny

> Use `EXP_LIST=occany` and `metric EXP_LIST=metric_occany`.

| Preset | `EXP_ID` | `POOLING_MODE` |
|:---|:---:|:---:|
| KITTI 5-frame geometry | `0` | `separate` |
| KITTI 1-frame geometry | `1` | `separate` |
| nuScenes 5-frame geometry | `2` | `separate` |
| nuScenes surround geometry | `3` | `separate` |
| KITTI 5-frame distill semantic | `4` | `separate` |
| nuScenes surround distill semantic | `5` | `separate` |
| KITTI 5-frame pretrained semantic | `6` | `separate` |
| nuScenes surround pretrained semantic | `7` | `separate` |

#### OccAny+

> Use `EXP_LIST=occany_plus` and `metric EXP_LIST=metric_occany_plus`.

| Preset | `EXP_ID` | `POOLING_MODE` |
|:---|:---:|:---:|
| KITTI 5-frame geometry | `0` | `separate` |
| nuScenes surround geometry | `1` | `separate` |
| KITTI 5-frame distill semantic | `2` | `unified` |
| nuScenes surround distill semantic | `3` | `unified` |
| KITTI 5-frame pretrained semantic | `4` | `unified` |
| nuScenes surround pretrained semantic | `5` | `unified` |

### SLURM Workflow

Some extraction presets can generate up to 180 views, so extraction can be slow. The provided `slurm/eval_occany.slurm` script runs a 20-task array in parallel by default (`#SBATCH --array=0-19` with `WORLD=20`).

Each example below submits the extraction job first and then chains the metric job with `--dependency=afterany:$(...)`, so the metric job waits until the full extraction array finishes. The public SLURM wrappers keep the Karolina-HPC defaults (`-A eu-25-92`, `--partition=qgpu`, `--hint=nomultithread`, `--cpus-per-task=16`, and `conda activate occany`); update those settings to match your cluster.

**Command template:**

```bash
sbatch --dependency=afterany:$(sbatch --parsable \
    --export=EXP_LIST=<exp_list>,EXP_ID=<id>,WORLD=20 slurm/eval_occany.slurm) \
  --export=EXP_LIST=<metric_exp_list>,EXP_ID=<id>,USE_MAJORITY_POOLING=1,POOLING_MODE=<mode> \
  slurm/compute_metric.slurm
```

#### OccAny

> Use `EXP_LIST=occany`, `metric EXP_LIST=metric_occany`, and `POOLING_MODE=separate`.

| Preset | Paper | `EXP_ID` | Expected Metrics | Notes |
|:---|:---:|:---:|:---|:---|
| KITTI 5-frame geometry | Tab. 1 | `0` | P 36.79 · R 46.70 · IoU 25.91 | |
| KITTI 1-frame geometry | Tab. 2 | `1` | P 45.64 · R 33.66 · IoU 24.03 | |
| nuScenes 5-frame geometry | Tab. 1 | `2` | P 36.09 · R 40.39 · IoU 23.55 | |
| nuScenes surround geometry | Tab. 3 | `3` | P 45.04 · R 58.54 · IoU 34.15 | |
| KITTI 5-frame distill sem. | Tab. 5, 6 | `4` | mIoU 7.30 · mIoU*ˢᶜ* 13.54 | ≈ paper 7.28 / 13.53 |
| nuScenes surround distill sem. | Tab. 5, 6 | `5` | mIoU 6.65 · mIoU*ˢᶜ* 10.31 | ≈ paper 6.66 / 10.32 |
| KITTI 5-frame pretrained sem. | Tab. 7 | `6` | mIoU 7.62 · mIoU*ˢᶜ* 13.75 | ≈ paper 7.67 / 13.75 |
| nuScenes surround pretrained sem. | Tab. 7 | `7` | mIoU 7.42 · mIoU*ˢᶜ* 10.78 | |

#### OccAny+

> For OccAny+, geometry metrics use `separate` pooling while semantic metrics use `unified` pooling.
> Use `EXP_LIST=occany_plus` and `metric EXP_LIST=metric_occany_plus`.

| Preset | Paper | `EXP_ID` | Expected Metrics | `POOLING_MODE` |
|:---|:---:|:---:|:---|:---:|
| KITTI 5-frame geometry | Tab. 5 | `0` | P 38.11 · R 49.13 · IoU 27.33 | `separate` |
| nuScenes surround geometry | Tab. 5 | `1` | P 46.37 · R 54.67 · IoU 33.49 | `separate` |
| KITTI 5-frame distill sem. | Tab. 7 | `2` | mIoU 6.49 · mIoU*ˢᶜ* 13.31 | `unified` |
| nuScenes surround distill sem. | Tab. 7 | `3` | mIoU 7.20 · mIoU*ˢᶜ* 11.51 | `unified` |
| KITTI 5-frame pretrained sem. | Tab. 7 | `4` | mIoU 8.03 · mIoU*ˢᶜ* 13.17 | `unified` |
| nuScenes surround pretrained sem. | Tab. 7 | `5` | mIoU 9.45 · mIoU*ˢᶜ* 12.22 | `unified` |

## 🚗 Ego Trajectory Evaluation

This section covers ego-trajectory evaluation on NuScenes Vista validation sequences using `infer_trajectory.py` and the provided shell / SLURM wrappers. The reported metric is ADE (Average Displacement Error).

### SLURM Workflow

Submit the provided array job to evaluate both released model sizes:

```bash
sbatch slurm/eval_trajectory.slurm
```

The bundled wrapper maps the array tasks as follows:

- `SLURM_ARRAY_TASK_ID=0` → `./checkpoints/occany_plus_recon.pth` (OccAny 0.35B)
- `SLURM_ARRAY_TASK_ID=1` → `./checkpoints/occany_plus_recon_1B.pth` (OccAny 1.1B)

### Local Shell Workflow

Run the same evaluation locally without SLURM by setting the checkpoint explicitly.

#### OccAny 0.35B

```bash
OCCANY_PLUS_RECON_CKPT=./checkpoints/occany_plus_recon.pth \
  bash sh/infer_occany_nuscenes_traj.sh
```

#### OccAny 1.1B

```bash
OCCANY_PLUS_RECON_CKPT=./checkpoints/occany_plus_recon_1B.pth \
  bash sh/infer_occany_nuscenes_traj.sh
```

Outputs are written to `./outputs/<ckpt_name>_nuscenes_traj/`, including `ade_metrics.json` and trajectory plots.

### ADE Results

| Method | ADE (m) ↓ |
|:---|---:|
| GEM pseudo traj | 1.63 |
| DA3 0.35B + DA3 metric 0.35B | 2.44 |
| DA3 1.1B + DA3 metric 0.35B | 1.12 |
| OccAny 0.35B | 1.86 |
| **OccAny 1.1B** | **0.90** |

## 🏋️ Training

This repository ships public SLURM wrappers for the two-stage training pipelines:

- `slurm/train_occany.slurm` for **OccAny** (Must3R + SAM2)
- `slurm/train_occany_plus.slurm` for **OccAny+** (Depth Anything 3 + SAM3)

All public training SLURM scripts in this repository have been tested on 16 A100 40G GPUs across 2 nodes.

Both wrappers use the same array-task mapping:

- `0` = reconstruction stage
- `1` = generation stage

The shell entrypoints under `sh/` expect the processed training datasets referenced in those files to already exist under `$SCRATCH/data/...`. If your processed dataset roots live elsewhere, update the paths in the corresponding `sh/train_*.sh` file before launching training.

### Common Setup

All commands in this section assume your current working directory is the repository root. From there, define the output roots used by the training wrappers:

```bash
export PROJECT=$PWD
export SCRATCH=${SCRATCH:-$PROJECT}
```

The SLURM wrappers already activate the `occany` conda environment and write scheduler logs to `slurm/output/`.

If you want to launch training directly from an interactive shell without SLURM, activate the environment yourself and force single-node / single-GPU mode so the helper scripts use plain `python` instead of `srun python`:

```bash
conda activate occany
export NUM_NODE=1
export NUM_GPU_PER_NODE=1
```

You can also override script defaults inline, for example:

```bash
BATCH_SIZE=1 N_WORKERS=4 bash sh/train_occany_recon.sh
```

Training recipe note: the original **OccAny** training was run in three stages: **sequence-only reconstruction**, **sequence-only generation**, and **sequence + surround reconstruction**. In this public codebase, we simplify the recipe to two stages: **sequence + surround reconstruction** followed by **sequence + surround generation**. This simplified two-stage recipe is also the one used for **OccAny+**.

### Prepare Training Datasets

For a compact overview of the scripts under `dataset_setup/`, see [`dataset_setup/README.md`](dataset_setup/README.md). Dataset-specific caveats for DDAD and PandaSet live in [`dataset_setup/ddad/README.md`](dataset_setup/ddad/README.md) and [`dataset_setup/pandaset/README.md`](dataset_setup/pandaset/README.md).

The training shell entrypoints expect the processed datasets below to exist under `$SCRATCH/data/`:

```text
$SCRATCH/data/
├── ddad_processed/
├── once_processed/
├── pandaset_processed/
├── vkitti_processed/
└── waymo_processed/
```

The commands below keep the raw archives under `$PROJECT/data/raw/` and write the preprocessed training samples to `$SCRATCH/data/`, which matches the default roots used by `sh/train_occany*.sh`.

#### Common Roots

```bash
export PROJECT=$PWD
export SCRATCH=${SCRATCH:-$PROJECT}

mkdir -p \
  "$PROJECT/data/raw" \
  "$SCRATCH/data/waymo_processed" \
  "$SCRATCH/data/vkitti_processed" \
  "$SCRATCH/data/ddad_processed" \
  "$SCRATCH/data/pandaset_processed" \
  "$SCRATCH/data/once_processed"
```

If you prefer to keep the raw datasets elsewhere, pass the explicit raw root to each preprocessing command instead of relying on the defaults.

#### Waymo Open Dataset

1. Accept the Waymo Open Dataset license, then download the **Perception v1.4.2** `training/*.tfrecord` files into:

   ```text
   $PROJECT/data/raw/waymo/training/
   ```

   You can use `dataset_setup/waymo/download_waymo.sh` as a starting point.

2. Install the extra dependency required by `dataset_setup/waymo/preprocess_waymo.py`:

   ```bash
   pip install gcsfs waymo-open-dataset-tf-2-12-0==1.6.4 --no-cache-dir
   ```

3. Preprocess the dataset:

   ```bash
   python dataset_setup/waymo/preprocess_waymo.py \
     --waymo_dir "$PROJECT/data/raw/waymo/training" \
     --output_dir "$SCRATCH/data/waymo_processed" \
     --workers 16
   ```

#### VKITTI2

1. Download and extract **Virtual KITTI 2** so that the raw root looks like:

   ```text
   $PROJECT/data/raw/vkitti/VirtualKitti2/
   ├── Scene01/
   ├── Scene02/
   └── ...
   ```

2. Preprocess the dataset:

   ```bash
   python dataset_setup/vkitti/preprocess_vkitti.py \
     --vkitti_dir "$PROJECT/data/raw/vkitti/VirtualKitti2" \
     --output_dir "$SCRATCH/data/vkitti_processed" \
     --workers 16
   ```

#### DDAD

1. Download and extract DDAD to a raw root such as:

   ```text
   $PROJECT/data/raw/DDAD/
   ```

2. Install TRI-ML's `dgp` package and the protobuf version expected by `dataset_setup/ddad/preprocess.py`. See [`dataset_setup/ddad/README.md`](dataset_setup/ddad/README.md) for environment-specific installation notes and the protobuf pin used here.

3. Preprocess the dataset:

   ```bash
   python dataset_setup/ddad/preprocess.py \
     --ddad_root "$PROJECT/data/raw/DDAD" \
     --preprocessed_root "$SCRATCH/data/ddad_processed" \
     --n_workers 16
   ```

#### ONCE

1. Download the ONCE archives from the official source and place the tar files under:

   ```text
   $PROJECT/data/raw/once_archives/
   ```

2. Extract them so that the raw dataset root becomes:

   ```text
   $PROJECT/data/raw/ONCE/
   └── data/
       ├── <sequence_id>/
       ├── train_split.txt
       ├── val_split.txt
       └── ...
   ```

   The helper `dataset_setup/once/extract.sh` shows one parallel extraction approach, but it contains site-specific paths, so update its `SOURCE` / `DEST` variables before using it.

3. Preprocess the dataset:

   ```bash
   python dataset_setup/once/preprocess.py \
     --root "$PROJECT/data/raw/ONCE" \
     --preprocessed_root "$SCRATCH/data/once_processed" \
     --n_workers 16
   ```

#### PandaSet

1. Download and extract PandaSet to a raw root such as:

```text
$PROJECT/data/raw/PandaSet/
```

2. Install the `pandaset-devkit` dependency. [`dataset_setup/pandaset/README.md`](dataset_setup/pandaset/README.md) includes a concrete environment example and notes about the optional pair-generation helper.

3. Preprocess the dataset:

   ```bash
   python dataset_setup/pandaset/preprocess.py \
     --root "$PROJECT/data/raw/PandaSet" \
     --save_dir "$SCRATCH/data/pandaset_processed"
   ```

The public training scripts expect the processed output under:

```text
$SCRATCH/data/pandaset_processed/
```

`dataset_setup/pandaset/make_pairs.py` is optional and only applies if you maintain a JPEG-exported processed tree. The current `preprocess.py` writes `.npz` samples.

#### Create Training Sequences

After preprocessing, generate the training sequence pickle files consumed by `WaymoSeqMultiView`, `VKittiSeqMultiView`, `DDADSeqMultiView`, `PandasetSeqMultiView`, and `OnceSeqMultiView`.

The intended batch entrypoint is:

```bash
sbatch slurm/make_seqs.slurm
```

Important notes:

- `slurm/make_seqs.slurm` launches temporal sequence generation for `waymo`, `once`, `ddad`, `pandaset`, `vkitti`, and `kitti`, plus surround sequence generation for the multi-camera datasets `waymo`, `once`, `ddad`, and `pandaset`.
- `sh/make_seqs.sh` calls the bundled `dataset_setup/base_make_seq.py`.
- With the public scripts as shipped, the expected sequence filenames are:
  - `seq_exact_len_sub5_stride9_all.pkl` for Waymo, DDAD, PandaSet, and ONCE temporal training
  - `seq_exact_len_sub5_stride9.pkl` for VKITTI and KITTI temporal runs
  - `seq_surround_all.pkl` for Waymo, DDAD, PandaSet, and ONCE surround training
- Single-camera datasets (`kitti`, `vkitti`) skip surround mode by design.

Once the processed roots and sequence pickle files are in place, the default training wrappers can read them directly from `$SCRATCH/data/...` without any further path edits.

### OccAny (Must3R + SAM2)

#### Download the Must3R Base Checkpoint

OccAny reconstruction and generation both rely on the Must3R base weights referenced by `sh/train_occany_recon.sh` and `sh/train_occany_gen.sh`:

```bash
mkdir -p checkpoints
curl -L https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512.pth \
  -o checkpoints/MUSt3R_512.pth
```

#### Run the Reconstruction Stage

`TRAIN_TASK_ID=0` dispatches `slurm/train_occany.slurm` to `sh/train_occany_recon.sh`:

```bash
sbatch --array=0 slurm/train_occany.slurm
```

Without SLURM, run the same stage directly with:

```bash
bash sh/train_occany_recon.sh
```

With the default script values, checkpoints and TensorBoard logs are written to:

```bash
$PROJECT/tb_log_occany/occany_recon
```

For **OccAny**, the final checkpoint for both reconstruction and generation is the **last checkpoint**, i.e. `checkpoint-last.pth`.

#### Run the Generation Stage

Before launching generation, point the helper script at the reconstruction checkpoint you just trained. By default it uses:

```bash
checkpoints/occany_recon.pth
```

In practice, `checkpoints/occany_recon.pth` should be a copy or symlink to that last reconstruction checkpoint.

If you want a different path, override `OCCANY_RECON_CKPT` inline:

```bash
OCCANY_RECON_CKPT=/path/to/occany_recon.pth bash sh/train_occany_gen.sh
```

Keep the Must3R base checkpoint available at `checkpoints/MUSt3R_512.pth`, or override `MUST3R_PRETRAINED_CKPT`. The generation stage still loads the base Must3R checkpoint in addition to `--pretrained_occany`.

Then launch the generation stage with:

```bash
sbatch --array=1 slurm/train_occany.slurm
```

Without SLURM, run the same stage directly with:

```bash
BATCH_SIZE=2 bash sh/train_occany_gen.sh
```

With the default script values, generation outputs are written to:

```bash
$PROJECT/tb_log_occany/occany_gen
```

### OccAny+ (Depth Anything 3 + SAM3)

#### Run the Reconstruction Stage

`TRAIN_TASK_ID=0` dispatches `slurm/train_occany_plus.slurm` to `sh/train_occany_plus_recon.sh`:

```bash
sbatch --array=0 slurm/train_occany_plus.slurm
```

Without SLURM, run the same stage directly with:

```bash
bash sh/train_occany_plus_recon.sh
```

With the default script values, checkpoints and TensorBoard logs are written to:

```bash
$PROJECT/tb_log_occany/occany_plus_recon
```

For **OccAny+**, we use the checkpoint at **epoch 50** as the final checkpoint for both reconstruction comparison and generation handoff for comparison convenience: all OccAny+ experiments run past 50 epochs within about **2 days on 16 A100 40GB GPUs**.

#### Run the Generation Stage

Before launching generation, point the helper script at the reconstruction checkpoint you just trained. By default it uses:

```bash
checkpoints/occany_plus_recon.pth
```

In practice, `checkpoints/occany_plus_recon.pth` should usually be a copy or symlink of `checkpoint-50.pth`.

If you want a different path, override `OCCANY_PLUS_RECON_CKPT` inline:

```bash
OCCANY_PLUS_RECON_CKPT=/path/to/occany_plus_recon.pth bash sh/train_occany_plus_gen.sh
```

Then launch the generation stage with:

```bash
sbatch --array=1 slurm/train_occany_plus.slurm
```

Without SLURM, run the same stage directly with:

```bash
bash sh/train_occany_plus_gen.sh
```

With the default script values, generation outputs are written to:

```bash
$PROJECT/tb_log_occany/occany_plus_gen
```

### Training Outputs

For both training backends, `--output_dir` is the canonical experiment directory. It stores:

- TensorBoard event files
- `log.txt`
- `checkpoint-last.pth`
- `checkpoint-final.pth`
- periodic `checkpoint-<epoch>.pth` snapshots

To inspect an experiment with TensorBoard, point `--logdir` at the same `--output_dir` used for training. For example:

```bash
tensorboard --logdir "$PROJECT/tb_log_occany/occany_recon"
```

## 📄 License

This project is licensed under the Apache License 2.0, see the [LICENSE](LICENSE.txt) file for details.

## 🙏 Acknowledgments
We thank the authors of these excellent open-source projects:

[Dust3r](https://github.com/naver/dust3r) · [Must3r](https://github.com/naver/must3r) · [Depth-Anything-3](https://github.com/ByteDance-Seed/depth-anything-3) · [SAM2](https://github.com/facebookresearch/sam2) · [SAM3](https://github.com/facebookresearch/sam3) · [viser](https://github.com/nerfstudio-project/viser)

## 📝 Citation

If you find this work or code useful, please cite the paper and consider starring the repository:

```bibtex
@inproceedings{cao2026occany,
  title={OccAny: Generalized Unconstrained Urban 3D Occupancy},
  author={Anh-Quan Cao and Tuan-Hung Vu},
  booktitle={CVPR},
  year={2026}
}
```
