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

OccAny provides demo inputs, pretrained checkpoints, inference scripts, evaluation utilities, and visualization tools for urban 3D occupancy under unconstrained camera inputs. This public release includes two model variants:

- **OccAny**, based on Must3R + SAM2
- **OccAny+**, based on Depth Anything 3 + SAM3

The repository also includes sample RGB scenes in `demo_data/input`, pretrained weights in `checkpoints/`, and viewers for both point-cloud and voxel-grid outputs.

## 1. Citation

If you find this work or code useful, please cite the paper and consider starring the repository:

```bibtex
@inproceedings{cao2026occany,
  title={OccAny: Generalized Unconstrained Urban 3D Occupancy},
  author={Anh-Quan Cao and Tuan-Hung Vu},
  booktitle={CVPR},
  year={2026}
}
```

## 2. Table of contents

- [3. Roadmap](#3-roadmap)
- [4. Installation](#4-installation)
- [5. Checkpoints](#5-checkpoints)
- [6. Quick start](#6-quick-start)
- [7. Key inference flags](#7-key-inference-flags)
- [8. Visualization](#8-visualization)
- [9. Evaluation](#9-evaluation)
- [10. License](#10-license)
- [11. Acknowledgments](#11-acknowledgments)

## 3. Roadmap

- [x] Inference code for OccAny (Must3R + SAM2) and OccAny+ (DA3 + SAM3)
- [x] Pretrained checkpoints
- [x] Evaluation code for nuScenes and KITTI
- [ ] Dataset preparation scripts for Waymo, PandaSet, DDAD, VKitti, ONCE
- [ ] Training code for OccAny (Must3R + SAM2) and OccAny+ (DA3 + SAM3)

## 4. Installation

The commands below create the environment used for the public release and keep all required third-party dependencies local to this repository.

### 4.1 Clone the repository

```bash
git clone https://github.com/valeoai/OccAny.git
cd OccAny
```

### 4.2 Create a Python environment

```bash
conda create -n occany python=3.12 -y
conda activate occany
python -m pip install --upgrade pip setuptools wheel ninja
```

### 4.3 Install PyTorch and CUDA

```bash
conda install -c nvidia cuda-toolkit=12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install xformers==0.0.29.post2
```

### 4.4 Install shared Python dependencies

```bash
pip install -r requirements.txt
```

### 4.5 Install `torch-scatter`

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
pip install torch-scatter --no-cache-dir --no-build-isolation
```

### 4.6 Use the vendored third-party code

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

### 4.7 Compile CroCo's `curope` extension (recommended)

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

### 4.8 Optional sanity check

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

## 5. Checkpoints

Model checkpoints are hosted on Hugging Face:

- [anhquancao/OccAny/tree/main/checkpoints](https://huggingface.co/anhquancao/OccAny/tree/main/checkpoints)

Download checkpoints with:

```bash
cd OccAny
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='anhquancao/OccAny', repo_type='model', local_dir='.', allow_patterns='checkpoints/*')"
```

Expected files under `checkpoints/`:

- `occany_da3_gen.pth`
- `occany_da3_recon.pth`
- `occany_must3r.pth`
- `groundingdino_swinb_cogcoor.pth`
- `sam2.1_hiera_large.pt`

## 6. Quick start

After installation and checkpoint download, you can run the demo commands below from the repo root as-is. By default:

- RGB inputs are read from `./demo_data/input`
- Outputs are written to `./demo_data/output`
- The repo already includes sample scenes such as `kitti_08_1390` and `nuscenes_scenes-0039`

### 6.1 Inference recipes

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

- `pts3d_render.npy` for reconstruction views
- `pts3d_render_gen.npy` for generated views when `--gen` is enabled
- `pts3d_render_recon_gen.npy` for the merged point-cloud output
- `voxel_predictions.pkl` for voxelized predictions and visualization metadata

`inference.py` currently uses an urban voxel grid tuned for the included demo scenes:

- `voxel_size = 0.4`
- `occ_size = [200, 200, 24]`
- `voxel_origin = [-40.0, -40.0, -3.6]`

If you need a different dataset convention or voxel layout, update these values in `inference.py` before running inference. Two common presets are:

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


## 7. Key inference flags

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




## 8. Visualization

### 8.1 Point-cloud visualization with `viser`

Use `vis_viser.py` to inspect the saved `pts3d_*.npy` point-cloud outputs interactively:

```bash
python vis_viser.py --input_folder ./demo_data/output
```

You can point `--input_folder` either to the output root or directly to a single scene folder. In the viewer, the common dropdown options are:

- `render` for reconstruction output
- `render_gen` for generated-view output
- `render_recon_gen` for the combined output

### 8.2 Voxel visualization with `mayavi`

`vis_voxel.py` renders voxel predictions to image files. Install `mayavi` separately if you want to use this path:

```bash
pip install mayavi
python vis_voxel.py --input_root ./demo_data/output --dataset nuscenes
```

Helpful notes:

- The script writes rendered images to `./output` by default
- If the requested `--prediction_key` is missing, it automatically falls back to the best available `render*` grid
- Use `--dataset kitti` for KITTI-style scenes and `--dataset nuscenes` for nuScenes-style surround-view scenes
- Add `--save_input_images` if you also want stacked input RGB images next to the voxel render




## 9. Evaluation

This section covers the end-to-end evaluation workflow for KITTI and nuScenes using the provided shell and SLURM wrappers.


### 9.1 Download evaluation datasets
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


### 9.2 Prepare output and dataset roots

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






### 9.3 Local shell workflow

> [!CAUTION] 
> Evaluation can take a very long time on a single process because some extraction presets generate up to 180 novel views. We therefore provide the SLURM commands in the [9.4 SLURM](#94-slurm) section, which run 20 processes in parallel for occupancy extraction. We have only tested the SLURM path but the local shell should output the same results.

> [!NOTE]
> To maximize performance, some presets sample novel-view densely and therefore generate roughly 150 to 180 views. You can reduce runtime by lowering `-vpi` (the number of generated views per reconstruction view). In general, the total number of novel views is `n_recon x args.vpi x (3 if args.rot > 0 else 1)`.

Evaluation is a two-step workflow:

1. Run `extract_output_occany.py` through `sh/eval_occany.sh` (or `slurm/eval_occany.slurm`) to save voxel predictions.
2. Run `compute_metrics_from_saved_voxels.py` through `sh/compute_metric.sh` (or `slurm/compute_metric.slurm`) to compute SSC metrics from the saved `voxel_predictions.pkl` files.


Run both commands in each block from the repo root. Each block below mirrors the corresponding SLURM example without the `sbatch` wrapper.

#### OccAny

##### KITTI 5-frame geometry

```bash
EXP_LIST=occany EXP_ID=0 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=separate EXP_LIST=metric_occany EXP_ID=0 bash sh/compute_metric.sh
```

##### KITTI 1-frame geometry

```bash
EXP_LIST=occany EXP_ID=1 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=separate EXP_LIST=metric_occany EXP_ID=1 bash sh/compute_metric.sh
```

##### nuScenes 5-frame geometry

```bash
EXP_LIST=occany EXP_ID=2 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=separate EXP_LIST=metric_occany EXP_ID=2 bash sh/compute_metric.sh
```

##### nuScenes surround geometry

```bash
EXP_LIST=occany EXP_ID=3 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=separate EXP_LIST=metric_occany EXP_ID=3 bash sh/compute_metric.sh
```

##### KITTI 5-frame distill semantic

```bash
EXP_LIST=occany EXP_ID=4 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=separate EXP_LIST=metric_occany EXP_ID=4 bash sh/compute_metric.sh
```


##### nuScenes surround distill semantic

```bash
EXP_LIST=occany EXP_ID=5 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=separate EXP_LIST=metric_occany EXP_ID=5 bash sh/compute_metric.sh
```



##### KITTI 5-frame pretrained semantic

```bash
EXP_LIST=occany EXP_ID=6 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=separate EXP_LIST=metric_occany EXP_ID=6 bash sh/compute_metric.sh
```



##### nuScenes surround pretrained semantic
```bash
EXP_LIST=occany EXP_ID=7 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=separate EXP_LIST=metric_occany EXP_ID=7 bash sh/compute_metric.sh
```



#### OccAny+

##### KITTI 5-frame geometry

```bash
EXP_LIST=occany_plus EXP_ID=0 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=separate EXP_LIST=metric_occany_plus EXP_ID=0 bash sh/compute_metric.sh
```

##### nuScenes surround geometry

```bash
EXP_LIST=occany_plus EXP_ID=1 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=separate EXP_LIST=metric_occany_plus EXP_ID=1 bash sh/compute_metric.sh
```

##### KITTI 5-frame distill semantic

```bash
EXP_LIST=occany_plus EXP_ID=2 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=unified EXP_LIST=metric_occany_plus EXP_ID=2 bash sh/compute_metric.sh
```

##### nuScenes surround distill semantic

```bash
EXP_LIST=occany_plus EXP_ID=3 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=unified EXP_LIST=metric_occany_plus EXP_ID=3 bash sh/compute_metric.sh
```

##### KITTI 5-frame pretrained semantic

```bash
EXP_LIST=occany_plus EXP_ID=4 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=unified EXP_LIST=metric_occany_plus EXP_ID=4 bash sh/compute_metric.sh
```

##### nuScenes surround pretrained semantic

```bash
EXP_LIST=occany_plus EXP_ID=5 bash sh/eval_occany.sh
USE_MAJORITY_POOLING=1 POOLING_MODE=unified EXP_LIST=metric_occany_plus EXP_ID=5 bash sh/compute_metric.sh
```


### 9.4 SLURM

Some extraction presets can generate up to 180 views, so extraction can be slow. The provided `slurm/eval_occany.slurm` script therefore runs a 20-task array in parallel by default (`#SBATCH --array=0-19` with `WORLD=20`). 

Each example below submits the extraction job first and then chains the metric job with `--dependency=afterany:$(...)`, so the metric job waits until the full extraction array finishes. The public SLURM wrappers keep the Karolina-HPC defaults (`-A eu-25-92`, `--partition=qgpu`, `--hint=nomultithread`, `--cpus-per-task=16`, and `conda activate occany`); update those settings to match your cluster.

#### OccAny

##### KITTI 5-frame geometry (Tab. 1)
Precision, Recall, IoU: 36.79, 46.70, 25.91
```bash
sbatch --dependency=afterany:$(sbatch --parsable --export=EXP_LIST=occany,EXP_ID=0,WORLD=20 slurm/eval_occany.slurm) --export=EXP_LIST=metric_occany,EXP_ID=0,USE_MAJORITY_POOLING=1,POOLING_MODE=separate slurm/compute_metric.slurm
```

##### KITTI 1-frame geometry (Tab. 2)
Precision, Recall, IoU: 45.64, 33.66, 24.03
```bash
sbatch --dependency=afterany:$(sbatch --parsable --export=EXP_LIST=occany,EXP_ID=1,WORLD=20 slurm/eval_occany.slurm) --export=EXP_LIST=metric_occany,EXP_ID=1,USE_MAJORITY_POOLING=1,POOLING_MODE=separate slurm/compute_metric.slurm
```

##### nuScenes 5-frame geometry (Tab. 1)
Precision, Recall, IoU: 36.09, 40.39, 23.55
```bash
sbatch --dependency=afterany:$(sbatch --parsable --export=EXP_LIST=occany,EXP_ID=2,WORLD=20 slurm/eval_occany.slurm) --export=EXP_LIST=metric_occany,EXP_ID=2,USE_MAJORITY_POOLING=1,POOLING_MODE=separate slurm/compute_metric.slurm
```

##### nuScenes surround geometry (Tab. 3)
Precision, Recall, IoU: 45.04, 58.54, 34.15
```bash
sbatch --dependency=afterany:$(sbatch --parsable --export=EXP_LIST=occany,EXP_ID=3,WORLD=20 slurm/eval_occany.slurm) --export=EXP_LIST=metric_occany,EXP_ID=3,USE_MAJORITY_POOLING=1,POOLING_MODE=separate slurm/compute_metric.slurm
```

##### KITTI 5-frame distill semantic (Tab. 5, 6)
mIoU: 7.30, mIoU^{sc}: 13.54 (Slightly higher than 7.28, 13.53 in paper)
```bash
sbatch --dependency=afterany:$(sbatch --parsable --export=EXP_LIST=occany,EXP_ID=4,WORLD=20 slurm/eval_occany.slurm) --export=EXP_LIST=metric_occany,EXP_ID=4,USE_MAJORITY_POOLING=1,POOLING_MODE=separate slurm/compute_metric.slurm
```

##### nuScenes surround distill semantic (Tab. 5, 6)
mIoU: 6.65, mIoU^{sc}: 10.31 (Slight variation w.r.t 6.66, 10.32 in paper)
```bash
sbatch --dependency=afterany:$(sbatch --parsable --export=EXP_LIST=occany,EXP_ID=5,WORLD=20 slurm/eval_occany.slurm) --export=EXP_LIST=metric_occany,EXP_ID=5,USE_MAJORITY_POOLING=1,POOLING_MODE=separate slurm/compute_metric.slurm
```


##### KITTI 5-frame pretrained semantic (Tab. 7)
mIoU: 7.62, mIoU^{sc}: 13.75 (Slight variation w.r.t 7.67, 13.75 in paper)
```bash
sbatch --dependency=afterany:$(sbatch --parsable --export=EXP_LIST=occany,EXP_ID=6,WORLD=20 slurm/eval_occany.slurm) --export=EXP_LIST=metric_occany,EXP_ID=6,USE_MAJORITY_POOLING=1,POOLING_MODE=separate slurm/compute_metric.slurm
```

##### nuScenes surround pretrained semantic (Tab. 7)
mIoU: 7.42, mIoU^{sc}: 10.78
```bash
sbatch --dependency=afterany:$(sbatch --parsable --export=EXP_LIST=occany,EXP_ID=7,WORLD=20 slurm/eval_occany.slurm) --export=EXP_LIST=metric_occany,EXP_ID=7,USE_MAJORITY_POOLING=1,POOLING_MODE=separate slurm/compute_metric.slurm
```

#### OccAny+

For OccAny+, geometry metrics use separate pooling, while semantic metrics use unified pooling (pool geometry first, then semantics).
##### KITTI 5-frame geometry (Tab. 5)
Precision, Recall, IoU: 38.11, 49.13, 27.33
```bash
sbatch --dependency=afterany:$(sbatch --parsable --export=EXP_LIST=occany_plus,EXP_ID=0,WORLD=20 slurm/eval_occany.slurm) --export=EXP_LIST=metric_occany_plus,EXP_ID=0,USE_MAJORITY_POOLING=1,POOLING_MODE=separate slurm/compute_metric.slurm
```

##### nuScenes surround geometry (Tab. 5)
Precision, Recall, IoU: 46.37, 54.67, 33.49
```bash
sbatch --dependency=afterany:$(sbatch --parsable --export=EXP_LIST=occany_plus,EXP_ID=1,WORLD=20 slurm/eval_occany.slurm) --export=EXP_LIST=metric_occany_plus,EXP_ID=1,USE_MAJORITY_POOLING=1,POOLING_MODE=separate slurm/compute_metric.slurm
```

##### KITTI 5-frame distill semantic (Tab. 7)
mIoU: 6.49, mIoU^{sc}: 13.31 (Slight variation w.r.t. 6.48, 13.30 in the paper)
```bash
sbatch --dependency=afterany:$(sbatch --parsable --export=EXP_LIST=occany_plus,EXP_ID=2,WORLD=20 slurm/eval_occany.slurm) --export=EXP_LIST=metric_occany_plus,EXP_ID=2,USE_MAJORITY_POOLING=1,POOLING_MODE=unified slurm/compute_metric.slurm
```

##### nuScenes surround distill semantic (Tab. 7)
mIoU: 7.20, mIoU^{sc}: 11.51
```bash
sbatch --dependency=afterany:$(sbatch --parsable --export=EXP_LIST=occany_plus,EXP_ID=3,WORLD=20 slurm/eval_occany.slurm) --export=EXP_LIST=metric_occany_plus,EXP_ID=3,USE_MAJORITY_POOLING=1,POOLING_MODE=unified slurm/compute_metric.slurm
```

##### KITTI 5-frame pretrained semantic (Tab. 7)
mIoU: 8.03, mIoU^{sc}: 13.17
```bash
sbatch --dependency=afterany:$(sbatch --parsable --export=EXP_LIST=occany_plus,EXP_ID=4,WORLD=20 slurm/eval_occany.slurm) --export=EXP_LIST=metric_occany_plus,EXP_ID=4,USE_MAJORITY_POOLING=1,POOLING_MODE=unified slurm/compute_metric.slurm
```

##### nuScenes surround pretrained semantic (Tab. 7)
mIoU: 9.45, mIoU^{sc}: 12.22
```bash
sbatch --dependency=afterany:$(sbatch --parsable --export=EXP_LIST=occany_plus,EXP_ID=5,WORLD=20 slurm/eval_occany.slurm) --export=EXP_LIST=metric_occany_plus,EXP_ID=5,USE_MAJORITY_POOLING=1,POOLING_MODE=unified slurm/compute_metric.slurm
```


## 10. License

This project is licensed under the Apache License 2.0, see the [LICENSE](LICENSE.txt) file for details.

## 11. Acknowledgments
We thank the authors of these excellent repositories: [Dust3r](https://github.com/naver/dust3r), [Must3r](https://github.com/naver/must3r), [Depth-Anything-3](https://github.com/ByteDance-Seed/depth-anything-3), [SAM2](https://github.com/facebookresearch/sam2), [SAM3](https://github.com/facebookresearch/sam3), and [viser](https://github.com/nerfstudio-project/viser).
