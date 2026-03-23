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

**Paper and arXiv links will be added once they are public.**

**TL;DR:** A unified framework for generalized unconstrained urban 3D occupancy prediction.
<!-- <p align="center">
  <img src="/assets/teaser.jpg" alt="Pipeline">
</p> -->

</div>




https://github.com/user-attachments/assets/a3abde5b-c170-4b5d-818c-262b0df2546f



---

OccAny provides demo and inference code for urban 3D occupancy under unconstrained inputs. This repository currently includes two model variants:

- **OccAny** which is based on Must3R and SAM2,
- **OccAny+** which is based on Depth Anything 3 and SAM3

The repository includes sample RGB inputs in `demo_data/input`, pretrained weights in `checkpoints/`, and visualization tools for both point clouds and voxel grids.

## Citation

If you find this work or code useful, please cite the paper and consider starring the repository:

```bibtex
@inproceedings{cao2026occany,
  title={OccAny: Generalized Unconstrained Urban 3D Occupancy},
  author={Anh-Quan Cao and Tuan-Hung Vu},
  booktitle={CVPR},
  year={2026}
}
```

## Table of contents

- [Installation](#installation)
- [Checkpoints](#checkpoints)
- [Quick start](#quick-start)
- [Inference recipes](#inference-recipes)
- [Key inference flags](#key-inference-flags)
- [Visualization](#visualization)
- [Outputs](#outputs)

## 📝 TO-DO List

- [x] Inference code for OccAny (Must3R + SAM2) and OccAny+ (DA3 + SAM3)
- [x] Pretrained checkpoints
- [ ] Evaluation code for nuScenes and KITTI
- [ ] Dataset preparation scripts for Waymo, PandaSet, DDAD, VKitti, ONCE
- [ ] Training code for OccAny (Must3R + SAM2) and OccAny+ (DA3 + SAM3)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/valeoai/OccAny.git
cd OccAny
```

### 2. Create a Python environment

```bash
conda create -n occany python=3.12 -y
conda activate occany
python -m pip install --upgrade pip setuptools wheel ninja
```

### 3. Install PyTorch and CUDA

```bash
conda install -c nvidia cuda-toolkit=12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install xformers==0.0.29.post2
```

### 4. Install shared Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Install `torch-scatter`

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
pip install torch-scatter --no-cache-dir --no-build-isolation
```

### 6. Use the vendored third-party code

OccAny relies on the copies bundled in `third_party/`:

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

### 7. Compile CroCo's `curope` extension (recommended)

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

### 8. Optional sanity check

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

## Checkpoints

Download checkpoints with:

```bash
cd OccAny
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='anhquancao/OccAny', repo_type='model', local_dir='.', allow_patterns='checkpoints/*')"
```

Expected files under `checkpoints/`:

- `groundingdino_swinb_cogcoor.pth`
- `occany_da3_gen.pth`
- `occany_da3_recon.pth`
- `occany_must3r.pth`
- `sam2.1_hiera_large.pt`

## Inference

After installation, the demo commands below can be run as-is. By default:

- RGB inputs are read from `./demo_data/input`
- Outputs are written to `./demo_data/output`
- The repo already includes sample input scenes such as `kitti_08_1390` and `nuscenes_scenes-0039`

### OccAny+ (Depth Anything 3 + SAM3)

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

### OccAny (Must3R + SAM2)

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

## Key inference flags

The most commonly adjusted flags fall into three groups: common flags, semantic flags, and generation-specific flags. If you only want reconstruction output, omit `--gen` and any flag whose scope below is `Generation` or `Generation + semantic`.

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

## Visualization

### Point-cloud visualization with `viser`

Use `vis_viser.py` to inspect the saved `pts3d_*.npy` outputs interactively:

```bash
python vis_viser.py --input_folder ./demo_data/output
```

You can point `--input_folder` either to the output root or directly to a single scene folder. In the viewer, the common dropdown options are:

- `render` for reconstruction output
- `render_gen` for generated-view output
- `render_recon_gen` for the combined output

### Voxel visualization with `mayavi`

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

## Outputs

Each processed scene is written under `./demo_data/output/<frame_id>_<model>/`. Typical artifacts include:

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

## License

This project is licensed under the Apache License 2.0, see the [LICENSE](LICENSE.txt) file for details.

## Acknowledgments
We thanks the authors of these great repositories [Dust3r](https://github.com/naver/dust3r), [Must3r](https://github.com/naver/must3r), [Depth-Anything-3](https://github.com/ByteDance-Seed/depth-anything-3), [SAM2](https://github.com/facebookresearch/sam2), [SAM3](https://github.com/facebookresearch/sam3) and [viser](https://github.com/nerfstudio-project/viser).
