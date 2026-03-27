# Dataset setup overview

This directory collects the public dataset utilities used by OccAny for evaluation and training.

## What lives here

- `base_make_seq.py` builds the sequence pickle files consumed by the training wrappers in `sh/train_occany*.sh`.
- `extract_occ3d.sh`, `occ3d_nuscenes.md`, and `nuscenes/download.py` help with the evaluation-only Occ3D nuScenes assets.
- `waymo/`, `vkitti/`, `ddad/`, `once/`, and `pandaset/` contain dataset-specific preprocessing helpers.
- The `make_pairs.py` helpers under some dataset folders are optional legacy utilities for JPEG-exported trees. They are separate from the training sequence pickle generation done by `base_make_seq.py`.

## Common conventions

- Keep raw archives under a project-local location such as `$PROJECT/data/raw/...`.
- Write processed outputs to `$SCRATCH/data/<dataset>_processed` if you want the bundled training scripts to work without path edits.
- Most preprocessors in this tree convert raw data into compressed per-camera `.npz` files that contain image and geometry metadata.

The main README documents the end-to-end training and evaluation flows. Use this file as a quick map of the dataset utilities and the filenames that need to line up with the training scripts.

## Dataset entrypoints

### Evaluation-only nuScenes assets

- `nuscenes/download.py` downloads the official nuScenes camera data.
- `extract_occ3d.sh` is a site-specific helper for unpacking Occ3D nuScenes voxel annotations.
- `occ3d_nuscenes.md` captures one environment-specific extraction example.

These tools support the evaluation pipeline in the root [`README.md`](../README.md). They are not part of the default training wrappers.

### Training datasets

- Waymo: `waymo/preprocess_waymo.py`
- VKITTI2: `vkitti/preprocess_vkitti.py`
- DDAD: `ddad/preprocess.py`
- ONCE: `once/preprocess.py`
- PandaSet: `pandaset/preprocess.py`

Additional dataset-specific notes:

- DDAD dependency notes: [`ddad/README.md`](ddad/README.md)
- PandaSet dependency notes: [`pandaset/README.md`](pandaset/README.md)

## Extra dependencies by dataset

- Waymo preprocessing needs `gcsfs` and `waymo-open-dataset-tf-2-12-0==1.6.4`.
- DDAD preprocessing needs TRI-ML's `dgp` package and `protobuf==3.20.1`.
- PandaSet preprocessing needs the `pandaset-devkit`.

VKITTI2 and ONCE use the shipped scripts directly once the raw data is extracted into the expected folder structure.

## Sequence files expected by training

After preprocessing, generate the sequence pickle files consumed by the training wrappers. The public entrypoints are:

```bash
bash sh/make_seqs.sh
sbatch slurm/make_seqs.slurm
```

`sh/make_seqs.sh` calls the bundled `dataset_setup/base_make_seq.py`. With the public scripts as shipped, the expected filenames are:

- `seq_exact_len_sub5_stride9_all.pkl` for Waymo, DDAD, PandaSet, and ONCE temporal training
- `seq_exact_len_sub5_stride9.pkl` for VKITTI and KITTI temporal runs
- `seq_surround_all.pkl` for Waymo, DDAD, PandaSet, and ONCE surround training

Single-camera datasets (`kitti`, `vkitti`) skip surround mode by design.

`base_make_seq.py` also exposes `occ3d_nuscenes` and `occ3d_nuscenes_all` surround modes if you want to build surround-view sequence files for custom experiments, but those modes are not wired into `slurm/make_seqs.slurm` by default.

## Minimal workflow

1. Download or extract the raw dataset into your preferred raw-data root.
2. Run the dataset-specific preprocessor to populate `$SCRATCH/data/<dataset>_processed`.
3. Generate the required sequence pickle files with `sh/make_seqs.sh` or `slurm/make_seqs.slurm`.
4. Verify that the filenames expected by the training wrappers are present before launching `sh/train_occany*.sh`.

For the exact shell commands used by the public release, see the training section of the root [`README.md`](../README.md).
