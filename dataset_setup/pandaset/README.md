# PandaSet dataset setup

This folder contains the PandaSet preprocessing helpers used in this repository. For the cross-dataset overview, see [`../README.md`](../README.md).

- `preprocess.py` converts extracted PandaSet data into compressed per-camera `.npz` samples under `<save_dir>/<scene>/`. Each saved file is named `<frame>_<camera_idx>.npz` and contains `image`, `depthmap`, `intrinsics`, and `cam2world`.
- `make_pairs.py` is an optional helper that creates forecast triplets from a processed PandaSet tree that still contains `<frame>_<camera>.jpg` files.

## Raw dataset layout

Pass `--root` to the extracted PandaSet root understood by `pandaset-devkit`. If `--root` is omitted, `preprocess.py` defaults to:

```text
$SCRATCH/data/PandaSet/
```

## Output layout

`preprocess.py` writes per-camera samples like:

```text
<save_dir>/
└── <scene>/
    ├── 000000_0.npz
    ├── 000000_1.npz
    └── ...
```

Each `.npz` contains `image`, `depthmap`, `intrinsics`, and `cam2world`.

## Training sequence files

Preprocessing alone is not enough for training. The current `sh/train_occany*.sh` wrappers do **not** use the dataset-class default `pandaset_seq_video.pkl`; they explicitly look for both of these files under the processed root:

```text
$SCRATCH/data/pandaset_processed/seq_exact_len_sub5_stride9_all.pkl
$SCRATCH/data/pandaset_processed/seq_surround_all.pkl
```

The bundled `sh/make_seqs.sh` delegates sequence generation to the bundled `dataset_setup/base_make_seq.py`, so after running it, verify that those exact filenames exist before launching training.

## Dependencies

Install the Scale API PandaSet devkit before running `preprocess.py`:

```bash
git clone https://github.com/scaleapi/pandaset-devkit.git
cd pandaset-devkit/python
pip install --user --no-cache-dir -e .
```

## Quick start

1. Extract PandaSet to a raw root such as:

```text
$PROJECT/data/raw/PandaSet/
```

2. Run the preprocessor:

```bash
python dataset_setup/pandaset/preprocess.py \
  --root "$PROJECT/data/raw/PandaSet" \
  --save_dir "$SCRATCH/data/pandaset_processed"
```

3. To shard work across multiple processes, launch the script with distinct `--pid` values and a shared `--nproc`, for example:

```bash
python dataset_setup/pandaset/preprocess.py \
  --root "$PROJECT/data/raw/PandaSet" \
  --save_dir "$SCRATCH/data/pandaset_processed" \
  --pid 0 \
  --nproc 4
```

Repeat the same command with `--pid 1`, `--pid 2`, and `--pid 3` in parallel.

## Karolina example

```bash
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

git clone https://github.com/scaleapi/pandaset-devkit.git
cd pandaset-devkit/python
pip install --user --no-cache-dir -e .

python dataset_setup/pandaset/preprocess.py \
  --root "$PROJECT/data/raw/PandaSet" \
  --save_dir "$SCRATCH/data/pandaset_processed"
```

## Optional pair generation

If you still maintain a JPEG-exported processed PandaSet tree, you can build forecast pairs with:

```bash
python dataset_setup/pandaset/make_pairs.py --subsampling_rate 1 --max_stride 9
```

`make_pairs.py` reads `$SCRATCH/data/pandaset_processed` and writes `pandaset_pairs_forecast_video_sub*_stride*.npz`. It expects JPEGs named like `000000_0.jpg`, while the current `preprocess.py` writes `.npz` samples only, so use this helper only if you generate those JPEGs separately. It is separate from the training-sequence pickles consumed by `sh/train_occany*.sh`.
