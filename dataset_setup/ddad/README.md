# DDAD dataset setup

This folder contains the DDAD preprocessing helpers used in this repository. For the cross-dataset overview, see [`../README.md`](../README.md).

- `preprocess.py` converts raw DDAD into compressed per-camera `.npz` samples under `<preprocessed_root>/<split>_<scene>/`. Each saved file is named `<frame>_<camera_idx>.npz` and contains `image`, `depthmap`, `intrinsics`, and `cam2world`.
- `make_pairs.py` is an optional helper that creates forecast triplets from a processed DDAD tree that still contains `camera_XX_000000.jpg` files.

## Raw dataset layout

Pass `--ddad_root` as the directory that contains `ddad.json`, for example:

```text
$PROJECT/data/raw/DDAD/
└── ddad.json
```

## Output layout

`preprocess.py` writes per-camera samples like:

```text
<preprocessed_root>/
├── train_<scene>/
│   ├── 000000_0.npz
│   ├── 000000_1.npz
│   └── ...
└── val_<scene>/
    ├── 000000_0.npz
    ├── 000000_1.npz
    └── ...
```

Each `.npz` contains `image`, `depthmap`, `intrinsics`, and `cam2world`.

## Training sequence files

Preprocessing alone is not enough for training. The current `sh/train_occany*.sh` wrappers do **not** use the dataset-class default `ddad_seq_video.pkl`; they explicitly look for both of these files under the processed root:

```text
$SCRATCH/data/ddad_processed/seq_exact_len_sub5_stride9_all.pkl
$SCRATCH/data/ddad_processed/seq_surround_all.pkl
```

The bundled `sh/make_seqs.sh` delegates sequence generation to the bundled `dataset_setup/base_make_seq.py`, so after running it, verify that those exact filenames exist before launching training.

## Dependencies

`preprocess.py` depends on TRI-ML's `dgp` package and works with `protobuf==3.20.1`.

## Jeanzay

1. Load the environment and set a user install prefix.

```bash
module load pytorch-gpu/py3/2.2.0
mkdir -p $TRG_WORK/python_envs/dgp
export PYTHONUSERBASE=$TRG_WORK/python_envs/dgp
```

2. Install `dgp` and pin `protobuf`.

```bash
git clone https://github.com/TRI-ML/dgp.git
cd dgp
pip install --user --no-cache-dir -r requirements.txt -r requirements-dev.txt
pip install --user --no-cache-dir -e .
pip install --user --no-cache-dir protobuf==3.20.1
```

3. Run the DDAD preprocessor. A single invocation processes both the `train` and `val` splits.

```bash
python dataset_setup/ddad/preprocess.py \
  --ddad_root "$PROJECT/data/raw/DDAD" \
  --preprocessed_root "$SCRATCH/data/ddad_processed" \
  --n_workers 16
```

## Karolina

1. Load the environment and set a user install prefix.

```bash
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
export PYTHONUSERBASE="$HOME/.local"
```

2. Install `dgp` and pin `protobuf`.

```bash
git clone https://github.com/TRI-ML/dgp.git
cd dgp
pip install --user --no-cache-dir -r requirements.txt -r requirements-dev.txt
pip install --user --no-cache-dir -e .
pip install --user --no-cache-dir protobuf==3.20.1
```

3. Run the DDAD preprocessor. A single invocation processes both the `train` and `val` splits.

```bash
python dataset_setup/ddad/preprocess.py \
  --ddad_root "$PROJECT/data/raw/DDAD" \
  --preprocessed_root "$SCRATCH/data/ddad_processed" \
  --n_workers 16
```

## Optional pair generation

If you still maintain a JPEG-exported processed DDAD tree, you can build forecast pairs with:

```bash
python dataset_setup/ddad/make_pairs.py --subsampling_rate 1 --max_stride 9
```

`make_pairs.py` reads `$SCRATCH/data/ddad_processed` and writes `ddad_pairs_forecast_video_sub*_stride*.npz`. It expects JPEGs named like `camera_01_000000.jpg`, while the current `preprocess.py` writes `.npz` samples only, so this helper is only relevant if you generate those JPEGs separately. It is separate from the training-sequence pickles consumed by `sh/train_occany*.sh`.
