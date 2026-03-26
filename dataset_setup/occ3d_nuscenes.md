# Occ3D NuScenes Dataset Extraction

## Source and Target Directories

**Source:** `/lustre/fsstor/projects/rech/trg/uyl37fq/Occ3D_nuscenes/voxel04`
**Target:** `/lustre/fsn1/projects/rech/kvd/uyl37fq/data/occ3d_nuscenes`

## Source Files

```
-rw-r----- 1 uyl37fq trg 144M May  6  2023 annotations.json
-rw-r----- 1 uyl37fq trg 2.6G Jun  3 17:52 gts.tar.gz
-rw-r--r-- 1 uyl37fq kvd  29G Oct 24 10:03 imgs.tar.gz
```

## Extraction Instructions

Run the extraction script:

```bash
bash datasets_preprocess/nuscenes/extract_occ3d.sh
```

This will:
1. Create the target directory if it doesn't exist
2. Copy `annotations.json` (144M) to the target directory
3. Extract `gts.tar.gz` (2.6G) to the target directory with progress bar
4. Extract `imgs.tar.gz` (29G) to the target directory with progress bar

**Progress Display:**
- The script uses `pv` (Pipe Viewer) to show real-time progress with transfer rate and ETA
- If `pv` is not available, it falls back to verbose tar output with file count progress
- Progress format: `[current size] [progress bar] [percentage] [transfer rate] [ETA]`

**Note:** The imgs.tar.gz extraction may take 10-30 minutes depending on I/O speed.

## Create training data format 
```bash 
python datasets_preprocess/nuscenes/preprocess_nuscenes.py \
    --split val \
    --output_resolution 1024 576 \
    --debug
```

## Test the generated file
```bash
python datasets_preprocess/test_project_lidar.py \
    --seq_dir /scratch/project/eu-25-92/data/occ3d_nuscenes_processed/scene-0555 \
    --frame_ids 002362_5 002364_5
```