#!/bin/bash
module load pv/1.8.5

# Define paths
SOURCE_DIR=/lustre/fsstor/projects/rech/trg/uyl37fq/Occ3D_nuscenes/voxel04
TARGET_DIR=/lustre/fsn1/projects/rech/kvd/uyl37fq/data/occ3d_nuscenes

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "========================================"
echo "Occ3D NuScenes Dataset Extraction"
echo "========================================"
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"
echo "========================================"
echo ""

# Copy annotations.json
echo "[1/3] Copying annotations.json (144M)..."
cp -v "$SOURCE_DIR/annotations.json" "$TARGET_DIR/"
if [ $? -eq 0 ]; then
    echo "✓ annotations.json copied successfully"
else
    echo "✗ Failed to copy annotations.json"
    exit 1
fi
echo ""

# Extract gts.tar.gz with progress
echo "[2/3] Extracting gts.tar.gz (2.6G)..."
echo "Progress:"
pv "$SOURCE_DIR/gts.tar.gz" | tar -xzf - -C "$TARGET_DIR" 2>&1 | grep -v "tar:"
if [ ${PIPESTATUS[1]} -eq 0 ]; then
    echo "✓ gts.tar.gz extracted successfully"
else
    echo "⚠ pv not available, using tar with verbose mode..."
    tar -xzvf "$SOURCE_DIR/gts.tar.gz" -C "$TARGET_DIR" | pv -l -s $(tar -tzf "$SOURCE_DIR/gts.tar.gz" | wc -l) > /dev/null
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ gts.tar.gz extracted successfully"
    else
        echo "✗ Failed to extract gts.tar.gz"
        exit 1
    fi
fi
echo ""

# Extract imgs.tar.gz with progress
echo "[3/3] Extracting imgs.tar.gz (29G) - this will take a while..."
echo "Progress:"
pv "$SOURCE_DIR/imgs.tar.gz" | tar -xzf - -C "$TARGET_DIR" 2>&1 | grep -v "tar:"
if [ ${PIPESTATUS[1]} -eq 0 ]; then
    echo "✓ imgs.tar.gz extracted successfully"
else
    echo "⚠ pv not available, using tar with verbose mode..."
    tar -xzvf "$SOURCE_DIR/imgs.tar.gz" -C "$TARGET_DIR" | pv -l -s $(tar -tzf "$SOURCE_DIR/imgs.tar.gz" | wc -l) > /dev/null
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ imgs.tar.gz extracted successfully"
    else
        echo "✗ Failed to extract imgs.tar.gz"
        exit 1
    fi
fi
echo ""

echo "========================================"
echo "✓ All files extracted successfully!"
echo "========================================"
echo ""
echo "Target directory contents:"
ls -lh "$TARGET_DIR"
echo ""
echo "Disk usage:"
du -sh "$TARGET_DIR"
