#!/bin/bash

# Default paths
SOURCE="${ONCE_ARCHIVE_DIR:-$PROJECT/data/raw/once_archives}"
DEST="${ONCE_RAW_DIR:-$PROJECT/data/raw/ONCE}"

# Find all tar files in SOURCE directory
TAR_FILES=$(find "$SOURCE" -type f -regex '.*\.tar$')

# Echo the list of tar files
echo "Found the following tar files:"
echo "$TAR_FILES"
echo "Total: $(echo "$TAR_FILES" | wc -l) files"

# Create results file if it doesn't exist
if [ -f "$DEST/res.txt" ]; then
    rm "$DEST/res.txt"
fi
touch "$DEST/res.txt"

# Function to extract a tar file
extract_tar() {
    local tar_file=$1
    local filename=$(basename "$tar_file")
    
    echo "Extracting $filename..."
    
    # Create target directory if it doesn't exist
    mkdir -p "$DEST"
    
    # Extract the tar file
    if [[ "$tar_file" == *.tar.gz || "$tar_file" == *.tgz ]]; then
        tar -xzf "$tar_file" -C "$DEST"
    else
        tar -xf "$tar_file" -C "$DEST"
    fi
    
    # Check if extraction was successful
    if [ $? -eq 0 ]; then
        # Record successful extraction
        echo "$filename: DONE" >> $DEST/res.txt
        echo "Finished extracting $filename"
    else
        echo "$filename: FAILED" >> $DEST/res.txt
        echo "Failed to extract $filename"
    fi
}

# Export the function so it can be used by parallel
export -f extract_tar
export DEST

# Run extraction in parallel with 16 workers using GNU parallel if available
if command -v parallel >/dev/null 2>&1; then
    echo "$TAR_FILES" | parallel -j 16 --will-cite --line-buffer --eta 'extract_tar {}'
else
    echo "GNU parallel not found; falling back to xargs."
    echo "$TAR_FILES" | xargs -I{} -P 16 bash -c 'extract_tar "{}"'
fi

echo "All extractions completed. Results saved in $DEST/res.txt"