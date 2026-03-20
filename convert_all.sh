#!/bin/bash

# Configuration
DATA_ROOT="/local_data/lim2045/vla/libero_data"
HF_USER="LucaMacesanu"

# Find every .hdf5 file
find "$DATA_ROOT" -name "*.hdf5" | while read -r hdf5_path; do
    
    # Extract the directory name (e.g., "open_the_middle_drawer_demo")
    # This removes the full path and the filename, leaving just the folder name
    TASK_NAME=$(basename "$(dirname "$hdf5_path")")
    
    # Construct the repo_id
    REPO_ID="${HF_USER}/${TASK_NAME}"

    echo "===================================================="
    echo "Task: $TASK_NAME"
    echo "Input: $hdf5_path"
    echo "Repo ID: $REPO_ID"
    echo "===================================================="

    # Run the conversion
    uv run convert_hdf5_to_lerobot.py \
        --input "$hdf5_path" \
        --repo_id "$REPO_ID"

    # Check for errors
    if [ $? -ne 0 ]; then
        echo "FAILED: $TASK_NAME"
    fi
    
    echo -e "\n"
done