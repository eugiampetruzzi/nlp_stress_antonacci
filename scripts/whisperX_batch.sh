#!/bin/bash

#=============================================================================
# SLURM JOB DIRECTIVES
#=============================================================================
#
#SBATCH --job-name=WhisperX
#SBATCH --output=whisperx_job_%A_%a.out  # %A = job ID, %a = task ID
#SBATCH --error=whisperx_job_%A_%a.err   # Separate error log for each task
#SBATCH --time=01:00:00               # Max runtime *per file* (1 hour)
#SBATCH --cpus-per-task=4             # CPUs per task
#SBATCH --mem=16G                     # Memory per task
#SBATCH --partition=gpu               # Partition (change to your cluster's GPU partition)
#
# --- ⚠️ SET YOUR ARRAY LIMITS HERE ---
#
# Set the array index to match the number of files you have (e.g., 0-499 for 500 files)
# You can run 'find /path/to/audio/files -name "*.wav" | wc -l' to get the count
# Using '%10' limits this to 10 jobs running at the same time. Adjust as needed.
#
#SBATCH --array=0-499%10

#=============================================================================
# USER-CONFIGURABLE VARIABLES
#=============================================================================
#
# --- ⚠️ PASTE YOUR TOKEN HERE ---
# Paste your PyAnnote/Hugging Face token inside the quotes
PYANNOTE_HF_TOKEN=""

# --- Paths ---
# Path to your python virtual environment
VENV_DIR="$HOME/whisperx-venv"
# Path to the directory containing audio files
AUDIO_DIR="/path/to/audio/files"
# Path to the directory where transcripts will be saved
OUTPUT_DIR="/path/to/output/directory"

# --- Diarization Parameters ---
# (Optional) Specify number of speakers if known
MIN_SPEAKERS=2
MAX_SPEAKERS=2

#=============================================================================
# SCRIPT LOGIC (JOB ARRAY)
#=============================================================================
#
# This script no longer uses a "for" loop.
# Instead, SLURM runs this *entire script* once for each number in the
# --array (e.g., 500 times), and each run gets a unique $SLURM_ARRAY_TASK_ID.

# 1. Activate the virtual environment
echo "Activating virtual environment at $VENV_DIR..."
source "$VENV_DIR/bin/activate"

# 2. Create the output directory (run this command *once* manually in your
#    terminal *before* submitting the job, not here)
#    mkdir -p "$OUTPUT_DIR"

# 3. Get the list of all audio files into a BASH array
echo "Finding all audio files in $AUDIO_DIR..."
mapfile -t AUDIO_FILES < <(find "$AUDIO_DIR" -name "*.wav" -o -name "*.mp3" -o -name "*.wma")

# 4. Get the specific file for *this* job task
TASK_ID=$SLURM_ARRAY_TASK_ID
AUDIO_FILE=${AUDIO_FILES[$TASK_ID]}

# 5. Check if this task ID is valid (e.g., if array is 0-999 but we only have 500 files)
if [ -z "$AUDIO_FILE" ]; then
    echo "Task ID $TASK_ID is out of bounds (no file). Exiting."
    exit 0
fi

# 6. Extract filename and Subject ID
FILENAME=$(basename "$AUDIO_FILE")

# ⚠️ This line is fragile and assumes all filenames start with
# an 8-character ID (e.g., "ELS_123.").
# You may need to change this based on your naming convention.
SUB_ID=$(echo "$FILENAME" | cut -c 1-8)

OUTPUT_FILE="$OUTPUT_DIR/${SUB_ID}_whisperx_transcript.txt"

# 7. Check if the output file already exists (skip if it does)
if [ -f "$OUTPUT_FILE" ]; then
    echo "Output file already exists for $FILENAME (Task $TASK_ID). Skipping."
else
    echo "Processing $FILENAME (Task $TASK_ID)..."
    # 8. Run the Python script for this *one* file
    # (srun is not needed here; SLURM is already managing the job)
    python whisperX_process.py \
        "$AUDIO_FILE" \
        "$SUB_ID" \
        --pyannote_token "$PYANNOTE_HF_TOKEN" \
        --min_speakers "$MIN_SPEAKERS" \
        --max_speakers "$MAX_SPEAKERS" \
        --output_dir "$OUTPUT_DIR"
fi

# 9. Deactivate the virtual environment
deactivate

echo "Task $TASK_ID complete."