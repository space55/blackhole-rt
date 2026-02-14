#!/usr/bin/env bash
# ============================================================================
# submit_render.sh — Submit a black hole frame sequence to the Slurm cluster
# ============================================================================
# This is the main entry point for distributed rendering. It submits a Slurm
# job array where each task renders one frame.
#
# Usage:
#   ./submit_render.sh [options]
#
# Options:
#   -n NUM       Number of frames to render          (default: 60)
#   -t0 FLOAT    Starting time value                 (default: 0.0)
#   -dt FLOAT    Time step per frame                 (default: 0.5)
#   -p PREFIX    Frame filename prefix               (default: frame)
#   -d DIR       Project directory on compute nodes  (default: /opt/bhrt)
#   -s SCENE     Scene file name (relative to build) (default: scene.txt)
#   -o DIR       Output subdirectory under build/    (default: frames)
#   -P PARTITION Slurm partition to use              (default: gpu)
#   -c CPUS      CPUs per frame task                 (default: 4)
#   -m MEMORY    Memory per task (e.g. 4G)           (default: 4G)
#   -T TIME      Max wall time per frame             (default: 01:00:00)
#   --cpu        Submit to cpu only partition.       (default: off)
#   --exr        Also output multi-layer OpenEXR per frame (.exr)
#   --hdr        Also output Radiance HDR per frame (.hdr)
#   --dry-run    Show the sbatch command without submitting
#   -h           Show this help
# ============================================================================

set -euo pipefail

# ---------- Defaults ---------------------------------------------------------
NUM_FRAMES=60
TIME_START=0.0
TIME_STEP=0.5
PREFIX="frame"
PROJECT_DIR="/opt/bhrt"
SCENE_FILE="scene.txt"
OUTPUT_DIR="frames"
PARTITION="gpu"
CPUS=4
MEMORY="4G"
WALL_TIME="01:00:00"
USE_GPU=true
EXR_OUTPUT=false
HDR_OUTPUT=false
DRY_RUN=false

# ---------- Parse arguments --------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n)         NUM_FRAMES="$2"; shift 2 ;;
        -t0)        TIME_START="$2"; shift 2 ;;
        -dt)        TIME_STEP="$2"; shift 2 ;;
        -p)         PREFIX="$2"; shift 2 ;;
        -d)         PROJECT_DIR="$2"; shift 2 ;;
        -s)         SCENE_FILE="$2"; shift 2 ;;
        -o)         OUTPUT_DIR="$2"; shift 2 ;;
        -P)         PARTITION="$2"; shift 2 ;;
        -c)         CPUS="$2"; shift 2 ;;
        -m)         MEMORY="$2"; shift 2 ;;
        -T)         WALL_TIME="$2"; shift 2 ;;
        --cpu)      USE_GPU=false; PARTITION="cpu"; shift ;;
        --exr)      EXR_OUTPUT=true; shift ;;
        --hdr)      HDR_OUTPUT=true; shift ;;
        --dry-run)  DRY_RUN=true; shift ;;
        -h|--help)
            head -27 "$0" | tail -24
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ---------- Resolve paths ----------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/render_frame.sbatch"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "Error: Batch script not found: $SBATCH_SCRIPT" >&2
    exit 1
fi

LAST_FRAME=$((NUM_FRAMES - 1))
LOG_DIR="$PROJECT_DIR/build/logs"

# ---------- Pre-flight checks -----------------------------------------------
echo "============================================"
echo " BHRT Slurm Render Submission"
echo "============================================"
echo " Frames:     0..$LAST_FRAME ($NUM_FRAMES total)"
echo " Time range: $TIME_START to $(python3 -c "print(f'{${TIME_START} + ${LAST_FRAME} * ${TIME_STEP}:.2f}')")"
echo " Time step:  $TIME_STEP"
echo " Scene:      $SCENE_FILE"
echo " Output:     $PROJECT_DIR/build/$OUTPUT_DIR/"
echo " Partition:  $PARTITION"
echo " CPUs/task:  $CPUS"
echo " Memory:     $MEMORY"
echo " Wall time:  $WALL_TIME"
echo " GPU:        $USE_GPU"
echo " EXR output: $EXR_OUTPUT"
echo " HDR output: $HDR_OUTPUT"
echo "============================================"

# Check cluster is reachable
if ! command -v sbatch &>/dev/null; then
    echo "Error: sbatch not found. Is Slurm installed?" >&2
    exit 1
fi

if ! sinfo &>/dev/null; then
    echo "Error: Cannot reach Slurm controller. Check slurmctld is running." >&2
    exit 1
fi

echo
echo "Cluster status:"
sinfo -o '  %12P %6a %10F %8c %10m %10G' --noheader
echo

# ---------- Validate resources against cluster capacity ----------------------
# Query the target partition for what's actually available to avoid
# "Requested node configuration is not available" errors.
PART_INFO=$(sinfo -p "$PARTITION" -o '%c %m %G' --noheader 2>/dev/null | head -1 | tr -s ' ')
if [[ -z "$PART_INFO" ]]; then
    echo "Error: Partition '$PARTITION' not found. Available partitions:" >&2
    sinfo -o '  %P' --noheader >&2
    exit 1
fi

MAX_CPUS=$(echo "$PART_INFO" | awk '{print $1}')
MAX_MEM_MB=$(echo "$PART_INFO" | awk '{print $2}')
AVAIL_GRES=$(echo "$PART_INFO" | awk '{print $3}')

# Auto-fix CPUs if requesting more than the node has
if [[ "$CPUS" -gt "$MAX_CPUS" ]]; then
    echo "Warning: Requested $CPUS CPUs but partition '$PARTITION' has max $MAX_CPUS. Clamping." >&2
    CPUS=$MAX_CPUS
fi

# Auto-fix memory: convert requested memory to MB for comparison
REQ_MEM_MB=$(echo "$MEMORY" | awk '{
    val=$1; gsub(/[^0-9.]/,"",val);
    if ($1 ~ /[Gg]/) val=val*1024;
    if ($1 ~ /[Tt]/) val=val*1024*1024;
    printf "%d", val
}')
if [[ "$REQ_MEM_MB" -gt "$MAX_MEM_MB" ]]; then
    NEW_MEM="$((MAX_MEM_MB / 1024))G"
    echo "Warning: Requested $MEMORY but partition '$PARTITION' has max ${MAX_MEM_MB}MB. Clamping to $NEW_MEM." >&2
    MEMORY=$NEW_MEM
fi

# Check GPU availability if requesting a GPU
if $USE_GPU; then
    if [[ "$AVAIL_GRES" == "(null)" || -z "$AVAIL_GRES" ]]; then
        echo "Note: No Slurm GRES configured on partition '$PARTITION'." >&2
        echo "  Skipping --gres=gpu:1 flag (OK for WSL2 nodes — CUDA will use the GPU automatically)." >&2
        USE_GPU=false
    fi
fi

echo "Resolved resources: CPUs=$CPUS, Mem=$MEMORY, GPU=$USE_GPU"
echo

# ---------- Build sbatch command ---------------------------------------------
SBATCH_CMD=(
    sbatch
    --job-name="bhrt-render"
    --array="0-${LAST_FRAME}"
    --partition="$PARTITION"
    --cpus-per-task="$CPUS"
    --mem="$MEMORY"
    --time="$WALL_TIME"
    --output="$LOG_DIR/frame_%A_%a.out"
    --error="$LOG_DIR/frame_%A_%a.err"
)

if $USE_GPU; then
    SBATCH_CMD+=(--gres=gpu:1)
fi

# Export configuration as environment variables
SBATCH_CMD+=(
    --export="ALL,BHRT_PROJECT_DIR=$PROJECT_DIR,BHRT_SCENE_FILE=$SCENE_FILE,BHRT_TIME_START=$TIME_START,BHRT_TIME_STEP=$TIME_STEP,BHRT_PREFIX=$PREFIX,BHRT_OUTPUT_DIR=$OUTPUT_DIR,BHRT_EXR=$EXR_OUTPUT,BHRT_HDR=$HDR_OUTPUT"
)

SBATCH_CMD+=("$SBATCH_SCRIPT")

# ---------- Submit or dry-run ------------------------------------------------
if $DRY_RUN; then
    echo "[DRY RUN] Would execute:"
    echo "  ${SBATCH_CMD[*]}"
    echo
    echo "No jobs submitted."
else
    # Create log directory on the head node
    mkdir -p "$LOG_DIR"

    echo "Submitting $NUM_FRAMES tasks..."
    JOB_OUTPUT=$("${SBATCH_CMD[@]}")
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+$')

    echo "$JOB_OUTPUT"
    echo
    echo "Monitor progress with:"
    echo "  squeue -j $JOB_ID                # job status"
    echo "  sacct -j $JOB_ID --format=JobID,State,Elapsed,MaxRSS,NodeList"
    echo "  tail -f $LOG_DIR/frame_${JOB_ID}_*.out  # live output"
    echo
    echo "Cancel all tasks:   scancel $JOB_ID"
    echo "Cancel one frame:   scancel ${JOB_ID}_<TASK_ID>"
    echo
    echo "When done, collect frames:"
    echo "  ./slurm/collect_frames.sh -n $NUM_FRAMES -d $PROJECT_DIR"

    # Save job metadata for collect_frames.sh
    cat > "$PROJECT_DIR/build/.slurm_job_info" <<EOF
JOB_ID=$JOB_ID
NUM_FRAMES=$NUM_FRAMES
TIME_START=$TIME_START
TIME_STEP=$TIME_STEP
PREFIX=$PREFIX
OUTPUT_DIR=$OUTPUT_DIR
PROJECT_DIR=$PROJECT_DIR
EXR_OUTPUT=$EXR_OUTPUT
HDR_OUTPUT=$HDR_OUTPUT
SUBMITTED_AT=$(date -Iseconds)
EOF
fi
