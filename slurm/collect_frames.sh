#!/usr/bin/env bash
# ============================================================================
# collect_frames.sh — Gather rendered frames from Slurm compute nodes
# ============================================================================
# After a Slurm render job completes, this script collects frames from all
# compute nodes back to the head node (or a local machine) and verifies
# that every frame was rendered successfully.
#
# In a shared-filesystem setup (NFS/Ceph), frames are already in place and
# this script only verifies completeness. For Tailscale-only setups without
# a shared filesystem, it uses rsync over SSH to pull frames.
#
# Usage:
#   ./collect_frames.sh [options]
#
# Options:
#   -n NUM         Number of frames expected        (default: from .slurm_job_info)
#   -d DIR         Project directory                 (default: /opt/bhrt)
#   -o DIR         Output frames subdir              (default: frames)
#   -p PREFIX      Frame filename prefix             (default: frame)
#   -L LOCAL_DIR   Local directory to copy into      (default: ./build/frames)
#   --shared-fs    Skip rsync, just verify in-place  (shared filesystem mode)
#   --nodes LIST   Comma-separated list of node hostnames/IPs to pull from
#   -h             Show this help
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ---------- Defaults ---------------------------------------------------------
PROJECT_DIR="/opt/bhrt"
OUTPUT_DIR="frames"
PREFIX="frame"
LOCAL_DIR=""
SHARED_FS=false
NODES=""
NUM_FRAMES=""

# ---------- Parse arguments --------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n)         NUM_FRAMES="$2"; shift 2 ;;
        -d)         PROJECT_DIR="$2"; shift 2 ;;
        -o)         OUTPUT_DIR="$2"; shift 2 ;;
        -p)         PREFIX="$2"; shift 2 ;;
        -L)         LOCAL_DIR="$2"; shift 2 ;;
        --shared-fs) SHARED_FS=true; shift ;;
        --nodes)    NODES="$2"; shift 2 ;;
        -h|--help)
            head -26 "$0" | tail -23
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ---------- Load job metadata if available -----------------------------------
JOB_INFO="$PROJECT_DIR/build/.slurm_job_info"
if [[ -f "$JOB_INFO" ]]; then
    echo "Loading job info from $JOB_INFO ..."
    source "$JOB_INFO"
    NUM_FRAMES="${NUM_FRAMES:-$NUM_FRAMES}"
    PREFIX="${PREFIX:-$PREFIX}"
    OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR}"
fi

if [[ -z "$NUM_FRAMES" ]]; then
    echo "Error: Number of frames not specified. Use -n or submit a job first." >&2
    exit 1
fi

REMOTE_FRAMES_DIR="$PROJECT_DIR/build/$OUTPUT_DIR"
LOCAL_DIR="${LOCAL_DIR:-./build/$OUTPUT_DIR}"

echo "============================================"
echo " BHRT Frame Collection"
echo "============================================"
echo " Expected frames: $NUM_FRAMES"
echo " Remote dir:      $REMOTE_FRAMES_DIR"
echo " Local dir:       $LOCAL_DIR"
echo " Shared FS:       $SHARED_FS"
echo "============================================"
echo

# ---------- Discover nodes ---------------------------------------------------
if [[ -z "$NODES" ]] && ! $SHARED_FS; then
    # Try to discover nodes from Slurm
    if command -v sinfo &>/dev/null; then
        NODES=$(sinfo --noheader --format="%n" | sort -u | tr '\n' ',' | sed 's/,$//')
        echo "Auto-discovered nodes from Slurm: $NODES"
    fi
fi

# ---------- Collect frames via rsync -----------------------------------------
if ! $SHARED_FS && [[ -n "$NODES" ]]; then
    mkdir -p "$LOCAL_DIR"

    IFS=',' read -ra NODE_LIST <<< "$NODES"
    for node in "${NODE_LIST[@]}"; do
        node=$(echo "$node" | tr -d ' ')
        echo -n "  Syncing from $node ... "

        if rsync -az --ignore-existing \
            "$node:$REMOTE_FRAMES_DIR/${PREFIX}_*.tga" \
            "$LOCAL_DIR/" 2>/dev/null; then
            SYNCED=$(ls "$LOCAL_DIR/${PREFIX}_"*.tga 2>/dev/null | wc -l | tr -d ' ')
            echo -e "${GREEN}OK${NC} ($SYNCED frames total so far)"
        else
            echo -e "${YELLOW}SKIP${NC} (unreachable or no frames)"
        fi
    done
    echo
    FRAMES_DIR="$LOCAL_DIR"
elif $SHARED_FS; then
    FRAMES_DIR="$REMOTE_FRAMES_DIR"
else
    # No nodes specified, assume local
    FRAMES_DIR="$LOCAL_DIR"
    if [[ ! -d "$FRAMES_DIR" ]]; then
        FRAMES_DIR="$REMOTE_FRAMES_DIR"
    fi
fi

# ---------- Verify completeness ----------------------------------------------
echo "Verifying frame completeness in $FRAMES_DIR ..."
echo

MISSING=()
FOUND=0
TOTAL_SIZE=0

for i in $(seq 0 $((NUM_FRAMES - 1))); do
    FNAME="${PREFIX}_$(printf '%04d' "$i").tga"
    FPATH="$FRAMES_DIR/$FNAME"

    if [[ -f "$FPATH" ]]; then
        SIZE=$(stat -f%z "$FPATH" 2>/dev/null || stat --format=%s "$FPATH" 2>/dev/null || echo 0)
        if [[ "$SIZE" -gt 0 ]]; then
            FOUND=$((FOUND + 1))
            TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
        else
            MISSING+=("$FNAME (empty file)")
        fi
    else
        MISSING+=("$FNAME")
    fi
done

# ---------- Report -----------------------------------------------------------
echo "============================================"
echo " Frame Verification Report"
echo "============================================"
echo -e " Found:   ${GREEN}$FOUND${NC} / $NUM_FRAMES"

if [[ ${#MISSING[@]} -eq 0 ]]; then
    TOTAL_MB=$(echo "scale=1; $TOTAL_SIZE / 1048576" | bc 2>/dev/null || echo "?")
    echo -e " Status:  ${GREEN}ALL FRAMES PRESENT${NC}"
    echo " Total:   ${TOTAL_MB} MB"
    echo "============================================"
    echo
    echo "Ready to encode video:"
    echo "  ./make_video.sh -i $FRAMES_DIR"
else
    echo -e " Missing: ${RED}${#MISSING[@]}${NC} frames"
    echo "============================================"
    echo
    echo "Missing frames:"
    for m in "${MISSING[@]}"; do
        echo -e "  ${RED}✗${NC} $m"
    done
    echo
    echo "To re-render missing frames, resubmit with specific array indices:"

    # Build a compact array index list for resubmission
    MISSING_INDICES=()
    for m in "${MISSING[@]}"; do
        IDX=$(echo "$m" | grep -oP '\d{4}' | head -1 | sed 's/^0*//' )
        [[ -z "$IDX" ]] && IDX=0
        MISSING_INDICES+=("$IDX")
    done
    IDX_LIST=$(IFS=','; echo "${MISSING_INDICES[*]}")
    echo "  sbatch --array=$IDX_LIST slurm/render_frame.sbatch"

    exit 1
fi
