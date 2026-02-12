#!/usr/bin/env bash
# ============================================================================
# render_frame.sbatch — Slurm batch script for rendering one frame
# ============================================================================
# This script is submitted by submit_render.sh as a Slurm job array.
# Each array task renders a single frame of the black hole animation.
#
# Environment variables (set by submit_render.sh):
#   BHRT_PROJECT_DIR   — path to the project root  (default: /opt/bhrt)
#   BHRT_SCENE_FILE    — base scene file            (default: scene.txt)
#   BHRT_TIME_START    — starting time value         (default: 0.0)
#   BHRT_TIME_STEP     — time increment per frame    (default: 0.5)
#   BHRT_PREFIX        — output frame name prefix    (default: frame)
#   BHRT_OUTPUT_DIR    — where to write frames       (default: frames)
# ============================================================================

#SBATCH --job-name=bhrt-render
#SBATCH --output=/opt/bhrt/build/logs/frame_%A_%a.out
#SBATCH --error=/opt/bhrt/build/logs/frame_%A_%a.err
#SBATCH --ntasks=1
# NOTE: --cpus-per-task, --mem, --time, --partition, and --gres are set by
# submit_render.sh on the sbatch command line so they can be easily overridden.
# Do NOT hardcode them here or they will conflict with CLI arguments.

# ---------- Configuration (from environment or defaults) ---------------------
PROJECT_DIR="${BHRT_PROJECT_DIR:-/opt/bhrt}"
SCENE_FILE="${BHRT_SCENE_FILE:-scene.txt}"
TIME_START="${BHRT_TIME_START:-0.0}"
TIME_STEP="${BHRT_TIME_STEP:-0.5}"
PREFIX="${BHRT_PREFIX:-frame}"
OUTPUT_DIR="${BHRT_OUTPUT_DIR:-frames}"

BUILD_DIR="$PROJECT_DIR/build"
BINARY="$BUILD_DIR/bhrt3"
SCENE_SRC="$BUILD_DIR/$SCENE_FILE"

# ---------- Validate ---------------------------------------------------------
if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: Renderer binary not found: $BINARY" >&2
    exit 1
fi
if [[ ! -f "$SCENE_SRC" ]]; then
    echo "ERROR: Scene file not found: $SCENE_SRC" >&2
    exit 1
fi

# ---------- Determine frame index from Slurm array task ID ------------------
FRAME_INDEX=${SLURM_ARRAY_TASK_ID}

# Compute the time value for this frame
TIME_VAL=$(python3 -c "print(f'{${TIME_START} + ${FRAME_INDEX} * ${TIME_STEP}:.6f}')")
OUT_FILE="${OUTPUT_DIR}/${PREFIX}_$(printf '%04d' "$FRAME_INDEX").tga"

echo "=== BHRT Frame Render ==="
echo "Node:        $(hostname) ($(tailscale ip -4 2>/dev/null || echo 'no-tailscale'))"
echo "Frame:       $FRAME_INDEX"
echo "Time:        $TIME_VAL"
echo "Output:      $OUT_FILE"
echo "Scene:       $SCENE_SRC"
echo "CPUs:        ${SLURM_CPUS_PER_TASK:-1}"
echo "========================="

# ---------- Create per-task scene file with patched time and output ----------
TASK_SCENE="$BUILD_DIR/_slurm_scene_${SLURM_JOB_ID}_${FRAME_INDEX}.txt"

# Patch scene file: override time and output_file
python3 - "$SCENE_SRC" "$TASK_SCENE" "$TIME_VAL" "$OUT_FILE" <<'PATCHER'
import re, sys
src, dst, time_val, out_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
overrides = {"time": time_val, "output_file": out_file}
with open(src) as f:
    lines = f.readlines()
with open(dst, "w") as f:
    for line in lines:
        stripped = line.split("#")[0].strip()
        matched = False
        for key, val in overrides.items():
            if re.match(rf"^{re.escape(key)}\s*=", stripped):
                f.write(re.sub(r"=\s*\S+", f"= {val}", line.split("#")[0]).rstrip() + "\n")
                matched = True
                break
        if not matched:
            f.write(line)
PATCHER

# ---------- Ensure output directory exists -----------------------------------
mkdir -p "$BUILD_DIR/$OUTPUT_DIR"

# ---------- Run the renderer -------------------------------------------------
START_TIME=$(date +%s%N)

cd "$BUILD_DIR"
"$BINARY" "$TASK_SCENE"
EXIT_CODE=$?

END_TIME=$(date +%s%N)
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))

# ---------- Clean up temp scene file -----------------------------------------
rm -f "$TASK_SCENE"

# ---------- Report -----------------------------------------------------------
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "SUCCESS: Frame $FRAME_INDEX rendered in ${ELAPSED_MS}ms -> $OUT_FILE"
else
    echo "FAILED: Frame $FRAME_INDEX exited with code $EXIT_CODE" >&2
fi

exit $EXIT_CODE
