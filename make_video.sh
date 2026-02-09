#!/usr/bin/env bash
#
# Assemble rendered frames into a video using ffmpeg.
#
# Usage:
#   ./make_video.sh [options]
#
# Options:
#   -i DIR      Input frames directory  (default: build/frames)
#   -p PREFIX   Frame filename prefix   (default: frame)
#   -o FILE     Output video filename   (default: blackhole.mp4)
#   -r FPS      Framerate               (default: 30)
#   -c CRF      Quality (0=lossless, 23=default, 51=worst)  (default: 18)
#

set -euo pipefail

# Defaults
FRAMES_DIR="build/frames"
PREFIX="frame"
OUTPUT="blackhole.mp4"
FPS=24
CRF=18

# Parse arguments
while getopts "i:p:o:r:c:h" opt; do
    case "$opt" in
        i) FRAMES_DIR="$OPTARG" ;;
        p) PREFIX="$OPTARG" ;;
        o) OUTPUT="$OPTARG" ;;
        r) FPS="$OPTARG" ;;
        c) CRF="$OPTARG" ;;
        h)
            head -14 "$0" | tail -12
            exit 0
            ;;
        *) exit 1 ;;
    esac
done

# Validate
if ! command -v ffmpeg &>/dev/null; then
    echo "Error: ffmpeg not found. Install with: brew install ffmpeg" >&2
    exit 1 
fi

if [[ ! -d "$FRAMES_DIR" ]]; then
    echo "Error: frames directory not found: $FRAMES_DIR" >&2
    echo "Run render_sequence.py first." >&2
    exit 1
fi

NUM_FRAMES=$(ls "$FRAMES_DIR"/${PREFIX}_*.tga 2>/dev/null | wc -l | tr -d ' ')
if [[ "$NUM_FRAMES" -eq 0 ]]; then
    echo "Error: no ${PREFIX}_*.tga files found in $FRAMES_DIR" >&2
    exit 1
fi

echo "Encoding $NUM_FRAMES frames from $FRAMES_DIR/${PREFIX}_NNNN.tga"
echo "  -> $OUTPUT  (${FPS} fps, CRF ${CRF})"
echo

ffmpeg -y \
    -framerate "$FPS" \
    -i "${FRAMES_DIR}/${PREFIX}_%04d.tga" \
    -c:v libx264 \
    -preset slow \
    -crf "$CRF" \
    -pix_fmt yuv420p \
    -movflags +faststart \
    "$OUTPUT"

echo
echo "Done! Video saved to $OUTPUT"
