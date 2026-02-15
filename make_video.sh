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
#   -f FORMAT   Input frame format: tga, exr, hdr  (default: tga)
#

set -euo pipefail

# Defaults (auto-detect flared frames if present)
FRAMES_DIR=""
PREFIX="frame"
OUTPUT="blackhole.mp4"
FPS=24
CRF=18
FORMAT="tga"

# Prefer flared frames if present
if [[ -d build/frames_flared_tga && "${FORMAT}" == "tga" && $(ls build/frames_flared_tga/frame_*.tga 2>/dev/null | wc -l) -gt 0 ]]; then
    FRAMES_DIR="build/frames_flared_tga"
elif [[ -d build/frames_flared && ( "${FORMAT}" == "exr" || "${FORMAT}" == "hdr" ) && $(ls build/frames_flared/frame_*.${FORMAT} 2>/dev/null | wc -l) -gt 0 ]]; then
    FRAMES_DIR="build/frames_flared"
else
    FRAMES_DIR="build/frames"
fi

# Parse arguments
while getopts "i:p:o:r:c:f:h" opt; do
    case "$opt" in
        i) FRAMES_DIR="$OPTARG" ;;
        p) PREFIX="$OPTARG" ;;
        o) OUTPUT="$OPTARG" ;;
        r) FPS="$OPTARG" ;;
        c) CRF="$OPTARG" ;;
        f) FORMAT="$OPTARG" ;;
        h)
            echo "Usage: $0 [options]"
            echo "  -i DIR      Input frames directory  (default: build/frames_flared[_tga] if present, else build/frames)"
            echo "  -p PREFIX   Frame filename prefix   (default: frame)"
            echo "  -o FILE     Output video filename   (default: blackhole.mp4)"
            echo "  -r FPS      Framerate               (default: 30)"
            echo "  -c CRF      Quality (0=lossless, 23=default, 51=worst)  (default: 18)"
            echo "  -f FORMAT   Input frame format: tga, exr, hdr  (default: tga)"
            echo
            echo "If build/frames_flared[_tga] exists and contains frames, it is used by default."
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


NUM_FRAMES=$(ls "$FRAMES_DIR"/${PREFIX}_*.${FORMAT} 2>/dev/null | wc -l | tr -d ' ')
if [[ "$NUM_FRAMES" -eq 0 ]]; then
    echo "Error: no ${PREFIX}_*.${FORMAT} files found in $FRAMES_DIR" >&2
    exit 1
fi

# Warn if using non-flared frames
if [[ "$FRAMES_DIR" == "build/frames" ]]; then
    echo "Warning: using non-flared frames from build/frames. Run flare_sequence.py for lens flare."
fi

echo "Encoding $NUM_FRAMES frames from $FRAMES_DIR/${PREFIX}_NNNN.${FORMAT}"
echo "  -> $OUTPUT  (${FPS} fps, CRF ${CRF})"
echo

# EXR/HDR inputs are linear float — apply a basic tonemap for video output
EXTRA_INPUT_ARGS=()
EXTRA_FILTER_ARGS=""
if [[ "$FORMAT" == "exr" || "$FORMAT" == "hdr" ]]; then
    # Apply Reinhard tonemap and convert to sRGB for video encoding
    EXTRA_FILTER_ARGS="-vf zscale=transfer=bt709:primaries=bt709:matrix=bt709,tonemap=reinhard:desat=0,format=yuv420p"
    echo "  (applying tonemap for linear HDR -> SDR video)"
fi

ffmpeg -y \
    -framerate "$FPS" \
    ${EXTRA_INPUT_ARGS[@]+"${EXTRA_INPUT_ARGS[@]}"} \
    -i "${FRAMES_DIR}/${PREFIX}_%04d.${FORMAT}" \
    -c:v libx264 \
    -preset slow \
    -crf "$CRF" \
    ${EXTRA_FILTER_ARGS:--pix_fmt yuv420p} \
    -movflags +faststart \
    "$OUTPUT"

echo
echo "Done! Video saved to $OUTPUT"
