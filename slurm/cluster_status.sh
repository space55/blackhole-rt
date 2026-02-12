#!/usr/bin/env bash
# ============================================================================
# cluster_status.sh — Quick health check for the BHRT Slurm + Tailscale cluster
# ============================================================================
# Shows:
#   1. Tailscale mesh status
#   2. Slurm partition & node state
#   3. Running / pending render jobs
#   4. Rendered frames progress (if a job is active)
#
# Usage:
#   ./cluster_status.sh [-d PROJECT_DIR] [-w] [-h]
#
# Options:
#   -d DIR   Project directory   (default: /opt/bhrt)
#   -w       Watch mode — refresh every 5 seconds
#   -h       Show this help
# ============================================================================

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

PROJECT_DIR="/opt/bhrt"
WATCH=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d) PROJECT_DIR="$2"; shift 2 ;;
        -w) WATCH=true; shift ;;
        -h|--help)
            head -17 "$0" | tail -14
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

show_status() {
    clear 2>/dev/null || true
    echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║          BHRT Cluster Status — $(date '+%H:%M:%S')            ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
    echo

    # ---- 1. Tailscale Mesh --------------------------------------------------
    echo -e "${CYAN}━━━ Tailscale Mesh ━━━${NC}"
    if command -v tailscale &>/dev/null; then
        TS_SELF_IP=$(tailscale ip -4 2>/dev/null || echo "disconnected")
        echo "  This node: $TS_SELF_IP ($(hostname -s))"
        echo

        # List all peers
        tailscale status 2>/dev/null | while IFS= read -r line; do
            if echo "$line" | grep -q "active"; then
                echo -e "  ${GREEN}●${NC} $line"
            elif echo "$line" | grep -q "idle"; then
                echo -e "  ${YELLOW}○${NC} $line"
            elif echo "$line" | grep -q "offline"; then
                echo -e "  ${RED}✗${NC} $line"
            else
                echo "  $line"
            fi
        done
    else
        echo -e "  ${RED}Tailscale not installed${NC}"
    fi
    echo

    # ---- 2. Slurm Partitions & Nodes ----------------------------------------
    echo -e "${CYAN}━━━ Slurm Cluster ━━━${NC}"
    if command -v sinfo &>/dev/null && sinfo &>/dev/null; then
        echo
        sinfo --format="  %-12P %-6a %-10F %-8c %-10m %-8G" 2>/dev/null || echo "  (cannot reach slurmctld)"
        echo
        echo "  Node details:"
        sinfo --Node --format="    %-15N %-10T %-6c %-10m %-20f" --noheader 2>/dev/null || true
    else
        echo -e "  ${RED}Slurm not available or slurmctld not running${NC}"
    fi
    echo

    # ---- 3. Job Queue -------------------------------------------------------
    echo -e "${CYAN}━━━ Render Jobs ━━━${NC}"
    if command -v squeue &>/dev/null && squeue &>/dev/null; then
        BHRT_JOBS=$(squeue --name=bhrt-render --noheader 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$BHRT_JOBS" -gt 0 ]]; then
            RUNNING=$(squeue --name=bhrt-render --states=RUNNING --noheader 2>/dev/null | wc -l | tr -d ' ')
            PENDING=$(squeue --name=bhrt-render --states=PENDING --noheader 2>/dev/null | wc -l | tr -d ' ')
            echo -e "  Active render jobs: ${GREEN}$RUNNING running${NC}, ${YELLOW}$PENDING pending${NC}"
            echo
            squeue --name=bhrt-render --format="  %-10i %-8j %-8T %-10M %-6D %-20R" 2>/dev/null | head -20
            if [[ "$BHRT_JOBS" -gt 20 ]]; then
                echo "  ... and $((BHRT_JOBS - 20)) more"
            fi
        else
            echo "  No active render jobs."
        fi
    else
        echo "  (squeue unavailable)"
    fi
    echo

    # ---- 4. Frame Progress --------------------------------------------------
    echo -e "${CYAN}━━━ Frame Progress ━━━${NC}"
    JOB_INFO="$PROJECT_DIR/build/.slurm_job_info"
    FRAMES_DIR="$PROJECT_DIR/build/frames"

    if [[ -f "$JOB_INFO" ]]; then
        source "$JOB_INFO"
        EXPECTED=${NUM_FRAMES:-0}
        PREFIX=${PREFIX:-frame}

        if [[ -d "$FRAMES_DIR" ]]; then
            RENDERED=$(ls "$FRAMES_DIR/${PREFIX}_"*.tga 2>/dev/null | wc -l | tr -d ' ')
        else
            RENDERED=0
        fi

        if [[ "$EXPECTED" -gt 0 ]]; then
            PCT=$((RENDERED * 100 / EXPECTED))
            BAR_WIDTH=40
            FILLED=$((PCT * BAR_WIDTH / 100))
            EMPTY=$((BAR_WIDTH - FILLED))
            BAR=$(printf '%0.s█' $(seq 1 $FILLED 2>/dev/null) 2>/dev/null || true)
            BAR+=$(printf '%0.s░' $(seq 1 $EMPTY 2>/dev/null) 2>/dev/null || true)

            echo "  Job ID:     ${JOB_ID:-?}"
            echo "  Submitted:  ${SUBMITTED_AT:-?}"
            echo -e "  Progress:   [${GREEN}${BAR}${NC}] ${PCT}%  (${RENDERED}/${EXPECTED})"
        else
            echo "  No frame count in job info."
        fi
    elif [[ -d "$FRAMES_DIR" ]]; then
        RENDERED=$(ls "$FRAMES_DIR/"*.tga 2>/dev/null | wc -l | tr -d ' ')
        echo "  Found $RENDERED .tga frames in $FRAMES_DIR (no job metadata)"
    else
        echo "  No frames directory or job info found."
    fi
    echo

    # ---- 5. Recent Errors ---------------------------------------------------
    LOG_DIR="$PROJECT_DIR/build/logs"
    if [[ -d "$LOG_DIR" ]]; then
        ERR_FILES=$(find "$LOG_DIR" -name "*.err" -size +0c -newer "$LOG_DIR" 2>/dev/null | head -5)
        if [[ -n "$ERR_FILES" ]]; then
            echo -e "${CYAN}━━━ Recent Errors ━━━${NC}"
            for ef in $ERR_FILES; do
                echo -e "  ${RED}$(basename "$ef"):${NC}"
                tail -3 "$ef" | sed 's/^/    /'
            done
            echo
        fi
    fi
}

# ---------- Main -------------------------------------------------------------
if $WATCH; then
    while true; do
        show_status
        sleep 5
    done
else
    show_status
fi
