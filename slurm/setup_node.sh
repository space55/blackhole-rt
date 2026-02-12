#!/usr/bin/env bash
# ============================================================================
# setup_node.sh — Bootstrap a compute node for the BHRT Slurm cluster
# ============================================================================
# Run this on every machine (head node AND compute nodes) that will
# participate in the render cluster.
#
# Prerequisites:
#   - A Linux machine (Ubuntu/Debian or RHEL/Fedora family)
#   - Tailscale already installed and joined to your tailnet
#   - SSH access between nodes via Tailscale IPs
#
# Usage:
#   sudo ./setup_node.sh [--head]    # pass --head on the controller node
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

IS_HEAD=false
[[ "${1:-}" == "--head" ]] && IS_HEAD=true

log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[error]${NC} $*" >&2; }

# ---------- Detect package manager -------------------------------------------
if command -v apt-get &>/dev/null; then
    PKG_MGR="apt"
elif command -v dnf &>/dev/null; then
    PKG_MGR="dnf"
elif command -v yum &>/dev/null; then
    PKG_MGR="yum"
else
    err "Unsupported package manager. Install packages manually."
    exit 1
fi

install_pkg() {
    case "$PKG_MGR" in
        apt) apt-get install -y "$@" ;;
        dnf) dnf install -y "$@" ;;
        yum) yum install -y "$@" ;;
    esac
}

# ---------- 1. System dependencies ------------------------------------------
log "Installing system dependencies..."

case "$PKG_MGR" in
    apt)
        apt-get update -qq
        install_pkg build-essential cmake g++ libgomp1 munge slurmd slurmctld
        ;;
    dnf|yum)
        install_pkg gcc-c++ cmake make munge munge-libs slurm slurm-slurmd
        $IS_HEAD && install_pkg slurm-slurmctld
        ;;
esac

# ---------- 2. Tailscale verification ----------------------------------------
log "Verifying Tailscale connectivity..."
if ! command -v tailscale &>/dev/null; then
    err "Tailscale is not installed. Install from https://tailscale.com/download"
    exit 1
fi

TS_IP=$(tailscale ip -4 2>/dev/null || true)
if [[ -z "$TS_IP" ]]; then
    err "Tailscale is not connected. Run 'sudo tailscale up' first."
    exit 1
fi
log "Tailscale IP: $TS_IP"
TS_HOSTNAME=$(tailscale status --self --json | python3 -c "import sys,json; print(json.load(sys.stdin)['Self']['HostName'])" 2>/dev/null || hostname -s)
log "Tailscale hostname: $TS_HOSTNAME"

# ---------- 3. MUNGE authentication ------------------------------------------
log "Configuring MUNGE authentication..."

MUNGE_KEY="/etc/munge/munge.key"
if [[ ! -f "$MUNGE_KEY" ]]; then
    if $IS_HEAD; then
        log "Generating new MUNGE key (head node)..."
        dd if=/dev/urandom bs=1 count=1024 > "$MUNGE_KEY" 2>/dev/null
        chown munge:munge "$MUNGE_KEY"
        chmod 400 "$MUNGE_KEY"
        log "IMPORTANT: Copy $MUNGE_KEY to all compute nodes:"
        echo "    scp $MUNGE_KEY <node>:$MUNGE_KEY"
    else
        warn "No MUNGE key found. Copy the key from the head node:"
        echo "    scp head-node:$MUNGE_KEY $MUNGE_KEY"
        echo "    chown munge:munge $MUNGE_KEY && chmod 400 $MUNGE_KEY"
    fi
fi

systemctl enable munge
systemctl restart munge
log "MUNGE daemon running."

# ---------- 4. Slurm directories --------------------------------------------
log "Creating Slurm directories..."
mkdir -p /var/spool/slurmd /var/spool/slurmctld /var/log/slurm
chown slurm:slurm /var/spool/slurmctld /var/log/slurm 2>/dev/null || true

# ---------- 5. Slurm config -------------------------------------------------
SLURM_CONF="/etc/slurm/slurm.conf"
TEMPLATE_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -f "$TEMPLATE_DIR/slurm.conf.template" && ! -f "$SLURM_CONF" ]]; then
    log "Installing slurm.conf template to $SLURM_CONF ..."
    mkdir -p /etc/slurm
    cp "$TEMPLATE_DIR/slurm.conf.template" "$SLURM_CONF"
    warn "Edit $SLURM_CONF and fill in your node list + Tailscale addresses!"
elif [[ -f "$SLURM_CONF" ]]; then
    log "Existing $SLURM_CONF found — skipping. Edit manually if needed."
else
    warn "No slurm.conf template found. Create $SLURM_CONF manually."
fi

# ---------- 6. Shared project directory --------------------------------------
PROJECT_DIR="/opt/bhrt"
log "Setting up shared project directory at $PROJECT_DIR ..."
mkdir -p "$PROJECT_DIR"

# If this script is run from the project tree, sync the source
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$SCRIPT_DIR/CMakeLists.txt" ]]; then
    log "Copying project source from $SCRIPT_DIR ..."
    rsync -a --exclude='build/' --exclude='.git/' "$SCRIPT_DIR/" "$PROJECT_DIR/"
fi

# ---------- 7. Build the renderer --------------------------------------------
log "Building bhrt3 renderer..."
cd "$PROJECT_DIR"
mkdir -p build && cd build

# Detect CUDA
BUILD_GPU_FLAG=""
if command -v nvcc &>/dev/null; then
    log "CUDA detected — building with GPU support."
    BUILD_GPU_FLAG="-DUSE_GPU=ON"
fi

cmake .. -DCMAKE_BUILD_TYPE=Release $BUILD_GPU_FLAG
cmake --build . -j"$(nproc)"

if [[ -x "$PROJECT_DIR/build/bhrt3" ]]; then
    log "Build successful: $PROJECT_DIR/build/bhrt3"
else
    err "Build failed. Check the output above."
    exit 1
fi

# ---------- 8. Shared output directory ---------------------------------------
FRAMES_DIR="$PROJECT_DIR/build/frames"
mkdir -p "$FRAMES_DIR"
log "Frames output directory: $FRAMES_DIR"

# ---------- 9. Start Slurm daemons ------------------------------------------
if $IS_HEAD; then
    log "Starting Slurm controller (slurmctld)..."
    systemctl enable slurmctld
    systemctl restart slurmctld
fi

log "Starting Slurm compute daemon (slurmd)..."
systemctl enable slurmd
systemctl restart slurmd

# ---------- 10. Summary -----------------------------------------------------
echo
echo "============================================"
echo " Node setup complete!"
echo "============================================"
echo " Tailscale IP:     $TS_IP"
echo " Tailscale host:   $TS_HOSTNAME"
echo " Node role:        $($IS_HEAD && echo "HEAD + COMPUTE" || echo "COMPUTE")"
echo " Project dir:      $PROJECT_DIR"
echo " Renderer binary:  $PROJECT_DIR/build/bhrt3"
echo " Frames dir:       $FRAMES_DIR"
echo "============================================"
echo
if $IS_HEAD; then
    echo "Next steps:"
    echo "  1. Edit /etc/slurm/slurm.conf with your node list"
    echo "  2. Copy /etc/munge/munge.key to all compute nodes"
    echo "  3. Run this script (without --head) on each compute node"
    echo "  4. Verify with: sinfo"
fi
