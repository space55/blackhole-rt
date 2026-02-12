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
#   sudo ./setup_node.sh [--head] [--node-name NAME]
#
#   --head          Run as the Slurm controller node
#   --node-name NAME  Set this node's Slurm NodeName (must match slurm.conf).
#                     Also sets the OS hostname so slurmd can self-identify.
#                     If omitted, the current hostname is used.
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

IS_HEAD=false
NODE_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --head)      IS_HEAD=true; shift ;;
        --node-name) NODE_NAME="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

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

# ---------- 2b. Set hostname to match Slurm NodeName -------------------------
# slurmd determines its identity by matching the OS hostname against NodeName
# entries in slurm.conf.  If they don't match you get:
#   "fatal: Unable to determine this slurmd's NodeName"
if [[ -n "$NODE_NAME" ]]; then
    log "Setting OS hostname to '$NODE_NAME' (to match slurm.conf NodeName)..."
    hostnamectl set-hostname "$NODE_NAME" 2>/dev/null || hostname "$NODE_NAME"
    # Persist in /etc/hostname for next boot
    echo "$NODE_NAME" > /etc/hostname
    # Add to /etc/hosts so the name resolves locally
    if ! grep -q "$NODE_NAME" /etc/hosts; then
        echo "127.0.1.1  $NODE_NAME" >> /etc/hosts
    fi
    log "Hostname set to: $(hostname)"
else
    NODE_NAME=$(hostname -s)
    warn "No --node-name given. Using current hostname '$NODE_NAME'."
    warn "Make sure a NodeName=$NODE_NAME entry exists in slurm.conf!"
fi

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

    # On the head node, auto-fill SlurmctldHost with this node's name
    if $IS_HEAD; then
        sed -i "s/^SlurmctldHost=.*/SlurmctldHost=$NODE_NAME/" "$SLURM_CONF"
        log "Set SlurmctldHost=$NODE_NAME in $SLURM_CONF"
    fi
    warn "Edit $SLURM_CONF and fill in your node list + Tailscale addresses!"
elif [[ -f "$SLURM_CONF" ]]; then
    log "Existing $SLURM_CONF found."
    # If running as head, ensure SlurmctldHost matches this node
    if $IS_HEAD; then
        CURRENT_CTL=$(grep -Po '(?<=^SlurmctldHost=)\S+' "$SLURM_CONF" 2>/dev/null || true)
        if [[ "$CURRENT_CTL" != "$NODE_NAME" ]]; then
            log "Updating SlurmctldHost from '$CURRENT_CTL' to '$NODE_NAME' ..."
            sed -i "s/^SlurmctldHost=.*/SlurmctldHost=$NODE_NAME/" "$SLURM_CONF"
        fi
    fi
else
    warn "No slurm.conf template found. Create $SLURM_CONF manually."
fi

# ---------- 5a. Fix ProctrackType if plugin doesn't exist --------------------
# proctrack/linuxprocs was removed in newer Slurm. Auto-detect a working plugin.
if [[ -f "$SLURM_CONF" ]]; then
    CURRENT_PROCTRACK=$(grep -Po '(?<=^ProctrackType=)\S+' "$SLURM_CONF" 2>/dev/null || true)
    if [[ -n "$CURRENT_PROCTRACK" ]]; then
        # Find the Slurm plugin directory
        SLURM_PLUGIN_DIR=""
        for d in /usr/lib64/slurm /usr/lib/x86_64-linux-gnu/slurm /usr/lib/slurm /usr/lib/*/slurm; do
            if [[ -d "$d" ]]; then
                SLURM_PLUGIN_DIR="$d"
                break
            fi
        done

        PLUGIN_FILE=$(echo "$CURRENT_PROCTRACK" | tr '/' '_')
        PLUGIN_EXISTS=false
        if [[ -n "$SLURM_PLUGIN_DIR" && -f "$SLURM_PLUGIN_DIR/${PLUGIN_FILE}.so" ]]; then
            PLUGIN_EXISTS=true
        fi

        if ! $PLUGIN_EXISTS; then
            # Pick the best available alternative
            NEW_PROCTRACK=""
            for candidate in proctrack/cgroup proctrack/pgid; do
                CAND_FILE=$(echo "$candidate" | tr '/' '_')
                if [[ -n "$SLURM_PLUGIN_DIR" && -f "$SLURM_PLUGIN_DIR/${CAND_FILE}.so" ]]; then
                    NEW_PROCTRACK="$candidate"
                    break
                fi
            done
            # If we couldn't find the plugin dir, default to pgid (universally available)
            [[ -z "$NEW_PROCTRACK" ]] && NEW_PROCTRACK="proctrack/pgid"

            log "ProctrackType=$CURRENT_PROCTRACK plugin not found, switching to $NEW_PROCTRACK"
            sed -i "s|^ProctrackType=.*|ProctrackType=$NEW_PROCTRACK|" "$SLURM_CONF"

            # Also fix TaskPlugin if it references task/cgroup but cgroup isn't available
            if [[ "$NEW_PROCTRACK" == "proctrack/pgid" ]]; then
                sed -i 's|TaskPlugin=task/affinity,task/cgroup|TaskPlugin=task/affinity|' "$SLURM_CONF" 2>/dev/null || true
            fi
        else
            log "ProctrackType=$CURRENT_PROCTRACK plugin found — OK."
        fi
    fi
fi

# ---------- 5b. Auto-register this node in slurm.conf -----------------------
# If there's no NodeName entry for this machine, add one automatically using
# detected hardware specs and the Tailscale IP.
if [[ -f "$SLURM_CONF" ]] && ! grep -q "^NodeName=$NODE_NAME " "$SLURM_CONF"; then
    log "No NodeName=$NODE_NAME found in $SLURM_CONF — adding automatically..."

    NODE_CPUS=$(nproc)
    NODE_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    NODE_MEM_MB=$((NODE_MEM_KB / 1024))
    # Reserve ~5% for OS overhead, Slurm wants RealMemory in MB
    NODE_REAL_MEM=$(( NODE_MEM_MB * 95 / 100 ))

    # Detect CPU topology — Slurm is strict about Sockets×Cores×Threads matching
    NODE_SOCKETS=$(lscpu | awk -F: '/^Socket\(s\)/{gsub(/ /,"",$2); print $2}')
    NODE_CORES=$(lscpu | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2); print $2}')
    NODE_THREADS=$(lscpu | awk -F: '/^Thread\(s\) per core/{gsub(/ /,"",$2); print $2}')
    # Fallback if lscpu parsing fails
    NODE_SOCKETS=${NODE_SOCKETS:-1}
    NODE_CORES=${NODE_CORES:-$NODE_CPUS}
    NODE_THREADS=${NODE_THREADS:-1}
    log "CPU topology: Sockets=$NODE_SOCKETS CoresPerSocket=$NODE_CORES ThreadsPerCore=$NODE_THREADS (total=$NODE_CPUS)"

    NODE_GRES=""
    if command -v nvidia-smi &>/dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$GPU_COUNT" -gt 0 ]]; then
            NODE_GRES="Gres=gpu:$GPU_COUNT"
            # Ensure GresTypes=gpu is uncommented
            if grep -q "^# *GresTypes=gpu" "$SLURM_CONF"; then
                sed -i 's/^# *GresTypes=gpu/GresTypes=gpu/' "$SLURM_CONF"
                log "Enabled GresTypes=gpu in $SLURM_CONF"
            fi
        fi
    fi

    NODE_LINE="NodeName=$NODE_NAME NodeAddr=$TS_IP CPUs=$NODE_CPUS Sockets=$NODE_SOCKETS CoresPerSocket=$NODE_CORES ThreadsPerCore=$NODE_THREADS RealMemory=$NODE_REAL_MEM $NODE_GRES State=UNKNOWN"
    echo "" >> "$SLURM_CONF"
    echo "# Auto-added by setup_node.sh on $(date -Iseconds)" >> "$SLURM_CONF"
    echo "$NODE_LINE" >> "$SLURM_CONF"
    log "Added: $NODE_LINE"

    # Ensure a default partition exists that includes this node
    if ! grep -q "^PartitionName=.*Nodes=.*$NODE_NAME" "$SLURM_CONF"; then
        # Check if there's an existing default partition to append to
        EXISTING_PARTITION=$(grep "^PartitionName=.*Default=YES" "$SLURM_CONF" 2>/dev/null | head -1 || true)
        if [[ -n "$EXISTING_PARTITION" ]]; then
            # Append this node to the existing partition's Nodes= list
            UPDATED=$(echo "$EXISTING_PARTITION" | sed "s/Nodes=\([^ ]*\)/Nodes=\1,$NODE_NAME/")
            sed -i "s|^${EXISTING_PARTITION}|${UPDATED}|" "$SLURM_CONF"
            log "Appended $NODE_NAME to existing default partition."
        else
            PART_NAME="compute"
            [[ -n "$NODE_GRES" ]] && PART_NAME="gpu"
            PART_LINE="PartitionName=$PART_NAME Nodes=$NODE_NAME Default=YES MaxTime=INFINITE State=UP"
            echo "$PART_LINE" >> "$SLURM_CONF"
            log "Added partition: $PART_LINE"
        fi
    fi
else
    [[ -f "$SLURM_CONF" ]] && log "NodeName=$NODE_NAME already present in $SLURM_CONF."
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
# Create a systemd override to pass the explicit node name to slurmd.
# This avoids the "Unable to determine this slurmd's NodeName" fatal error
# when the OS hostname doesn't exactly match a NodeName in slurm.conf.
#
# We also set Type=simple because -D keeps slurmd in the foreground.
# The stock unit often uses Type=forking, which causes a startup timeout
# when combined with -D (systemd waits for a fork that never happens).
mkdir -p /etc/systemd/system/slurmd.service.d
cat > /etc/systemd/system/slurmd.service.d/nodename.conf <<EOF
[Service]
Type=simple
PIDFile=
ExecStart=
ExecStart=/usr/sbin/slurmd -D -N $NODE_NAME \$SLURMD_OPTIONS
EOF
systemctl daemon-reload
systemctl enable slurmd
systemctl restart slurmd
log "slurmd started with NodeName=$NODE_NAME"

# ---------- 10. Summary -----------------------------------------------------
echo
echo "============================================"
echo " Node setup complete!"
echo "============================================"
echo " Slurm NodeName:   $NODE_NAME"
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
