#!/usr/bin/env bash
# ============================================================================
# add_node.sh — Register a remote compute node from the head node
# ============================================================================
# Run this on the HEAD NODE to add a Tailscale-connected machine to the
# Slurm cluster.  It will:
#   1. SSH into the remote node to detect hardware (CPUs, memory, GPUs)
#   2. Add a NodeName entry to the head's /etc/slurm/slurm.conf
#   3. Add the node to a partition (existing or new)
#   4. Push the updated slurm.conf to the remote node
#   5. Restart slurmctld and the remote slurmd
#
# Usage:
#   sudo ./add_node.sh <tailscale-host-or-ip> [--node-name NAME] [--partition PART]
#
# Examples:
#   sudo ./add_node.sh gpu-server-1
#   sudo ./add_node.sh 100.64.0.5 --node-name gpu01 --partition gpu
#   sudo ./add_node.sh my-desktop --node-name desktop01
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[add-node]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[error]${NC} $*" >&2; }

# ---------- Parse arguments --------------------------------------------------
REMOTE_HOST=""
NODE_NAME=""
PARTITION=""
SSH_USER="root"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node-name)  NODE_NAME="$2"; shift 2 ;;
        --partition)  PARTITION="$2"; shift 2 ;;
        --ssh-user)   SSH_USER="$2"; shift 2 ;;
        -h|--help)
            head -18 "$0" | tail -15
            exit 0
            ;;
        -*)
            err "Unknown option: $1"
            exit 1
            ;;
        *)
            REMOTE_HOST="$1"; shift
            ;;
    esac
done

if [[ -z "$REMOTE_HOST" ]]; then
    err "Usage: $0 <tailscale-host-or-ip> [--node-name NAME] [--partition PART]"
    exit 1
fi

SLURM_CONF="/etc/slurm/slurm.conf"
if [[ ! -f "$SLURM_CONF" ]]; then
    err "$SLURM_CONF not found. Run setup_node.sh --head first."
    exit 1
fi

# ---------- 1. Verify connectivity ------------------------------------------
log "Testing SSH connectivity to $REMOTE_HOST ..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$SSH_USER@$REMOTE_HOST" "echo ok" &>/dev/null; then
    err "Cannot SSH to $SSH_USER@$REMOTE_HOST"
    echo "  Make sure:"
    echo "    - Tailscale is running on both machines"
    echo "    - SSH keys are set up (ssh-copy-id $SSH_USER@$REMOTE_HOST)"
    exit 1
fi
log "SSH connection OK."

# ---------- 2. Detect remote hardware ---------------------------------------
log "Detecting hardware on $REMOTE_HOST ..."

REMOTE_INFO=$(ssh "$SSH_USER@$REMOTE_HOST" bash -s <<'DETECT'
set -e
CPUS=$(nproc)
MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
MEM_MB=$((MEM_KB / 1024))
REAL_MEM=$(( MEM_MB * 95 / 100 ))
SOCKETS=$(lscpu | awk -F: '/^Socket\(s\)/{gsub(/ /,"",$2); print $2}')
CORES=$(lscpu | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2); print $2}')
THREADS=$(lscpu | awk -F: '/^Thread\(s\) per core/{gsub(/ /,"",$2); print $2}')
SOCKETS=${SOCKETS:-1}
CORES=${CORES:-$CPUS}
THREADS=${THREADS:-1}
HOSTNAME_SHORT=$(hostname -s)

GPU_COUNT=0
GPU_DEVICES=""
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    # List /dev/nvidia* device files for gres.conf
    if [[ "$GPU_COUNT" -gt 0 ]]; then
        GPU_DEVICES=$(ls /dev/nvidia[0-9]* 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    fi
fi

TS_IP=$(tailscale ip -4 2>/dev/null || echo "")

# Detect WSL2 — GPU is accessed via /dev/dxg, not /dev/nvidia*
IS_WSL="no"
if grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL="yes"
fi

echo "CPUS=$CPUS"
echo "REAL_MEM=$REAL_MEM"
echo "SOCKETS=$SOCKETS"
echo "CORES=$CORES"
echo "THREADS=$THREADS"
echo "GPU_COUNT=$GPU_COUNT"
echo "GPU_DEVICES=$GPU_DEVICES"
echo "TS_IP=$TS_IP"
echo "HOSTNAME_SHORT=$HOSTNAME_SHORT"
echo "IS_WSL=$IS_WSL"
DETECT
)

# Source the values
eval "$REMOTE_INFO"

log "Remote hardware:"
echo "  Hostname:     $HOSTNAME_SHORT"
echo "  Tailscale IP: $TS_IP"
echo "  CPUs:         $CPUS (${SOCKETS}s × ${CORES}c × ${THREADS}t)"
echo "  Memory:       ${REAL_MEM} MB"
echo "  GPUs:         $GPU_COUNT"
if [[ "$IS_WSL" == "yes" ]]; then
    echo "  Platform:     WSL2 (GPU via /dev/dxg — no Slurm GRES needed)"
fi

# Default node name to remote hostname
NODE_NAME="${NODE_NAME:-$HOSTNAME_SHORT}"

# Use Tailscale IP; fall back to the host argument
NODE_ADDR="${TS_IP:-$REMOTE_HOST}"

# ---------- 3. Check for duplicates -----------------------------------------
if grep -q "^NodeName=$NODE_NAME " "$SLURM_CONF"; then
    warn "NodeName=$NODE_NAME already exists in $SLURM_CONF:"
    grep "^NodeName=$NODE_NAME " "$SLURM_CONF" | sed 's/^/  /'
    echo
    read -rp "  Overwrite? [y/N] " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        sed -i "/^NodeName=$NODE_NAME /d" "$SLURM_CONF"
        # Also remove the auto-added comment line above if present
        sed -i "/^# Auto-added.*$NODE_NAME/d" "$SLURM_CONF"
    else
        log "Skipped. No changes made."
        exit 0
    fi
fi

# ---------- 4. Build the NodeName line ---------------------------------------
NODE_GRES=""
if [[ "$GPU_COUNT" -gt 0 && "$IS_WSL" != "yes" ]]; then
    NODE_GRES="Gres=gpu:$GPU_COUNT"
    # Ensure GresTypes=gpu is enabled (handle various comment styles)
    if grep -q "^# *GresTypes=gpu" "$SLURM_CONF"; then
        sed -i 's/^# *GresTypes=gpu.*/GresTypes=gpu/' "$SLURM_CONF"
        log "Enabled GresTypes=gpu in $SLURM_CONF"
    elif ! grep -q "^GresTypes=gpu" "$SLURM_CONF"; then
        # Not present at all — add it
        sed -i '/^SchedulerType=/a GresTypes=gpu' "$SLURM_CONF"
        log "Added GresTypes=gpu to $SLURM_CONF"
    fi
fi

NODE_LINE="NodeName=$NODE_NAME NodeAddr=$NODE_ADDR CPUs=$CPUS Sockets=$SOCKETS CoresPerSocket=$CORES ThreadsPerCore=$THREADS RealMemory=$REAL_MEM $NODE_GRES State=UNKNOWN"

echo "" >> "$SLURM_CONF"
echo "# Auto-added by add_node.sh on $(date -Iseconds) — $NODE_NAME" >> "$SLURM_CONF"
echo "$NODE_LINE" >> "$SLURM_CONF"
log "Added: $NODE_LINE"

# ---------- 5. Add to partition ----------------------------------------------
# Auto-pick partition name if not specified
if [[ -z "$PARTITION" ]]; then
    if [[ "$GPU_COUNT" -gt 0 ]]; then
        PARTITION="gpu"
    else
        PARTITION="compute"
    fi
fi

if grep -q "^PartitionName=$PARTITION " "$SLURM_CONF"; then
    # Append to existing partition
    if ! grep "^PartitionName=$PARTITION " "$SLURM_CONF" | grep -q "$NODE_NAME"; then
        EXISTING=$(grep "^PartitionName=$PARTITION " "$SLURM_CONF")
        UPDATED=$(echo "$EXISTING" | sed "s/Nodes=\([^ ]*\)/Nodes=\1,$NODE_NAME/")
        sed -i "s|^PartitionName=$PARTITION .*|$UPDATED|" "$SLURM_CONF"
        log "Appended $NODE_NAME to partition '$PARTITION'."
    else
        log "$NODE_NAME already in partition '$PARTITION'."
    fi
else
    # Create new partition
    DEFAULT="NO"
    # If no default partition exists, make this one default
    if ! grep -q "^PartitionName=.*Default=YES" "$SLURM_CONF"; then
        DEFAULT="YES"
    fi
    PART_LINE="PartitionName=$PARTITION Nodes=$NODE_NAME Default=$DEFAULT MaxTime=INFINITE State=UP"
    echo "$PART_LINE" >> "$SLURM_CONF"
    log "Created partition: $PART_LINE"
fi

# ---------- 6. Create gres.conf for GPU nodes --------------------------------
# slurmd REQUIRES a gres.conf to report GPUs; without it the node reports 0 GPUs
# and slurmctld marks it as 'invalid' ("gres/gpu count reported lower than configured").
#
# Strategy:
#   1. Try to install the NVML plugin so AutoDetect=nvml works (cleanest).
#   2. If the plugin isn't available, fall back to a manual Name=gpu Count=N line.
GRES_CONF="/etc/slurm/gres.conf"
if [[ "$GPU_COUNT" -gt 0 && "$IS_WSL" == "yes" ]]; then
    log "WSL2 detected — skipping Slurm GRES configuration."
    log "GPU is available to all processes via /dev/dxg (Windows paravirtualization)."
    log "CUDA workloads will use the GPU automatically without Slurm allocation."
    # Write a no-op gres.conf so slurmd doesn't complain about a missing file
    ssh "$SSH_USER@$REMOTE_HOST" "mkdir -p /etc/slurm && echo '# WSL2: GPU managed by Windows host, no GRES needed' > /etc/slurm/gres.conf"
elif [[ "$GPU_COUNT" -gt 0 ]]; then
    log "Configuring gres.conf for $GPU_COUNT GPU(s) on $REMOTE_HOST ..."

    # Try to install the NVML plugin on the remote node
    HAS_NVML=$(ssh "$SSH_USER@$REMOTE_HOST" bash <<'CHECK_NVML'
# Check if the gpu/nvml plugin .so exists somewhere in slurm lib dirs
FOUND=false
for d in /usr/lib64/slurm /usr/lib/x86_64-linux-gnu/slurm /usr/lib/slurm /usr/lib/*/slurm; do
    if [[ -f "$d/gpu_nvml.so" ]]; then
        FOUND=true
        break
    fi
done

if ! $FOUND; then
    # Try installing the package
    if command -v apt-get &>/dev/null; then
        apt-get install -y slurm-wlm-nvidia-plugin 2>/dev/null && FOUND=true
    elif command -v dnf &>/dev/null; then
        dnf install -y slurm-plugins 2>/dev/null && FOUND=true
    elif command -v yum &>/dev/null; then
        yum install -y slurm-plugins 2>/dev/null && FOUND=true
    fi
fi

# Re-check after install attempt
if ! $FOUND; then
    for d in /usr/lib64/slurm /usr/lib/x86_64-linux-gnu/slurm /usr/lib/slurm /usr/lib/*/slurm; do
        if [[ -f "$d/gpu_nvml.so" ]]; then
            FOUND=true
            break
        fi
    done
fi

$FOUND && echo "yes" || echo "no"
CHECK_NVML
)

    if [[ "$HAS_NVML" == *"yes"* ]]; then
        log "NVML plugin available — using AutoDetect=nvml"
        GRES_CONTENT="# Auto-generated by add_node.sh on $(date -Iseconds)
AutoDetect=nvml"
    else
        log "NVML plugin not available — using manual GPU config with File="
        # Discover /dev/nvidia* device files for File= directive
        REMOTE_GPU_FILES=$(ssh "$SSH_USER@$REMOTE_HOST" 'ls /dev/nvidia[0-9]* 2>/dev/null | tr "\n" "," | sed "s/,$//"' || true)

        if [[ -z "$REMOTE_GPU_FILES" ]]; then
            # /dev/nvidia0 doesn't exist — try nvidia-modprobe or manual mknod
            log "No /dev/nvidia[0-9]* device files found — attempting to create them..."
            ssh "$SSH_USER@$REMOTE_HOST" bash <<'CREATE_DEV'
set -e
# Method 1: nvidia-modprobe (preferred)
if command -v nvidia-modprobe &>/dev/null; then
    echo "Using nvidia-modprobe..."
    nvidia-modprobe
elif apt-get install -y nvidia-modprobe 2>/dev/null; then
    echo "Installed nvidia-modprobe..."
    nvidia-modprobe
elif dnf install -y nvidia-modprobe 2>/dev/null || yum install -y nvidia-modprobe 2>/dev/null; then
    echo "Installed nvidia-modprobe..."
    nvidia-modprobe
else
    # Method 2: manual mknod using the nvidia major number
    echo "nvidia-modprobe not available — creating device nodes manually..."
    modprobe nvidia 2>/dev/null || true
    nvidia-smi > /dev/null 2>&1 || true
    MAJOR=$(grep nvidia /proc/devices 2>/dev/null | head -1 | awk '{print $1}')
    if [[ -n "$MAJOR" ]]; then
        GPU_IDX=0
        while [[ $GPU_IDX -lt $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) ]]; do
            if [[ ! -e "/dev/nvidia${GPU_IDX}" ]]; then
                mknod -m 666 "/dev/nvidia${GPU_IDX}" c "$MAJOR" "$GPU_IDX"
                echo "Created /dev/nvidia${GPU_IDX} (major=$MAJOR minor=$GPU_IDX)"
            fi
            GPU_IDX=$((GPU_IDX + 1))
        done
        if [[ ! -e /dev/nvidiactl ]]; then
            mknod -m 666 /dev/nvidiactl c "$MAJOR" 255
            echo "Created /dev/nvidiactl"
        fi
    else
        echo "ERROR: nvidia not found in /proc/devices" >&2
        exit 1
    fi
fi
CREATE_DEV
            REMOTE_GPU_FILES=$(ssh "$SSH_USER@$REMOTE_HOST" 'ls /dev/nvidia[0-9]* 2>/dev/null | tr "\n" "," | sed "s/,$//"' || true)
        fi

        if [[ -n "$REMOTE_GPU_FILES" ]]; then
            GRES_CONTENT="# Auto-generated by add_node.sh on $(date -Iseconds)
NodeName=$NODE_NAME Name=gpu File=$REMOTE_GPU_FILES"
        else
            err "GPU detected by nvidia-smi but /dev/nvidia* device files could not be created."
            err "Install nvidia-modprobe:  apt install nvidia-modprobe && nvidia-modprobe"
            err "Or create manually:  MAJOR=\$(grep nvidia /proc/devices | head -1 | awk '{print \$1}')"
            err "                     mknod -m 666 /dev/nvidia0 c \$MAJOR 0"
            exit 1
        fi
    fi

    # Write gres.conf on the remote node
    ssh "$SSH_USER@$REMOTE_HOST" "mkdir -p /etc/slurm && cat > /etc/slurm/gres.conf" <<EOF
$GRES_CONTENT
EOF
    log "Remote gres.conf written."

    # Ensure the head node also has a matching gres.conf
    echo "$GRES_CONTENT" > "$GRES_CONF"
    log "Head gres.conf updated."

    # Verify GPU visibility on the remote node
    log "Verifying GPU access on $REMOTE_HOST ..."
    NVIDIA_CHECK=$(ssh "$SSH_USER@$REMOTE_HOST" "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -3 || echo 'nvidia-smi-failed'" || true)
    if [[ "$NVIDIA_CHECK" == *"nvidia-smi-failed"* ]]; then
        warn "nvidia-smi failed on $REMOTE_HOST. Check that NVIDIA drivers are installed."
    else
        log "nvidia-smi reports: $NVIDIA_CHECK"
    fi

    # Note: slurmd -C only shows Gres with AutoDetect=nvml.
    # With manual gres.conf (Name=gpu Count=N), slurmd reads the config
    # at startup and reports it to slurmctld — slurmd -C won't list it,
    # but that's expected.
    if [[ "$HAS_NVML" == *"yes"* ]]; then
        REMOTE_GRES_CHECK=$(ssh "$SSH_USER@$REMOTE_HOST" "slurmd -C 2>&1 | grep -i gres || echo 'no-gres-detected'" || true)
        if [[ "$REMOTE_GRES_CHECK" == *"no-gres-detected"* ]]; then
            warn "AutoDetect=nvml is set but slurmd -C didn't detect GPUs."
            warn "Falling back to manual gres.conf ..."
            GRES_CONTENT="# Auto-generated by add_node.sh — nvml fallback
Name=gpu Count=$GPU_COUNT"
            ssh "$SSH_USER@$REMOTE_HOST" "cat > /etc/slurm/gres.conf" <<EOF
$GRES_CONTENT
EOF
            echo "$GRES_CONTENT" > "$GRES_CONF"
            log "Switched to manual gres.conf: Name=gpu Count=$GPU_COUNT"
        else
            log "NVML auto-detection OK: $REMOTE_GRES_CHECK"
        fi
    else
        log "Using manual gres.conf (Name=gpu Count=$GPU_COUNT) — OK."
    fi
fi

# ---------- 7. Push config to remote and all known nodes ---------------------
log "Pushing updated slurm.conf to $REMOTE_HOST ..."
scp "$SLURM_CONF" "$SSH_USER@$REMOTE_HOST:/etc/slurm/slurm.conf"

# Also push to all other known nodes so configs stay in sync
log "Syncing slurm.conf to other cluster nodes ..."
KNOWN_NODES=$(grep -oP '(?<=^NodeName=)\S+' "$SLURM_CONF" | grep -v "^$NODE_NAME$" || true)
for node in $KNOWN_NODES; do
    ADDR=$(grep "^NodeName=$node " "$SLURM_CONF" | grep -oP '(?<=NodeAddr=)\S+' || true)
    TARGET="${ADDR:-$node}"
    if scp -o ConnectTimeout=5 "$SLURM_CONF" "$SSH_USER@$TARGET:/etc/slurm/slurm.conf" 2>/dev/null; then
        echo "  ✓ $node ($TARGET) — slurm.conf"
    else
        echo "  ✗ $node ($TARGET) — unreachable, update manually"
    fi
    # Push gres.conf too if it exists
    if [[ -f "$GRES_CONF" ]]; then
        scp -o ConnectTimeout=5 "$GRES_CONF" "$SSH_USER@$TARGET:/etc/slurm/gres.conf" 2>/dev/null || true
    fi
done

# ---------- 8. Restart slurmctld on head and slurmd on remote ----------------
log "Restarting slurmctld on head node ..."
systemctl restart slurmctld

log "Setting hostname and restarting slurmd on $REMOTE_HOST ..."
ssh "$SSH_USER@$REMOTE_HOST" bash -s "$NODE_NAME" <<'REMOTE_RESTART'
NODE_NAME="$1"

# Set hostname to match NodeName
hostnamectl set-hostname "$NODE_NAME" 2>/dev/null || hostname "$NODE_NAME"
echo "$NODE_NAME" > /etc/hostname
grep -q "$NODE_NAME" /etc/hosts || echo "127.0.1.1  $NODE_NAME" >> /etc/hosts

# Create/update systemd override for slurmd
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
echo "slurmd restarted with NodeName=$NODE_NAME"
REMOTE_RESTART

# ---------- 9. Verify -------------------------------------------------------
log "Waiting for node to register ..."
sleep 3

echo
echo "============================================"
echo " Node added successfully!"
echo "============================================"
echo " NodeName:     $NODE_NAME"
echo " NodeAddr:     $NODE_ADDR"
echo " CPUs:         $CPUS"
echo " Memory:       ${REAL_MEM} MB"
echo " GPUs:         $GPU_COUNT"
echo " Partition:    $PARTITION"
echo "============================================"
echo

sinfo -N -o '%15N %10T %6c %10m %20f %10G' 2>/dev/null || true
echo
log "Done! If the node shows as 'down', try:"
echo "  sudo scontrol update NodeName=$NODE_NAME State=RESUME"
