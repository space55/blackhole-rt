# BHRT Distributed Rendering with Slurm + Tailscale

Distribute black hole ray tracing frame renders across multiple machines
connected via a [Tailscale](https://tailscale.com/) mesh, scheduled by
[Slurm](https://slurm.schedmd.com/).

## Architecture

```
┌────────────────────────────────────────────────────────┐
│  Your machine (macOS)                                  │
│    submit_render.sh  ──sbatch──►  Slurm controller     │
│    collect_frames.sh ◄──rsync──  compute nodes         │
│    make_video.sh     ──ffmpeg──►  blackhole.mp4        │
└────────────────────────────────────────────────────────┘
        │ Tailscale VPN (100.x.y.z)
        ├───── node01 (8 CPU)   ── renders frames 0-19
        ├───── node02 (16 CPU)  ── renders frames 20-39
        ├───── gpu01  (GPU)     ── renders frames 40-59
        └───── ...
```

Each frame is an independent Slurm **array task**. Slurm distributes tasks
across available nodes automatically. Frames are collected back to the head
node and assembled into a video.

## Quick Start

### 1. Set up Tailscale on every machine

```bash
# Install: https://tailscale.com/download
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Verify connectivity between nodes:
tailscale ping <other-node>
```

### 2. Bootstrap the head node

```bash
# On the machine that will run slurmctld:
sudo ./slurm/setup_node.sh --head
```

This installs Slurm + MUNGE, builds the renderer at `/opt/bhrt/`, and
generates a MUNGE key.

### 3. Configure the cluster

Edit `/etc/slurm/slurm.conf` on the head node:

```conf
# Set the head node hostname:
SlurmctldHost=your-head-node-tailscale-hostname

# Add your nodes (use Tailscale IPs from `tailscale ip -4`):
NodeName=node01  NodeAddr=100.64.0.1  CPUs=8   RealMemory=16000  State=UNKNOWN
NodeName=node02  NodeAddr=100.64.0.2  CPUs=16  RealMemory=32000  State=UNKNOWN
NodeName=gpu01   NodeAddr=100.64.0.10 CPUs=16  RealMemory=64000  Gres=gpu:1  State=UNKNOWN

# Define partitions:
PartitionName=cpu  Nodes=node01,node02  Default=YES  MaxTime=INFINITE  State=UP
PartitionName=gpu  Nodes=gpu01          Default=NO   MaxTime=INFINITE  State=UP
```

### 4. Set up compute nodes

```bash
# Copy MUNGE key from head to each compute node:
scp /etc/munge/munge.key node01:/etc/munge/munge.key
scp /etc/munge/munge.key node02:/etc/munge/munge.key

# Copy slurm.conf to each compute node:
scp /etc/slurm/slurm.conf node01:/etc/slurm/slurm.conf
scp /etc/slurm/slurm.conf node02:/etc/slurm/slurm.conf

# Run setup on each compute node:
ssh node01 'sudo /opt/bhrt/slurm/setup_node.sh'
ssh node02 'sudo /opt/bhrt/slurm/setup_node.sh'
```

### 5. Verify the cluster

```bash
sinfo                          # Should show all nodes in "idle" state
./slurm/cluster_status.sh      # Pretty-printed cluster overview
```

### 6. Submit a render

```bash
# Render 120 frames, distributed across all nodes:
./slurm/submit_render.sh -n 120 -dt 0.5 -t0 0.0

# With GPU acceleration:
./slurm/submit_render.sh -n 120 --gpu

# Dry run (preview the sbatch command):
./slurm/submit_render.sh -n 120 --dry-run
```

### 7. Monitor progress

```bash
./slurm/cluster_status.sh -w   # Live dashboard (refreshes every 5s)
squeue -u $USER                # Raw Slurm queue
```

### 8. Collect frames and make the video

```bash
# Pull frames from all nodes (or verify if using shared FS):
./slurm/collect_frames.sh

# If using a shared filesystem:
./slurm/collect_frames.sh --shared-fs

# Assemble into video:
./make_video.sh -i build/frames -o blackhole.mp4
```

## File Overview

| File                  | Purpose                                                                         |
| --------------------- | ------------------------------------------------------------------------------- |
| `slurm.conf.template` | Slurm configuration template — edit and copy to `/etc/slurm/slurm.conf`         |
| `setup_node.sh`       | Bootstrap script for each node (installs deps, builds renderer, starts daemons) |
| `render_frame.sbatch` | Slurm batch script — renders a single frame (one per array task)                |
| `submit_render.sh`    | Main entry point — submits a job array to Slurm                                 |
| `collect_frames.sh`   | Gathers frames from nodes, verifies completeness                                |
| `cluster_status.sh`   | Live cluster dashboard (Tailscale + Slurm + frame progress)                     |

## Options Reference

### `submit_render.sh`

| Flag        | Default   | Description                       |
| ----------- | --------- | --------------------------------- |
| `-n`        | 60        | Number of frames                  |
| `-t0`       | 0.0       | Starting time value               |
| `-dt`       | 0.5       | Time step per frame               |
| `-p`        | frame     | Filename prefix                   |
| `-d`        | /opt/bhrt | Project directory on nodes        |
| `-P`        | cpu       | Slurm partition                   |
| `-c`        | 4         | CPUs per task                     |
| `-m`        | 4G        | Memory per task                   |
| `-T`        | 01:00:00  | Wall time limit per frame         |
| `--gpu`     | off       | Use GPU partition + request 1 GPU |
| `--dry-run` | off       | Preview without submitting        |

### `collect_frames.sh`

| Flag          | Default        | Description                 |
| ------------- | -------------- | --------------------------- |
| `-n`          | from job info  | Expected frame count        |
| `-d`          | /opt/bhrt      | Project directory           |
| `-L`          | ./build/frames | Local collection directory  |
| `--shared-fs` | off            | Skip rsync, verify in-place |
| `--nodes`     | auto-detect    | Comma-separated node list   |

## Networking Notes

- All Slurm traffic (slurmctld ↔ slurmd) flows through the Tailscale mesh
- Node addresses in `slurm.conf` must be Tailscale IPs (100.x.y.z) or hostnames
- SSH for `rsync` in `collect_frames.sh` also routes through Tailscale
- No port forwarding or public IPs required
- Tailscale ACLs can restrict which nodes can communicate

## Troubleshooting

### Nodes show as "down" in `sinfo`

```bash
# Check slurmd is running on the node:
ssh node01 'systemctl status slurmd'

# Check Tailscale connectivity:
tailscale ping node01

# Resume a drained node:
sudo scontrol update NodeName=node01 State=RESUME
```

### MUNGE authentication errors

```bash
# Verify same key on all nodes:
md5sum /etc/munge/munge.key    # must match everywhere

# Restart MUNGE:
sudo systemctl restart munge
munge -n | unmunge             # test locally
ssh node01 'munge -n' | unmunge  # test cross-node
```

### Frames missing after render

```bash
# Check which tasks failed:
sacct -j <JOB_ID> --format=JobID,State,ExitCode,NodeList

# View error logs:
cat /opt/bhrt/build/logs/frame_<JOB_ID>_<TASK_ID>.err

# Re-render only missing frames:
./slurm/collect_frames.sh -n 120   # reports missing indices
sbatch --array=5,12,47 slurm/render_frame.sbatch
```

### Clock skew between nodes

MUNGE requires clocks to be within 5 minutes. Use NTP:

```bash
sudo timedatectl set-ntp true
timedatectl status
```
