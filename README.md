# BHRT3 — Black Hole Ray Tracer

A physically-based Kerr black hole ray tracer that renders images of spinning black holes with accretion disks. Uses full Kerr-Schild geodesic integration, volumetric radiative transfer, Novikov-Thorne temperature profiles, and relativistic beaming/redshift.

![Black hole render](build/output.tga)

## Features

- **Kerr metric** — full spinning black hole spacetime (spin up to a = 0.999M)
- **Volumetric accretion disk** with radiative transfer, opacity, and blackbody emission
- **Flat disk mode** — thin, highly textured, Interstellar/Gargantua-style appearance with procedural streak patterns and arc fragmentation
- **Disk stipple** — multi-octave procedural noise for particulate clumps and specs
- **Relativistic effects** — gravitational redshift, Doppler beaming, light bending
- **Multi-format output** — TGA, JPEG, Radiance HDR, and OpenEXR (float32, multi-layer)
- **Log tonemap** — configurable compression for cinematic look
- **Anti-aliasing** — NxN stratified supersampling (up to 16spp)
- **GPU acceleration** — optional CUDA backend for NVIDIA GPUs
- **CPU parallelism** — OpenMP for multi-core rendering
- **Animation** — time parameter for disk rotation with differential Keplerian orbits
- **Equirectangular** — supports 360° × 180° panoramic renders for VR

> **Note:** Post-processing (bloom, lens flares, colour grading) is handled
> externally by [flaresim](flaresim/) and compositing tools, keeping the
> renderer output clean for maximum flexibility.

## Requirements

- C++20 compiler (GCC, Clang, or MSVC)
- CMake 3.18+
- A sky map image (equirectangular JPEG, e.g. `hubble-skymap.jpg`)

### Optional

- **OpenMP** — multi-threaded CPU rendering (strongly recommended)
- **OpenEXR 3** — for `.exr` output with separate disk/sky/alpha layers
- **CUDA toolkit** — for GPU-accelerated rendering

### macOS (Homebrew)

```bash
brew install cmake openexr libomp
```

### Ubuntu/Debian

```bash
sudo apt install cmake libopenexr-dev libomp-dev
```

## Building

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

The binary `bhrt3` will be created in the `build/` directory.

### Build options

| Option             | Default   | Description                                    |
| ------------------ | --------- | ---------------------------------------------- |
| `USE_GPU`          | `OFF`     | Enable CUDA GPU acceleration                   |
| `CMAKE_BUILD_TYPE` | `Release` | `Release` for optimised, `Debug` for debugging |

```bash
# GPU build (requires CUDA toolkit)
cmake .. -DUSE_GPU=ON

# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

## Usage

```bash
cd build
./bhrt3 scene.txt
```

If no scene file is specified, `bhrt3` looks for `scene.txt` in the current directory. Missing keys use built-in defaults.

## Scene Configuration

All parameters are set in a plain-text `scene.txt` file using `key = value` syntax. Lines starting with `#` are comments.

### Output

| Key             | Default      | Description                                 |
| --------------- | ------------ | ------------------------------------------- |
| `output_width`  | `1024`       | Image width in pixels                       |
| `output_height` | `512`        | Image height in pixels                      |
| `output_file`   | `output.tga` | Primary output (TGA, tonemapped 8-bit)      |
| `hdr_output`    | _(empty)_    | Radiance HDR output path (raw linear float) |
| `exr_output`    | _(empty)_    | OpenEXR output path (float32, multi-layer)  |
| `jpg_output`    | _(empty)_    | JPEG thumbnail output path                  |

### Sky Map

| Key              | Default             | Description                               |
| ---------------- | ------------------- | ----------------------------------------- |
| `sky_image`      | `hubble-skymap.jpg` | Equirectangular sky map image             |
| `sky_brightness` | `1.0`               | Sky brightness multiplier (0 = black sky) |
| `sky_pitch`      | `0.0`               | Sky rotation pitch (degrees)              |
| `sky_yaw`        | `0.0`               | Sky rotation yaw (degrees)                |
| `sky_roll`       | `0.0`               | Sky rotation roll (degrees)               |
| `sky_offset_u`   | `0.0`               | Horizontal pan of sky texture (0–1 wraps) |
| `sky_offset_v`   | `0.0`               | Vertical pan of sky texture (0–1 wraps)   |

### Camera

| Key            | Default | Description                               |
| -------------- | ------- | ----------------------------------------- |
| `camera_x`     | `-25.0` | Camera X position (Cartesian, units of M) |
| `camera_y`     | `5.0`   | Camera Y position                         |
| `camera_z`     | `0.0`   | Camera Z position                         |
| `camera_pitch` | `10.0`  | Camera pitch (degrees)                    |
| `camera_yaw`   | `90.0`  | Camera yaw (degrees)                      |
| `camera_roll`  | `0.0`   | Camera roll (degrees)                     |
| `fov_x`        | `360.0` | Horizontal field of view (degrees)        |
| `fov_y`        | `180.0` | Vertical field of view (degrees)          |

### Black Hole

| Key       | Default | Description                                     |
| --------- | ------- | ----------------------------------------------- |
| `bh_mass` | `1.0`   | Mass parameter M                                |
| `bh_spin` | `0.999` | Spin parameter a (\|a\| < M, 0 = Schwarzschild) |

### Ray Integration

| Key             | Default | Description                             |
| --------------- | ------- | --------------------------------------- |
| `base_dt`       | `0.1`   | Base integration step size              |
| `max_affine`    | `100.0` | Maximum affine parameter (ray lifetime) |
| `escape_radius` | `50.0`  | Rays beyond this radius are escaped     |

### Accretion Disk

| Key                    | Default | Description                                   |
| ---------------------- | ------- | --------------------------------------------- |
| `disk_inner_r`         | `-1`    | Inner edge radius (-1 = auto ISCO)            |
| `disk_outer_r`         | `20.0`  | Outer edge radius (units of M)                |
| `disk_thickness`       | `0.5`   | Half-thickness scale height                   |
| `disk_density`         | `20.0`  | Base density ρ₀                               |
| `disk_opacity`         | `0.5`   | Absorption coefficient κ₀                     |
| `disk_emission_boost`  | `10.0`  | Brightness multiplier                         |
| `disk_color_variation` | `0.7`   | 0 = physical blackbody, 1 = cinematic colour  |
| `disk_turbulence`      | `0.0`   | 0 = smooth disk, 1 = torn debris ring         |
| `disk_stipple`         | `0.0`   | 0 = smooth, 1 = fully particulate             |
| `disk_flat_mode`       | `0`     | 0 = volumetric, 1 = thin/flat Gargantua-style |

### Tone Mapping & Output

| Key                   | Default | Description                                          |
| --------------------- | ------- | ---------------------------------------------------- |
| `tonemap_compression` | `1.0`   | Log compression (0 = linear, 1 = heavy, cinematic)   |
| `exposure`            | `1.0`   | Output exposure multiplier (TGA/JPEG only)           |
| `aa_samples`          | `1`     | NxN supersampling grid (1 = off, 2 = 4spp, 3 = 9spp) |

### Animation

| Key    | Default | Description                                             |
| ------ | ------- | ------------------------------------------------------- |
| `time` | `0.0`   | Frame time for disk rotation (increment ~1.0 per frame) |

## EXR Output Layers

When `exr_output` is set, the renderer writes a multi-layer float32 OpenEXR file containing clean, unprocessed linear radiance:

| Channel       | Description                                              |
| ------------- | -------------------------------------------------------- |
| `R`, `G`, `B` | Combined beauty pass (disk + sky × brightness)           |
| `A`           | Always 1.0 (fully composited image)                      |
| `disk.R/G/B`  | Raw disk emission only                                   |
| `disk.A`      | Disk opacity (0 = transparent, 1 = opaque or black hole) |
| `sky.R/G/B`   | Sky contribution (brightness-scaled)                     |

These separate layers allow independent colour grading, bloom, lens flares, sky replacement, and compositing in software like DaVinci Resolve, Nuke, or After Effects. The companion [flaresim](flaresim/) tool reads these EXR files directly.

## Animation Workflow

### Render a sequence

```bash
python render_sequence.py -n 120 -dt 0.5
```

This renders 120 frames, incrementing the `time` parameter by 0.5 each frame. Frames are saved to `build/frames/`.

### Assemble into video

```bash
./make_video.sh -r 24 -o blackhole.mp4
```

Uses ffmpeg to encode the frame sequence into an H.264 video.

### Distributed rendering

For rendering across multiple machines, see [slurm/README.md](slurm/README.md).

## Post-Processing Pipeline

bhrt3 outputs clean, unprocessed linear HDR data. All post-processing is handled by external tools:

1. **bhrt3** → renders raw EXR with separate disk/sky layers
2. **[flaresim](flaresim/)** → adds physically-based lens flares, bloom, and ghost reflections
3. **Compositing** (Resolve, Nuke, etc.) → final colour grading, sky replacement, LUTs

This separation keeps each stage clean and allows non-destructive iteration on the look.

## Project Structure

```
├── CMakeLists.txt          # Main build configuration
├── scene.txt               # Scene description (key=value)
├── render_sequence.py      # Animation frame renderer
├── make_video.sh           # ffmpeg video assembler
├── inc/                    # Header files
│   ├── physics.h           # Kerr geodesics, disk physics, radiative transfer
│   ├── scene.h             # Scene config struct
│   ├── sky.h               # Sky map sampling
│   ├── types.h             # Shared types (dvec3, etc.)
│   ├── vec_math.h          # CPU/GPU vector/matrix math
│   └── common.h            # Common utilities
├── src/                    # C++ source files
│   ├── main.cpp            # Render loop, output writers
│   ├── scene.cpp           # Scene file parser
│   ├── sky.cpp             # Sky image loader
│   └── stb_impl.cpp        # stb library implementations
├── gpu/                    # CUDA GPU backend
│   ├── gpu_render.cu       # GPU render kernel
│   └── gpu_render.h        # GPU interface
├── lib/                    # Third-party libraries (header-only)
│   ├── stb_image.h         # Image loading
│   └── stb_image_write.h   # Image writing (TGA, HDR, JPEG)
├── flaresim/               # Lens flare simulator (separate build)
│   ├── src/                # Flare sim source code
│   ├── lenses/             # Lens prescriptions (.lens files)
│   └── README.md           # Flaresim documentation
├── tools/                  # Utility programs
│   └── flaretest/          # EXR test pattern generator
└── slurm/                  # HPC cluster scripts (Slurm + Tailscale)
```

## License

This project is provided as-is for educational and research purposes.
