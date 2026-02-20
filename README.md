# BHRT3 — Black Hole Ray Tracer

A physically-based Kerr black hole ray tracer that renders images of spinning black holes with accretion disks. Uses full Kerr-Schild geodesic integration, volumetric radiative transfer, Novikov-Thorne temperature profiles, and relativistic beaming/redshift.

![Black hole render](demo/final.png)

## Features

- **Kerr metric** — full spinning black hole spacetime (spin up to a = 0.999M)
- **Volumetric accretion disk** with radiative transfer, opacity, and blackbody emission
- **Flat disk mode** — thin, highly textured, Interstellar/Gargantua-style appearance with procedural streak patterns and arc fragmentation
- **Disk stipple** — multi-octave procedural noise for particulate clumps and specs
- **Relativistic effects** — gravitational redshift, Doppler beaming, light bending
- **Multi-format output** — TGA, JPEG, Radiance HDR, and OpenEXR (float32, multi-layer)
- **Log tonemap** — configurable compression for cinematic look
- **Anti-aliasing** — NxN stratified supersampling (up to 16spp)
- **GPU acceleration** — optional CUDA backend for NVIDIA GPUs, or Metal backend for Apple Silicon Macs
- **Compile-time precision** — switchable fp64 (reference quality) / fp32 (fast GPU) via `USE_FLOAT`
- **LOD anti-aliasing** — procedural textures fade smoothly based on texel-to-pixel ratio, eliminating moiré and shimmer at all distances
- **CPU parallelism** — OpenMP for multi-core rendering
- **Animation** — time parameter for disk rotation with differential Keplerian orbits
- **Equirectangular** — supports 360° × 180° panoramic renders for VR

> **Note:** Post-processing (bloom, lens flares, film grain, colour grading) is handled
> externally by [flaresim](flaresim/), [tools/filmgrain](tools/filmgrain/), and
> compositing tools, keeping the renderer output clean for maximum flexibility.

## Requirements

- C++20 compiler (GCC, Clang, or MSVC)
- CMake 3.18+
- A sky map image (equirectangular JPEG, e.g. `hubble-skymap.jpg`)

### Optional

- **OpenMP** — multi-threaded CPU rendering (strongly recommended)
- **OpenEXR 3** — for `.exr` output with separate disk/sky/alpha layers
- **CUDA toolkit** — for GPU-accelerated rendering (NVIDIA GPUs)
- **Xcode** — for Metal GPU-accelerated rendering (Apple Silicon / macOS)

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
cmake --build . -j
```

The binary `bhrt3` will be created in the `build/` directory.

### Build options

| Option             | Default   | Description                                                                         |
| ------------------ | --------- | ----------------------------------------------------------------------------------- |
| `USE_GPU`          | `OFF`     | Enable CUDA GPU acceleration (NVIDIA)                                               |
| `USE_METAL`        | `OFF`     | Enable Metal GPU acceleration (Apple Silicon / macOS)                               |
| `USE_FLOAT`        | `OFF`     | Use fp32 instead of fp64 for all physics (~2× faster on GPU, 32× on consumer cards) |
| `CMAKE_BUILD_TYPE` | `Release` | `Release` for optimised, `Debug` for debugging                                      |

> `USE_GPU` and `USE_METAL` are mutually exclusive — enable only one at a time.

```bash
# CUDA GPU build (requires CUDA toolkit)
cmake .. -DUSE_GPU=ON

# CUDA GPU + single-precision (fastest on consumer GPUs)
cmake .. -DUSE_GPU=ON -DUSE_FLOAT=ON

# Metal GPU build (macOS — requires Xcode with metal compiler)
mkdir -p build-metal && cd build-metal
cmake .. -DUSE_METAL=ON
cmake --build . -j

# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

> **Important:** CMake caches option values. When switching `USE_FLOAT` or
> `USE_GPU` on/off, delete the build directory (or at least `CMakeCache.txt`)
> and reconfigure from scratch to ensure the flags take effect.

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

| Key             | Default | Description                                            |
| --------------- | ------- | ------------------------------------------------------ |
| `base_dt`       | `0.1`   | Base integration step size                             |
| `max_affine`    | `100.0` | Maximum affine parameter (ray lifetime)                |
| `escape_radius` | `50.0`  | Rays beyond this radius are considered escaped         |
| `max_iter`      | `50000` | Hard iteration cap per ray (bounds photon-sphere cost) |

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

This renders 120 frames, incrementing the `time` parameter by 0.5 each frame. Frames are saved to `build/frames/`. Supports TGA, EXR, and HDR output formats.

### Apply lens flares to all frames

```bash
python flare_sequence.py
python flare_sequence.py --jobs 4 -- --flare_gain 5000 --bloom_strength 3.0
```

Reads numbered EXR frames from `build/frames/`, runs each through [flaresim](flaresim/) to add lens-flare layers, and writes results to `build/frames_flared/`. Extra `--key value` pairs after `--` are forwarded as flaresim CLI overrides.

### Apply film grain

```bash
tools/filmgrain/build/filmgrain -s 0.12 build/output.tga output_grain.tga
```

See [tools/filmgrain](#filmgrain) below.

### Assemble into video

```bash
./make_video.sh -r 24 -o blackhole.mp4
```

Uses ffmpeg to encode the frame sequence into an H.264 video. Auto-detects flared frames if present. Options:

| Flag | Default         | Description                                  |
| ---- | --------------- | -------------------------------------------- |
| `-i` | _(auto-detect)_ | Input frames directory                       |
| `-p` | `frame`         | Frame filename prefix                        |
| `-o` | `blackhole.mp4` | Output video filename                        |
| `-r` | `24`            | Framerate (fps)                              |
| `-c` | `18`            | H.264 CRF quality (0 = lossless, 51 = worst) |
| `-f` | `tga`           | Input frame format (`tga`, `exr`, `hdr`)     |

### Distributed rendering

For rendering across multiple machines, see [slurm/README.md](slurm/README.md).

## GPU Rendering

Both GPU backends use a persistent-thread work-stealing design:

- Global atomic work counter distributes pixels dynamically
- Threads that finish cheap sky pixels immediately pick up new work
- Eliminates the "tail-end" stall from uneven per-pixel cost
- Live progress reporting (CUDA: zero-copy pinned memory; Metal: shared MTLBuffer polling)

Physics code in `physics.h` is shared across CPU, CUDA, and Metal via the `BH_FUNC` macro (`__host__ __device__` under nvcc, empty otherwise). The `BH_THREAD` macro handles Metal Shading Language's address space qualifiers transparently.

### CUDA (NVIDIA GPUs)

Requires the CUDA toolkit. Build with `cmake .. -DUSE_GPU=ON`.

When `USE_FLOAT=ON`, all physics computation uses `float` instead of `double`. On consumer NVIDIA GPUs (GeForce), this can be **32× faster** because fp64 throughput is heavily throttled. On datacenter GPUs (A100, H100) the speedup is ~2×. There is a slight loss of accuracy near the photon sphere.

### Metal (Apple Silicon / macOS)

Requires Xcode (the full IDE install — Command Line Tools alone do not include the `metal` shader compiler). Build with:

```bash
mkdir -p build-metal && cd build-metal
cmake .. -DUSE_METAL=ON
cmake --build . -j
```

Metal always uses fp32 (`BH_USE_FLOAT` is forced on) because Apple Silicon has no fp64 compute capability. The Metal shader (`.metal` → `.air` → `.metallib`) is compiled as part of the CMake build. At runtime, `bhrt3` loads `metal_render.metallib` from the working directory or the default Metal library.

> **Tip:** Use a separate build directory (`build-metal/`) so you can switch between CPU and Metal builds without reconfiguring.

## Post-Processing Pipeline

bhrt3 outputs clean, unprocessed linear HDR data. All post-processing is handled by external tools:

1. **bhrt3** → renders raw EXR with separate disk/sky layers
2. **[flaresim](flaresim/)** → adds physically-based lens flares, bloom, and ghost reflections
3. **[filmgrain](tools/filmgrain/)** → applies photographic film grain to TGA output
4. **Compositing** (Resolve, Nuke, etc.) → final colour grading, sky replacement, LUTs

This separation keeps each stage clean and allows non-destructive iteration on the look.

## Tools

Standalone utility programs in the [tools/](tools/) directory, each with their own CMakeLists.txt.

### flaretest

Generates a minimal test EXR file with a single extremely bright pixel — useful for validating flaresim's ghost tracing and bloom pipeline without a full bhrt3 render.

```bash
cd tools/flaretest && mkdir -p build && cd build
cmake .. && cmake --build .
./flaretest [output.exr]
```

### filmgrain

Applies photographic silver-halide film grain to a TGA image. The grain is primarily luminance noise (like real film crystal density variation) with a subtle, separately-generated chroma offset modelling the independent emulsion layers of colour negative stock. Grain is spatially clustered (not pixel-level white noise), shaped by an exposure response curve: minimal in deep shadows, peaks in mid-tones, gentle rolloff in highlights.

```bash
cd tools/filmgrain && mkdir -p build && cd build
cmake .. && cmake --build .
./filmgrain [options] input.tga [output.tga]
```

| Option    | Default | Description                                          |
| --------- | ------- | ---------------------------------------------------- |
| `-s`      | `0.06`  | Grain strength (0.0–1.0)                             |
| `-g`      | `1.8`   | Grain clump size (>1 = coarser, mimics 35mm scan)    |
| `-chroma` | `0.12`  | Chroma noise as fraction of luma strength (0 = mono) |
| `-seed`   | `42`    | RNG seed (same seed = same grain pattern)            |

## Flaresim

The [flaresim/](flaresim/) directory contains a physically-based lens flare simulator that reads EXR renders from bhrt3 and adds:

- **Ghost reflections** — traced through a real lens prescription using Fresnel equations
- **Bloom** — energy-conserving kernel convolution
- **Chromatic aberration** — wavelength-dependent ghost positioning and intensity

Lens prescriptions (`.lens` files) are in [flaresim/lenses/](flaresim/lenses/). Includes a Cooke triplet and double-Gauss design. See [flaresim/README.md](flaresim/README.md) for full documentation.

## Slurm Cluster Rendering

The [slurm/](slurm/) directory contains scripts for distributing frame renders across multiple machines connected via Tailscale, scheduled by Slurm. Each frame is an independent array task. See [slurm/README.md](slurm/README.md) for setup and usage.

## Reference

`kerr-image.c` is the original reference implementation by David A. Madore (2011, Public Domain) that inspired the geodesic integration approach used in bhrt3.

## Project Structure

```
├── CMakeLists.txt          # Main build configuration
├── scene.txt               # Scene description (key=value)
├── render_sequence.py      # Animation: render numbered frames
├── flare_sequence.py       # Animation: apply flaresim to all frames
├── make_video.sh           # Animation: assemble frames into video
├── kerr-image.c            # Reference: Madore's original Kerr tracer
├── hubble-skymap.jpg       # Default sky map
├── inc/                    # Shared header files
│   ├── physics.h           # Kerr geodesics, disk physics, radiative transfer, LOD
│   ├── scene.h             # Scene config struct
│   ├── sky.h               # Sky map sampling
│   ├── types.h             # Shared type aliases
│   ├── vec_math.h          # CPU/GPU vector/matrix math, bh_real precision typedef
│   └── common.h            # Common utilities
├── src/                    # C++ source files
│   ├── main.cpp            # CPU render loop, tone mapping, output writers
│   ├── scene.cpp           # Scene file parser
│   ├── sky.cpp             # Sky image loader
│   └── stb_impl.cpp        # stb library implementations
├── gpu/                    # GPU backends (CUDA + Metal)
│   ├── gpu_render.cu       # CUDA render kernel (persistent-thread work-stealing)
│   ├── gpu_render.h        # GPU interface and scene param struct
│   ├── metal_render.metal  # Metal compute kernel (shared physics via physics.h)
│   ├── metal_render.mm     # Metal host launch code (Objective-C++)
│   └── metal_render.h      # Metal interface header
├── lib/                    # Third-party libraries (header-only)
│   ├── stb_image.h         # Image loading
│   └── stb_image_write.h   # Image writing (TGA, HDR, JPEG)
├── flaresim/               # Lens flare simulator (separate build)
│   ├── src/                # Flare sim source (ghost tracing, bloom, Fresnel)
│   ├── lenses/             # Lens prescriptions (.lens files)
│   └── flaresim.conf       # Default flaresim settings
├── tools/                  # Standalone utility programs
│   ├── flaretest/          # EXR test pattern generator
│   └── filmgrain/          # Film grain applicator
└── slurm/                  # HPC cluster scripts (Slurm + Tailscale)
    ├── submit_render.sh    # Submit array job
    ├── render_frame.sbatch # Per-frame Slurm job script
    ├── collect_frames.sh   # Rsync frames from compute nodes
    ├── setup_node.sh       # Bootstrap a new compute node
    ├── add_node.sh         # Register node with Slurm controller
    ├── cluster_status.sh   # Check node/job status
    └── slurm.conf.template # Slurm config template
```

## License

This project is available under the MIT license
