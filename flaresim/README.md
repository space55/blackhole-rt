# Flaresim — Physically-Based Lens Flare Simulator

Reads a linear-light EXR image (e.g. from [bhrt3](../README.md)), extracts bright pixels, traces ghost reflections through a real lens prescription, and composites the result as additional EXR layers alongside the original channels.

Lens flares are computed by sequential ray tracing through every ghost bounce pair in the lens system, using physically-correct Fresnel reflectance, Cauchy chromatic dispersion, and multi-layer AR coating models.

## Features

- **Ghost reflections** — traces all C(N,2) bounce pairs through the full lens prescription
- **Fresnel reflectance** — polarisation-averaged Fresnel coefficients at every surface
- **Chromatic dispersion** — Cauchy model via Abbe number; separate R/G/B wavelengths
- **AR coatings** — multi-layer anti-reflection coating simulation per surface
- **Multi-scale bloom** — multi-octave warm chromatic bloom with configurable radius
- **Ghost smoothing** — adaptive tent-filter splatting with optional box-blur passes
- **Area normalization** — per-pair defocus compensation (ILM/Weta-style technique)
- **Multi-layer EXR I/O** — reads bhrt3 layers, writes `flare.R/G/B` alongside originals
- **Optional TGA output** — composited beauty + flare, tonemapped to 8-bit
- **GPU acceleration** — optional CUDA backend for ghost ray tracing on NVIDIA GPUs
- **CPU parallelism** — OpenMP for multi-core rendering
- **Config file + CLI overrides** — same `key = value` format as bhrt3

## Requirements

- C++20 compiler
- CMake 3.18+
- OpenEXR 3

### Optional

- **OpenMP** — multi-threaded CPU rendering (strongly recommended)
- **CUDA toolkit** — for GPU-accelerated ghost tracing

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
cd flaresim
mkdir -p build && cd build
cmake ..
cmake --build .
```

The binary `flaresim` will be created in `flaresim/build/`.

### Build options

| Option              | Default   | Description                                    |
| ------------------- | --------- | ---------------------------------------------- |
| `FLARESIM_USE_CUDA` | `OFF`     | Enable CUDA GPU ghost ray tracing              |
| `CMAKE_BUILD_TYPE`  | `Release` | `Release` for optimised, `Debug` for debugging |

```bash
# GPU build (requires CUDA toolkit — Linux/Windows only, not macOS)
cmake .. -DFLARESIM_USE_CUDA=ON
```

## Usage

```bash
cd flaresim/build
./flaresim flaresim.conf
```

Flaresim auto-detects config files in the current directory, searching for `flaresim.conf`, `flare.conf`, `flaresim.cfg`, or `flare.cfg`. You can also pass a config path explicitly, and override any key from the command line:

```bash
./flaresim flaresim.conf --flare_gain 5000 --threshold 2.0
```

### Typical workflow

```bash
# 1. Render a scene with bhrt3 (outputs multi-layer EXR)
cd build && ./bhrt3 scene.txt

# 2. Run flaresim on the EXR output
cd ../flaresim/build
./flaresim flaresim.conf
```

## Configuration

All parameters use `key = value` syntax. Lines starting with `#` are comments.

### Input / Output

| Key         | Default | Description                                          |
| ----------- | ------- | ---------------------------------------------------- |
| `input`     | —       | Input EXR file (must have R, G, B channels)          |
| `output`    | —       | Output EXR file (original layers + `flare.R/G/B`)    |
| `lens`      | —       | Lens prescription file (`.lens`)                     |
| `tga`       | —       | Optional composited TGA output (tonemapped 8-bit)    |
| `debug_tga` | —       | Optional debug TGA showing only bright source pixels |

### Optics

| Key         | Default | Description                             |
| ----------- | ------- | --------------------------------------- |
| `fov`       | `60`    | Horizontal field of view (degrees)      |
| `rays`      | `64`    | Entrance pupil grid per dimension (N×N) |
| `min_ghost` | `1e-7`  | Ghost pair pre-filter threshold         |

### Source Extraction

| Key          | Default | Description                             |
| ------------ | ------- | --------------------------------------- |
| `threshold`  | `3.0`   | Bright pixel luminance threshold        |
| `downsample` | `4`     | Downsample bright pixels by this factor |

### Flare Appearance

| Key              | Default | Description                           |
| ---------------- | ------- | ------------------------------------- |
| `flare_gain`     | `1000`  | Ghost intensity multiplier            |
| `sky_brightness` | `1.0`   | Scale sky background layers in output |

### Bloom

| Key               | Default | Description                                           |
| ----------------- | ------- | ----------------------------------------------------- |
| `bloom_strength`  | `0.0`   | Bloom intensity (0 = off, 1.0 = strong, 3.0+ = heavy) |
| `bloom_radius`    | `0.04`  | Base blur radius as fraction of image diagonal        |
| `bloom_passes`    | `3`     | Box-blur passes (approximates Gaussian)               |
| `bloom_octaves`   | `5`     | Multi-scale octaves (more = wider glow)               |
| `bloom_chromatic` | `1`     | Warm chromatic shift (white → yellow → orange → red)  |
| `bloom_threshold` | `-1`    | Bloom bright-pixel cutoff (-1 = use main threshold)   |

### Ghost Smoothing

| Key                 | Default | Description                                   |
| ------------------- | ------- | --------------------------------------------- |
| `ghost_blur`        | `0.003` | Blur radius as fraction of diagonal (0 = off) |
| `ghost_blur_passes` | `3`     | Box-blur passes (3 ≈ Gaussian)                |

### Ghost Normalization

| Key               | Default | Description                                      |
| ----------------- | ------- | ------------------------------------------------ |
| `ghost_normalize` | `1`     | Per-pair defocus area compensation (recommended) |
| `max_area_boost`  | `100`   | Cap on defocus correction factor                 |

### TGA Output

| Key        | Default | Description                                 |
| ---------- | ------- | ------------------------------------------- |
| `exposure` | `1.0`   | Exposure multiplier for TGA output          |
| `tonemap`  | `1.0`   | Tonemap compression (0 = linear, 1 = heavy) |

## Lens Prescriptions

Lens files use a simple text format (`.lens`). Each file defines an ordered sequence of optical surfaces:

```
name: Double Gauss 58mm f/2
focal_length: 58.0

surfaces:
# radius    thickness   ior     abbe    semi_ap  coating
  39.68     8.36        1.670   47.2    25.0     1
  152.73    0.20        1.000   0.0     25.0     0
  stop      12.13       1.000   0.0     12.0     0
  -25.46    1.82        1.603   38.0    14.0     1
  ...
```

| Column      | Description                                                         |
| ----------- | ------------------------------------------------------------------- |
| `radius`    | Signed radius of curvature in mm (0 = flat, `stop` = aperture stop) |
| `thickness` | Axial distance to next surface (mm)                                 |
| `ior`       | Refractive index after this surface (d-line, 1.0 = air)             |
| `abbe`      | Abbe number for Cauchy dispersion (0 = non-dispersive)              |
| `semi_ap`   | Clear semi-diameter (mm)                                            |
| `coating`   | AR coating layers (0 = uncoated, 1+ = coated)                       |

### Included lenses

| File                                     | Design                       | Elements        | Ghost pairs |
| ---------------------------------------- | ---------------------------- | --------------- | ----------- |
| `doublegauss.lens`                       | Double Gauss 58mm f/2        | 6 (11 surfaces) | 55          |
| `cooketriplet.lens`                      | Cooke Triplet 50mm f/3.5     | 3 (7 surfaces)  | 21          |
| `arri-zeiss-master-prime-t1.3-50mm.lens` | ARRI/Zeiss Master Prime 50mm | complex         | many        |
| `canon-ef-200-400-f4.lens`               | Canon EF 200-400mm f/4       | complex         | many        |

### Converting lens data

Two Python converters are included for importing lens prescriptions from [photonstophotos.net](https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/):

```bash
# From the Optical Bench .txt export format:
python lenses/convert_ob.py input.txt -o output.lens

# From pasted table data:
python lenses/convert_p2p.py input.txt -o output.lens
```

## EXR Layer Output

Flaresim preserves all input EXR channels and adds:

| Channel          | Description                                     |
| ---------------- | ----------------------------------------------- |
| `flare.R/G/B`    | Ghost reflection layer (additive, linear light) |
| _(all original)_ | All input channels passed through unchanged     |

The `flare` layer can be composited additively onto the beauty pass in any grading/compositing tool.

## GPU Acceleration

When built with `-DFLARESIM_USE_CUDA=ON`, ghost ray tracing runs on the GPU. The program automatically falls back to CPU if CUDA is unavailable at runtime.

The CUDA kernel traces all rays for all ghost pairs in parallel, while the adaptive splatting step runs on the CPU. This provides significant speedups for high `rays` grid sizes and complex lenses with many ghost pairs.

> **Note:** CUDA is not available on macOS. Use Metal for local GPU work or
> CUDA on Linux/Windows nodes in a cluster setup (see [slurm/](../slurm/README.md)).

## Project Structure

```
flaresim/
├── CMakeLists.txt          # Build configuration (optional CUDA)
├── flaresim.conf           # Example configuration file
├── README.md               # This file
├── src/
│   ├── main.cpp            # Entry point, EXR I/O, CLI, config parser
│   ├── lens.cpp            # Lens prescription loader
│   ├── lens.h              # Lens system data structures
│   ├── trace.cpp           # Sequential ray tracing through lens
│   ├── trace.h             # Ray tracing interface
│   ├── ghost.cpp           # Ghost enumeration, rendering, splatting
│   ├── ghost.h             # Ghost renderer interface
│   ├── fresnel.h           # Fresnel reflectance + Cauchy dispersion
│   ├── vec3.h              # 3D vector math
│   ├── gpu_ghosts.h        # CUDA ghost interface (GPU structs + dispatch)
│   ├── gpu_ghosts.cu       # CUDA ghost ray tracing kernel
│   └── stb_impl.cpp        # stb library implementations
└── lenses/
    ├── doublegauss.lens     # Double Gauss 58mm f/2
    ├── cooketriplet.lens    # Cooke Triplet 50mm f/3.5
    ├── arri-zeiss-master-prime-t1.3-50mm.lens
    ├── canon-ef-200-400-f4.lens
    ├── test.lens            # Minimal test lens
    ├── convert_ob.py        # Optical Bench format converter
    └── convert_p2p.py       # Photons to Photos table converter
```
