# BHRT Tools — Post-Processing Utilities

Standalone command-line tools for post-processing EXR renders from the BHRT3 black hole ray tracer.

## Building

```bash
cd tools
mkdir -p build && cd build
cmake ..
cmake --build .
```

### Requirements

- C++20 compiler
- CMake 3.18+
- OpenEXR 3 (required)
- OpenMP (optional, for parallelism)

---

## bhrt-grade

Unified post-processor combining bloom, lens flare, and grading in a single pass. This avoids the texture-blurring problem that arises when chaining `bhrt-bloom` and `bhrt-lensflare` separately (bloom softens the beauty pass, then flare reads from the softened result, compounding blur; exposure/tonemap also get applied twice).

### Usage

```bash
bhrt-grade input.exr output.[tga|exr|hdr] [options]
```

### Processing Pipeline

1. **Sky brightness** — Recomposites from `disk.*` + `sky.*` layers.
2. **Exposure** — Multiplies beauty RGB.
3. **Bright pixel extraction** — One shared threshold pass feeds both bloom and flare.
4. **Bloom** — Multi-octave box-blur with optional chromatic warm shift.
5. **Lens flare** — Ghosts, halo, starburst, anamorphic streak.
6. **Composite** — Adds all effects onto the original beauty pass (single pass).
7. **Tonemap** — Log compression.
8. **Output** — TGA/EXR/HDR.

### Options

**General:**

| Option                 | Default   | Description                                                      |
| ---------------------- | --------- | ---------------------------------------------------------------- |
| `--threshold <f>`      | `1.0`     | Shared luminance threshold for bloom + flare sources             |
| `--exposure <f>`       | `1.0`     | Exposure multiplier                                              |
| `--tonemap <f>`        | `0` (off) | Log tonemap compression                                          |
| `--sky-brightness <f>` | `1.0`     | Sky brightness multiplier (requires `disk.*`/`sky.*` EXR layers) |

**Bloom:**

| Option                 | Default | Description                                                |
| ---------------------- | ------- | ---------------------------------------------------------- |
| `--bloom-strength <f>` | `0.8`   | Bloom intensity (0 = off)                                  |
| `--bloom-radius <f>`   | `0.02`  | Base radius as fraction of diagonal                        |
| `--bloom-passes <n>`   | `3`     | Box-blur passes (1–10)                                     |
| `--bloom-octaves <n>`  | `4`     | Scale octaves (1–6)                                        |
| `--chromatic`          | off     | Chromatic bloom shift (white→yellow→orange→red per octave) |

**Ghost reflections:**

| Option                  | Default | Description                                |
| ----------------------- | ------- | ------------------------------------------ |
| `--ghosts <n>`          | `0`     | Number of ghost reflections (0 = off)      |
| `--ghost-dispersal <f>` | `0.35`  | Spacing between ghosts along the flip axis |
| `--ghost-intensity <f>` | `0.15`  | Ghost brightness                           |
| `--ghost-chromatic <f>` | `0.01`  | Chromatic aberration spread per ghost      |

**Halo ring:**

| Option                 | Default | Description                         |
| ---------------------- | ------- | ----------------------------------- |
| `--halo-radius <f>`    | `0.45`  | Ring radius as fraction of diagonal |
| `--halo-width <f>`     | `0.07`  | Ring softness (Gaussian σ)          |
| `--halo-intensity <f>` | `0`     | Ring brightness (0 = off)           |

**Starburst diffraction:**

| Option                      | Default | Description                          |
| --------------------------- | ------- | ------------------------------------ |
| `--starburst-rays <n>`      | `0`     | Number of diffraction rays (0 = off) |
| `--starburst-intensity <f>` | `0.3`   | Ray brightness                       |
| `--starburst-length <f>`    | `0.3`   | Ray length as fraction of diagonal   |
| `--starburst-width <f>`     | `0.008` | Angular width of each ray            |

**Anamorphic streak:**

| Option                    | Default       | Description                              |
| ------------------------- | ------------- | ---------------------------------------- |
| `--streak`                | off           | Enable horizontal anamorphic streak      |
| `--streak-intensity <f>`  | `0.25`        | Streak brightness                        |
| `--streak-length <f>`     | `0.5`         | Streak length as fraction of image width |
| `--streak-tint-r/g/b <f>` | `0.6 0.7 1.0` | Streak colour (default: cool blue)       |

**Presets:**

| Option           | Description                                                               |
| ---------------- | ------------------------------------------------------------------------- |
| `--interstellar` | Full cinematic look: chromatic bloom + streak + ghosts + halo + starburst |

### Examples

**Full cinematic grade in one shot:**

```bash
bhrt-grade output.exr final.tga --interstellar
```

**Interstellar with custom sky brightness:**

```bash
bhrt-grade output.exr final.tga --interstellar --sky-brightness 0.3
```

**Bloom only (no flare):**

```bash
bhrt-grade output.exr result.tga --bloom-strength 1.5 --bloom-radius 0.08 --chromatic --exposure 3.0 --tonemap 1.0
```

**Flare only (no bloom):**

```bash
bhrt-grade output.exr result.tga --bloom-strength 0 --ghosts 6 --halo-intensity 0.15 --streak --exposure 2.0 --tonemap 1.0
```

**Bloom + streak (no ghosts/halo):**

```bash
bhrt-grade output.exr result.tga --chromatic --bloom-strength 2.0 --streak --streak-intensity 0.4 --exposure 3.0 --tonemap 1.2
```
