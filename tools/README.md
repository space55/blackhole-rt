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

## bhrt-bloom

Reads a linear-light EXR file and applies configurable bloom, exposure, tonemapping, and sky brightness adjustments. Supports outputting to multiple formats.

### Usage

```bash
bhrt-bloom input.exr output.[tga|exr|hdr] [options]
```

The output format is determined by file extension:

| Extension | Format               | Notes                                      |
| --------- | -------------------- | ------------------------------------------ |
| `.tga`    | 8-bit TGA            | Auto-tonemaps if `--tonemap` not specified |
| `.exr`    | Float32 EXR          | Preserves all layers from input            |
| `.hdr`    | Float32 Radiance HDR | Beauty pass (R,G,B) only                   |

### Options

| Option                 | Default   | Description                                                                                                                                                  |
| ---------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--strength <f>`       | `0.8`     | Bloom mix intensity                                                                                                                                          |
| `--threshold <f>`      | `1.0`     | Luminance cutoff — pixels brighter than this contribute to bloom                                                                                             |
| `--radius <f>`         | `0.02`    | Blur radius as fraction of image diagonal (0.01 = tight, 0.1 = wide)                                                                                         |
| `--passes <n>`         | `3`       | Number of box-blur passes (1–10, more = smoother Gaussian)                                                                                                   |
| `--tonemap <f>`        | `0` (off) | Log tonemap compression (same formula as renderer: 0 = off, 1 = heavy)                                                                                       |
| `--exposure <f>`       | `1.0`     | Linear exposure multiplier applied before bloom                                                                                                              |
| `--sky-brightness <f>` | `1.0`     | Sky brightness multiplier (requires `disk.*` and `sky.*` EXR layers)                                                                                         |
| `--octaves <n>`        | `4`       | Bloom scale octaves (1–6). More = wider atmospheric wash                                                                                                     |
| `--chromatic`          | off       | Chromatic bloom: shifts warm (white→yellow→orange→red) at wider scales                                                                                       |
| `--interstellar`       |           | Preset: overexposed fire-glow look (sets strength=2.5 threshold=0.3 radius=0.06 octaves=5 exposure=3.0 chromatic=on tonemap=1.2). Individual flags override. |
| `--help`               |           | Print usage and exit                                                                                                                                         |

### Processing Pipeline

Operations are applied in this order:

1. **Sky brightness** — Recomposites the beauty pass from disk + sky layers: `beauty = disk + sky × sky_brightness`. Requires `disk.R/G/B` and `sky.R/G/B` layers in the input EXR (written by bhrt3).
2. **Exposure** — Multiplies beauty RGB by the exposure value.
3. **Bloom** — Extracts pixels above threshold, applies multi-pass box blur (Gaussian approximation), composites back at the given strength.
4. **Tonemap** — Applies logarithmic compression: `log(1 + cx) / log(1 + c)` where `c = 10^(2 × compression) - 1`.
5. **Output** — Writes the result in the format determined by file extension. TGA auto-tonemaps with compression=1.0 if no explicit `--tonemap` was given.

### Examples

**Basic bloom to TGA:**

```bash
bhrt-bloom output.exr result.tga --strength 0.8 --threshold 1.0
```

**Heavy cinematic bloom with wide radius:**

```bash
bhrt-bloom output.exr result.tga --strength 1.5 --radius 0.1 --threshold 0.5
```

**Adjust sky brightness and export tonemapped TGA:**

```bash
bhrt-bloom output.exr result.tga --sky-brightness 0.3 --tonemap 1.0
```

**Re-grade to a new EXR (preserves all layers):**

```bash
bhrt-bloom output.exr graded.exr --exposure 2.0 --sky-brightness 0.5 --strength 0.0
```

**Bloom-only pass (no tonemap) to HDR:**

```bash
bhrt-bloom output.exr bloomed.hdr --strength 1.0 --threshold 0.8
```

**Interstellar preset (one-shot cinematic fire-glow):**

```bash
bhrt-bloom output.exr interstellar.tga --interstellar
```

**Interstellar preset with custom sky brightness:**

```bash
bhrt-bloom output.exr interstellar.tga --interstellar --sky-brightness 0.3
```

**Chromatic bloom with manual tuning:**

```bash
bhrt-bloom output.exr result.tga --chromatic --strength 2.0 --exposure 4.0 --radius 0.06 --octaves 5 --threshold 0.5
```

### EXR Layer Handling

When the input EXR contains separate layers written by bhrt3:

| Layer         | Usage                                                                |
| ------------- | -------------------------------------------------------------------- |
| `R`, `G`, `B` | Beauty pass — modified by all operations                             |
| `A`           | Alpha — passed through unchanged                                     |
| `disk.R/G/B`  | Raw disk emission — used to recomposite with `--sky-brightness`      |
| `disk.A`      | Disk opacity — passed through unchanged                              |
| `sky.R/G/B`   | Sky contribution — scaled by `--sky-brightness`, then passed through |

If `--sky-brightness` is not 1.0, the beauty pass is recomposited from the disk and sky layers. The `sky.*` channels in the output are also updated to reflect the new brightness.

When outputting to EXR, **all channels** from the input are preserved in the output, including any layers not listed above.

If the input EXR doesn't have `disk.*`/`sky.*` layers (e.g. from a different renderer), `--sky-brightness` is skipped with a warning.
