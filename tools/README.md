# Tools

Utility programs for testing and debugging the bhrt3 / flaresim pipeline.

## flaretest

Generates a minimal test EXR file with a single extremely bright pixel — useful for validating that flaresim's ghost tracing and bloom pipeline are working correctly without needing a full bhrt3 render.

### Building

```bash
cd tools/flaretest
mkdir -p build && cd build
cmake .. && cmake --build .
```

### Usage

```bash
./flaretest [output.exr]
```

Writes a 512×256 black EXR with a single pixel at (384, 128) set to luminance 500.0. If no output path is given, defaults to `flaretest.exr`.

### Example: test flaresim with it

```bash
# Generate test pattern
cd tools/flaretest/build
./flaretest test.exr

# Run flaresim on the test pattern
cd ../../../flaresim/build
./flaresim --input ../../tools/flaretest/build/test.exr \
           --output flare_test_output.exr \
           --lens ../lenses/doublegauss.lens \
           --fov 60
```

This produces an EXR with ghost reflections from a single point source — ideal for inspecting individual ghost shapes and chromatic structure.

## filmgrain

Applies photographic film grain to a TGA (or any stb-supported) image. The grain model is based on a highlight roll-off response curve so dark/mid areas get more visible grain, like real silver-halide film stock.

### Building

```bash
cd tools/filmgrain
mkdir -p build && cd build
cmake .. && cmake --build .
```

### Usage

```bash
./filmgrain [options] input.tga [output.tga]
```

| Option       | Default         | Description                                |
| ------------ | --------------- | ------------------------------------------ |
| `-s <val>`   | `0.15`          | Grain strength (0.0–1.0)                   |
| `-g <val>`   | `1.0`           | Grain size in pixels (>1 = coarser clumps) |
| `-seed <n>`  | `42`            | RNG seed (deterministic for animation)     |
| `-mono`      | _(off)_         | Monochromatic grain (default: per-channel) |

If no output path is given, defaults to `input_grain.tga`.

### Examples

```bash
# Subtle grain
./filmgrain -s 0.10 ../../build/output.tga

# Heavy, coarse, monochromatic grain (high-ISO look)
./filmgrain -s 0.30 -g 2.0 -mono ../../build/output.tga output_grainy.tga

# Consistent grain across animation frames (same seed per frame for temporal stability)
./filmgrain -s 0.12 -seed 100 frame_0001.tga frame_0001_grain.tga
```
