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

Applies photographic silver-halide film grain to a TGA (or any stb-supported) image. The grain is primarily luminance noise (crystal density variation) with a subtle, independently-generated chroma offset modelling the separate emulsion layers of colour negative stock. Grain is spatially clustered and blurred to mimic real grain clumps at typical 35mm scan resolution. An exposure-dependent response curve shapes visibility: minimal in deep shadows (unexposed silver), peaks in mid-tones (~0.35), gentle rolloff in highlights.

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

| Option        | Default | Description                                          |
| ------------- | ------- | ---------------------------------------------------- |
| `-s <val>`    | `0.06`  | Grain strength (0.0–1.0)                             |
| `-g <val>`    | `1.8`   | Grain clump size (>1 = coarser, mimics 35mm scan)    |
| `-chroma <f>` | `0.12`  | Chroma noise as fraction of luma strength (0 = mono) |
| `-seed <n>`   | `42`    | RNG seed (deterministic for animation)               |

If no output path is given, defaults to `input_grain.tga`.

### Examples

```bash
# Subtle, film-like grain (default settings)
./filmgrain ../../build/output.tga

# Stronger grain with no chroma noise (pure luminance, like B&W stock)
./filmgrain -s 0.12 -chroma 0 ../../build/output.tga output_grainy.tga

# Heavy, coarse grain (pushed high-ISO look)
./filmgrain -s 0.20 -g 3.0 ../../build/output.tga output_grainy.tga

# Consistent grain across animation frames (same seed per frame for temporal stability)
./filmgrain -s 0.06 -seed 100 frame_0001.tga frame_0001_grain.tga
```
