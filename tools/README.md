# Tools

Utility programs for testing and debugging the bhrt3 / flaresim pipeline.

## flaretest

Generates a minimal test EXR file with a single extremely bright pixel — useful for validating that flaresim's ghost tracing and bloom pipeline are working correctly without needing a full bhrt3 render.

### Building

```bash
cd tools/flaretest
mkdir -p build && cd build
cmake ..
cmake --build .
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
