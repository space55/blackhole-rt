// filmgrain — applies photographic film grain to a TGA image
//
// Usage:  filmgrain [options] input.tga [output.tga]
//
//   -s <strength>   grain strength 0.0–1.0  (default 0.15)
//   -g <size>       grain size in pixels     (default 1.0, >1 = coarser)
//   -seed <n>       RNG seed                 (default 42)
//   -mono           monochromatic grain      (default: per-channel)
//
// The grain model is based on Gaussian noise shaped by a highlight roll-off
// curve so dark areas get more visible grain (like real silver-halide film).
// Grain size > 1 uses a simple box-filtered jittered grid to emulate larger
// grain clumps without a full FFT convolution.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

// stb_image for loading, stb_image_write for saving
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ---------------------------------------------------------------------------
// Simple xoshiro128** PRNG (fast, good quality, seedable)
// ---------------------------------------------------------------------------
struct Rng
{
    uint32_t s[4];

    void seed(uint64_t v)
    {
        // SplitMix64 to initialise state from a single seed
        for (int i = 0; i < 4; i++)
        {
            v += 0x9e3779b97f4a7c15ULL;
            uint64_t z = v;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            z = z ^ (z >> 31);
            s[i] = (uint32_t)(z & 0xFFFFFFFF);
        }
    }

    uint32_t rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }

    uint32_t next()
    {
        const uint32_t result = rotl(s[1] * 5, 7) * 9;
        const uint32_t t = s[1] << 9;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 11);
        return result;
    }

    // Uniform [0, 1)
    float uniform() { return (next() >> 8) * (1.0f / 16777216.0f); }

    // Approximate Gaussian via Box-Muller
    float gaussian()
    {
        float u1 = std::max(uniform(), 1e-10f);
        float u2 = uniform();
        return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307f * u2);
    }
};

// ---------------------------------------------------------------------------
// Film grain response curve: more grain in shadows/midtones, less in
// highlights (like real film where dense silver areas show less grain).
// ---------------------------------------------------------------------------
static inline float grain_response(float luminance)
{
    // Bell-shaped curve peaked at ~0.35, tapering to 0 at black and white.
    // This mimics the characteristic curve of most film stocks.
    float x = std::clamp(luminance, 0.0f, 1.0f);
    // Lifted shadows: even deep blacks get some grain
    return 0.3f + 0.7f * (4.0f * x * (1.0f - x));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    // Defaults
    float strength = 0.15f;
    float grain_size = 1.0f;
    uint64_t rng_seed = 42;
    bool mono = false;
    const char *input_path = nullptr;
    const char *output_path = nullptr;

    // Parse arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc)
            strength = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc)
            grain_size = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc)
            rng_seed = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "-mono") == 0)
            mono = true;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            printf("Usage: filmgrain [options] input.tga [output.tga]\n");
            printf("  -s <strength>   grain strength 0.0-1.0  (default 0.15)\n");
            printf("  -g <size>       grain size in pixels     (default 1.0)\n");
            printf("  -seed <n>       RNG seed                 (default 42)\n");
            printf("  -mono           monochromatic grain\n");
            return 0;
        }
        else if (!input_path)
            input_path = argv[i];
        else if (!output_path)
            output_path = argv[i];
    }

    if (!input_path)
    {
        fprintf(stderr, "Error: no input file specified. Use -h for help.\n");
        return 1;
    }

    // Default output: input_grain.tga
    char default_out[1024];
    if (!output_path)
    {
        // Strip extension, append _grain.tga
        const char *dot = strrchr(input_path, '.');
        size_t base_len = dot ? (size_t)(dot - input_path) : strlen(input_path);
        snprintf(default_out, sizeof(default_out), "%.*s_grain.tga",
                 (int)base_len, input_path);
        output_path = default_out;
    }

    // Load image
    int width, height, channels;
    unsigned char *pixels = stbi_load(input_path, &width, &height, &channels, 3);
    if (!pixels)
    {
        fprintf(stderr, "Error: could not load '%s'\n", input_path);
        return 1;
    }
    printf("Loaded: %s (%dx%d, %d channels)\n", input_path, width, height, channels);
    printf("Grain:  strength=%.3f  size=%.1f  seed=%llu  %s\n",
           strength, grain_size, (unsigned long long)rng_seed,
           mono ? "monochromatic" : "per-channel");

    // Allocate grain field.  When grain_size > 1, we generate a coarser
    // grid and replicate values to simulate larger clumps.
    const int gw = std::max(1, (int)ceilf(width / grain_size));
    const int gh = std::max(1, (int)ceilf(height / grain_size));
    const int grain_channels = mono ? 1 : 3;

    std::vector<float> grain_field((size_t)gw * gh * grain_channels);

    Rng rng;
    rng.seed(rng_seed);
    for (size_t i = 0; i < grain_field.size(); i++)
        grain_field[i] = rng.gaussian();

    // Apply grain to each pixel
    const float inv_255 = 1.0f / 255.0f;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            unsigned char *px = pixels + (y * width + x) * 3;

            // Luminance for response curve (Rec. 709)
            float lum = (0.2126f * px[0] + 0.7152f * px[1] + 0.0722f * px[2]) * inv_255;
            float response = grain_response(lum) * strength;

            // Look up grain from the (possibly coarser) grid
            int gx = std::min((int)(x / grain_size), gw - 1);
            int gy = std::min((int)(y / grain_size), gh - 1);
            int gi = gy * gw + gx;

            for (int c = 0; c < 3; c++)
            {
                float v = px[c] * inv_255;
                float g = grain_field[gi * grain_channels + (mono ? 0 : c)];
                v += g * response;
                px[c] = (unsigned char)std::clamp((int)(v * 255.0f + 0.5f), 0, 255);
            }
        }
    }

    // Write output
    int ok = stbi_write_tga(output_path, width, height, 3, pixels);
    stbi_image_free(pixels);

    if (ok)
        printf("Wrote: %s\n", output_path);
    else
    {
        fprintf(stderr, "Error: failed to write '%s'\n", output_path);
        return 1;
    }

    return 0;
}
