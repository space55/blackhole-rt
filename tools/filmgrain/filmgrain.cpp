// filmgrain — applies photographic film grain to a TGA image
//
// Usage:  filmgrain [options] input.tga [output.tga]
//
//   -s <strength>   grain strength 0.0–1.0  (default 0.06)
//   -g <size>       grain clump size         (default 1.8, mimics 35mm scan)
//   -chroma <f>     chroma noise fraction    (default 0.12, of luma strength)
//   -seed <n>       RNG seed                 (default 42)
//
// Models real silver-halide film grain:
//  • Primarily luminance noise (silver crystal density variation)
//  • Subtle, largely-correlated chroma offset (separate emulsion layers)
//  • Spatially clustered (grain clumps, not pixel-level white noise)
//  • Response shaped by exposure: minimal in deep shadows (unexposed
//    silver), peaks in mid-tones, gentle rolloff in highlights
//  • Slightly warm bias in shadows, neutral in highlights

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
// Film grain response curve.
//
// Models the relationship between exposure and grain visibility in real film:
//  • Deep shadows (unexposed silver): very little grain — the emulsion
//    simply wasn't activated, so there's no crystal structure to see.
//  • Mid-tones (~0.3–0.5): peak grain — partially exposed crystals have
//    maximum density variation.
//  • Highlights (dense silver): grain decreases — heavy exposure saturates
//    the emulsion, averaging out crystal variation.
// ---------------------------------------------------------------------------
static inline float grain_response(float luminance)
{
    float x = std::clamp(luminance, 0.0f, 1.0f);
    // Ramp from near-zero in blacks, peak at ~0.35, gentle rolloff to highlights
    float shadow_ramp = std::clamp(x / 0.12f, 0.0f, 1.0f);                  // kills deep blacks
    float body = 4.0f * x * (1.0f - x);                                     // bell, peaks ~0.5
    float shift = std::max(0.0f, 1.0f - (x - 0.35f) * (x - 0.35f) * 12.0f); // sharper peak at 0.35
    return shadow_ramp * (0.5f * body + 0.5f * shift);
}

// ---------------------------------------------------------------------------
// 3×3 box blur on a float field (in-place via temp buffer).
// Wraps at edges for seamless tiling.
// ---------------------------------------------------------------------------
static void blur_field(std::vector<float> &field, int w, int h, int passes)
{
    std::vector<float> tmp(field.size());
    for (int p = 0; p < passes; p++)
    {
        for (int y = 0; y < h; y++)
        {
            int ym = (y > 0) ? y - 1 : h - 1;
            int yp = (y < h - 1) ? y + 1 : 0;
            for (int x = 0; x < w; x++)
            {
                int xm = (x > 0) ? x - 1 : w - 1;
                int xp = (x < w - 1) ? x + 1 : 0;
                // Weighted: centre 4, edges 2, corners 1 → /16
                tmp[y * w + x] =
                    (4.0f * field[y * w + x] +
                     2.0f * (field[y * w + xm] + field[y * w + xp] +
                             field[ym * w + x] + field[yp * w + x]) +
                     1.0f * (field[ym * w + xm] + field[ym * w + xp] +
                             field[yp * w + xm] + field[yp * w + xp])) /
                    16.0f;
            }
        }
        field.swap(tmp);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    // Defaults — tuned for subtle, film-like grain
    float strength = 0.06f;
    float grain_size = 1.8f;
    float chroma_frac = 0.12f; // chroma noise as fraction of luma noise
    uint64_t rng_seed = 42;
    const char *input_path = nullptr;
    const char *output_path = nullptr;

    // Parse arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc)
            strength = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc)
            grain_size = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "-chroma") == 0 && i + 1 < argc)
            chroma_frac = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc)
            rng_seed = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            printf("Usage: filmgrain [options] input.tga [output.tga]\n");
            printf("  -s <strength>   grain strength 0.0-1.0  (default 0.06)\n");
            printf("  -g <size>       grain clump size         (default 1.8)\n");
            printf("  -chroma <f>     chroma noise fraction    (default 0.12)\n");
            printf("  -seed <n>       RNG seed                 (default 42)\n");
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
    printf("Grain:  strength=%.3f  size=%.1f  chroma=%.2f  seed=%llu\n",
           strength, grain_size, chroma_frac, (unsigned long long)rng_seed);

    // Generate grain at the clump grid resolution.
    // 4 channels per cell: luminance, Cr, Cb, warmth-bias
    const int gw = std::max(1, (int)ceilf(width / grain_size));
    const int gh = std::max(1, (int)ceilf(height / grain_size));

    // Separate luminance and two chroma noise planes
    std::vector<float> grain_luma((size_t)gw * gh);
    std::vector<float> grain_cr((size_t)gw * gh); // red-cyan axis
    std::vector<float> grain_cb((size_t)gw * gh); // blue-yellow axis

    Rng rng;
    rng.seed(rng_seed);
    for (size_t i = 0; i < grain_luma.size(); i++)
        grain_luma[i] = rng.gaussian();

    // Chroma noise from a different seed region (offset the RNG state).
    // This ensures luma and chroma are uncorrelated, modelling the
    // physically separate emulsion layers in colour negative film.
    rng.seed(rng_seed + 0x1234ABCD);
    for (size_t i = 0; i < grain_cr.size(); i++)
        grain_cr[i] = rng.gaussian();
    rng.seed(rng_seed + 0x5678EF01);
    for (size_t i = 0; i < grain_cb.size(); i++)
        grain_cb[i] = rng.gaussian();

    // Spatially blur the noise fields to create clustered grain structure.
    // Real film grain is not pixel-level white noise — silver halide crystals
    // clump together, creating soft-edged ~2-4px structures at typical scan
    // resolutions.  One blur pass at the clump grid scale is sufficient.
    blur_field(grain_luma, gw, gh, 1);
    // Chroma noise gets extra blur (colour variation is always lower-frequency
    // than luminance variation in real film, due to dye cloud spreading).
    blur_field(grain_cr, gw, gh, 2);
    blur_field(grain_cb, gw, gh, 2);

    // Renormalize after blur so strength parameter is consistent
    auto renorm = [](std::vector<float> &f)
    {
        double sum2 = 0;
        for (float v : f)
            sum2 += (double)v * v;
        float rms = sqrtf((float)(sum2 / f.size()));
        if (rms > 1e-6f)
            for (float &v : f)
                v /= rms;
    };
    renorm(grain_luma);
    renorm(grain_cr);
    renorm(grain_cb);

    // Apply grain to each pixel
    const float inv_255 = 1.0f / 255.0f;
    const float chroma_str = strength * chroma_frac;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            unsigned char *px = pixels + (y * width + x) * 3;

            // Linearize (approximate sRGB → linear)
            float r = px[0] * inv_255;
            float g = px[1] * inv_255;
            float b = px[2] * inv_255;

            // Luminance for response curve (Rec. 709)
            float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            float response = grain_response(lum);

            // Look up grain from the clump grid (nearest neighbour)
            int gx = std::min((int)(x / grain_size), gw - 1);
            int gy = std::min((int)(y / grain_size), gh - 1);
            int gi = gy * gw + gx;

            float luma_g = grain_luma[gi] * strength * response;
            float cr_g = grain_cr[gi] * chroma_str * response;
            float cb_g = grain_cb[gi] * chroma_str * response;

            // Subtle warm bias in shadows: grain shifts slightly toward
            // amber in dark areas (models the orange mask of colour neg film)
            float warm = 0.012f * strength * std::max(0.0f, 1.0f - lum * 3.0f);

            // Apply: luma noise to all channels equally, plus subtle
            // chroma offsets on R and B (Cr/Cb decomposition)
            r += luma_g + cr_g + warm;
            g += luma_g;
            b += luma_g + cb_g;

            px[0] = (unsigned char)std::clamp((int)(r * 255.0f + 0.5f), 0, 255);
            px[1] = (unsigned char)std::clamp((int)(g * 255.0f + 0.5f), 0, 255);
            px[2] = (unsigned char)std::clamp((int)(b * 255.0f + 0.5f), 0, 255);
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
