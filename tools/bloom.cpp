// ============================================================================
// bhrt-bloom — EXR bloom / glow post-processing tool
//
// Reads a linear-light EXR rendered by the black-hole raytracer (or any
// EXR with an R,G,B beauty pass) and applies a configurable multi-pass
// bloom effect.  All original layers/channels are preserved in the output.
//
// Usage:
//   bhrt-bloom input.exr output.tga [options]
//
// Output format is determined by file extension:
//   .exr  — float32 EXR (preserves all layers)
//   .tga  — 8-bit TGA (tonemapped if --tonemap not set, auto-applies 1.0)
//   .hdr  — float32 Radiance HDR (beauty pass only)
//
// Options:
//   --strength  <f>   Bloom mix strength            (default: 0.8)
//   --threshold <f>   Luminance threshold for bloom  (default: 1.0)
//   --radius    <f>   Blur radius as fraction of     (default: 0.02)
//                     image diagonal
//   --passes    <n>   Number of box-blur passes      (default: 3)
//   --tonemap   <f>   Optional log tonemap before    (default: 0, off)
//                     output (compression parameter)
//   --exposure  <f>   Exposure multiplier            (default: 1.0)
//   --sky-brightness <f>  Sky brightness multiplier   (default: 1.0)
//   --octaves   <n>   Bloom scale octaves (1-6)      (default: 4)
//                     More octaves = wider atmospheric wash
//   --chromatic       Enable chromatic bloom: bloom shifts from
//                     white → yellow → orange → red as it spreads,
//                     simulating the overexposed fire-like glow
//                     of Interstellar's accretion disk.
//   --interstellar    Convenience preset that sets:
//                     strength=2.5 threshold=0.3 radius=0.06
//                     octaves=5 exposure=3.0 chromatic=on
//                     tonemap=1.2 (for TGA), individual options
//                     override preset values.
//   --help            Print this help and exit
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfArray.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// Helpers
// ============================================================================

static void print_usage(const char *prog)
{
    printf("Usage: %s input.exr output.[tga|exr|hdr] [options]\n", prog);
    printf("\nOutput format is chosen by file extension (.tga, .exr, .hdr)\n");
    printf("\nOptions:\n");
    printf("  --strength  <f>   Bloom mix strength            (default: 0.8)\n");
    printf("  --threshold <f>   Luminance threshold for bloom  (default: 1.0)\n");
    printf("  --radius    <f>   Blur radius / image diagonal   (default: 0.02)\n");
    printf("  --passes    <n>   Box-blur passes (1-10)         (default: 3)\n");
    printf("  --tonemap   <f>   Log tonemap compression (0=off)(default: 0)\n");
    printf("  --exposure  <f>   Exposure multiplier            (default: 1.0)\n");
    printf("  --sky-brightness <f>  Sky brightness multiplier   (default: 1.0)\n");
    printf("  --octaves   <n>   Bloom octaves (1=single, 4=Interstellar) (default: 4)\n");
    printf("  --chromatic       Warm chromatic bloom (white→yellow→orange→red)\n");
    printf("  --interstellar    Preset: overexposed fire-glow look\n");
    printf("  --help            Print this help\n");
}

struct BloomParams
{
    std::string input_file;
    std::string output_file;
    float strength = 0.8f;
    float threshold = 1.0f;
    float radius = 0.02f;
    int passes = 3;
    float tonemap = 0.0f;
    float exposure = 1.0f;
    float sky_brightness = 1.0f;
    int octaves = 4;
    bool chromatic = false;
};

static bool parse_args(int argc, char *argv[], BloomParams &p)
{
    if (argc < 3)
        return false;

    p.input_file = argv[1];
    p.output_file = argv[2];

    for (int i = 3; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--help"))
            return false;
        else if (!strcmp(argv[i], "--strength") && i + 1 < argc)
            p.strength = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--threshold") && i + 1 < argc)
            p.threshold = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--radius") && i + 1 < argc)
            p.radius = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--passes") && i + 1 < argc)
            p.passes = std::clamp(atoi(argv[++i]), 1, 10);
        else if (!strcmp(argv[i], "--tonemap") && i + 1 < argc)
            p.tonemap = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--exposure") && i + 1 < argc)
            p.exposure = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--sky-brightness") && i + 1 < argc)
            p.sky_brightness = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--octaves") && i + 1 < argc)
            p.octaves = std::clamp(atoi(argv[++i]), 1, 6);
        else if (!strcmp(argv[i], "--chromatic"))
            p.chromatic = true;
        else if (!strcmp(argv[i], "--interstellar"))
        {
            // Set defaults for the Interstellar look — individual flags
            // that appear AFTER --interstellar on the command line will
            // override these, and flags that appeared BEFORE are preserved
            // because we only set values that are still at their defaults.
            // To keep it simple, we just set everything; user overrides later.
            p.strength = 2.5f;
            p.threshold = 0.3f;
            p.radius = 0.06f;
            p.octaves = 5;
            p.exposure = 3.0f;
            p.chromatic = true;
            p.tonemap = 1.2f;
        }
        else
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

// ============================================================================
// EXR I/O
// ============================================================================

struct EXRImage
{
    int width = 0;
    int height = 0;
    Imf::Header header;

    // Per-channel float buffers, keyed by channel name.
    // We store ALL channels from the file so non-beauty channels
    // (disk.*, sky.*, A, etc.) are passed through unchanged.
    struct Channel
    {
        std::string name;
        std::vector<float> data;
    };
    std::vector<Channel> channels;

    // Convenience indices into channels[] for R, G, B (-1 = missing)
    int idx_R = -1, idx_G = -1, idx_B = -1;

    // Separate layer indices (from bhrt3 renderer)
    int idx_disk_R = -1, idx_disk_G = -1, idx_disk_B = -1;
    int idx_sky_R = -1, idx_sky_G = -1, idx_sky_B = -1;

    size_t num_pixels() const { return (size_t)width * height; }
};

static bool load_exr(const char *path, EXRImage &img)
{
    try
    {
        Imf::InputFile file(path);
        const Imf::Header &hdr = file.header();

        Imath::Box2i dw = hdr.dataWindow();
        img.width = dw.max.x - dw.min.x + 1;
        img.height = dw.max.y - dw.min.y + 1;
        img.header = hdr;

        const Imf::ChannelList &ch_list = hdr.channels();
        size_t np = img.num_pixels();

        // Enumerate all channels
        for (auto it = ch_list.begin(); it != ch_list.end(); ++it)
        {
            int idx = (int)img.channels.size();
            img.channels.push_back({it.name(), std::vector<float>(np, 0.0f)});

            if (it.name() == std::string("R"))
                img.idx_R = idx;
            if (it.name() == std::string("G"))
                img.idx_G = idx;
            if (it.name() == std::string("B"))
                img.idx_B = idx;
            if (it.name() == std::string("disk.R"))
                img.idx_disk_R = idx;
            if (it.name() == std::string("disk.G"))
                img.idx_disk_G = idx;
            if (it.name() == std::string("disk.B"))
                img.idx_disk_B = idx;
            if (it.name() == std::string("sky.R"))
                img.idx_sky_R = idx;
            if (it.name() == std::string("sky.G"))
                img.idx_sky_G = idx;
            if (it.name() == std::string("sky.B"))
                img.idx_sky_B = idx;
        }

        // Build framebuffer — read everything as FLOAT regardless of storage type
        Imf::FrameBuffer fb;
        for (auto &ch : img.channels)
        {
            fb.insert(ch.name.c_str(),
                      Imf::Slice(Imf::FLOAT,
                                 (char *)(ch.data.data() - (size_t)dw.min.x - (size_t)dw.min.y * img.width),
                                 sizeof(float),
                                 sizeof(float) * img.width));
        }

        file.setFrameBuffer(fb);
        file.readPixels(dw.min.y, dw.max.y);

        printf("Loaded EXR: %s  (%dx%d, %zu channels)\n",
               path, img.width, img.height, img.channels.size());
        for (auto &ch : img.channels)
            printf("  channel: %s\n", ch.name.c_str());

        return true;
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "ERROR loading EXR: %s (%s)\n", path, e.what());
        return false;
    }
}

static bool save_exr(const char *path, const EXRImage &img)
{
    try
    {
        // Build a new header preserving the original's metadata but updating
        // channel list to match our data (all FLOAT now)
        Imf::Header header(img.width, img.height);
        header.compression() = img.header.compression();

        // Copy custom attributes (e.g. comments, chromaticities)
        for (auto it = img.header.begin(); it != img.header.end(); ++it)
        {
            // Skip attributes that the Header constructor already set
            if (!strcmp(it.name(), "channels") ||
                !strcmp(it.name(), "compression") ||
                !strcmp(it.name(), "dataWindow") ||
                !strcmp(it.name(), "displayWindow") ||
                !strcmp(it.name(), "lineOrder") ||
                !strcmp(it.name(), "pixelAspectRatio") ||
                !strcmp(it.name(), "screenWindowCenter") ||
                !strcmp(it.name(), "screenWindowWidth"))
                continue;
            header.insert(it.name(), it.attribute());
        }

        for (auto &ch : img.channels)
            header.channels().insert(ch.name.c_str(), Imf::Channel(Imf::FLOAT));

        Imf::FrameBuffer fb;
        for (auto &ch : img.channels)
        {
            fb.insert(ch.name.c_str(),
                      Imf::Slice(Imf::FLOAT,
                                 (char *)ch.data.data(),
                                 sizeof(float),
                                 sizeof(float) * img.width));
        }

        Imf::OutputFile file(path, header);
        file.setFrameBuffer(fb);
        file.writePixels(img.height);

        printf("Wrote EXR: %s\n", path);
        return true;
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "ERROR writing EXR: %s (%s)\n", path, e.what());
        return false;
    }
}

// ============================================================================
// Box-blur bloom (matches the main renderer's implementation)
//
// Extended-box algorithm from "Fast Almost-Gaussian Filtering"
// (Kovesi 2010): N passes of box blur approximate a Gaussian.
// ============================================================================

static void compute_box_radii(float sigma, int num_passes, std::vector<int> &radii)
{
    radii.resize(num_passes);
    float w_ideal = std::sqrt(12.0f * sigma * sigma / num_passes + 1.0f);
    int w_lo = ((int)w_ideal) | 1;
    if (w_lo < 1)
        w_lo = 1;
    int w_hi = w_lo + 2;
    float target_var = sigma * sigma;
    float var_lo = (w_lo * w_lo - 1) / 12.0f;
    float var_hi = (w_hi * w_hi - 1) / 12.0f;
    int n_hi = (var_hi > var_lo + 1e-6f)
                   ? std::clamp((int)std::round((target_var - num_passes * var_lo) / (var_hi - var_lo)),
                                0, num_passes)
                   : 0;
    for (int i = 0; i < num_passes; ++i)
        radii[i] = ((i < n_hi) ? w_hi : w_lo) / 2;
}

static void box_blur_pass(const std::vector<float> &src_r,
                          const std::vector<float> &src_g,
                          const std::vector<float> &src_b,
                          std::vector<float> &dst_r,
                          std::vector<float> &dst_g,
                          std::vector<float> &dst_b,
                          int width, int height, int radius)
{
    const float inv_w = 1.0f / (2 * radius + 1);
    size_t np = (size_t)width * height;

    // Temp buffers for separable pass
    std::vector<float> tmp_r(np), tmp_g(np), tmp_b(np);

    // ---- Horizontal pass ----
#pragma omp parallel for
    for (int y = 0; y < height; ++y)
    {
        const int row = y * width;
        float sr = 0, sg = 0, sb = 0;
        for (int k = -radius; k <= radius; ++k)
        {
            int sx = std::clamp(k, 0, width - 1);
            sr += src_r[row + sx];
            sg += src_g[row + sx];
            sb += src_b[row + sx];
        }
        tmp_r[row] = sr * inv_w;
        tmp_g[row] = sg * inv_w;
        tmp_b[row] = sb * inv_w;

        for (int x = 1; x < width; ++x)
        {
            int add = std::min(x + radius, width - 1);
            sr += src_r[row + add];
            sg += src_g[row + add];
            sb += src_b[row + add];
            int rem = std::clamp(x - radius - 1, 0, width - 1);
            sr -= src_r[row + rem];
            sg -= src_g[row + rem];
            sb -= src_b[row + rem];
            tmp_r[row + x] = sr * inv_w;
            tmp_g[row + x] = sg * inv_w;
            tmp_b[row + x] = sb * inv_w;
        }
    }

    // ---- Vertical pass ----
#pragma omp parallel for
    for (int x = 0; x < width; ++x)
    {
        float sr = 0, sg = 0, sb = 0;
        for (int k = -radius; k <= radius; ++k)
        {
            int sy = std::clamp(k, 0, height - 1);
            sr += tmp_r[sy * width + x];
            sg += tmp_g[sy * width + x];
            sb += tmp_b[sy * width + x];
        }
        dst_r[x] = sr * inv_w;
        dst_g[x] = sg * inv_w;
        dst_b[x] = sb * inv_w;

        for (int y = 1; y < height; ++y)
        {
            int add = std::min(y + radius, height - 1);
            sr += tmp_r[add * width + x];
            sg += tmp_g[add * width + x];
            sb += tmp_b[add * width + x];
            int rem = std::clamp(y - radius - 1, 0, height - 1);
            sr -= tmp_r[rem * width + x];
            sg -= tmp_g[rem * width + x];
            sb -= tmp_b[rem * width + x];
            dst_r[y * width + x] = sr * inv_w;
            dst_g[y * width + x] = sg * inv_w;
            dst_b[y * width + x] = sb * inv_w;
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char *argv[])
{
    BloomParams params;
    if (!parse_args(argc, argv, params))
    {
        print_usage(argv[0]);
        return 1;
    }

    printf("bhrt-bloom — EXR post-processor\n");
    printf("  input:     %s\n", params.input_file.c_str());
    printf("  output:    %s\n", params.output_file.c_str());
    printf("  strength:  %.3f\n", params.strength);
    printf("  threshold: %.3f\n", params.threshold);
    printf("  radius:    %.4f\n", params.radius);
    printf("  passes:    %d\n", params.passes);
    printf("  tonemap:   %.3f%s\n", params.tonemap, params.tonemap < 1e-6f ? " (off)" : "");
    printf("  exposure:  %.3f\n", params.exposure);
    printf("  sky_brt:   %.3f\n", params.sky_brightness);
    printf("  octaves:   %d\n", params.octaves);
    printf("  chromatic: %s\n", params.chromatic ? "ON (warm shift)" : "off");

    // --- Load EXR --------------------------------------------------------
    EXRImage img;
    if (!load_exr(params.input_file.c_str(), img))
        return 1;

    if (img.idx_R < 0 || img.idx_G < 0 || img.idx_B < 0)
    {
        fprintf(stderr, "ERROR: input EXR must have R, G, B channels\n");
        return 1;
    }

    float *beauty_r = img.channels[img.idx_R].data.data();
    float *beauty_g = img.channels[img.idx_G].data.data();
    float *beauty_b = img.channels[img.idx_B].data.data();
    size_t np = img.num_pixels();

    auto t0 = std::chrono::steady_clock::now();

    // --- Apply sky brightness --------------------------------------------
    // The EXR stores disk.* and sky.* as separate layers.  The beauty R,G,B
    // was written as  disk + sky * original_brightness.  To re-grade the sky
    // we recomposite:  beauty = disk + sky * new_brightness.
    // The sky.* channels in the output are also scaled accordingly.
    bool has_layers = (img.idx_disk_R >= 0 && img.idx_disk_G >= 0 && img.idx_disk_B >= 0 &&
                       img.idx_sky_R >= 0 && img.idx_sky_G >= 0 && img.idx_sky_B >= 0);

    if (std::abs(params.sky_brightness - 1.0f) > 1e-6f)
    {
        if (!has_layers)
        {
            fprintf(stderr, "WARNING: --sky-brightness requires disk.* and sky.* layers in the EXR.\n"
                            "         These layers are not present — sky brightness will not be applied.\n");
        }
        else
        {
            printf("Applying sky brightness: %.3f\n", params.sky_brightness);
            float *disk_r = img.channels[img.idx_disk_R].data.data();
            float *disk_g = img.channels[img.idx_disk_G].data.data();
            float *disk_b = img.channels[img.idx_disk_B].data.data();
            float *sky_r = img.channels[img.idx_sky_R].data.data();
            float *sky_g = img.channels[img.idx_sky_G].data.data();
            float *sky_b = img.channels[img.idx_sky_B].data.data();

#pragma omp parallel for
            for (int i = 0; i < (int)np; ++i)
            {
                // Recomposite beauty from layers
                beauty_r[i] = disk_r[i] + sky_r[i] * params.sky_brightness;
                beauty_g[i] = disk_g[i] + sky_g[i] * params.sky_brightness;
                beauty_b[i] = disk_b[i] + sky_b[i] * params.sky_brightness;

                // Update sky channels to reflect new brightness
                sky_r[i] *= params.sky_brightness;
                sky_g[i] *= params.sky_brightness;
                sky_b[i] *= params.sky_brightness;
            }
        }
    }

    // --- Apply exposure --------------------------------------------------
    if (std::abs(params.exposure - 1.0f) > 1e-6f)
    {
        printf("Applying exposure: %.3f\n", params.exposure);
#pragma omp parallel for
        for (int i = 0; i < (int)np; ++i)
        {
            beauty_r[i] *= params.exposure;
            beauty_g[i] *= params.exposure;
            beauty_b[i] *= params.exposure;
        }
    }

    // --- Multi-octave bloom -----------------------------------------------
    // Each octave blurs at a progressively wider radius (2.5× per octave) and
    // contributes with decreasing weight, capped at 25% of the image diagonal
    // to keep the glow bright but spatially contained.
    //   octave 0: radius × 1      weight 1.0   (tight halo)
    //   octave 1: radius × 2.5    weight 0.6   (medium spread)
    //   octave 2: radius × 6.25   weight 0.36  (wide glow)
    //   octave 3: radius × 15.6   weight 0.22  (atmospheric)
    //   ...
    if (params.strength > 1e-6f)
    {
        const float diag = std::sqrt((float)(img.width * img.width + img.height * img.height));
        const int base_kernel = std::max((int)(params.radius * diag), 1);

        printf("Bloom: %d octaves, base radius=%d, strength=%.2f\n",
               params.octaves, base_kernel, params.strength);

        // Extract bright pixels (threshold) — shared across all octaves
        std::vector<float> bright_r(np, 0), bright_g(np, 0), bright_b(np, 0);
#pragma omp parallel for
        for (int i = 0; i < (int)np; ++i)
        {
            float lum = 0.2126f * beauty_r[i] + 0.7152f * beauty_g[i] + 0.0722f * beauty_b[i];
            if (lum > params.threshold)
            {
                float scale = (lum - params.threshold) / std::max(lum, 1e-12f);
                bright_r[i] = beauty_r[i] * scale;
                bright_g[i] = beauty_g[i] * scale;
                bright_b[i] = beauty_b[i] * scale;
            }
        }

        // Accumulate bloom from all octaves
        std::vector<float> accum_r(np, 0), accum_g(np, 0), accum_b(np, 0);

        float octave_weight = 1.0f;
        const float weight_decay = 0.6f; // each octave contributes 60% of the previous

        // Chromatic bloom: per-octave RGB tint that shifts from white-hot
        // centre → yellow → orange → deep red at the widest scales,
        // simulating the look of an overexposed incandescent source.
        //    octave 0: (1.00, 1.00, 1.00) — white core
        //    octave 1: (1.00, 0.85, 0.50) — warm yellow
        //    octave 2: (1.00, 0.60, 0.20) — orange
        //    octave 3: (1.00, 0.35, 0.08) — deep orange
        //    octave 4: (0.90, 0.18, 0.03) — red
        //    octave 5: (0.70, 0.08, 0.01) — deep red
        static const float chroma_r[] = {1.00f, 1.00f, 1.00f, 1.00f, 0.90f, 0.70f};
        static const float chroma_g[] = {1.00f, 0.85f, 0.60f, 0.35f, 0.18f, 0.08f};
        static const float chroma_b[] = {1.00f, 0.50f, 0.20f, 0.08f, 0.03f, 0.01f};

        for (int oct = 0; oct < params.octaves; ++oct)
        {
            // Each octave: radius scales by 2.5^oct
            float oct_radius = (float)base_kernel;
            for (int k = 0; k < oct; ++k)
                oct_radius *= 2.5f;
            // Cap kernel at 25% of image diagonal — glow, not diffusion
            int oct_kernel = std::min((int)oct_radius, (int)(diag * 0.25f));

            float oct_sigma = oct_kernel / 3.0f;
            std::vector<int> box_radii;
            compute_box_radii(oct_sigma, params.passes, box_radii);

            // Per-octave colour tint (only active when --chromatic is on)
            int ci = std::min(oct, 5);
            float tint_r = params.chromatic ? chroma_r[ci] : 1.0f;
            float tint_g = params.chromatic ? chroma_g[ci] : 1.0f;
            float tint_b = params.chromatic ? chroma_b[ci] : 1.0f;

            printf("  octave %d/%d: radius=%d, sigma=%.1f, weight=%.3f",
                   oct + 1, params.octaves, oct_kernel, oct_sigma, octave_weight);
            if (params.chromatic)
                printf(", tint=(%.2f, %.2f, %.2f)", tint_r, tint_g, tint_b);
            printf(", box=[");
            for (int i = 0; i < params.passes; ++i)
                printf("%s%d", i ? ", " : "", box_radii[i]);
            printf("]\n");

            // Copy bright pixels for this octave's blur
            std::vector<float> bloom_r(bright_r), bloom_g(bright_g), bloom_b(bright_b);
            std::vector<float> tmp_r(np), tmp_g(np), tmp_b(np);

            for (int pass = 0; pass < params.passes; ++pass)
            {
                box_blur_pass(bloom_r, bloom_g, bloom_b,
                              tmp_r, tmp_g, tmp_b,
                              img.width, img.height, box_radii[pass]);
                std::swap(bloom_r, tmp_r);
                std::swap(bloom_g, tmp_g);
                std::swap(bloom_b, tmp_b);
            }

            // Accumulate this octave with per-channel tint
            const float wr = octave_weight * tint_r;
            const float wg = octave_weight * tint_g;
            const float wb = octave_weight * tint_b;
#pragma omp parallel for
            for (int i = 0; i < (int)np; ++i)
            {
                accum_r[i] += wr * bloom_r[i];
                accum_g[i] += wg * bloom_g[i];
                accum_b[i] += wb * bloom_b[i];
            }

            octave_weight *= weight_decay;
        }

        // Composite accumulated bloom onto beauty
#pragma omp parallel for
        for (int i = 0; i < (int)np; ++i)
        {
            beauty_r[i] += params.strength * accum_r[i];
            beauty_g[i] += params.strength * accum_g[i];
            beauty_b[i] += params.strength * accum_b[i];
        }

        printf("Bloom applied (%d octaves).\n", params.octaves);
    }

    // --- Optional tonemap ------------------------------------------------
    if (params.tonemap > 1e-6f)
    {
        const float c = std::pow(10.0f, params.tonemap * 2.0f) - 1.0f;
        const float norm = 1.0f / std::log(1.0f + c);
        printf("Applying tonemap: compression=%.2f  (c=%.2f)\n", params.tonemap, c);

#pragma omp parallel for
        for (int i = 0; i < (int)np; ++i)
        {
            beauty_r[i] = std::log(1.0f + c * std::max(beauty_r[i], 0.0f)) * norm;
            beauty_g[i] = std::log(1.0f + c * std::max(beauty_g[i], 0.0f)) * norm;
            beauty_b[i] = std::log(1.0f + c * std::max(beauty_b[i], 0.0f)) * norm;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    printf("Processing time: %.2f seconds\n", elapsed);

    // --- Write output ----------------------------------------------------
    auto ends_with = [](const std::string &s, const char *suffix) -> bool
    {
        size_t len = strlen(suffix);
        return s.size() >= len && s.compare(s.size() - len, len, suffix) == 0;
    };

    const std::string &out = params.output_file;

    if (ends_with(out, ".tga"))
    {
        // TGA: 8-bit LDR — auto-apply tonemap if user didn't specify one
        if (params.tonemap < 1e-6f)
        {
            const float c = std::pow(10.0f, 1.0f * 2.0f) - 1.0f; // default compression=1.0
            const float norm = 1.0f / std::log(1.0f + c);
            printf("Auto-applying tonemap for TGA (compression=1.0, c=%.2f)\n", c);
#pragma omp parallel for
            for (int i = 0; i < (int)np; ++i)
            {
                beauty_r[i] = std::log(1.0f + c * std::max(beauty_r[i], 0.0f)) * norm;
                beauty_g[i] = std::log(1.0f + c * std::max(beauty_g[i], 0.0f)) * norm;
                beauty_b[i] = std::log(1.0f + c * std::max(beauty_b[i], 0.0f)) * norm;
            }
        }

        // Convert to 8-bit
        std::vector<unsigned char> pixels(np * 3);
#pragma omp parallel for
        for (int i = 0; i < (int)np; ++i)
        {
            pixels[i * 3 + 0] = (unsigned char)std::clamp((int)(beauty_r[i] * 255.0f + 0.5f), 0, 255);
            pixels[i * 3 + 1] = (unsigned char)std::clamp((int)(beauty_g[i] * 255.0f + 0.5f), 0, 255);
            pixels[i * 3 + 2] = (unsigned char)std::clamp((int)(beauty_b[i] * 255.0f + 0.5f), 0, 255);
        }

        if (stbi_write_tga(out.c_str(), img.width, img.height, 3, pixels.data()))
            printf("Wrote TGA: %s\n", out.c_str());
        else
        {
            fprintf(stderr, "ERROR: failed to write TGA: %s\n", out.c_str());
            return 1;
        }
    }
    else if (ends_with(out, ".hdr"))
    {
        // Radiance HDR: interleaved float RGB
        std::vector<float> hdr_pixels(np * 3);
#pragma omp parallel for
        for (int i = 0; i < (int)np; ++i)
        {
            hdr_pixels[i * 3 + 0] = beauty_r[i];
            hdr_pixels[i * 3 + 1] = beauty_g[i];
            hdr_pixels[i * 3 + 2] = beauty_b[i];
        }
        if (stbi_write_hdr(out.c_str(), img.width, img.height, 3, hdr_pixels.data()))
            printf("Wrote HDR: %s\n", out.c_str());
        else
        {
            fprintf(stderr, "ERROR: failed to write HDR: %s\n", out.c_str());
            return 1;
        }
    }
    else
    {
        // Default: EXR (preserves all layers)
        if (!save_exr(out.c_str(), img))
            return 1;
    }

    return 0;
}
