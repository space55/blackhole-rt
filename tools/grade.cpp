// ============================================================================
// bhrt-grade — unified EXR post-processor (bloom + lens flare + grading)
//
// Reads a linear-light EXR rendered by the BHRT3 black hole raytracer and
// applies bloom AND lens flare in a single pass, so both effects share the
// same clean bright-pixel extraction from the raw source data.  This avoids
// the texture-blurring problem that arises when chaining separate tools
// (bloom softens the image, then flare samples from the softened result).
//
// Processing pipeline:
//   1. Sky brightness (recomposite from disk.*/sky.* layers)
//   2. Exposure
//   3. Bright pixel extraction (shared by bloom + flare)
//   4. Bloom (multi-octave, optional chromatic warm shift)
//   5. Lens flare (ghosts, halo, starburst, anamorphic streak)
//   6. Composite bloom + flare onto original beauty pass
//   7. Tonemap
//   8. Output
//
// Usage:
//   bhrt-grade input.exr output.[tga|exr|hdr] [options]
//
// The --interstellar preset activates bloom + flare with cinematic defaults.
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Parameters
// ============================================================================

struct GradeParams
{
    std::string input_file;
    std::string output_file;

    // General
    float threshold = 1.0f;
    float exposure = 1.0f;
    float tonemap = 0.0f;
    float sky_brightness = 1.0f;

    // Bloom
    float bloom_strength = 0.8f;
    float bloom_radius = 0.02f;
    int bloom_passes = 3;
    int bloom_octaves = 4;
    bool bloom_chromatic = false;

    // Ghosts
    int ghosts = 0; // 0 = off by default
    float ghost_dispersal = 0.35f;
    float ghost_intensity = 0.15f;
    float ghost_chromatic = 0.01f;

    // Halo
    float halo_radius = 0.45f;
    float halo_width = 0.07f;
    float halo_intensity = 0.0f; // 0 = off by default

    // Starburst
    int starburst_rays = 0; // 0 = off
    float starburst_intensity = 0.3f;
    float starburst_length = 0.3f;
    float starburst_width = 0.008f;

    // Anamorphic streak
    bool streak = false;
    float streak_intensity = 0.25f;
    float streak_length = 0.5f;
    float streak_tint_r = 0.6f;
    float streak_tint_g = 0.7f;
    float streak_tint_b = 1.0f;
};

static void print_usage(const char *prog)
{
    printf("Usage: %s input.exr output.[tga|exr|hdr] [options]\n\n", prog);
    printf("Unified post-processor: bloom + lens flare + grading.\n\n");
    printf("General:\n");
    printf("  --threshold <f>          Luminance threshold (shared)  (default: 1.0)\n");
    printf("  --exposure  <f>          Exposure multiplier           (default: 1.0)\n");
    printf("  --tonemap   <f>          Log tonemap (0=off)           (default: 0)\n");
    printf("  --sky-brightness <f>     Sky brightness multiplier     (default: 1.0)\n");
    printf("\nBloom:\n");
    printf("  --bloom-strength <f>     Bloom intensity (0=off)       (default: 0.8)\n");
    printf("  --bloom-radius   <f>     Base radius / diagonal        (default: 0.02)\n");
    printf("  --bloom-passes   <n>     Box-blur passes (1-10)        (default: 3)\n");
    printf("  --bloom-octaves  <n>     Scale octaves (1-6)           (default: 4)\n");
    printf("  --chromatic              Warm chromatic bloom shift     (default: off)\n");
    printf("\nGhost reflections:\n");
    printf("  --ghosts <n>             Number of ghosts (0=off)      (default: 0)\n");
    printf("  --ghost-dispersal <f>    Spacing                       (default: 0.35)\n");
    printf("  --ghost-intensity <f>    Brightness                    (default: 0.15)\n");
    printf("  --ghost-chromatic <f>    Chromatic spread               (default: 0.01)\n");
    printf("\nHalo ring:\n");
    printf("  --halo-radius <f>        Ring radius (frac of diag)    (default: 0.45)\n");
    printf("  --halo-width <f>         Ring softness                 (default: 0.07)\n");
    printf("  --halo-intensity <f>     Ring brightness (0=off)       (default: 0)\n");
    printf("\nStarburst diffraction:\n");
    printf("  --starburst-rays <n>     Number of rays (0=off)        (default: 0)\n");
    printf("  --starburst-intensity <f> Ray brightness               (default: 0.3)\n");
    printf("  --starburst-length <f>   Ray length (frac of diag)     (default: 0.3)\n");
    printf("  --starburst-width <f>    Angular width per ray          (default: 0.008)\n");
    printf("\nAnamorphic streak:\n");
    printf("  --streak                 Enable streak                 (default: off)\n");
    printf("  --streak-intensity <f>   Streak brightness             (default: 0.25)\n");
    printf("  --streak-length <f>      Length (frac of width)         (default: 0.5)\n");
    printf("  --streak-tint-r/g/b <f>  Streak colour                 (default: 0.6 0.7 1.0)\n");
    printf("\nPresets:\n");
    printf("  --interstellar           Full cinematic look (bloom + flare)\n");
    printf("  --help                   Print this help\n");
}

static bool parse_args(int argc, char *argv[], GradeParams &p)
{
    if (argc < 3)
        return false;

    p.input_file = argv[1];
    p.output_file = argv[2];

    for (int i = 3; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--help"))
            return false;
        // General
        else if (!strcmp(argv[i], "--threshold") && i + 1 < argc)
            p.threshold = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--exposure") && i + 1 < argc)
            p.exposure = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--tonemap") && i + 1 < argc)
            p.tonemap = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--sky-brightness") && i + 1 < argc)
            p.sky_brightness = (float)atof(argv[++i]);
        // Bloom
        else if (!strcmp(argv[i], "--bloom-strength") && i + 1 < argc)
            p.bloom_strength = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--bloom-radius") && i + 1 < argc)
            p.bloom_radius = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--bloom-passes") && i + 1 < argc)
            p.bloom_passes = std::clamp(atoi(argv[++i]), 1, 10);
        else if (!strcmp(argv[i], "--bloom-octaves") && i + 1 < argc)
            p.bloom_octaves = std::clamp(atoi(argv[++i]), 1, 6);
        else if (!strcmp(argv[i], "--chromatic"))
            p.bloom_chromatic = true;
        // Ghosts
        else if (!strcmp(argv[i], "--ghosts") && i + 1 < argc)
            p.ghosts = std::clamp(atoi(argv[++i]), 0, 20);
        else if (!strcmp(argv[i], "--ghost-dispersal") && i + 1 < argc)
            p.ghost_dispersal = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--ghost-intensity") && i + 1 < argc)
            p.ghost_intensity = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--ghost-chromatic") && i + 1 < argc)
            p.ghost_chromatic = (float)atof(argv[++i]);
        // Halo
        else if (!strcmp(argv[i], "--halo-radius") && i + 1 < argc)
            p.halo_radius = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--halo-width") && i + 1 < argc)
            p.halo_width = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--halo-intensity") && i + 1 < argc)
            p.halo_intensity = (float)atof(argv[++i]);
        // Starburst
        else if (!strcmp(argv[i], "--starburst-rays") && i + 1 < argc)
            p.starburst_rays = std::clamp(atoi(argv[++i]), 0, 32);
        else if (!strcmp(argv[i], "--starburst-intensity") && i + 1 < argc)
            p.starburst_intensity = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--starburst-length") && i + 1 < argc)
            p.starburst_length = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--starburst-width") && i + 1 < argc)
            p.starburst_width = (float)atof(argv[++i]);
        // Streak
        else if (!strcmp(argv[i], "--streak"))
            p.streak = true;
        else if (!strcmp(argv[i], "--streak-intensity") && i + 1 < argc)
            p.streak_intensity = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--streak-length") && i + 1 < argc)
            p.streak_length = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--streak-tint-r") && i + 1 < argc)
            p.streak_tint_r = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--streak-tint-g") && i + 1 < argc)
            p.streak_tint_g = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--streak-tint-b") && i + 1 < argc)
            p.streak_tint_b = (float)atof(argv[++i]);
        // Presets
        else if (!strcmp(argv[i], "--interstellar"))
        {
            // Bloom: overexposed chromatic fire-glow
            p.bloom_strength = 2.5f;
            p.bloom_radius = 0.06f;
            p.bloom_octaves = 5;
            p.bloom_chromatic = true;

            // Flare: barely-there — a hint of optics, not a light show
            p.streak = true;
            p.streak_intensity = 0.05f;
            p.streak_length = 0.20f;
            p.streak_tint_r = 0.5f;
            p.streak_tint_g = 0.65f;
            p.streak_tint_b = 1.0f;
            p.ghosts = 2;
            p.ghost_intensity = 0.015f;
            p.ghost_dispersal = 0.30f;
            p.ghost_chromatic = 0.005f;
            p.halo_intensity = 0.025f;
            p.halo_radius = 0.40f;
            p.starburst_rays = 4;
            p.starburst_intensity = 0.03f;
            p.starburst_length = 0.15f;

            // Grading
            p.threshold = 1.5f;
            p.exposure = 3.0f;
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
    int width = 0, height = 0;
    Imf::Header header;

    struct Channel
    {
        std::string name;
        std::vector<float> data;
    };
    std::vector<Channel> channels;

    int idx_R = -1, idx_G = -1, idx_B = -1;
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
        Imf::Header header(img.width, img.height);
        header.compression() = img.header.compression();

        for (auto it = img.header.begin(); it != img.header.end(); ++it)
        {
            if (!strcmp(it.name(), "channels") || !strcmp(it.name(), "compression") ||
                !strcmp(it.name(), "dataWindow") || !strcmp(it.name(), "displayWindow") ||
                !strcmp(it.name(), "lineOrder") || !strcmp(it.name(), "pixelAspectRatio") ||
                !strcmp(it.name(), "screenWindowCenter") || !strcmp(it.name(), "screenWindowWidth"))
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
// Bloom — multi-octave box-blur with optional chromatic warm shift
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

    std::vector<float> tmp_r(np), tmp_g(np), tmp_b(np);

    // Horizontal pass
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

    // Vertical pass
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

// Generate multi-octave bloom into accumulator buffers.
// Does NOT modify the beauty pass — caller composites.
static void generate_bloom(const float *bright_r, const float *bright_g,
                           const float *bright_b,
                           float *out_r, float *out_g, float *out_b,
                           int w, int h, size_t np, const GradeParams &p)
{
    if (p.bloom_strength < 1e-6f)
        return;

    const float diag = std::sqrt((float)(w * w + h * h));
    const int base_kernel = std::max((int)(p.bloom_radius * diag), 1);

    printf("Bloom: %d octaves, base radius=%d, strength=%.2f%s\n",
           p.bloom_octaves, base_kernel, p.bloom_strength,
           p.bloom_chromatic ? ", chromatic" : "");

    // Chromatic tint table: white-hot → yellow → orange → red
    static const float chroma_r[] = {1.00f, 1.00f, 1.00f, 1.00f, 0.90f, 0.70f};
    static const float chroma_g[] = {1.00f, 0.85f, 0.60f, 0.35f, 0.18f, 0.08f};
    static const float chroma_b[] = {1.00f, 0.50f, 0.20f, 0.08f, 0.03f, 0.01f};

    float octave_weight = 1.0f;
    const float weight_decay = 0.6f;

    for (int oct = 0; oct < p.bloom_octaves; ++oct)
    {
        float oct_radius = (float)base_kernel;
        for (int k = 0; k < oct; ++k)
            oct_radius *= 2.5f;
        int oct_kernel = std::min((int)oct_radius, (int)(diag * 0.25f));

        float oct_sigma = oct_kernel / 3.0f;
        std::vector<int> box_radii;
        compute_box_radii(oct_sigma, p.bloom_passes, box_radii);

        int ci = std::min(oct, 5);
        float tint_r = p.bloom_chromatic ? chroma_r[ci] : 1.0f;
        float tint_g = p.bloom_chromatic ? chroma_g[ci] : 1.0f;
        float tint_b = p.bloom_chromatic ? chroma_b[ci] : 1.0f;

        printf("  octave %d/%d: radius=%d, sigma=%.1f, weight=%.3f",
               oct + 1, p.bloom_octaves, oct_kernel, oct_sigma, octave_weight);
        if (p.bloom_chromatic)
            printf(", tint=(%.2f, %.2f, %.2f)", tint_r, tint_g, tint_b);
        printf(", box=[");
        for (int i = 0; i < p.bloom_passes; ++i)
            printf("%s%d", i ? ", " : "", box_radii[i]);
        printf("]\n");

        std::vector<float> bloom_r(bright_r, bright_r + np);
        std::vector<float> bloom_g(bright_g, bright_g + np);
        std::vector<float> bloom_b(bright_b, bright_b + np);
        std::vector<float> tmp_r(np), tmp_g(np), tmp_b(np);

        for (int pass = 0; pass < p.bloom_passes; ++pass)
        {
            box_blur_pass(bloom_r, bloom_g, bloom_b,
                          tmp_r, tmp_g, tmp_b,
                          w, h, box_radii[pass]);
            std::swap(bloom_r, tmp_r);
            std::swap(bloom_g, tmp_g);
            std::swap(bloom_b, tmp_b);
        }

        const float wr = octave_weight * tint_r * p.bloom_strength;
        const float wg = octave_weight * tint_g * p.bloom_strength;
        const float wb = octave_weight * tint_b * p.bloom_strength;
#pragma omp parallel for
        for (int i = 0; i < (int)np; ++i)
        {
            out_r[i] += wr * bloom_r[i];
            out_g[i] += wg * bloom_g[i];
            out_b[i] += wb * bloom_b[i];
        }

        octave_weight *= weight_decay;
    }
}

// ============================================================================
// Lens Flare Components
// ============================================================================

// Bilinear sample at (u,v) in [0,1]×[0,1]
static inline void sample_bilinear(const float *buf_r, const float *buf_g,
                                   const float *buf_b,
                                   int w, int h,
                                   float u, float v,
                                   float &out_r, float &out_g, float &out_b)
{
    float fx = u * (w - 1);
    float fy = v * (h - 1);
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = x0 + 1, y1 = y0 + 1;

    if (x0 < 0 || y0 < 0 || x1 >= w || y1 >= h)
    {
        out_r = out_g = out_b = 0;
        return;
    }

    float dx = fx - x0, dy = fy - y0;
    float w00 = (1 - dx) * (1 - dy);
    float w10 = dx * (1 - dy);
    float w01 = (1 - dx) * dy;
    float w11 = dx * dy;

    int i00 = y0 * w + x0, i10 = y0 * w + x1;
    int i01 = y1 * w + x0, i11 = y1 * w + x1;

    out_r = w00 * buf_r[i00] + w10 * buf_r[i10] + w01 * buf_r[i01] + w11 * buf_r[i11];
    out_g = w00 * buf_g[i00] + w10 * buf_g[i10] + w01 * buf_g[i01] + w11 * buf_g[i11];
    out_b = w00 * buf_b[i00] + w10 * buf_b[i10] + w01 * buf_b[i01] + w11 * buf_b[i11];
}

// Horizontal 1D box blur
static void hbox_blur(const std::vector<float> &src,
                      std::vector<float> &dst,
                      int width, int height, int radius)
{
    const float inv = 1.0f / (2 * radius + 1);
#pragma omp parallel for
    for (int y = 0; y < height; ++y)
    {
        const int row = y * width;
        float s = 0;
        for (int k = -radius; k <= radius; ++k)
            s += src[row + std::clamp(k, 0, width - 1)];
        dst[row] = s * inv;
        for (int x = 1; x < width; ++x)
        {
            s += src[row + std::min(x + radius, width - 1)];
            s -= src[row + std::clamp(x - radius - 1, 0, width - 1)];
            dst[row + x] = s * inv;
        }
    }
}

// --- Ghosts ---
static void generate_ghosts(const float *bright_r, const float *bright_g,
                            const float *bright_b,
                            float *out_r, float *out_g, float *out_b,
                            int w, int h, const GradeParams &p)
{
    if (p.ghosts <= 0 || p.ghost_intensity < 1e-6f)
        return;

    printf("  Generating %d ghost reflections (dispersal=%.2f, intensity=%.3f)\n",
           p.ghosts, p.ghost_dispersal, p.ghost_intensity);

#pragma omp parallel for
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            float u = ((float)x + 0.5f) / w;
            float v = ((float)y + 0.5f) / h;

            float ghost_u = 1.0f - u;
            float ghost_v = 1.0f - v;
            float du = ghost_u - 0.5f;
            float dv = ghost_v - 0.5f;

            float acc_r = 0, acc_g = 0, acc_b = 0;

            for (int g = 0; g < p.ghosts; ++g)
            {
                float scale = 1.0f - p.ghost_dispersal * (g + 1);
                float su = 0.5f + du * scale;
                float sv = 0.5f + dv * scale;

                float chroma = p.ghost_chromatic * (g + 1);
                float sr, sg, sb, tr, tg, tb;

                sample_bilinear(bright_r, bright_r, bright_r, w, h,
                                0.5f + (su - 0.5f) * (1.0f + chroma),
                                0.5f + (sv - 0.5f) * (1.0f + chroma),
                                sr, tg, tb);
                sample_bilinear(bright_g, bright_g, bright_g, w, h,
                                su, sv, tr, sg, tb);
                sample_bilinear(bright_b, bright_b, bright_b, w, h,
                                0.5f + (su - 0.5f) * (1.0f - chroma),
                                0.5f + (sv - 0.5f) * (1.0f - chroma),
                                tr, tg, sb);

                float dist = std::sqrt((su - 0.5f) * (su - 0.5f) + (sv - 0.5f) * (sv - 0.5f));
                float falloff = 1.0f - std::clamp(dist * 2.0f, 0.0f, 1.0f);
                falloff *= falloff;

                float ghost_weight = 1.0f / (1.0f + g * 0.5f);

                acc_r += sr * falloff * ghost_weight;
                acc_g += sg * falloff * ghost_weight;
                acc_b += sb * falloff * ghost_weight;
            }

            int idx = y * w + x;
            out_r[idx] += p.ghost_intensity * acc_r;
            out_g[idx] += p.ghost_intensity * acc_g;
            out_b[idx] += p.ghost_intensity * acc_b;
        }
    }
}

// --- Halo ---
static void generate_halo(const float *bright_r, const float *bright_g,
                          const float *bright_b,
                          float *out_r, float *out_g, float *out_b,
                          int w, int h, const GradeParams &p)
{
    if (p.halo_intensity < 1e-6f)
        return;

    size_t np = (size_t)w * h;
    const float aspect = (float)w / h;

    printf("  Generating halo ring (radius=%.2f, width=%.3f, intensity=%.3f)\n",
           p.halo_radius, p.halo_width, p.halo_intensity);

    double sum_lum = 0;
#pragma omp parallel for reduction(+ : sum_lum)
    for (int i = 0; i < (int)np; ++i)
        sum_lum += 0.2126 * bright_r[i] + 0.7152 * bright_g[i] + 0.0722 * bright_b[i];
    float avg_lum = (float)(sum_lum / np);
    float lum_scale = std::min(avg_lum * 50.0f, 3.0f);

    if (lum_scale < 1e-6f)
        return;

#pragma omp parallel for
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            float u = ((float)x + 0.5f) / w - 0.5f;
            float v = ((float)y + 0.5f) / h - 0.5f;
            u *= aspect;

            float dist = std::sqrt(u * u + v * v);

            float ring = std::exp(-0.5f * ((dist - p.halo_radius) * (dist - p.halo_radius)) / (p.halo_width * p.halo_width));
            if (ring < 0.001f)
                continue;

            float ring_r = std::exp(-0.5f * ((dist - p.halo_radius * 1.04f) * (dist - p.halo_radius * 1.04f)) / (p.halo_width * p.halo_width));
            float ring_g = ring;
            float ring_b = std::exp(-0.5f * ((dist - p.halo_radius * 0.96f) * (dist - p.halo_radius * 0.96f)) / (p.halo_width * p.halo_width));

            int idx = y * w + x;
            float intensity = p.halo_intensity * lum_scale;
            out_r[idx] += intensity * ring_r;
            out_g[idx] += intensity * ring_g;
            out_b[idx] += intensity * ring_b;
        }
    }
}

// --- Starburst ---
static void generate_starburst(const float *bright_r, const float *bright_g,
                               const float *bright_b,
                               float *out_r, float *out_g, float *out_b,
                               int w, int h, const GradeParams &p)
{
    if (p.starburst_rays <= 0 || p.starburst_intensity < 1e-6f)
        return;

    size_t np = (size_t)w * h;
    const float diag = std::sqrt((float)(w * w + h * h));
    const int ray_len = std::max((int)(p.starburst_length * diag), 1);
    const int num_rays = p.starburst_rays;

    printf("  Generating %d-ray starburst (length=%d px, intensity=%.3f)\n",
           num_rays, ray_len, p.starburst_intensity);

    std::vector<float> ray_dx(num_rays), ray_dy(num_rays);
    for (int r = 0; r < num_rays; ++r)
    {
        float angle = (float)(r * M_PI / num_rays);
        ray_dx[r] = std::cos(angle);
        ray_dy[r] = std::sin(angle);
    }

    std::vector<float> star_r(np, 0), star_g(np, 0), star_b(np, 0);

    for (int r = 0; r < num_rays; ++r)
    {
        float dx = ray_dx[r];
        float dy = ray_dy[r];

        std::vector<float> cur_r(bright_r, bright_r + np);
        std::vector<float> cur_g(bright_g, bright_g + np);
        std::vector<float> cur_b(bright_b, bright_b + np);
        std::vector<float> tmp_r(np), tmp_g(np), tmp_b(np);

        int num_passes = std::max(1, (int)std::ceil(std::log2((float)ray_len)));
        int step = 1;

        for (int pass = 0; pass < num_passes; ++pass)
        {
            float sdx = dx * step;
            float sdy = dy * step;

#pragma omp parallel for
            for (int y0 = 0; y0 < h; ++y0)
            {
                for (int x0 = 0; x0 < w; ++x0)
                {
                    int idx = y0 * w + x0;
                    int sx1 = x0 + (int)sdx, sy1 = y0 + (int)sdy;
                    int sx2 = x0 - (int)sdx, sy2 = y0 - (int)sdy;

                    float rv = cur_r[idx], gv = cur_g[idx], bv = cur_b[idx];
                    if (sx1 >= 0 && sx1 < w && sy1 >= 0 && sy1 < h)
                    {
                        int oi = sy1 * w + sx1;
                        rv += cur_r[oi];
                        gv += cur_g[oi];
                        bv += cur_b[oi];
                    }
                    if (sx2 >= 0 && sx2 < w && sy2 >= 0 && sy2 < h)
                    {
                        int oi = sy2 * w + sx2;
                        rv += cur_r[oi];
                        gv += cur_g[oi];
                        bv += cur_b[oi];
                    }
                    tmp_r[idx] = rv * 0.333333f;
                    tmp_g[idx] = gv * 0.333333f;
                    tmp_b[idx] = bv * 0.333333f;
                }
            }

            std::swap(cur_r, tmp_r);
            std::swap(cur_g, tmp_g);
            std::swap(cur_b, tmp_b);
            step *= 2;
        }

#pragma omp parallel for
        for (int i = 0; i < (int)np; ++i)
        {
            star_r[i] += cur_r[i];
            star_g[i] += cur_g[i];
            star_b[i] += cur_b[i];
        }
    }

    float scale = p.starburst_intensity / num_rays;
#pragma omp parallel for
    for (int i = 0; i < (int)np; ++i)
    {
        out_r[i] += star_r[i] * scale;
        out_g[i] += star_g[i] * scale;
        out_b[i] += star_b[i] * scale;
    }
}

// --- Anamorphic streak ---
static void generate_streak(const float *bright_r, const float *bright_g,
                            const float *bright_b,
                            float *out_r, float *out_g, float *out_b,
                            int w, int h, const GradeParams &p)
{
    if (!p.streak || p.streak_intensity < 1e-6f)
        return;

    size_t np = (size_t)w * h;
    const int streak_radius = std::max((int)(p.streak_length * w * 0.5f), 1);

    printf("  Generating anamorphic streak (radius=%d px, intensity=%.3f, tint=%.2f,%.2f,%.2f)\n",
           streak_radius, p.streak_intensity, p.streak_tint_r, p.streak_tint_g, p.streak_tint_b);

    std::vector<float> str_r(np), str_g(np), str_b(np);
#pragma omp parallel for
    for (int i = 0; i < (int)np; ++i)
    {
        float lum = 0.2126f * bright_r[i] + 0.7152f * bright_g[i] + 0.0722f * bright_b[i];
        str_r[i] = lum * p.streak_tint_r;
        str_g[i] = lum * p.streak_tint_g;
        str_b[i] = lum * p.streak_tint_b;
    }

    std::vector<float> tmp(np);
    int passes = 4;
    int pass_radius = streak_radius / 2;

    for (int pass = 0; pass < passes; ++pass)
    {
        hbox_blur(str_r, tmp, w, h, pass_radius);
        std::swap(str_r, tmp);
        hbox_blur(str_g, tmp, w, h, pass_radius);
        std::swap(str_g, tmp);
        hbox_blur(str_b, tmp, w, h, pass_radius);
        std::swap(str_b, tmp);
    }

#pragma omp parallel for
    for (int i = 0; i < (int)np; ++i)
    {
        out_r[i] += str_r[i] * p.streak_intensity;
        out_g[i] += str_g[i] * p.streak_intensity;
        out_b[i] += str_b[i] * p.streak_intensity;
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char *argv[])
{
    GradeParams params;
    if (!parse_args(argc, argv, params))
    {
        print_usage(argv[0]);
        return 1;
    }

    printf("bhrt-grade — unified EXR post-processor\n");
    printf("  input:          %s\n", params.input_file.c_str());
    printf("  output:         %s\n", params.output_file.c_str());
    printf("  threshold:      %.2f\n", params.threshold);
    printf("  exposure:       %.2f\n", params.exposure);
    printf("  sky_brightness: %.2f\n", params.sky_brightness);
    printf("  bloom:          strength=%.2f radius=%.3f octaves=%d%s\n",
           params.bloom_strength, params.bloom_radius, params.bloom_octaves,
           params.bloom_chromatic ? " chromatic" : "");
    printf("  ghosts:         %d (dispersal=%.2f, intensity=%.3f)\n",
           params.ghosts, params.ghost_dispersal, params.ghost_intensity);
    printf("  halo:           radius=%.2f width=%.3f intensity=%.3f\n",
           params.halo_radius, params.halo_width, params.halo_intensity);
    printf("  starburst:      %d rays (intensity=%.3f, length=%.2f)\n",
           params.starburst_rays, params.starburst_intensity, params.starburst_length);
    printf("  streak:         %s (intensity=%.3f, length=%.2f, tint=%.2f,%.2f,%.2f)\n",
           params.streak ? "ON" : "off",
           params.streak_intensity, params.streak_length,
           params.streak_tint_r, params.streak_tint_g, params.streak_tint_b);
    printf("  tonemap:        %.2f%s\n", params.tonemap, params.tonemap < 1e-6f ? " (off)" : "");

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
    int w = img.width, h = img.height;

    auto t0 = std::chrono::steady_clock::now();

    // --- 1. Sky brightness -----------------------------------------------
    bool has_layers = (img.idx_disk_R >= 0 && img.idx_disk_G >= 0 && img.idx_disk_B >= 0 &&
                       img.idx_sky_R >= 0 && img.idx_sky_G >= 0 && img.idx_sky_B >= 0);

    if (std::abs(params.sky_brightness - 1.0f) > 1e-6f)
    {
        if (!has_layers)
        {
            fprintf(stderr, "WARNING: --sky-brightness requires disk.*/sky.* layers. Skipping.\n");
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
                beauty_r[i] = disk_r[i] + sky_r[i] * params.sky_brightness;
                beauty_g[i] = disk_g[i] + sky_g[i] * params.sky_brightness;
                beauty_b[i] = disk_b[i] + sky_b[i] * params.sky_brightness;
                sky_r[i] *= params.sky_brightness;
                sky_g[i] *= params.sky_brightness;
                sky_b[i] *= params.sky_brightness;
            }
        }
    }

    // --- 2. Exposure -----------------------------------------------------
    if (std::abs(params.exposure - 1.0f) > 1e-6f)
    {
        printf("Applying exposure: %.2f\n", params.exposure);
#pragma omp parallel for
        for (int i = 0; i < (int)np; ++i)
        {
            beauty_r[i] *= params.exposure;
            beauty_g[i] *= params.exposure;
            beauty_b[i] *= params.exposure;
        }
    }

    // --- 3. Shared bright pixel extraction --------------------------------
    // One threshold extraction feeds BOTH bloom and lens flare.
    printf("Extracting bright pixels (threshold=%.2f)...\n", params.threshold);
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

    // --- 4 & 5. Generate bloom + flare into a shared accumulator ---------
    // Both effects add into the same buffer, then composite once.
    std::vector<float> fx_r(np, 0), fx_g(np, 0), fx_b(np, 0);

    printf("--- Bloom ---\n");
    generate_bloom(bright_r.data(), bright_g.data(), bright_b.data(),
                   fx_r.data(), fx_g.data(), fx_b.data(),
                   w, h, np, params);

    printf("--- Lens Flare ---\n");
    generate_ghosts(bright_r.data(), bright_g.data(), bright_b.data(),
                    fx_r.data(), fx_g.data(), fx_b.data(),
                    w, h, params);

    generate_halo(bright_r.data(), bright_g.data(), bright_b.data(),
                  fx_r.data(), fx_g.data(), fx_b.data(),
                  w, h, params);

    generate_starburst(bright_r.data(), bright_g.data(), bright_b.data(),
                       fx_r.data(), fx_g.data(), fx_b.data(),
                       w, h, params);

    generate_streak(bright_r.data(), bright_g.data(), bright_b.data(),
                    fx_r.data(), fx_g.data(), fx_b.data(),
                    w, h, params);

    // --- 6. Composite FX onto beauty (single pass) -----------------------
    printf("Compositing effects onto beauty...\n");
#pragma omp parallel for
    for (int i = 0; i < (int)np; ++i)
    {
        beauty_r[i] += fx_r[i];
        beauty_g[i] += fx_g[i];
        beauty_b[i] += fx_b[i];
    }

    // --- 7. Tonemap ------------------------------------------------------
    if (params.tonemap > 1e-6f)
    {
        const float c = std::pow(10.0f, params.tonemap * 2.0f) - 1.0f;
        const float norm = 1.0f / std::log(1.0f + c);
        printf("Applying tonemap: compression=%.2f (c=%.2f)\n", params.tonemap, c);
#pragma omp parallel for
        for (int i = 0; i < (int)np; ++i)
        {
            beauty_r[i] = std::log(1.0f + c * std::max(beauty_r[i], 0.0f)) * norm;
            beauty_g[i] = std::log(1.0f + c * std::max(beauty_g[i], 0.0f)) * norm;
            beauty_b[i] = std::log(1.0f + c * std::max(beauty_b[i], 0.0f)) * norm;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    printf("Processing time: %.2f seconds\n",
           std::chrono::duration<double>(t1 - t0).count());

    // --- 8. Output -------------------------------------------------------
    auto ends_with = [](const std::string &s, const char *suffix) -> bool
    {
        size_t len = strlen(suffix);
        return s.size() >= len && s.compare(s.size() - len, len, suffix) == 0;
    };

    const std::string &out = params.output_file;

    if (ends_with(out, ".tga"))
    {
        if (params.tonemap < 1e-6f)
        {
            const float c = std::pow(10.0f, 1.0f * 2.0f) - 1.0f;
            const float norm = 1.0f / std::log(1.0f + c);
            printf("Auto-applying tonemap for TGA (compression=1.0)\n");
#pragma omp parallel for
            for (int i = 0; i < (int)np; ++i)
            {
                beauty_r[i] = std::log(1.0f + c * std::max(beauty_r[i], 0.0f)) * norm;
                beauty_g[i] = std::log(1.0f + c * std::max(beauty_g[i], 0.0f)) * norm;
                beauty_b[i] = std::log(1.0f + c * std::max(beauty_b[i], 0.0f)) * norm;
            }
        }

        std::vector<unsigned char> pixels(np * 3);
#pragma omp parallel for
        for (int i = 0; i < (int)np; ++i)
        {
            pixels[i * 3 + 0] = (unsigned char)std::clamp((int)(beauty_r[i] * 255.0f + 0.5f), 0, 255);
            pixels[i * 3 + 1] = (unsigned char)std::clamp((int)(beauty_g[i] * 255.0f + 0.5f), 0, 255);
            pixels[i * 3 + 2] = (unsigned char)std::clamp((int)(beauty_b[i] * 255.0f + 0.5f), 0, 255);
        }

        if (stbi_write_tga(out.c_str(), w, h, 3, pixels.data()))
            printf("Wrote TGA: %s\n", out.c_str());
        else
        {
            fprintf(stderr, "ERROR: failed to write TGA: %s\n", out.c_str());
            return 1;
        }
    }
    else if (ends_with(out, ".hdr"))
    {
        std::vector<float> hdr_pixels(np * 3);
#pragma omp parallel for
        for (int i = 0; i < (int)np; ++i)
        {
            hdr_pixels[i * 3 + 0] = beauty_r[i];
            hdr_pixels[i * 3 + 1] = beauty_g[i];
            hdr_pixels[i * 3 + 2] = beauty_b[i];
        }
        if (stbi_write_hdr(out.c_str(), w, h, 3, hdr_pixels.data()))
            printf("Wrote HDR: %s\n", out.c_str());
        else
        {
            fprintf(stderr, "ERROR: failed to write HDR: %s\n", out.c_str());
            return 1;
        }
    }
    else
    {
        if (!save_exr(out.c_str(), img))
            return 1;
    }

    return 0;
}
