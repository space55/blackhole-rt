// ============================================================================
// main.cpp — flaresim entry point
//
// Physically-based lens flare simulator.
//
// Reads a linear-light EXR, extracts bright pixels, traces ghost reflections
// through a user-specified lens system, and writes the result as additional
// EXR layers (flare.R, flare.G, flare.B) alongside the original channels.
//
// Usage:
//   flaresim input.exr output.exr --lens lenses/doublegauss.lens --fov 60
//
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfFrameBuffer.h>

#include "stb_image_write.h"

#include "lens.h"
#include "ghost.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- CLI parameters -----------------------------------------------------

struct Params
{
    std::string input_file;
    std::string output_file;
    std::string lens_file;

    float fov_deg = 60.0f;       // horizontal FOV in degrees
    float threshold = 3.0f;      // bright pixel luminance threshold
    int ray_grid = 64;           // entrance pupil grid per dimension
    float min_ghost = 1e-7f;     // ghost pair pre-filter threshold
    int downsample = 4;          // downsample bright pixels by this factor
    float exposure = 1.0f;       // exposure multiplier for TGA
    float tonemap = 1.0f;        // tonemap compression for TGA
    float flare_gain = 1000.0f;  // ghost intensity multiplier (default high for visibility)
    float sky_brightness = 1.0f; // sky layer multiplier (requires sky.R/G/B layers in EXR)

    // Bloom
    float bloom_strength = 0.0f;   // 0 = off; 1.0 = strong; 3.0+ = on-fire wash-out
    float bloom_radius = 0.04f;    // base radius as fraction of image diagonal
    int bloom_passes = 3;          // box-blur passes (approx Gaussian)
    int bloom_octaves = 5;         // multi-scale octaves (more = wider glow)
    bool bloom_chromatic = true;   // warm chromatic shift (white→yellow→orange→red)
    float bloom_threshold = -1.0f; // bloom threshold; -1 = use main threshold

    // Ghost area normalization (compensate for defocused ghosts)
    bool ghost_normalize = true;   // per-pair area correction
    float max_area_boost = 100.0f; // cap on area correction factor

    // Ghost smoothing
    float ghost_blur = 0.003f; // blur radius as fraction of image diagonal (0 = off)
    int ghost_blur_passes = 3; // box-blur passes (3 ≈ Gaussian)

    std::string tga_file;  // optional TGA output path
    std::string debug_tga; // optional debug TGA: only bright pixels above threshold
};

static void print_usage(const char *prog)
{
    printf("Usage: %s <config_file> [--key value ...] [options]\n\n", prog);
    printf("Physically-based lens flare simulator.\n\n");
    printf("The config file uses key = value format (same as bhrt3 scene files).\n");
    printf("Lines starting with # are comments.\n\n");
    printf("Config keys:\n");
    printf("  input           Input EXR file\n");
    printf("  output          Output EXR file\n");
    printf("  lens            Lens prescription file (.lens)\n");
    printf("  fov             Horizontal field of view in degrees (default: 60)\n");
    printf("  threshold       Bright pixel luminance threshold (default: 3.0)\n");
    printf("  rays            Entrance pupil grid size (default: 64)\n");
    printf("  min_ghost       Ghost pair pre-filter threshold (default: 1e-7)\n");
    printf("  downsample      Downsample bright pixels by factor (default: 4)\n");
    printf("  flare_gain      Ghost intensity multiplier (default: 1000)\n");
    printf("  sky_brightness  Scale sky background brightness (default: 1.0)\n");
    printf("  tga             Also write composited TGA (beauty + flare, tonemapped)\n");
    printf("  exposure        Exposure multiplier for TGA output (default: 1.0)\n");
    printf("  tonemap         Tonemap compression for TGA output (default: 1.0)\n");
    printf("  debug_tga       Write debug TGA: only bright pixels above threshold\n");
    printf("\nBloom (fiery wash-out glow):\n");
    printf("  bloom_strength  Bloom intensity, 0=off (default: 0)\n");
    printf("                    0.5 = subtle, 1.0 = strong, 3.0+ = on fire\n");
    printf("  bloom_radius    Base blur radius as frac of diagonal (default: 0.04)\n");
    printf("  bloom_passes    Box-blur passes, 1-10 (default: 3)\n");
    printf("  bloom_octaves   Multi-scale octaves, 1-6 (default: 5)\n");
    printf("  bloom_chromatic 1 = warm chromatic shift (default: 1)\n");
    printf("  bloom_threshold Bloom bright-pixel cutoff; -1 = use threshold (default: -1)\n");
    printf("\nGhost normalization (defocus compensation):\n");
    printf("  ghost_normalize  1 = per-pair area correction (default: 1)\n");
    printf("  max_area_boost   Cap on defocus boost factor (default: 100)\n");
    printf("\nGhost smoothing (smooth sparse ray splats):\n");
    printf("  ghost_blur       Blur radius as frac of diagonal; 0=off (default: 0.003)\n");
    printf("  ghost_blur_passes  Box-blur passes, 1-5 (default: 3)\n");
    printf("\nAll keys can also be passed as CLI overrides: --key value\n");
    printf("  e.g.: %s flare.conf --flare_gain 2000 --threshold 2.0\n", prog);
    printf("\n  --help          Print this help\n");
}

// ---- Config file parser (key = value, same format as bhrt3 scene) --------

static std::string trim(const std::string &s)
{
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
        return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static bool load_config(const char *path,
                        std::unordered_map<std::string, std::string> &kv)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        fprintf(stderr, "ERROR: cannot open config file: %s\n", path);
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        // Strip comments
        size_t hash = line.find('#');
        if (hash != std::string::npos)
            line = line.substr(0, hash);

        line = trim(line);
        if (line.empty())
            continue;

        size_t eq = line.find('=');
        if (eq == std::string::npos)
            continue;

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));

        if (!key.empty() && !val.empty())
            kv[key] = val;
    }
    return true;
}

static bool apply_params(const std::unordered_map<std::string, std::string> &kv,
                         Params &p)
{
    auto get_string = [&](const char *key, std::string &out)
    {
        auto it = kv.find(key);
        if (it != kv.end())
            out = it->second;
    };
    auto get_float = [&](const char *key, float &out)
    {
        auto it = kv.find(key);
        if (it != kv.end())
            out = (float)std::atof(it->second.c_str());
    };
    auto get_int = [&](const char *key, int &out)
    {
        auto it = kv.find(key);
        if (it != kv.end())
            out = std::atoi(it->second.c_str());
    };

    get_string("input", p.input_file);
    get_string("output", p.output_file);
    get_string("lens", p.lens_file);
    get_float("fov", p.fov_deg);
    get_float("threshold", p.threshold);
    get_int("rays", p.ray_grid);
    get_float("min_ghost", p.min_ghost);
    get_int("downsample", p.downsample);
    get_float("flare_gain", p.flare_gain);
    get_float("sky_brightness", p.sky_brightness);
    get_string("tga", p.tga_file);
    get_float("exposure", p.exposure);
    get_float("tonemap", p.tonemap);
    get_string("debug_tga", p.debug_tga);

    // Bloom
    get_float("bloom_strength", p.bloom_strength);
    get_float("bloom_radius", p.bloom_radius);
    get_int("bloom_passes", p.bloom_passes);
    get_int("bloom_octaves", p.bloom_octaves);
    get_float("bloom_threshold", p.bloom_threshold);
    {
        auto it = kv.find("bloom_chromatic");
        if (it != kv.end())
            p.bloom_chromatic = (std::atoi(it->second.c_str()) != 0);
    }

    // Ghost normalization
    get_float("max_area_boost", p.max_area_boost);
    {
        auto it = kv.find("ghost_normalize");
        if (it != kv.end())
            p.ghost_normalize = (std::atoi(it->second.c_str()) != 0);
    }

    // Ghost smoothing
    get_float("ghost_blur", p.ghost_blur);
    get_int("ghost_blur_passes", p.ghost_blur_passes);

    return true;
}

static bool parse_args(int argc, char *argv[], Params &p)
{
    if (argc < 2)
        return false;

    // First positional arg is the config file
    const char *config_path = nullptr;

    // Collect CLI overrides as key-value pairs
    std::unordered_map<std::string, std::string> cli_kv;

    int i = 1;
    while (i < argc)
    {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
            return false;

        // CLI key-value override: --key value
        if (argv[i][0] == '-' && argv[i][1] == '-' && i + 1 < argc)
        {
            // Strip leading "--" and convert hyphens to underscores
            std::string key(argv[i] + 2);
            for (auto &c : key)
                if (c == '-')
                    c = '_';
            cli_kv[key] = argv[i + 1];
            i += 2;
        }
        else if (!config_path)
        {
            config_path = argv[i];
            ++i;
        }
        else
        {
            fprintf(stderr, "ERROR: unexpected argument: %s\n", argv[i]);
            return false;
        }
    }

    // Load config file
    if (!config_path)
    {
        fprintf(stderr, "ERROR: no config file specified\n");
        return false;
    }

    std::unordered_map<std::string, std::string> file_kv;
    if (!load_config(config_path, file_kv))
        return false;

    // Apply file config, then CLI overrides on top
    apply_params(file_kv, p);

    // Merge CLI overrides into file_kv and re-apply
    for (auto &pair : cli_kv)
        file_kv[pair.first] = pair.second;
    apply_params(file_kv, p);

    // Clamp
    p.ray_grid = std::max(4, p.ray_grid);
    p.downsample = std::max(1, p.downsample);

    // Validate required fields
    if (p.input_file.empty())
    {
        fprintf(stderr, "ERROR: 'input' is required in config\n");
        return false;
    }
    if (p.output_file.empty())
    {
        fprintf(stderr, "ERROR: 'output' is required in config\n");
        return false;
    }
    if (p.lens_file.empty())
    {
        fprintf(stderr, "ERROR: 'lens' is required in config\n");
        return false;
    }

    return true;
}

// ---- EXR I/O ------------------------------------------------------------

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
    int idx_skyR = -1, idx_skyG = -1, idx_skyB = -1;

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
            img.channels.push_back({it.name(), std::vector<float>(np, 0)});

            if (it.name() == std::string("R"))
                img.idx_R = idx;
            if (it.name() == std::string("G"))
                img.idx_G = idx;
            if (it.name() == std::string("B"))
                img.idx_B = idx;
            if (it.name() == std::string("sky.R"))
                img.idx_skyR = idx;
            if (it.name() == std::string("sky.G"))
                img.idx_skyG = idx;
            if (it.name() == std::string("sky.B"))
                img.idx_skyB = idx;
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

        printf("Loaded EXR: %s (%d×%d, %zu channels)\n",
               path, img.width, img.height, img.channels.size());
        return true;
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "ERROR loading EXR: %s (%s)\n", path, e.what());
        return false;
    }
}

static bool save_exr(const char *path, const EXRImage &img,
                     const float *flare_r, const float *flare_g, const float *flare_b,
                     const float *bloom_r = nullptr, const float *bloom_g = nullptr,
                     const float *bloom_b = nullptr)
{
    try
    {
        Imf::Header header(img.width, img.height);
        header.compression() = img.header.compression();

        // Copy metadata from original
        for (auto it = img.header.begin(); it != img.header.end(); ++it)
        {
            if (!strcmp(it.name(), "channels") || !strcmp(it.name(), "compression") ||
                !strcmp(it.name(), "dataWindow") || !strcmp(it.name(), "displayWindow") ||
                !strcmp(it.name(), "lineOrder") || !strcmp(it.name(), "pixelAspectRatio") ||
                !strcmp(it.name(), "screenWindowCenter") || !strcmp(it.name(), "screenWindowWidth"))
                continue;
            header.insert(it.name(), it.attribute());
        }

        // Original channels
        for (auto &ch : img.channels)
            header.channels().insert(ch.name.c_str(), Imf::Channel(Imf::FLOAT));

        // Flare layers
        header.channels().insert("flare.R", Imf::Channel(Imf::FLOAT));
        header.channels().insert("flare.G", Imf::Channel(Imf::FLOAT));
        header.channels().insert("flare.B", Imf::Channel(Imf::FLOAT));

        // Bloom layers
        bool has_bloom = (bloom_r && bloom_g && bloom_b);
        if (has_bloom)
        {
            header.channels().insert("bloom.R", Imf::Channel(Imf::FLOAT));
            header.channels().insert("bloom.G", Imf::Channel(Imf::FLOAT));
            header.channels().insert("bloom.B", Imf::Channel(Imf::FLOAT));
        }

        Imf::FrameBuffer fb;

        for (auto &ch : img.channels)
        {
            fb.insert(ch.name.c_str(),
                      Imf::Slice(Imf::FLOAT,
                                 (char *)ch.data.data(),
                                 sizeof(float),
                                 sizeof(float) * img.width));
        }

        fb.insert("flare.R",
                  Imf::Slice(Imf::FLOAT, (char *)flare_r,
                             sizeof(float), sizeof(float) * img.width));
        fb.insert("flare.G",
                  Imf::Slice(Imf::FLOAT, (char *)flare_g,
                             sizeof(float), sizeof(float) * img.width));
        fb.insert("flare.B",
                  Imf::Slice(Imf::FLOAT, (char *)flare_b,
                             sizeof(float), sizeof(float) * img.width));

        if (has_bloom)
        {
            fb.insert("bloom.R",
                      Imf::Slice(Imf::FLOAT, (char *)bloom_r,
                                 sizeof(float), sizeof(float) * img.width));
            fb.insert("bloom.G",
                      Imf::Slice(Imf::FLOAT, (char *)bloom_g,
                                 sizeof(float), sizeof(float) * img.width));
            fb.insert("bloom.B",
                      Imf::Slice(Imf::FLOAT, (char *)bloom_b,
                                 sizeof(float), sizeof(float) * img.width));
        }

        Imf::OutputFile file(path, header);
        file.setFrameBuffer(fb);
        file.writePixels(img.height);

        printf("Wrote EXR: %s (original layers + flare.R/G/B%s)\n", path,
               has_bloom ? " + bloom.R/G/B" : "");
        return true;
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "ERROR writing EXR: %s (%s)\n", path, e.what());
        return false;
    }
}

// ---- Bloom — multi-octave box-blur with chromatic warm shift ------------

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

// Generate multi-octave bloom.
// Chromatic tint shifts white-hot → yellow → orange → deep red across octaves
// for a fiery wash-out look.  Higher bloom_strength values push bright regions
// well past 1.0 before tonemapping, creating the "on fire" overexposed look.
static void generate_bloom(const float *img_r, const float *img_g,
                           const float *img_b,
                           float *out_r, float *out_g, float *out_b,
                           int w, int h, float bloom_thresh,
                           float bloom_strength, float bloom_radius,
                           int bloom_passes, int bloom_octaves,
                           bool bloom_chromatic)
{
    if (bloom_strength < 1e-6f)
        return;

    size_t np = (size_t)w * h;
    const float diag = std::sqrt((float)(w * w + h * h));
    const int base_kernel = std::max((int)(bloom_radius * diag), 1);

    printf("Bloom: %d octaves, base radius=%d px, strength=%.2f%s, threshold=%.2f\n",
           bloom_octaves, base_kernel, bloom_strength,
           bloom_chromatic ? ", chromatic" : "", bloom_thresh);

    // Extract bright pixels into the bloom source buffer
    std::vector<float> bright_r(np), bright_g(np), bright_b(np);
#pragma omp parallel for
    for (int i = 0; i < (int)np; ++i)
    {
        float lum = 0.2126f * img_r[i] + 0.7152f * img_g[i] + 0.0722f * img_b[i];
        if (lum > bloom_thresh)
        {
            // Keep the excess above threshold — this drives the glow
            float excess = (lum - bloom_thresh) / std::max(lum, 1e-10f);
            bright_r[i] = img_r[i] * excess;
            bright_g[i] = img_g[i] * excess;
            bright_b[i] = img_b[i] * excess;
        }
    }

    // Chromatic tint table: white-hot → yellow → orange → deep red
    static const float chroma_r[] = {1.00f, 1.00f, 1.00f, 1.00f, 0.95f, 0.80f};
    static const float chroma_g[] = {1.00f, 0.90f, 0.65f, 0.40f, 0.20f, 0.10f};
    static const float chroma_b[] = {1.00f, 0.55f, 0.22f, 0.08f, 0.03f, 0.01f};

    float octave_weight = 1.0f;
    const float weight_decay = 0.55f;

    for (int oct = 0; oct < bloom_octaves; ++oct)
    {
        float oct_radius = (float)base_kernel;
        for (int k = 0; k < oct; ++k)
            oct_radius *= 2.5f;
        int oct_kernel = std::min((int)oct_radius, (int)(diag * 0.25f));

        float oct_sigma = oct_kernel / 3.0f;
        std::vector<int> box_radii;
        compute_box_radii(oct_sigma, bloom_passes, box_radii);

        int ci = std::min(oct, 5);
        float tint_r = bloom_chromatic ? chroma_r[ci] : 1.0f;
        float tint_g = bloom_chromatic ? chroma_g[ci] : 1.0f;
        float tint_b = bloom_chromatic ? chroma_b[ci] : 1.0f;

        printf("  octave %d/%d: radius=%d, sigma=%.1f, weight=%.3f",
               oct + 1, bloom_octaves, oct_kernel, oct_sigma, octave_weight);
        if (bloom_chromatic)
            printf(", tint=(%.2f, %.2f, %.2f)", tint_r, tint_g, tint_b);
        printf("\n");

        std::vector<float> blur_r(bright_r.begin(), bright_r.end());
        std::vector<float> blur_g(bright_g.begin(), bright_g.end());
        std::vector<float> blur_b(bright_b.begin(), bright_b.end());
        std::vector<float> tmp_r(np), tmp_g(np), tmp_b(np);

        for (int pass = 0; pass < bloom_passes; ++pass)
        {
            box_blur_pass(blur_r, blur_g, blur_b,
                          tmp_r, tmp_g, tmp_b,
                          w, h, box_radii[pass]);
            std::swap(blur_r, tmp_r);
            std::swap(blur_g, tmp_g);
            std::swap(blur_b, tmp_b);
        }

        const float wr = octave_weight * tint_r * bloom_strength;
        const float wg = octave_weight * tint_g * bloom_strength;
        const float wb = octave_weight * tint_b * bloom_strength;
#pragma omp parallel for
        for (int i = 0; i < (int)np; ++i)
        {
            out_r[i] += wr * blur_r[i];
            out_g[i] += wg * blur_g[i];
            out_b[i] += wb * blur_b[i];
        }

        octave_weight *= weight_decay;
    }

    // Bloom stats
    float bmax = 0;
    int bnonzero = 0;
    for (size_t i = 0; i < np; ++i)
    {
        float v = out_r[i] + out_g[i] + out_b[i];
        if (v > 0)
            ++bnonzero;
        bmax = std::max(bmax, v);
    }
    printf("Bloom stats: max=%.4f, nonzero=%d / %zu\n", bmax, bnonzero, np);
}

// ---- Bright pixel extraction --------------------------------------------

static std::vector<BrightPixel> extract_bright_pixels(
    const EXRImage &img, float threshold, int downsample,
    float fov_h, float fov_v)
{
    std::vector<BrightPixel> result;

    const float *R = img.channels[img.idx_R].data.data();
    const float *G = img.channels[img.idx_G].data.data();
    const float *B = img.channels[img.idx_B].data.data();

    int dw = std::max(1, img.width / downsample);
    int dh = std::max(1, img.height / downsample);

    float tan_half_h = std::tan(fov_h * 0.5f);
    float tan_half_v = std::tan(fov_v * 0.5f);

    for (int dy = 0; dy < dh; ++dy)
    {
        for (int dx = 0; dx < dw; ++dx)
        {
            // Average the downsampled block
            float sum_r = 0, sum_g = 0, sum_b = 0;
            int count = 0;

            int y0 = dy * downsample;
            int x0 = dx * downsample;
            int y1 = std::min(y0 + downsample, img.height);
            int x1 = std::min(x0 + downsample, img.width);

            for (int y = y0; y < y1; ++y)
            {
                for (int x = x0; x < x1; ++x)
                {
                    int i = y * img.width + x;
                    sum_r += R[i];
                    sum_g += G[i];
                    sum_b += B[i];
                    ++count;
                }
            }

            if (count == 0)
                continue;

            float avg_r = sum_r / count;
            float avg_g = sum_g / count;
            float avg_b = sum_b / count;

            float lum = 0.2126f * avg_r + 0.7152f * avg_g + 0.0722f * avg_b;
            if (lum <= threshold)
                continue;

            // Centre of the block → pixel coordinate
            float cx = (x0 + x1) * 0.5f;
            float cy = (y0 + y1) * 0.5f;

            // Pixel → angle (pinhole camera model)
            // ndc ∈ [-0.5, 0.5], angle = atan(ndc * 2 * tan(fov/2))
            float ndc_x = cx / img.width - 0.5f;
            float ndc_y = cy / img.height - 0.5f;

            BrightPixel bp;
            bp.angle_x = std::atan(ndc_x * 2.0f * tan_half_h);
            bp.angle_y = std::atan(ndc_y * 2.0f * tan_half_v);

            // Store the average radiance of the block.
            bp.r = avg_r;
            bp.g = avg_g;
            bp.b = avg_b;

            result.push_back(bp);
        }
    }

    return result;
}

// ---- Main ---------------------------------------------------------------

int main(int argc, char *argv[])
{
    Params params;
    if (!parse_args(argc, argv, params))
    {
        print_usage(argv[0]);
        return 1;
    }

    printf("flaresim — physically-based lens flare simulator\n");
    printf("  input:      %s\n", params.input_file.c_str());
    printf("  output:     %s\n", params.output_file.c_str());
    printf("  lens:       %s\n", params.lens_file.c_str());
    printf("  fov:        %.1f°\n", params.fov_deg);
    printf("  threshold:  %.2f\n", params.threshold);
    printf("  ray grid:   %d×%d\n", params.ray_grid, params.ray_grid);
    printf("  downsample: %d×\n", params.downsample);
    printf("  flare gain: %.1f\n", params.flare_gain);
    if (params.bloom_strength > 0)
    {
        printf("  bloom:      strength=%.2f, radius=%.3f, %d octaves, %d passes%s\n",
               params.bloom_strength, params.bloom_radius,
               params.bloom_octaves, params.bloom_passes,
               params.bloom_chromatic ? ", chromatic" : "");
    }
    if (params.sky_brightness != 1.0f)
        printf("  sky bright: %.3f\n", params.sky_brightness);

    // ---- Load lens system ----
    LensSystem lens;
    if (!lens.load(params.lens_file.c_str()))
        return 1;
    lens.print_summary();

    if (lens.focal_length <= 0)
    {
        fprintf(stderr, "ERROR: lens focal_length must be > 0\n");
        return 1;
    }

    // ---- Load input EXR ----
    EXRImage img;
    if (!load_exr(params.input_file.c_str(), img))
        return 1;

    if (img.idx_R < 0 || img.idx_G < 0 || img.idx_B < 0)
    {
        fprintf(stderr, "ERROR: input EXR must have R, G, B channels\n");
        return 1;
    }

    size_t np = img.num_pixels();

    // ---- Apply sky brightness scaling ----
    if (params.sky_brightness != 1.0f)
    {
        if (img.idx_skyR >= 0 && img.idx_skyG >= 0 && img.idx_skyB >= 0)
        {
            // We have separate sky layers: scale the sky contribution in the
            // beauty R/G/B channels.  beauty = (beauty - sky) + sky * factor
            //                                = beauty + sky * (factor - 1)
            float delta = params.sky_brightness - 1.0f;
            float *R = img.channels[img.idx_R].data.data();
            float *G = img.channels[img.idx_G].data.data();
            float *B = img.channels[img.idx_B].data.data();
            const float *sR = img.channels[img.idx_skyR].data.data();
            const float *sG = img.channels[img.idx_skyG].data.data();
            const float *sB = img.channels[img.idx_skyB].data.data();
            for (size_t i = 0; i < np; ++i)
            {
                R[i] += sR[i] * delta;
                G[i] += sG[i] * delta;
                B[i] += sB[i] * delta;
            }
            // Also scale the sky layers themselves for the EXR output
            float *swR = img.channels[img.idx_skyR].data.data();
            float *swG = img.channels[img.idx_skyG].data.data();
            float *swB = img.channels[img.idx_skyB].data.data();
            for (size_t i = 0; i < np; ++i)
            {
                swR[i] *= params.sky_brightness;
                swG[i] *= params.sky_brightness;
                swB[i] *= params.sky_brightness;
            }
            printf("Sky brightness: %.3f (using sky.R/G/B layers)\n", params.sky_brightness);
        }
        else
        {
            // No sky layers — fall back to scaling all pixels below threshold.
            // This is a rough approximation: dim pixels are assumed to be sky.
            float *R = img.channels[img.idx_R].data.data();
            float *G = img.channels[img.idx_G].data.data();
            float *B = img.channels[img.idx_B].data.data();
            int scaled = 0;
            for (size_t i = 0; i < np; ++i)
            {
                float lum = 0.2126f * R[i] + 0.7152f * G[i] + 0.0722f * B[i];
                if (lum <= params.threshold)
                {
                    R[i] *= params.sky_brightness;
                    G[i] *= params.sky_brightness;
                    B[i] *= params.sky_brightness;
                    ++scaled;
                }
            }
            printf("Sky brightness: %.3f (fallback: scaled %d/%zu pixels below threshold)\n",
                   params.sky_brightness, scaled, np);
        }
    }

    // ---- Compute FOV ----
    float fov_h = params.fov_deg * (float)M_PI / 180.0f;
    float aspect = (float)img.width / img.height;
    float fov_v = 2.0f * std::atan(std::tan(fov_h * 0.5f) / aspect);

    printf("FOV: %.1f° × %.1f° (aspect %.3f)\n",
           fov_h * 180 / M_PI, fov_v * 180 / M_PI, aspect);

    // ---- Extract bright pixels ----
    auto sources = extract_bright_pixels(img, params.threshold, params.downsample,
                                         fov_h, fov_v);
    printf("Bright sources after downsample: %zu\n", sources.size());

    // ---- Bright pixel diagnostics ----
    if (!sources.empty())
    {
        float min_lum = 1e30f, max_lum = 0, sum_lum = 0;
        float max_r = 0, max_g = 0, max_b = 0;
        for (auto &s : sources)
        {
            float lum = 0.2126f * s.r + 0.7152f * s.g + 0.0722f * s.b;
            min_lum = std::min(min_lum, lum);
            max_lum = std::max(max_lum, lum);
            sum_lum += lum;
            max_r = std::max(max_r, s.r);
            max_g = std::max(max_g, s.g);
            max_b = std::max(max_b, s.b);
        }
        printf("Bright source stats:\n");
        printf("  luminance: min=%.4f, max=%.4f, mean=%.4f\n",
               min_lum, max_lum, sum_lum / sources.size());
        printf("  max channel values: R=%.4f, G=%.4f, B=%.4f\n",
               max_r, max_g, max_b);
        printf("  angle range: x=[%.4f, %.4f]°, y=[%.4f, %.4f]°\n",
               sources.front().angle_x * 180.0f / (float)M_PI,
               sources.back().angle_x * 180.0f / (float)M_PI,
               sources.front().angle_y * 180.0f / (float)M_PI,
               sources.back().angle_y * 180.0f / (float)M_PI);
    }

    // ---- Write debug TGA: only bright pixels above threshold ----
    if (!params.debug_tga.empty())
    {
        const float *R = img.channels[img.idx_R].data.data();
        const float *G = img.channels[img.idx_G].data.data();
        const float *B = img.channels[img.idx_B].data.data();

        float img_max = 0;
        for (size_t i = 0; i < np; ++i)
        {
            float lum = 0.2126f * R[i] + 0.7152f * G[i] + 0.0722f * B[i];
            img_max = std::max(img_max, lum);
        }
        printf("Image luminance max: %.6f (threshold: %.6f)\n", img_max, params.threshold);

        int above_count = 0;
        std::vector<unsigned char> dbg(np * 3, 0);
        for (size_t i = 0; i < np; ++i)
        {
            float lum = 0.2126f * R[i] + 0.7152f * G[i] + 0.0722f * B[i];
            if (lum > params.threshold)
            {
                ++above_count;
                // Normalize to make bright pixels visible in 8-bit
                float scale = 1.0f / std::max(img_max, 1.0f);
                dbg[i * 3 + 0] = (unsigned char)std::clamp((int)(R[i] * scale * 255.0f + 0.5f), 0, 255);
                dbg[i * 3 + 1] = (unsigned char)std::clamp((int)(G[i] * scale * 255.0f + 0.5f), 0, 255);
                dbg[i * 3 + 2] = (unsigned char)std::clamp((int)(B[i] * scale * 255.0f + 0.5f), 0, 255);
            }
        }
        printf("Pixels above threshold: %d / %zu (%.2f%%)\n",
               above_count, np, 100.0f * above_count / np);

        if (stbi_write_tga(params.debug_tga.c_str(), img.width, img.height, 3, dbg.data()))
            printf("Wrote debug TGA: %s\n", params.debug_tga.c_str());
        else
            fprintf(stderr, "ERROR: failed to write debug TGA: %s\n", params.debug_tga.c_str());
    }

    if (sources.empty())
    {
        printf("No pixels above threshold — writing input unchanged (with empty flare layers).\n");
        std::vector<float> zeros(np, 0);
        save_exr(params.output_file.c_str(), img,
                 zeros.data(), zeros.data(), zeros.data());
        return 0;
    }

    // ---- Render ghosts ----
    std::vector<float> flare_r(np, 0), flare_g(np, 0), flare_b(np, 0);

    GhostConfig gcfg;
    gcfg.ray_grid = params.ray_grid;
    gcfg.min_intensity = params.min_ghost;
    gcfg.gain = params.flare_gain;
    gcfg.ghost_normalize = params.ghost_normalize;
    gcfg.max_area_boost = params.max_area_boost;

    render_ghosts(lens, sources, fov_h, fov_v,
                  flare_r.data(), flare_g.data(), flare_b.data(),
                  img.width, img.height, gcfg);

    // ---- Ghost blur (smooth sparse ray splats) ----
    if (params.ghost_blur > 0)
    {
        const float diag = std::sqrt((float)(img.width * img.width + img.height * img.height));
        int kernel = std::max((int)(params.ghost_blur * diag), 1);
        float sigma = kernel / 3.0f;
        int passes = std::clamp(params.ghost_blur_passes, 1, 5);

        std::vector<int> box_radii;
        compute_box_radii(sigma, passes, box_radii);

        printf("Ghost blur: radius=%d px (sigma=%.1f), %d passes\n",
               kernel, sigma, passes);

        std::vector<float> tmp_r(np), tmp_g(np), tmp_b(np);
        for (int pass = 0; pass < passes; ++pass)
        {
            box_blur_pass(flare_r, flare_g, flare_b,
                          tmp_r, tmp_g, tmp_b,
                          img.width, img.height, box_radii[pass]);
            std::swap(flare_r, tmp_r);
            std::swap(flare_g, tmp_g);
            std::swap(flare_b, tmp_b);
        }
    }

    // ---- Flare stats ----
    {
        float max_val = 0, sum_val = 0;
        int nonzero = 0;
        for (size_t i = 0; i < np; ++i)
        {
            float v = flare_r[i] + flare_g[i] + flare_b[i];
            if (v > 0)
            {
                ++nonzero;
                sum_val += v;
            }
            max_val = std::max(max_val, v);
        }
        printf("Flare stats: max=%.6f, mean=%.6e, nonzero=%d / %zu\n",
               max_val, nonzero > 0 ? sum_val / nonzero : 0.0f, nonzero, np);
    }

    // ---- Generate bloom ----
    std::vector<float> bloom_r(np, 0), bloom_g(np, 0), bloom_b(np, 0);
    if (params.bloom_strength > 0)
    {
        float bt = (params.bloom_threshold >= 0) ? params.bloom_threshold : params.threshold;
        const float *R = img.channels[img.idx_R].data.data();
        const float *G = img.channels[img.idx_G].data.data();
        const float *B = img.channels[img.idx_B].data.data();
        generate_bloom(R, G, B,
                       bloom_r.data(), bloom_g.data(), bloom_b.data(),
                       img.width, img.height, bt,
                       params.bloom_strength, params.bloom_radius,
                       std::clamp(params.bloom_passes, 1, 10),
                       std::clamp(params.bloom_octaves, 1, 6),
                       params.bloom_chromatic);
    }

    // ---- Write output EXR ----
    if (!save_exr(params.output_file.c_str(), img,
                  flare_r.data(), flare_g.data(), flare_b.data(),
                  params.bloom_strength > 0 ? bloom_r.data() : nullptr,
                  params.bloom_strength > 0 ? bloom_g.data() : nullptr,
                  params.bloom_strength > 0 ? bloom_b.data() : nullptr))
        return 1;

    // ---- Write composited TGA ----
    if (!params.tga_file.empty())
    {
        const float *beauty_r = img.channels[img.idx_R].data.data();
        const float *beauty_g = img.channels[img.idx_G].data.data();
        const float *beauty_b = img.channels[img.idx_B].data.data();

        float c = std::pow(10.0f, params.tonemap * 2.0f) - 1.0f;
        float norm = 1.0f / std::log(1.0f + c);
        float exp = params.exposure;

        std::vector<unsigned char> pixels(np * 3);
        for (size_t i = 0; i < np; ++i)
        {
            float r = (beauty_r[i] + flare_r[i] + bloom_r[i]) * exp;
            float g = (beauty_g[i] + flare_g[i] + bloom_g[i]) * exp;
            float b = (beauty_b[i] + flare_b[i] + bloom_b[i]) * exp;

            // Tonemap: log(1 + cx) / log(1 + c)
            r = std::log(1.0f + c * std::max(r, 0.0f)) * norm;
            g = std::log(1.0f + c * std::max(g, 0.0f)) * norm;
            b = std::log(1.0f + c * std::max(b, 0.0f)) * norm;

            pixels[i * 3 + 0] = (unsigned char)std::clamp((int)(r * 255.0f + 0.5f), 0, 255);
            pixels[i * 3 + 1] = (unsigned char)std::clamp((int)(g * 255.0f + 0.5f), 0, 255);
            pixels[i * 3 + 2] = (unsigned char)std::clamp((int)(b * 255.0f + 0.5f), 0, 255);
        }

        if (stbi_write_tga(params.tga_file.c_str(), img.width, img.height, 3, pixels.data()))
            printf("Wrote TGA: %s (exposure=%.2f, tonemap=%.2f)\n",
                   params.tga_file.c_str(), params.exposure, params.tonemap);
        else
            fprintf(stderr, "ERROR: failed to write TGA: %s\n", params.tga_file.c_str());
    }

    printf("Done.\n");
    return 0;
}
