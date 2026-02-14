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

#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfFrameBuffer.h>

#include "stb_image_write.h"

#include "lens.h"
#include "ghost.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- CLI parameters -----------------------------------------------------

struct Params
{
    std::string input_file;
    std::string output_file;
    std::string lens_file;

    float fov_deg = 60.0f;      // horizontal FOV in degrees
    float threshold = 3.0f;     // bright pixel luminance threshold
    int ray_grid = 64;          // entrance pupil grid per dimension
    float min_ghost = 1e-7f;    // ghost pair pre-filter threshold
    int downsample = 4;         // downsample bright pixels by this factor
    float exposure = 1.0f;      // exposure multiplier for TGA
    float tonemap = 1.0f;       // tonemap compression for TGA
    float flare_gain = 1000.0f; // ghost intensity multiplier (default high for visibility)
    std::string tga_file;       // optional TGA output path
    std::string debug_tga;      // optional debug TGA: only bright pixels above threshold
};

static void print_usage(const char *prog)
{
    printf("Usage: %s input.exr output.exr [options]\n\n", prog);
    printf("Physically-based lens flare simulator.\n\n");
    printf("Required:\n");
    printf("  --lens <file>         Lens prescription file (.lens)\n");
    printf("  --fov <degrees>       Horizontal field of view (default: 60)\n");
    printf("\nOptional:\n");
    printf("  --threshold <f>       Bright pixel threshold (default: 3.0)\n");
    printf("  --rays <n>            Entrance pupil grid size (default: 64)\n");
    printf("  --min-ghost <f>       Ghost pair pre-filter threshold (default: 1e-7)\n");
    printf("  --downsample <n>      Downsample bright pixels by factor (default: 4)\n");
    printf("  --flare-gain <f>      Ghost intensity multiplier (default: 1000)\n");
    printf("  --tga <file>          Also write composited TGA (beauty + flare, tonemapped)\n");
    printf("  --exposure <f>        Exposure multiplier for TGA output (default: 1.0)\n");
    printf("  --tonemap <f>         Tonemap compression for TGA output (default: 1.0)\n");
    printf("  --debug-tga <file>    Write debug TGA: only bright pixels above threshold\n");
    printf("  --help                Print this help\n");
}

static bool parse_args(int argc, char *argv[], Params &p)
{
    if (argc < 3)
        return false;

    p.input_file = argv[1];
    p.output_file = argv[2];

    for (int i = 3; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--help"))
            return false;
        else if (!strcmp(argv[i], "--lens") && i + 1 < argc)
            p.lens_file = argv[++i];
        else if (!strcmp(argv[i], "--fov") && i + 1 < argc)
            p.fov_deg = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--threshold") && i + 1 < argc)
            p.threshold = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--rays") && i + 1 < argc)
            p.ray_grid = std::max(4, atoi(argv[++i]));
        else if (!strcmp(argv[i], "--min-ghost") && i + 1 < argc)
            p.min_ghost = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--downsample") && i + 1 < argc)
            p.downsample = std::max(1, atoi(argv[++i]));
        else if (!strcmp(argv[i], "--flare-gain") && i + 1 < argc)
            p.flare_gain = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--tga") && i + 1 < argc)
            p.tga_file = argv[++i];
        else if (!strcmp(argv[i], "--exposure") && i + 1 < argc)
            p.exposure = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--tonemap") && i + 1 < argc)
            p.tonemap = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--debug-tga") && i + 1 < argc)
            p.debug_tga = argv[++i];
        else
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return false;
        }
    }

    if (p.lens_file.empty())
    {
        fprintf(stderr, "ERROR: --lens is required\n");
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
                     const float *flare_r, const float *flare_g, const float *flare_b)
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

        Imf::OutputFile file(path, header);
        file.setFrameBuffer(fb);
        file.writePixels(img.height);

        printf("Wrote EXR: %s (original layers + flare.R/G/B)\n", path);
        return true;
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "ERROR writing EXR: %s (%s)\n", path, e.what());
        return false;
    }
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

    render_ghosts(lens, sources, fov_h, fov_v,
                  flare_r.data(), flare_g.data(), flare_b.data(),
                  img.width, img.height, gcfg);

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

    // ---- Write output EXR ----
    if (!save_exr(params.output_file.c_str(), img,
                  flare_r.data(), flare_g.data(), flare_b.data()))
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
            float r = (beauty_r[i] + flare_r[i]) * exp;
            float g = (beauty_g[i] + flare_g[i]) * exp;
            float b = (beauty_b[i] + flare_b[i]) * exp;

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
