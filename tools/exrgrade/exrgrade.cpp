// exrgrade — per-layer exposure grading for multi-layer EXR files
//
// Reads an EXR, applies per-layer exposure multipliers, composites layers,
// and saves the result as a TGA (LDR, clamped to [0,1]).
//
// Usage:
//   exrgrade input.exr -o output.tga [layer=multiplier ...]
//
// Layers correspond to the EXR channel name prefix.  The renderer writes:
//   (default)  R, G, B, A         — beauty pass (disk + sky combined)
//   disk       disk.R/G/B/A       — accretion disk emission
//   sky        sky.R/G/B          — background sky
//
// Flaresim may add:
//   flare      flare.R/G/B        — lens flare
//   bloom      bloom.R/G/B        — bloom glow
//
// If no layer arguments are given, the default beauty-pass RGB is used at 1×.
//
// When layer arguments ARE given, only those layers are composited (additive).
// This lets you mix-and-match:
//
//   exrgrade render.exr -o bright_disk.tga disk=3.0 sky=0.5
//   exrgrade render.exr -o disk_only.tga disk=2.0
//   exrgrade render.exr -o beauty.tga default=1.5
//
// Options:
//   -o <path>       output TGA path (required)
//   -exposure <e>   exposure multiplier (default 1.0)
//   -tonemap <t>    tone-map compression 0-1 (default 1.0; matches renderer)
//                   Uses log(1 + c*x) / log(1 + c) where c = 10^(2t) - 1
//                   0 = linear (no tone mapping), 1 = heavy compression
//   -gamma <g>      output gamma (default 1.0, use 2.2 for sRGB)
//   -list           list all channels in the EXR and exit

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <set>

#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfFrameBuffer.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ---------------------------------------------------------------------------
// Channel storage
// ---------------------------------------------------------------------------
struct Channel
{
    std::string name;
    std::vector<float> data;
};

// ---------------------------------------------------------------------------
// Parse layer prefix from a channel name.
// "disk.R" -> "disk",  "R" -> "" (default layer)
// ---------------------------------------------------------------------------
static std::string layer_prefix(const std::string &ch_name)
{
    auto dot = ch_name.rfind('.');
    if (dot == std::string::npos)
        return ""; // top-level channel = default layer
    return ch_name.substr(0, dot);
}

// Suffix after the dot (or the whole name if no dot)
static std::string channel_suffix(const std::string &ch_name)
{
    auto dot = ch_name.rfind('.');
    if (dot == std::string::npos)
        return ch_name;
    return ch_name.substr(dot + 1);
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s input.exr -o output.tga [options] [layer=multiplier ...]\n"
            "\n"
            "Options:\n"
            "  -o <path>       Output TGA file (required)\n"
            "  -exposure <e>   Exposure multiplier (default 1.0)\n"
            "  -tonemap <t>    Tone-map compression 0-1 (default 1.0, matches renderer)\n"
            "                  0 = linear (no tone mapping), 1 = heavy compression\n"
            "  -gamma <g>      Output gamma (default 1.0; use 2.2 for sRGB)\n"
            "  -list           List all channels/layers in the EXR and exit\n"
            "\n"
            "Layer arguments:\n"
            "  default=1.5     Multiply the beauty pass (R,G,B) by 1.5\n"
            "  disk=3.0        Multiply disk.R/G/B by 3.0\n"
            "  sky=0.5         Multiply sky.R/G/B by 0.5\n"
            "  flare=2.0       Multiply flare.R/G/B by 2.0\n"
            "\n"
            "When layer arguments are given, only specified layers are composited\n"
            "(additive).  When none are given, the beauty pass is used at 1×.\n",
            prog);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    // Parse arguments
    const char *input_path = nullptr;
    const char *output_path = nullptr;
    float exposure = 1.0f;
    float tonemap_compression = 1.0f; // matches renderer/flaresim default
    float gamma_val = 1.0f;
    bool list_mode = false;
    std::map<std::string, float> layer_multipliers; // layer prefix -> multiplier

    // Helper: parse "-flag=value" or "-flag value" for numeric options.
    // Returns the float value, advancing i if the value is in the next arg.
    // Sets ok=true on success, ok=false if no value found.
    auto parse_flag_float = [&](int &i, const char *flag, bool &ok) -> float {
        const char *arg = argv[i];
        size_t flag_len = strlen(flag);
        if (strncmp(arg, flag, flag_len) == 0 && arg[flag_len] == '=')
        {
            ok = true;
            return (float)atof(arg + flag_len + 1); // -flag=value
        }
        if (strcmp(arg, flag) == 0 && i + 1 < argc)
        {
            ok = true;
            return (float)atof(argv[++i]); // -flag value
        }
        ok = false;
        return 0.0f;
    };

    for (int i = 1; i < argc; ++i)
    {
        std::string a(argv[i]);

        if (a == "-o" && i + 1 < argc)
        {
            output_path = argv[++i];
        }
        else if (a.rfind("-exposure", 0) == 0)
        {
            bool ok;
            exposure = parse_flag_float(i, "-exposure", ok);
            if (!ok) { fprintf(stderr, "Error: -exposure requires a value\n"); return 1; }
        }
        else if (a.rfind("-tonemap", 0) == 0)
        {
            bool ok;
            tonemap_compression = parse_flag_float(i, "-tonemap", ok);
            if (!ok) { fprintf(stderr, "Error: -tonemap requires a value\n"); return 1; }
        }
        else if (a.rfind("-gamma", 0) == 0)
        {
            bool ok;
            gamma_val = parse_flag_float(i, "-gamma", ok);
            if (!ok) { fprintf(stderr, "Error: -gamma requires a value\n"); return 1; }
        }
        else if (a == "-list")
        {
            list_mode = true;
        }
        else if (a == "-h" || a == "--help")
        {
            print_usage(argv[0]);
            return 0;
        }
        else if (a[0] == '-')
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
        else if (strchr(argv[i], '=') != nullptr)
        {
            // layer=multiplier
            auto eq = a.find('=');
            std::string layer = a.substr(0, eq);
            float mult = (float)atof(a.substr(eq + 1).c_str());
            layer_multipliers[layer] = mult;
        }
        else if (!input_path)
        {
            input_path = argv[i];
        }
        else
        {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!input_path)
    {
        fprintf(stderr, "Error: no input EXR specified.\n");
        print_usage(argv[0]);
        return 1;
    }

    // -----------------------------------------------------------------------
    // Load EXR
    // -----------------------------------------------------------------------
    int width, height;
    std::vector<Channel> channels;

    try
    {
        Imf::InputFile file(input_path);
        const Imf::Header &hdr = file.header();
        Imath::Box2i dw = hdr.dataWindow();
        width = dw.max.x - dw.min.x + 1;
        height = dw.max.y - dw.min.y + 1;
        const size_t num_pixels = (size_t)width * height;

        const Imf::ChannelList &ch_list = hdr.channels();
        for (auto it = ch_list.begin(); it != ch_list.end(); ++it)
        {
            channels.push_back({it.name(), std::vector<float>(num_pixels, 0.0f)});
        }

        Imf::FrameBuffer fb;
        for (auto &ch : channels)
        {
            fb.insert(ch.name.c_str(),
                      Imf::Slice(Imf::FLOAT,
                                 (char *)(ch.data.data() - (size_t)dw.min.x - (size_t)dw.min.y * width),
                                 sizeof(float),
                                 sizeof(float) * width));
        }
        file.setFrameBuffer(fb);
        file.readPixels(dw.min.y, dw.max.y);

        printf("Loaded: %s (%d x %d, %zu channels)\n",
               input_path, width, height, channels.size());
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "Error loading EXR '%s': %s\n", input_path, e.what());
        return 1;
    }

    // -----------------------------------------------------------------------
    // Build layer inventory
    // -----------------------------------------------------------------------
    // Discover all layers and their R/G/B channel indices
    struct LayerInfo
    {
        std::string prefix; // "" for default, "disk", "sky", etc.
        int idx_r = -1;
        int idx_g = -1;
        int idx_b = -1;
    };
    std::map<std::string, LayerInfo> layers;

    for (int i = 0; i < (int)channels.size(); ++i)
    {
        std::string prefix = layer_prefix(channels[i].name);
        std::string suffix = channel_suffix(channels[i].name);

        // Map "" prefix to display name "default"
        std::string display = prefix.empty() ? "default" : prefix;
        auto &li = layers[display];
        li.prefix = display;

        if (suffix == "R")
            li.idx_r = i;
        else if (suffix == "G")
            li.idx_g = i;
        else if (suffix == "B")
            li.idx_b = i;
    }

    // -----------------------------------------------------------------------
    // List mode
    // -----------------------------------------------------------------------
    if (list_mode)
    {
        printf("\nChannels in %s:\n", input_path);
        for (auto &ch : channels)
            printf("  %s\n", ch.name.c_str());

        printf("\nLayers (usable names for grading):\n");
        for (auto &[name, info] : layers)
        {
            printf("  %-12s  R:%s  G:%s  B:%s\n",
                   name.c_str(),
                   info.idx_r >= 0 ? channels[info.idx_r].name.c_str() : "(none)",
                   info.idx_g >= 0 ? channels[info.idx_g].name.c_str() : "(none)",
                   info.idx_b >= 0 ? channels[info.idx_b].name.c_str() : "(none)");
        }
        return 0;
    }

    // -----------------------------------------------------------------------
    // Validate output path
    // -----------------------------------------------------------------------
    if (!output_path)
    {
        fprintf(stderr, "Error: no output path specified (use -o output.tga).\n");
        print_usage(argv[0]);
        return 1;
    }

    // -----------------------------------------------------------------------
    // Validate layer multipliers
    // -----------------------------------------------------------------------
    for (auto &[name, mult] : layer_multipliers)
    {
        if (layers.find(name) == layers.end())
        {
            fprintf(stderr, "Warning: layer '%s' not found in EXR. Available layers:\n", name.c_str());
            for (auto &[n, _] : layers)
                fprintf(stderr, "  %s\n", n.c_str());
            return 1;
        }
    }

    // Default: if no layers specified, use the beauty pass at 1×
    if (layer_multipliers.empty())
    {
        layer_multipliers["default"] = 1.0f;
        printf("No layer arguments — using beauty pass (default) at 1.0x\n");
    }

    // Print grading plan
    printf("Compositing:\n");
    for (auto &[name, mult] : layer_multipliers)
        printf("  %-12s  x %.4f\n", name.c_str(), mult);
    printf("Exposure: %.2f  Tonemap: %.2f  Gamma: %.2f\n",
           exposure, tonemap_compression, gamma_val);

    // -----------------------------------------------------------------------
    // Composite: additive blend of selected layers with multipliers
    // -----------------------------------------------------------------------
    const size_t num_pixels = (size_t)width * height;
    std::vector<float> out_r(num_pixels, 0.0f);
    std::vector<float> out_g(num_pixels, 0.0f);
    std::vector<float> out_b(num_pixels, 0.0f);

    for (auto &[name, mult] : layer_multipliers)
    {
        auto &info = layers[name];
        const float *src_r = (info.idx_r >= 0) ? channels[info.idx_r].data.data() : nullptr;
        const float *src_g = (info.idx_g >= 0) ? channels[info.idx_g].data.data() : nullptr;
        const float *src_b = (info.idx_b >= 0) ? channels[info.idx_b].data.data() : nullptr;

        for (size_t i = 0; i < num_pixels; ++i)
        {
            if (src_r)
                out_r[i] += src_r[i] * mult;
            if (src_g)
                out_g[i] += src_g[i] * mult;
            if (src_b)
                out_b[i] += src_b[i] * mult;
        }
    }

    // -----------------------------------------------------------------------
    // Tone mapping / gamma -> 8-bit TGA
    // Same logarithmic tone mapper as the renderer and flaresim:
    //   output = log(1 + c * x) / log(1 + c)
    //   where c = 10^(tonemap_compression * 2) - 1
    // -----------------------------------------------------------------------
    const float c = std::pow(10.0f, tonemap_compression * 2.0f) - 1.0f;
    const float tm_norm = (c > 1e-6f) ? (1.0f / std::log(1.0f + c)) : 1.0f;
    const bool do_tonemap = (c > 1e-6f);
    const float inv_gamma = (gamma_val > 0.01f) ? (1.0f / gamma_val) : 1.0f;
    std::vector<uint8_t> pixels(num_pixels * 3);

    for (size_t i = 0; i < num_pixels; ++i)
    {
        // Apply exposure
        float r = std::max(out_r[i] * exposure, 0.0f);
        float g = std::max(out_g[i] * exposure, 0.0f);
        float b = std::max(out_b[i] * exposure, 0.0f);

        // Logarithmic tone mapping (matches renderer/flaresim)
        if (do_tonemap)
        {
            r = std::log(1.0f + c * r) * tm_norm;
            g = std::log(1.0f + c * g) * tm_norm;
            b = std::log(1.0f + c * b) * tm_norm;
        }

        // Clamp to [0,1]
        r = std::min(r, 1.0f);
        g = std::min(g, 1.0f);
        b = std::min(b, 1.0f);

        // Gamma
        if (std::abs(inv_gamma - 1.0f) > 1e-4f)
        {
            r = std::pow(r, inv_gamma);
            g = std::pow(g, inv_gamma);
            b = std::pow(b, inv_gamma);
        }

        pixels[i * 3 + 0] = (uint8_t)std::clamp((int)(r * 255.0f + 0.5f), 0, 255);
        pixels[i * 3 + 1] = (uint8_t)std::clamp((int)(g * 255.0f + 0.5f), 0, 255);
        pixels[i * 3 + 2] = (uint8_t)std::clamp((int)(b * 255.0f + 0.5f), 0, 255);
    }

    // -----------------------------------------------------------------------
    // Write TGA
    // -----------------------------------------------------------------------
    if (!stbi_write_tga(output_path, width, height, 3, pixels.data()))
    {
        fprintf(stderr, "Error: failed to write TGA '%s'\n", output_path);
        return 1;
    }

    printf("Wrote: %s (%d x %d)\n", output_path, width, height);
    return 0;
}
