#include "scene.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Trim leading/trailing whitespace
// ---------------------------------------------------------------------------
static std::string trim(const std::string &s)
{
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
        return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// ---------------------------------------------------------------------------
// Load scene config from a key = value text file
// ---------------------------------------------------------------------------
bool load_scene_config(const char *path, scene_config_s &cfg)
{
    std::ifstream file(path);
    if (!file.is_open())
        return false;

    std::unordered_map<std::string, std::string> kv;

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

    // Helper lambdas
    auto get_int = [&](const char *key, int &out)
    {
        auto it = kv.find(key);
        if (it != kv.end())
            out = std::stoi(it->second);
    };
    auto get_double = [&](const char *key, double &out)
    {
        auto it = kv.find(key);
        if (it != kv.end())
            out = std::stod(it->second);
    };
    auto get_string = [&](const char *key, std::string &out)
    {
        auto it = kv.find(key);
        if (it != kv.end())
            out = it->second;
    };

    // Output
    get_int("output_width", cfg.output_width);
    get_int("output_height", cfg.output_height);
    get_string("output_file", cfg.output_file);
    get_string("hdr_output", cfg.hdr_output);

    // Sky
    get_string("sky_image", cfg.sky_image);
    get_double("sky_brightness", cfg.sky_brightness);
    get_double("sky_pitch", cfg.sky_pitch);
    get_double("sky_yaw", cfg.sky_yaw);
    get_double("sky_roll", cfg.sky_roll);
    get_double("sky_offset_u", cfg.sky_offset_u);
    get_double("sky_offset_v", cfg.sky_offset_v);

    // Camera
    get_double("camera_x", cfg.camera_x);
    get_double("camera_y", cfg.camera_y);
    get_double("camera_z", cfg.camera_z);
    get_double("camera_pitch", cfg.camera_pitch);
    get_double("camera_yaw", cfg.camera_yaw);
    get_double("camera_roll", cfg.camera_roll);
    get_double("fov_x", cfg.fov_x);
    get_double("fov_y", cfg.fov_y);

    // Black hole
    get_double("bh_mass", cfg.bh_mass);
    get_double("bh_spin", cfg.bh_spin);

    // Ray integration
    get_double("base_dt", cfg.base_dt);
    get_double("max_affine", cfg.max_affine);
    get_double("escape_radius", cfg.escape_radius);

    // Disk geometry
    get_double("disk_outer_r", cfg.disk_outer_r);
    get_double("disk_thickness", cfg.disk_thickness);
    get_double("disk_density", cfg.disk_density);
    get_double("disk_opacity", cfg.disk_opacity);

    // Disk appearance
    get_double("disk_emission_boost", cfg.disk_emission_boost);
    get_double("disk_color_variation", cfg.disk_color_variation);
    get_double("disk_turbulence", cfg.disk_turbulence);

    // Tone mapping
    get_double("tonemap_compression", cfg.tonemap_compression);

    // Anti-aliasing
    get_int("aa_samples", cfg.aa_samples);
    if (cfg.aa_samples < 1)
        cfg.aa_samples = 1;

    // Bloom / lens flare
    get_double("bloom_strength", cfg.bloom_strength);
    get_double("bloom_threshold", cfg.bloom_threshold);
    get_double("bloom_radius", cfg.bloom_radius);

    // Animation
    get_double("time", cfg.time);

    return true;
}

// ---------------------------------------------------------------------------
// Pretty-print the config
// ---------------------------------------------------------------------------
void print_scene_config(const scene_config_s &cfg)
{
    printf("=== Scene Configuration ===\n");
    printf("  Output:    %d x %d  ->  %s\n", cfg.output_width, cfg.output_height, cfg.output_file.c_str());
    if (!cfg.hdr_output.empty())
        printf("  HDR out:   %s\n", cfg.hdr_output.c_str());
    printf("  Sky image: %s  brightness=%.2f  rot=(%.1f, %.1f, %.1f)  offset=(%.2f, %.2f)\n",
           cfg.sky_image.c_str(), cfg.sky_brightness,
           cfg.sky_pitch, cfg.sky_yaw, cfg.sky_roll,
           cfg.sky_offset_u, cfg.sky_offset_v);
    printf("  Camera:    pos=(%.2f, %.2f, %.2f)  rot=(%.1f, %.1f, %.1f)  fov=(%.1f, %.1f)\n",
           cfg.camera_x, cfg.camera_y, cfg.camera_z,
           cfg.camera_pitch, cfg.camera_yaw, cfg.camera_roll,
           cfg.fov_x, cfg.fov_y);
    printf("  Black hole: M=%.3f  a=%.4f\n", cfg.bh_mass, cfg.bh_spin);
    printf("  Integration: dt=%.3f  max_affine=%.1f  escape_r=%.1f\n",
           cfg.base_dt, cfg.max_affine, cfg.escape_radius);
    printf("  Disk:      outer_r=%.1f  thickness=%.2f  density=%.1f  opacity=%.2f\n",
           cfg.disk_outer_r, cfg.disk_thickness, cfg.disk_density, cfg.disk_opacity);
    printf("  Disk look: emission_boost=%.1f  color_variation=%.2f  turbulence=%.2f\n",
           cfg.disk_emission_boost, cfg.disk_color_variation, cfg.disk_turbulence);
    printf("  Tonemap:   compression=%.2f\n", cfg.tonemap_compression);
    printf("  AA:        %dx%d = %d samples/pixel\n", cfg.aa_samples, cfg.aa_samples, cfg.aa_samples * cfg.aa_samples);
    printf("  Bloom:     strength=%.2f  threshold=%.2f  radius=%.3f\n",
           cfg.bloom_strength, cfg.bloom_threshold, cfg.bloom_radius);
    printf("  Animation: time=%.4f\n", cfg.time);
    printf("===========================\n");
}
