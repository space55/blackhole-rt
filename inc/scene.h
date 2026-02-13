#ifndef _BH_SCENE_H
#define _BH_SCENE_H

#include <string>

// ---------------------------------------------------------------------------
// Scene configuration — all tweakable parameters in one place
// ---------------------------------------------------------------------------
struct scene_config_s
{
    // Output
    int output_width = 1024;
    int output_height = 512;
    std::string output_file = "output.tga";
    std::string hdr_output = ""; // empty = disabled; set to e.g. "output.hdr" to write Radiance HDR
    std::string exr_output = ""; // empty = disabled; set to e.g. "output.exr" to write OpenEXR
    std::string jpg_output = ""; // empty = disabled; set to e.g. "output.jpg" to write JPEG thumbnail

    // Sky map
    std::string sky_image = "hubble-skymap.jpg";
    double sky_brightness = 1.0;
    double sky_pitch = 0.0;
    double sky_yaw = 0.0;
    double sky_roll = 0.0;
    double sky_offset_u = 0.0;
    double sky_offset_v = 0.0;

    // Camera
    double camera_x = -25.0;
    double camera_y = 5.0;
    double camera_z = 0.0;
    double camera_pitch = 10.0;
    double camera_yaw = 90.0;
    double camera_roll = 0.0;
    double fov_x = 360.0;
    double fov_y = 180.0;

    // Black hole
    double bh_mass = 1.0;
    double bh_spin = 0.999;

    // Ray integration
    double base_dt = 0.1;
    double max_affine = 100.0;
    double escape_radius = 50.0;

    // Accretion disk geometry
    double disk_outer_r = 20.0;
    double disk_thickness = 0.5;
    double disk_density = 20.0;
    double disk_opacity = 0.5;

    // Accretion disk appearance
    double disk_emission_boost = 10.0;
    double disk_color_variation = 0.7;
    double disk_turbulence = 0.0;

    // Flat disk mode: 0 = normal volumetric disk, 1 = thin/flat with rich texture & high opacity
    int disk_flat_mode = 0;

    // Tone mapping
    double tonemap_compression = 1.0;
    double exposure = 1.0; // output exposure multiplier (>1 blows out brights, lifts shadows)

    // Anti-aliasing
    int aa_samples = 1; // NxN supersampling grid per pixel (1=off, 2=4spp, 3=9spp, 4=16spp)

    // Bloom / lens flare
    double bloom_strength = 0.0;  // 0 = off, 0.3 = subtle, 1.0 = heavy glow
    double bloom_threshold = 0.6; // HDR brightness above which bloom triggers
    double bloom_radius = 0.02;   // blur radius as fraction of image diagonal

    // Animation
    double time = 0.0; // frame time for disk rotation / animation
};

// Load a scene description from a key=value text file.
// Missing keys keep their default values.  Returns false on file-open error.
bool load_scene_config(const char *path, scene_config_s &cfg);

// Print the loaded configuration to stdout.
void print_scene_config(const scene_config_s &cfg);

#endif
