#!/usr/bin/env python3
"""
Render a sequence of black hole frames with an animated accretion disk.

Usage:
    python render_sequence.py [options]

Produces numbered frames in the build directory by incrementing
the 'time' parameter in scene.txt between renders.

Supports TGA (default), EXR (linear HDR), and HDR (Radiance) output formats.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time as time_mod

# Map format name → (scene key, extension, keys to disable)
FORMAT_TABLE = {
    "tga": ("output_file", ".tga", ["exr_output", "hdr_output", "jpg_output"]),
    "exr": ("exr_output", ".exr", ["output_file", "hdr_output", "jpg_output"]),
    "hdr": ("hdr_output", ".hdr", ["output_file", "exr_output", "jpg_output"]),
}


def patch_scene_file(src: str, dst: str, overrides: dict[str, str]) -> None:
    """Copy a scene file, overriding specific key = value lines."""
    with open(src) as f:
        lines = f.readlines()

    with open(dst, "w") as f:
        for line in lines:
            stripped = line.split("#")[0].strip()
            matched = False
            for key, val in overrides.items():
                if re.match(rf"^{re.escape(key)}\s*=", stripped):
                    # Preserve alignment: replace everything after '='
                    f.write(
                        re.sub(r"=\s*\S+", f"= {val}", line.split("#")[0]).rstrip()
                        + "\n"
                    )
                    matched = True
                    break
            if not matched:
                f.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an animated black hole sequence"
    )
    parser.add_argument(
        "-n",
        "--num-frames",
        type=int,
        default=60,
        help="Number of frames to render (default: 60)",
    )
    parser.add_argument(
        "-dt",
        "--time-step",
        type=float,
        default=0.5,
        help="Time increment per frame (default: 0.5)",
    )
    parser.add_argument(
        "-t0",
        "--time-start",
        type=float,
        default=0.0,
        help="Starting time value (default: 0.0)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="scene.txt",
        help="Base scene file to use (default: scene.txt)",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default="build",
        help="Build directory containing the bhrt3 binary (default: build)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="frames",
        help="Directory for output frames (default: frames)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Output filename prefix (default: frame)",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=list(FORMAT_TABLE.keys()),
        default="exr",
        help="Output format: tga, exr, or hdr (default: exr)",
    )
    args = parser.parse_args()

    fmt_key, fmt_ext, fmt_disable = FORMAT_TABLE[args.format]

    # Resolve paths relative to the script's directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(project_root, args.build_dir)
    scene_src = os.path.join(build_dir, args.scene)
    output_dir = os.path.join(build_dir, args.output_dir)
    binary = os.path.join(build_dir, "bhrt3")

    if not os.path.isfile(binary):
        print(f"Error: binary not found at {binary}", file=sys.stderr)
        print("Run 'cmake --build build' first.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(scene_src):
        print(f"Error: scene file not found at {scene_src}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    temp_scene = os.path.join(build_dir, "_render_sequence_tmp.txt")

    print(
        f"Rendering {args.num_frames} frames: t={args.time_start} to "
        f"t={args.time_start + (args.num_frames - 1) * args.time_step:.2f}, "
        f"dt={args.time_step}"
    )
    print(f"Output: {output_dir}/{args.prefix}_NNNN{fmt_ext} ({args.format.upper()})")
    print()

    try:
        wall_start = time_mod.monotonic()

        for i in range(args.num_frames):
            t = args.time_start + i * args.time_step
            out_file = os.path.join(args.output_dir, f"{args.prefix}_{i:04d}{fmt_ext}")

            overrides = {
                "time": f"{t:.6f}",
                fmt_key: out_file,
            }
            # Disable other output formats to avoid redundant writes
            for key in fmt_disable:
                overrides[key] = ""

            patch_scene_file(
                scene_src,
                temp_scene,
                overrides,
            )

            frame_start = time_mod.monotonic()

            result = subprocess.run(
                [binary, temp_scene],
                cwd=build_dir,
                capture_output=True,
                text=True,
            )

            frame_elapsed = time_mod.monotonic() - frame_start
            total_elapsed = time_mod.monotonic() - wall_start
            frames_done = i + 1
            frames_left = args.num_frames - frames_done
            avg_per_frame = total_elapsed / frames_done
            eta_secs = avg_per_frame * frames_left

            # Format elapsed and ETA as H:MM:SS or M:SS
            def fmt_time(s: float) -> str:
                s = int(s)
                if s >= 3600:
                    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
                return f"{s // 60}:{s % 60:02d}"

            print(
                f"[{frames_done}/{args.num_frames}] time={t:.4f}  ->  {out_file}  "
                f"({frame_elapsed:.1f}s, avg {avg_per_frame:.1f}s/frame, "
                f"elapsed {fmt_time(total_elapsed)}, ETA {fmt_time(eta_secs)})"
            )

            if result.returncode != 0:
                print(f"  ERROR (exit code {result.returncode}):", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                sys.exit(1)

    finally:
        # Clean up temp file
        if os.path.exists(temp_scene):
            os.remove(temp_scene)

    print(f"\nDone! {args.num_frames} frames written to {output_dir}/")


if __name__ == "__main__":
    main()
