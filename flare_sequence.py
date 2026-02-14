#!/usr/bin/env python3
"""
Run flaresim on every frame produced by render_sequence.py.

Usage:
    python flare_sequence.py [options]

Reads numbered EXR frames from the input directory (default: build/frames),
runs each through flaresim to add lens-flare layers, and writes the results
to an output directory (default: build/frames_flared).

Flaresim settings come from a config file (default: flaresim/flaresim.conf).
Any additional --key value pairs after -- are forwarded as CLI overrides.

Examples:
    python flare_sequence.py
    python flare_sequence.py --input-dir build/frames --output-dir build/frames_flared
    python flare_sequence.py --jobs 4 -- --flare_gain 5000 --bloom_strength 3.0
"""

import argparse
import glob
import os
import subprocess
import sys
import time as time_mod


def fmt_time(s: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    s = int(s)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run flaresim on a sequence of EXR frames",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="build/frames",
        help="Directory containing input EXR frames (default: build/frames)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="build/frames_flared",
        help="Directory for flared output frames (default: build/frames_flared)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="flaresim/flaresim.conf",
        help="Flaresim config file (default: flaresim/flaresim.conf)",
    )
    parser.add_argument(
        "--binary",
        type=str,
        default="flaresim/build/flaresim",
        help="Path to flaresim binary (default: flaresim/build/flaresim)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.exr",
        help="Glob pattern for input frames (default: *.exr)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel flaresim processes (default: 1)",
    )
    parser.add_argument(
        "--tga",
        action="store_true",
        help="Also write tonemapped TGA for each frame",
    )
    parser.add_argument(
        "--tga-dir",
        type=str,
        default=None,
        help="Directory for TGA outputs (default: <output-dir>_tga)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip frames whose output already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )

    # Everything after -- is forwarded to flaresim as extra CLI overrides
    args, extra = parser.parse_known_args()

    # Resolve paths relative to the project root (script's directory)
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(project_root, args.input_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    config = os.path.join(project_root, args.config)
    binary = os.path.join(project_root, args.binary)

    # Validate
    if not os.path.isfile(binary):
        print(f"Error: flaresim binary not found at {binary}", file=sys.stderr)
        print("Run 'cmake --build flaresim/build' first.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(config):
        print(f"Error: config file not found at {config}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(input_dir):
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover frames
    frames = sorted(glob.glob(os.path.join(input_dir, args.pattern)))
    if not frames:
        print(f"No files matching '{args.pattern}' in {input_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # TGA output directory
    tga_dir = None
    if args.tga:
        tga_dir = (
            os.path.join(project_root, args.tga_dir)
            if args.tga_dir
            else output_dir + "_tga"
        )
        os.makedirs(tga_dir, exist_ok=True)

    # Build work list: (input_path, output_path, tga_path_or_None)
    work = []
    for frame_path in frames:
        name = os.path.basename(frame_path)
        out_path = os.path.join(output_dir, name)
        tga_path = (
            os.path.join(tga_dir, os.path.splitext(name)[0] + ".tga")
            if tga_dir
            else None
        )
        if args.skip_existing and os.path.isfile(out_path):
            continue
        work.append((frame_path, out_path, tga_path))

    skipped = len(frames) - len(work)
    total = len(work)

    if total == 0:
        print("All frames already processed (use without --skip-existing to re-run).")
        return

    print(
        f"Flaresim sequence: {total} frames to process"
        + (f" ({skipped} skipped)" if skipped else "")
    )
    print(f"  Input:  {input_dir}/")
    print(f"  Output: {output_dir}/")
    if tga_dir:
        print(f"  TGA:    {tga_dir}/")
    print(f"  Config: {config}")
    if extra:
        print(f"  Extra:  {' '.join(extra)}")
    if args.jobs > 1:
        print(f"  Jobs:   {args.jobs}")
    print()

    if args.dry_run:
        for in_path, out_path, tga_path in work:
            cmd = [binary, config, "--input", in_path, "--output", out_path]
            if tga_path:
                cmd += ["--tga", tga_path]
            cmd += extra
            print(" ".join(cmd))
        return

    # --- Sequential execution (jobs == 1) ------------------------------------
    if args.jobs <= 1:
        wall_start = time_mod.monotonic()

        for i, (in_path, out_path, tga_path) in enumerate(work):
            frame_name = os.path.basename(in_path)
            frame_start = time_mod.monotonic()

            cmd = [binary, config, "--input", in_path, "--output", out_path]
            if tga_path:
                cmd += ["--tga", tga_path]
            cmd += extra

            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(config),
                capture_output=True,
                text=True,
            )

            frame_elapsed = time_mod.monotonic() - frame_start
            total_elapsed = time_mod.monotonic() - wall_start
            done = i + 1
            left = total - done
            avg = total_elapsed / done
            eta = avg * left

            print(
                f"[{done}/{total}] {frame_name}  "
                f"({frame_elapsed:.1f}s, avg {avg:.1f}s/frame, "
                f"elapsed {fmt_time(total_elapsed)}, ETA {fmt_time(eta)})"
            )

            if result.returncode != 0:
                print(f"  ERROR (exit {result.returncode}):", file=sys.stderr)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                sys.exit(1)
    else:
        # --- Parallel execution (jobs > 1) -----------------------------------
        from concurrent.futures import ProcessPoolExecutor, as_completed

        def run_one(idx_in_out):
            idx, in_path, out_path, tga_path = idx_in_out
            cmd = [binary, config, "--input", in_path, "--output", out_path]
            if tga_path:
                cmd += ["--tga", tga_path]
            cmd += extra
            t0 = time_mod.monotonic()
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(config),
                capture_output=True,
                text=True,
            )
            elapsed = time_mod.monotonic() - t0
            return (
                idx,
                os.path.basename(in_path),
                elapsed,
                result.returncode,
                result.stderr,
            )

        wall_start = time_mod.monotonic()
        done = 0

        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(run_one, (i, inp, outp, tga)): i
                for i, (inp, outp, tga) in enumerate(work)
            }

            for future in as_completed(futures):
                idx, name, elapsed, rc, stderr = future.result()
                done += 1
                total_elapsed = time_mod.monotonic() - wall_start
                left = total - done
                avg = total_elapsed / done
                eta = avg * left

                print(
                    f"[{done}/{total}] {name}  "
                    f"({elapsed:.1f}s, avg {avg:.1f}s/frame, "
                    f"elapsed {fmt_time(total_elapsed)}, ETA {fmt_time(eta)})"
                )

                if rc != 0:
                    print(f"  ERROR (exit {rc}):", file=sys.stderr)
                    if stderr:
                        print(stderr, file=sys.stderr)
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    sys.exit(1)

    print(f"\nDone! {total} flared frames written to {output_dir}/")


if __name__ == "__main__":
    main()
