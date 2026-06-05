#!/usr/bin/env python3
"""Utility to create or update otf_state.json from Slurm output files.

Greps for "Selected/Selecting structure with gamma = " lines and reconstructs
the gamma_max0_history in the OTF state file.
"""

import argparse
import json
import os
import re
import sys

import numpy

OTF_STATE_FILE = "otf_state.json"

# Matches both:
#   "Selected structure with gamma =  1.2345"  (print with comma, two spaces)
#   "Selecting structure with gamma = 1.2345"  (f-string, one space)
GAMMA_LINE_RE = re.compile(r"[Ss]elect(?:ed|ing) structure with gamma =\s+([\d.eE+\-]+)")


def parse_gammas(path):
    gammas = []
    with open(path) as f:
        for line in f:
            m = GAMMA_LINE_RE.search(line)
            if m:
                gammas.append(float(m.group(1)))
    return gammas


def apply_gamma_obs(state, obs, floor, window, percentile, factor):
    history = state.get("gamma_max0_history", [])
    history = (history + [float(obs)])[-window:]
    gamma_max0_new = max(factor * numpy.percentile(history, percentile), floor)
    state["gamma_max0_history"] = history
    state["gamma_max0"] = gamma_max0_new
    return gamma_max0_new


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create or update otf_state.json by extracting gamma observations "
            "from Slurm output files."
        )
    )
    parser.add_argument(
        "slurm_outputs", nargs="*",
        help="Slurm output file(s) to parse (processed in order)")
    parser.add_argument(
        "-o", "--output", default=OTF_STATE_FILE,
        help=f"Path to the state file (default: {OTF_STATE_FILE})")
    parser.add_argument(
        "--update", action="store_true",
        help="Append to existing state instead of starting fresh")
    parser.add_argument(
        "--gamma_max0_floor", type=float, required=True,
        help="Floor for gamma_max0 — should match --gamma_max used in the OTF run")
    parser.add_argument("--gamma_max0_window", type=int, default=5,
                        help="Rolling window size (default: 5)")
    parser.add_argument("--gamma_max0_percentile", type=float, default=75,
                        help="Percentile for threshold (default: 75)")
    parser.add_argument("--gamma_max0_factor", type=float, default=1.2,
                        help="Multiplication factor (default: 1.2)")
    args = parser.parse_args()

    # Default to slurm-*.out sorted numerically by the job number
    if not args.slurm_outputs:
        import glob
        _slurm_re = re.compile(r"slurm-(\d+)\.out$")
        args.slurm_outputs = sorted(
            glob.glob("slurm-*.out"),
            key=lambda p: int(_slurm_re.search(p).group(1)) if _slurm_re.search(p) else 0,
        )
        if args.slurm_outputs:
            print(f"No files specified — using: {args.slurm_outputs}")
        else:
            print("No slurm-*.out files found in current directory.", file=sys.stderr)

    # Load or initialise state
    if args.update and os.path.isfile(args.output):
        with open(args.output) as f:
            state = json.load(f)
        print(f"Loaded existing state from {args.output}")
        print(f"  gamma_max0_history = {state.get('gamma_max0_history', [])}")
        print(f"  gamma_max0         = {state.get('gamma_max0')}")
    else:
        if args.update:
            print(f"State file {args.output} not found — starting fresh.")
        state = {}

    # Collect gamma observations from all files in order
    all_gammas = []
    for path in args.slurm_outputs:
        if not os.path.isfile(path):
            print(f"Warning: {path} not found, skipping.", file=sys.stderr)
            continue
        gammas = parse_gammas(path)
        print(f"{path}: {len(gammas)} gamma observation(s) found")
        all_gammas.extend(gammas)

    print(f"Total observations to apply: {len(all_gammas)}")

    for obs in all_gammas:
        apply_gamma_obs(
            state, obs,
            floor=args.gamma_max0_floor,
            window=args.gamma_max0_window,
            percentile=args.gamma_max0_percentile,
            factor=args.gamma_max0_factor,
        )

    if all_gammas:
        print(f"Updated gamma_max0_history = {state['gamma_max0_history']}")
        print(f"Updated gamma_max0         = {state['gamma_max0']:.4f}")
    elif not state:
        print("No observations found and no existing state — writing empty state file.")

    with open(args.output, "w") as f:
        json.dump(state, f, indent=2)
    print(f"State written to {args.output}")


if __name__ == "__main__":
    main()
