#!/usr/bin/env python3
"""Repeat-run benchmark for the ink-detection inference entrypoint.

villa's AGENTS.md section 1.4 asks performance work to report the command line,
the input, the build type, iteration counts and summary stats rather than a
single number. This produces that.

The figure taken from each run is the inference loop's own throughput, as
tqdm reports it on the final progress line, so model load and imports are
excluded. Wall clock is recorded alongside it for context.

    python bench.py --input in.zarr --checkpoint step-075000.pth \
        --devices mps cpu --repeats 5
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RATE = re.compile(r"(\d+)/\1\s+\[[^\]]*?,\s*([\d.]+)\s*block/s\]")


def one_run(input_zarr, checkpoint, device, extra):
    out = Path(tempfile.mkdtemp(prefix="bench_")) / "out.tif"
    cmd = [
        sys.executable, "-m", "koine_machines.inference.infer",
        str(input_zarr), str(checkpoint), str(out),
        "--batch-size", "1", "--no-compile", "--device", device, *extra,
    ]
    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.monotonic() - started
    if proc.returncode != 0:
        sys.exit(f"run failed on {device}:\n{proc.stderr[-1500:]}")
    matches = RATE.findall(proc.stdout + proc.stderr)
    if not matches:
        sys.exit(f"no throughput line found for {device}")
    blocks, rate = matches[-1]
    return int(blocks), float(rate), wall, cmd


def summarise(values):
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--devices", nargs="+", default=["mps", "cpu"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("extra", nargs="*", default=[])
    args = ap.parse_args()

    import torch
    print(f"torch {torch.__version__}  repeats={args.repeats} (+{args.warmup} discarded warmup)")

    results = {}
    for device in args.devices:
        for _ in range(args.warmup):
            one_run(args.input, args.checkpoint, device, args.extra)
        rates, walls, blocks, cmd = [], [], None, None
        for _ in range(args.repeats):
            blocks, rate, wall, cmd = one_run(args.input, args.checkpoint, device, args.extra)
            rates.append(rate)
            walls.append(wall)
        results[device] = (summarise(rates), summarise(walls), blocks)
        print(f"  {device}: " + " ".join(f"{r:.2f}" for r in rates))
        print(f"    command: {' '.join(cmd)}")

    print(f"\n{'device':<8}{'blocks':>7}{'min':>9}{'median':>9}{'max':>9}{'mean':>9}{'stdev':>9}   wall median")
    for device, (rate, wall, blocks) in results.items():
        print(f"{device:<8}{blocks:>7}{rate['min']:>9.2f}{rate['median']:>9.2f}"
              f"{rate['max']:>9.2f}{rate['mean']:>9.2f}{rate['stdev']:>9.2f}   {wall['median']:.2f}s")
    if len(results) == 2 and "cpu" in results:
        other = [d for d in results if d != "cpu"][0]
        fast, slow = results[other][0]["median"], results["cpu"][0]["median"]
        print(f"\n{other} median / cpu median = {fast / slow:.2f}x")


if __name__ == "__main__":
    main()
