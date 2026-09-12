# vesuvius-mps-inference

[![tests](https://github.com/SurgeFok/vesuvius-mps-inference/actions/workflows/tests.yml/badge.svg)](https://github.com/SurgeFok/vesuvius-mps-inference/actions/workflows/tests.yml)

Ink detection inference in [villa](https://github.com/ScrollPrize/villa) runs on the CPU on
Apple Silicon while the GPU sits idle. It prints no error and no warning. This repository
fixes it. On a full PHerc0800 surface volume, the MPS path ran the inference loop **5.97x**
faster and the whole command **2.42x** faster than the CPU path on the same M5 Pro. The two
output images match to within one uint8 level.

Reported upstream as [ScrollPrize/villa#1764](https://github.com/ScrollPrize/villa/issues/1764).

## The bug

`ink-detection/koine_machines/inference/infer.py`, in `prepare_model_for_inference`:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

`infer_full3d_tifxyz.py` has the same line in `prepare_model`. There is no MPS branch, so on
any Mac both entrypoints take the CPU path. Autocast is gated the same way:

```python
autocast_enabled = bool(amp and device.type == "cuda")
...
torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype)
```

So a Mac user who follows the [First Letters workflow](https://scrollprize.substack.com/p/from-ct-scan-to-ancient-text-a-first)
downloads the published `ink_9um` checkpoints, runs the documented command, gets correct
output, and never learns that the run took several times longer than it needed to. The
guide's only tuning advice is to lower `--batch-size` if you run out of GPU memory, which
does not apply on the path they are actually on.

## Villa already had the answer

`vesuvius/src/vesuvius/models/utilities/get_accelerator.py` resolves cuda, then mps, then cpu,
and logs `Using MPS device (Apple Silicon)`. nnU-Net under `segmentation/models/arch/nnunet`
handles MPS too. The ink-detection entrypoints are a separate package (`koine_machines`) and
predate that helper, so they never picked it up.

`accelerator.py` here is that helper's logic, put where `koine_machines` can reach it, plus the
autocast set and a `--device` flag.

## The fix

| File | Change |
| --- | --- |
| `koine_machines/common/accelerator.py` | new. `select_device()` (cuda, mps, cpu), `AUTOCAST_DEVICE_TYPES`, `DEVICE_CHOICES`, `log_device()` |
| `inference/infer.py` | device via `select_device()`, `device_type=device.type` for autocast, new `--device`, CUDA-specific log wording made backend-neutral |
| `inference/infer_full3d_tifxyz.py` | same, with `device_preference` threaded through `prepare_model()` |

`--gpus` keeps its old meaning and its old errors. It indexes CUDA ordinals and feeds
`nn.DataParallel`, so it stays CUDA-only, and asking for `--gpus 0 --device mps` is now a
clear error instead of a confusing one.

Applied diff: [`mps-inference.patch`](mps-inference.patch), against `merge-ink-pipelines`.

## Measured, by running it

Reported the way villa's own `AGENTS.md` section 1.4 asks for performance work: command line,
input, build type, iteration count, and summary statistics rather than one number.

**Input:** a 512x512x20 uint8 zarr, giving 49 blocks per run.
**Checkpoint:** `hybrid_3d2d-seed42/step-075000.pth` from
[`scrollprize/ink_9um`](https://huggingface.co/scrollprize/ink_9um), 132 MB,
`patch_size [17,128,128]`, `mixed_precision fp16`.
**Machine:** Apple M5 Pro, 24 GB unified memory (17.8 GiB recommended MPS budget),
macOS 26.5.1, torch 2.14.0.
**Command:**

```bash
python -m koine_machines.inference.infer input.zarr step-075000.pth out.tif \
    --batch-size 1 --no-compile --device {mps|cpu}
```

**Iterations:** 5 timed runs per device, one warm-up run discarded first. The figure is the
inference loop's own throughput, so model load and imports are excluded.

| `--device` | blocks | min | median | max | mean | stdev | wall median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mps` | 49 | 33.79 | **33.88** | 33.99 | 33.88 | 0.08 | 25.13 s |
| `cpu` | 49 | 12.65 | **12.83** | 13.18 | 12.88 | 0.22 | 27.42 s |

Median against median, **2.64x**. The spreads do not overlap and are nowhere near each other,
so the gap is not a sampling artefact. MPS is also the steadier of the two, at 0.08 stdev
against 0.22.

The `cpu` row is what this machine did before the patch. Log lines separate them:
`mps autocast enabled for inference with dtype=float16` against
`Autocast disabled for inference (device=cpu)`.

Reproduce with [`bench.py`](bench.py):

```bash
python bench.py --input input.zarr --checkpoint step-075000.pth --devices mps cpu --repeats 5
```

**Output equivalence.** Comparing the two output TIFFs pixel by pixel gives `maxdiff=1` on
uint8 and a mean absolute difference of 0.084, with no pixel differing by more than one level.
That size of gap is what fp16 accumulation produces.

**Two caveats.** Wall clock is dominated by model load and imports at this input size (25.13 s
against 27.42 s median), which is why the throughput column is the one that means anything.
The full-segment follow-up below measures how that changes on real data. Every run used
`--no-compile`. `torch.compile` does work on MPS here, but I have not tested it together with
autocast.

### Full PHerc0800 segment follow-up

I repeated the comparison on the public PHerc0800 segment
`20251028222030-auto_grown_20251028222030940`, using its
`8.64um-1.2m-116keV-volume-20250521135224.zarr` surface volume. Level 0 is a
31x2580x2580 uint8 array. The checkpoint, command options, Mac, and measurement protocol are
the same as above: one discarded warm-up, then five timed runs per device with batch size 1
and compilation disabled. Each run scheduled 923 blocks.

| `--device` | min | median | p95 | max | mean | stdev | wall median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mps` | 97.66 | **97.94** | 97.97 | 97.97 | 97.89 | 0.13 | **32.92 s** |
| `cpu` | 16.25 | **16.40** | 16.89 | 16.91 | 16.55 | 0.29 | **79.67 s** |

Median against median, the inference loop is **5.97x** faster on MPS. Median wall time falls
from 79.67 to 32.92 seconds, a **2.42x** end-to-end speedup. All six runs on each backend,
including the warm-up, produced the same SHA-256 within that backend. Comparing the retained
MPS and CPU TIFFs across 6,656,400 pixels gives `maxdiff=1`, mean absolute difference 0.0277,
and zero pixels differing by more than one level.

As a separate CUDA compatibility check, I ran the patched code with the same input,
checkpoint, batch size, and measurement protocol on a Linux workstation with an 8 GB NVIDIA
GTX 1070. That host used driver 570.211.01 and torch 2.2.0+cu118. CUDA measured 49.29 to 50.66
blocks/s, with a median of 50.31 blocks/s and a median whole-command time of 22.25 seconds.
All six CUDA outputs had the same SHA-256. Against the retained Mac outputs, the CUDA TIFF
had `maxdiff=2` and mean absolute differences of 0.0298 from MPS and 0.0343 from CPU.

The CUDA result is not part of the MPS-versus-CPU speedup calculation because it came from a
different machine and torch build. It verifies that the unchanged CUDA path still completes
the same full-segment workload and produces numerically equivalent output.

The exact public input path is:

```text
s3://vesuvius-challenge-open-data/PHerc0800/segments/20251028222030-auto_grown_20251028222030940/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/
```

The per-run measurements, hashes, environment, and output comparison are in
[`benchmarks/full-pherc0800/results.json`](benchmarks/full-pherc0800/results.json).

## What I did not change

Both entrypoints keep `pin_memory` gated on `device.type == "cuda"`. Setting it under MPS makes
torch emit `'pin_memory' argument is set as true but not supported on MPS now, device pinned`
on every loader iteration, and pins nothing. `tests/test_accelerator.py` asserts that warning
still appears, so if a later torch adds support the test fails and the gate can be widened.

CPU autocast is also left off. It is bfloat16-only, and on the machines that reach the CPU
branch at all it is not reliably faster than plain float32.

The `koine_machines/inference/container/` tree is untouched. It targets CUDA k8s nodes by
design.

## Reproducing the run

You need `vesuvius/src` on `PYTHONPATH` and six packages that ink-detection's own dependency
list does not pull in: `requests`, `pynrrd`, `opencv-python-headless`, `imagecodecs`,
`dynamic-network-architectures`, `batchgenerators`. Install `imagecodecs` before you start.
Without it the run finishes inference and then fails writing the LZW output.

```bash
PYTHONPATH=/path/to/villa/vesuvius/src python -m koine_machines.inference.infer \
    input.zarr step-075000.pth out.tif --batch-size 1 --no-compile --device mps
```

## Tests

```bash
pip install pytest torch
python -m pytest tests/ -v
```

24 tests. The CPU cases and the error paths run anywhere, so CI is meaningful on a plain ubuntu
runner. The MPS cases skip unless Metal is present, so a green CI badge on its own does not
prove the hardware paths work. Both were checked on their respective hardware:

- Linux x86_64, no CUDA, no MPS: 14 passed, 10 skipped
- M5 Pro, macOS 26.5.1, MPS: 20 passed, 4 skipped
- GTX 1070, Ubuntu 24.04.3, CUDA 11.8: 17 passed, 7 skipped

Between the three, every test runs somewhere.

### CUDA hardware result

The CUDA run used an NVIDIA GeForce GTX 1070 (8 GB, compute capability 6.1), driver
570.211.01, torch 2.2.0+cu118 and Python 3.12.3. The full suite completed in 1.88 seconds:
17 passed and 7 skipped. The skips are the six MPS-only cases and the test that requires CUDA
to be absent.

Three hardware-gated tests cover the path the entrypoints use. `auto` and explicit `cuda`
both resolve to CUDA; `--gpus 0` resolves to `cuda:0`; and a real 3D convolution under CUDA
fp16 autocast produces finite output matching the fp32 reference within the pinned tolerance.
The pre-existing out-of-range ordinal test also ran against the real one-device count.

The five fake-backend tests remain because they cover decisions a one-GPU machine cannot
produce, such as selecting ordinals 2 and 3 and reporting a visible count of two. The hardware
run complements those tests rather than replacing them.

## Licence

MIT.
