# vesuvius-mps-inference

Ink detection inference in [villa](https://github.com/ScrollPrize/villa) runs on the CPU on
Apple Silicon while the GPU sits idle. It prints no error and no warning. This repository
fixes it and measures the result: **2.67x** on an M5 Pro, with output that matches the CPU
path to within one uint8 level.

I could not find an open issue for this, so as far as I can tell it is unreported.

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

Real checkpoint `hybrid_3d2d-seed42/step-075000.pth` from
[`scrollprize/ink_9um`](https://huggingface.co/scrollprize/ink_9um) (132 MB,
`patch_size [17,128,128]`, `mixed_precision fp16`), through the documented CLI, on an M5 Pro
with 24 GB unified memory, macOS 26.5.1, torch 2.14.0:

| `--device` | throughput | log line |
| --- | --- | --- |
| `mps` | **33.91 block/s** | `mps autocast enabled for inference with dtype=float16` |
| `cpu` | 12.70 block/s | `Autocast disabled for inference (device=cpu)` |

The `cpu` row is what this machine did before the patch. Comparing the two output TIFFs pixel
by pixel gives `maxdiff=1` on uint8 and a mean absolute difference of 0.084, with no pixel
differing by more than one level. That size of gap is what fp16 accumulation produces.

Two honest caveats. The input was a 512x512 synthetic volume giving 49 blocks, so model load
and imports dominate wall clock (23.1 s against 25.4 s) and the throughput row is the number
that means anything. A real segment amortises setup better, so read 2.67x as a floor. And I
ran with `--no-compile` throughout; `torch.compile` does work on MPS here, but I have not
tested it in combination.

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

21 tests. The CPU cases and the error paths run anywhere, so CI is meaningful on a plain ubuntu
runner. The MPS cases skip unless Metal is present, so a green CI badge on its own does not
prove the MPS path works. That part was checked on hardware:

- Linux x86_64, no CUDA, no MPS: 14 passed, 7 skipped
- M5 Pro, macOS 26.5.1, MPS: 20 passed, 1 skipped

Between the two, every test runs somewhere.

### The CUDA path is the gap

I have no machine with a usable CUDA GPU, so I could not run the CUDA branch. It is the
default path, and this patch touches it, so here is exactly what I did instead.

`select_device()` keeps the original control flow for CUDA and reproduces the two error
strings character for character, including `visible device count is N`. Five tests in
`TestCudaDecisionsWithoutCudaHardware` fake `torch.cuda.is_available` and
`torch.cuda.device_count` to check the decisions: cuda outranks mps under `auto`, `--gpus 2,3`
resolves to `cuda:2`, an out-of-range ordinal reports the visible count, `--device cpu`
overrides available CUDA, and the missing-CUDA message is byte-identical to the old one.

Those tests cover the resolver's logic on its own. Confirming that torch still behaves on real
CUDA hardware needs a CUDA box, and running `pytest tests/ -v` on one is the check I am missing.

## Licence

MIT.
