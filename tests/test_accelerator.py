"""Tests for the accelerator resolver.

The CPU-only cases run everywhere, including CI. The MPS and CUDA cases skip
unless the backend is actually present, so a green CI run does not by itself
prove the MPS path works. That was checked on hardware; see the README.
"""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from accelerator import (  # noqa: E402
    AUTOCAST_DEVICE_TYPES,
    DEVICE_CHOICES,
    autocast_supported,
    available_device_types,
    log_device,
    mps_available,
    select_device,
)

HAVE_CUDA = torch.cuda.is_available()
HAVE_MPS = mps_available()


def test_auto_prefers_cuda_then_mps_then_cpu():
    expected = "cuda" if HAVE_CUDA else ("mps" if HAVE_MPS else "cpu")
    assert select_device("auto").type == expected


def test_cpu_is_always_reachable():
    assert select_device("cpu").type == "cpu"


def test_available_device_types_always_ends_with_cpu():
    types = available_device_types()
    assert types[-1] == "cpu"
    assert ("cuda" in types) == HAVE_CUDA
    assert ("mps" in types) == HAVE_MPS


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="Unsupported device"):
        select_device("tpu")


def test_device_choices_match_the_resolver():
    assert DEVICE_CHOICES == ("auto", "cuda", "mps", "cpu")
    for choice in DEVICE_CHOICES:
        if choice == "auto":
            continue
        try:
            assert select_device(choice).type == choice
        except ValueError as exc:
            assert "not available" in str(exc)


def test_autocast_set_excludes_cpu():
    # CPU autocast is bfloat16-only and is not a win on the machines that reach
    # the CPU branch, so the entrypoints must not switch it on there.
    assert AUTOCAST_DEVICE_TYPES == {"cuda", "mps"}
    assert not autocast_supported(torch.device("cpu"))
    assert autocast_supported(torch.device("cuda"))
    assert autocast_supported(torch.device("mps"))


def test_log_device_never_raises():
    log_device(select_device("auto"))
    log_device(torch.device("cpu"))


class TestGpuIdsStayCudaOnly:
    """--gpus indexes CUDA ordinals and feeds DataParallel."""

    @pytest.mark.skipif(HAVE_CUDA, reason="needs a machine without CUDA")
    def test_gpu_ids_without_cuda_raise(self):
        with pytest.raises(ValueError, match="CUDA is not available"):
            select_device("auto", gpu_ids=(0,))

    def test_gpu_ids_with_a_non_cuda_device_raise(self):
        with pytest.raises(ValueError, match="CUDA ordinals"):
            select_device("mps", gpu_ids=(0,))
        with pytest.raises(ValueError, match="CUDA ordinals"):
            select_device("cpu", gpu_ids=(0,))

    @pytest.mark.skipif(not HAVE_CUDA, reason="needs CUDA")
    def test_out_of_range_gpu_id_raises(self):
        too_high = torch.cuda.device_count()
        with pytest.raises(ValueError, match="unavailable"):
            select_device("auto", gpu_ids=(too_high,))


@pytest.mark.skipif(not HAVE_MPS, reason="needs Apple Silicon")
class TestMpsBranch:
    def test_auto_selects_mps_when_cuda_is_absent(self):
        if HAVE_CUDA:
            pytest.skip("CUDA outranks MPS")
        assert select_device("auto").type == "mps"

    def test_explicit_mps_works(self):
        assert select_device("mps").type == "mps"

    @pytest.mark.parametrize("dtype", [None, torch.float16, torch.bfloat16])
    def test_autocast_forward_matches_fp32(self, dtype):
        """The branch the entrypoints now take: autocast on the chosen device."""
        device = torch.device("mps")
        model = torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, 3, padding=1), torch.nn.ReLU()
        ).to(device)
        x = torch.randn(2, 1, 16, 64, 64, device=device)
        with torch.no_grad():
            reference = model(x).float()
            kwargs = {"dtype": dtype} if dtype is not None else {}
            with torch.autocast(device_type=device.type, enabled=True, **kwargs):
                out = model(x)
        torch.mps.synchronize()
        assert torch.isfinite(out).all()
        assert torch.allclose(out.float(), reference, atol=2e-2, rtol=2e-2)

    def test_pin_memory_is_still_unsupported_on_mps(self):
        """Why the entrypoints keep pin_memory CUDA-only.

        If torch ever supports it, this test fails and the gate can widen.
        """
        import warnings

        from torch.utils.data import DataLoader, TensorDataset

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loader = DataLoader(
                TensorDataset(torch.randn(4, 3)), batch_size=2, pin_memory=True
            )
            for _ in loader:
                pass
        assert any("not supported on MPS" in str(w.message) for w in caught)


class TestCudaDecisionsWithoutCudaHardware:
    """The CUDA branch is the default path, and no machine here has a usable
    CUDA GPU to test it on, so the decisions are checked against a faked
    backend. This covers the resolver's logic only. It says nothing about
    whether torch itself still works, which real CUDA hardware would."""

    @staticmethod
    def _fake_cuda(monkeypatch, count):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: count > 0)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: count)

    def test_auto_prefers_cuda_over_mps(self, monkeypatch):
        self._fake_cuda(monkeypatch, 2)
        assert select_device("auto").type == "cuda"

    def test_gpu_ids_select_the_first_ordinal(self, monkeypatch):
        self._fake_cuda(monkeypatch, 4)
        device = select_device("auto", gpu_ids=(2, 3))
        assert (device.type, device.index) == ("cuda", 2)

    def test_out_of_range_ordinal_reports_the_visible_count(self, monkeypatch):
        self._fake_cuda(monkeypatch, 2)
        with pytest.raises(ValueError) as excinfo:
            select_device("auto", gpu_ids=(0, 7))
        message = str(excinfo.value)
        # Same wording the entrypoints raised before the refactor.
        assert "[7]" in message and "visible device count is 2" in message

    def test_explicit_cpu_still_wins_over_available_cuda(self, monkeypatch):
        self._fake_cuda(monkeypatch, 1)
        assert select_device("cpu").type == "cpu"

    def test_missing_cuda_message_is_unchanged(self, monkeypatch):
        self._fake_cuda(monkeypatch, 0)
        with pytest.raises(ValueError) as excinfo:
            select_device("auto", gpu_ids=(0,))
        assert str(excinfo.value) == (
            "--gpus was provided, but CUDA is not available in this environment."
        )
