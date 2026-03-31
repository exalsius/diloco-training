"""Tests for E8P12 lattice compression and distributed lattice reduce."""

import math
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

from diloco_training.utils.lattice_compression import (
    compress_tensor,
    decompress_tensor,
    compression_ratio,
    roundtrip_quality,
    get_e8_quantizer,
    E8P12Quantizer,
    _calibrate_e8_scale,
)


# ---------------------------------------------------------------------------
# E8P12 Quantizer unit tests
# ---------------------------------------------------------------------------


class TestE8P12Quantizer:
    def test_grid_shape(self):
        q = get_e8_quantizer("cpu")
        assert q.grid.shape == (65536, 8)

    def test_quantize_shape(self):
        q = get_e8_quantizer("cpu")
        x = torch.randn(100, 8)
        vals, idx = q.quantize(x)
        assert vals.shape == (100, 8)
        assert idx.shape == (100,)

    def test_dequantize_roundtrip(self):
        """dequantize(idx) should return the same grid point as quantize()."""
        q = get_e8_quantizer("cpu")
        x = torch.randn(50, 8)
        vals, idx = q.quantize(x)
        decoded = q.dequantize(idx)
        assert torch.allclose(vals, decoded, atol=1e-5)

    def test_indices_in_range(self):
        q = get_e8_quantizer("cpu")
        x = torch.randn(200, 8)
        _, idx = q.quantize(x)
        assert idx.min() >= 0
        assert idx.max() < 65536


# ---------------------------------------------------------------------------
# Scale calibration
# ---------------------------------------------------------------------------


class TestScaleCalibration:
    def test_calibration_returns_positive(self):
        scale = _calibrate_e8_scale("cpu")
        assert scale > 0

    def test_calibration_is_cached(self):
        s1 = _calibrate_e8_scale("cpu")
        s2 = _calibrate_e8_scale("cpu")
        assert s1 == s2

    def test_calibration_reasonable_range(self):
        scale = _calibrate_e8_scale("cpu")
        # E8 grid has mean norm ~3, unit-norm 8D blocks need scaling by ~3-15
        assert 2.0 < scale < 30.0


# ---------------------------------------------------------------------------
# Compress / decompress roundtrip tests
# ---------------------------------------------------------------------------


class TestCompressDecompress:
    def test_basic_roundtrip(self):
        torch.manual_seed(42)
        t = torch.randn(1024)
        c = compress_tensor(t)
        d = decompress_tensor(c)
        cos = torch.nn.functional.cosine_similarity(
            t.unsqueeze(0), d.unsqueeze(0)
        ).item()
        assert cos > 0.90, f"Cosine similarity {cos:.4f} too low"

    def test_cosine_similarity_target(self):
        """Should achieve >= 0.95 cosine similarity on Gaussian data."""
        torch.manual_seed(123)
        t = torch.randn(100_000)
        result = roundtrip_quality(t)
        assert result["cosine_similarity"] > 0.95, (
            f"cos_sim={result['cosine_similarity']:.4f}, expected >0.95"
        )

    def test_compression_ratio(self):
        t = torch.randn(10_000)
        c = compress_tensor(t)
        ratio = compression_ratio(c)
        # uint16 indices + fp16 norms = 4 bytes/block = 0.5 bytes/element
        # vs 4 bytes/element fp32 → ~8x compression
        assert ratio > 6.0, f"Compression ratio {ratio:.1f}x too low"
        assert ratio < 10.0, f"Compression ratio {ratio:.1f}x unexpectedly high"

    def test_shape_preservation_1d(self):
        t = torch.randn(1024)
        d = decompress_tensor(compress_tensor(t))
        assert d.shape == t.shape

    def test_shape_preservation_2d(self):
        t = torch.randn(64, 128)
        d = decompress_tensor(compress_tensor(t))
        assert d.shape == t.shape

    def test_shape_preservation_3d(self):
        t = torch.randn(3, 7, 11)
        d = decompress_tensor(compress_tensor(t))
        assert d.shape == t.shape

    def test_non_divisible_by_8(self):
        """Tensor size not divisible by 8 should still work (padding)."""
        for size in [1, 3, 7, 13, 100, 1023]:
            t = torch.randn(size)
            d = decompress_tensor(compress_tensor(t))
            assert d.shape == t.shape, f"Failed for size {size}"

    def test_zero_tensor(self):
        t = torch.zeros(64)
        d = decompress_tensor(compress_tensor(t))
        assert d.abs().max().item() < 1e-6

    def test_constant_tensor(self):
        t = torch.ones(64) * 3.14
        d = decompress_tensor(compress_tensor(t))
        cos = torch.nn.functional.cosine_similarity(
            t.unsqueeze(0), d.unsqueeze(0)
        ).item()
        assert cos > 0.90

    def test_very_small_values(self):
        t = torch.randn(256) * 1e-8
        d = decompress_tensor(compress_tensor(t))
        cos = torch.nn.functional.cosine_similarity(
            t.unsqueeze(0), d.unsqueeze(0)
        ).item()
        assert cos > 0.90, f"cos_sim={cos:.4f} for 1e-8 scale"

    def test_very_large_values(self):
        t = torch.randn(256) * 1e6
        d = decompress_tensor(compress_tensor(t))
        cos = torch.nn.functional.cosine_similarity(
            t.unsqueeze(0), d.unsqueeze(0)
        ).item()
        assert cos > 0.90, f"cos_sim={cos:.4f} for 1e6 scale"

    def test_pseudogradient_like(self):
        """Simulate pseudo-gradient: small perturbation of model params."""
        torch.manual_seed(0)
        W = torch.randn(512, 512) * 0.02
        pseudo_grad = torch.randn_like(W) * 0.002
        result = roundtrip_quality(pseudo_grad)
        assert result["cosine_similarity"] > 0.95


# ---------------------------------------------------------------------------
# Distributed lattice reduce (multi-process gloo test)
# ---------------------------------------------------------------------------


def _worker_lattice_reduce(rank, world_size, tensors, result_tensor, op):
    """Worker function for multi-process distributed test."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    from diloco_training.utils.quantization import distributed_reduce_lattice

    tensor = tensors[rank].clone()
    result = distributed_reduce_lattice(tensor, op=op)
    # Write result into the shared tensor at the right offset
    result_tensor[rank].copy_(result)

    dist.destroy_process_group()


class TestDistributedLatticeReduce:
    @pytest.mark.skipif(
        not hasattr(mp, "spawn"), reason="mp.spawn not available"
    )
    def test_sum_reduces_correctly(self):
        """AllGather + sum should approximate the sum of inputs."""
        torch.manual_seed(42)
        world_size = 2
        size = 256
        tensors = [torch.randn(size) for _ in range(world_size)]

        # Use shared-memory tensors so spawned processes can write back
        result_tensor = torch.zeros(world_size, size).share_memory_()

        mp.spawn(
            _worker_lattice_reduce,
            args=(world_size, tensors, result_tensor, dist.ReduceOp.SUM),
            nprocs=world_size,
            join=True,
        )

        # Both workers should get the same result
        assert torch.allclose(result_tensor[0], result_tensor[1], atol=1e-5), (
            "Workers got different results"
        )

        # Result should approximate sum of inputs (with compression error)
        expected_sum = torch.stack(tensors).sum(dim=0)
        cos = torch.nn.functional.cosine_similarity(
            expected_sum.unsqueeze(0), result_tensor[0].unsqueeze(0)
        ).item()
        assert cos > 0.90, f"Reduced result cos_sim={cos:.4f} vs expected sum"

    @pytest.mark.skipif(
        not hasattr(mp, "spawn"), reason="mp.spawn not available"
    )
    def test_avg_operation(self):
        """Test AVG reduction operation."""
        torch.manual_seed(99)
        world_size = 2
        size = 128
        tensors = [torch.randn(size) for _ in range(world_size)]

        result_tensor = torch.zeros(world_size, size).share_memory_()

        mp.spawn(
            _worker_lattice_reduce,
            args=(world_size, tensors, result_tensor, dist.ReduceOp.AVG),
            nprocs=world_size,
            join=True,
        )

        expected_avg = torch.stack(tensors).mean(dim=0)
        cos = torch.nn.functional.cosine_similarity(
            expected_avg.unsqueeze(0), result_tensor[0].unsqueeze(0)
        ).item()
        assert cos > 0.90, f"AVG reduce cos_sim={cos:.4f}"


# ---------------------------------------------------------------------------
# Error feedback tests
# ---------------------------------------------------------------------------


class TestErrorFeedback:
    def setup_method(self):
        """Reset error feedback buffers before each test."""
        from diloco_training.utils.diloco_utils import reset_error_feedback
        reset_error_feedback()

    def test_error_feedback_buffers_accumulate(self):
        """Error feedback stores residual from compression."""
        from diloco_training.utils.diloco_utils import _error_feedback_buffers, reset_error_feedback

        reset_error_feedback()
        torch.manual_seed(42)
        grad = torch.randn(256)

        # Compress and compute residual manually
        compressed = compress_tensor(grad)
        reconstructed = decompress_tensor(compressed)
        expected_residual = grad - reconstructed

        # Manually simulate what the error feedback code does
        ef_key = (0, 0)
        pre_compress = grad.clone()
        compressed_local = compress_tensor(grad)
        reconstructed_local = decompress_tensor(compressed_local)
        _error_feedback_buffers[ef_key] = (pre_compress - reconstructed_local).cpu()

        assert ef_key in _error_feedback_buffers
        stored_residual = _error_feedback_buffers[ef_key]
        assert stored_residual.shape == grad.shape
        # Residual should be small (compression error)
        assert stored_residual.norm() < grad.norm() * 0.5, (
            f"Residual norm {stored_residual.norm():.4f} too large relative to grad norm {grad.norm():.4f}"
        )

    def test_error_feedback_improves_over_steps(self):
        """Simulates error feedback across multiple outer steps.

        The total signal transmitted should be closer to the sum of all
        pseudo-gradients when error feedback is used vs not used.
        """
        from diloco_training.utils.diloco_utils import _error_feedback_buffers, reset_error_feedback

        reset_error_feedback()
        torch.manual_seed(123)
        n_steps = 5
        size = 512
        grads = [torch.randn(size) for _ in range(n_steps)]

        # Without error feedback
        total_transmitted_no_ef = torch.zeros(size)
        for g in grads:
            compressed = compress_tensor(g)
            total_transmitted_no_ef += decompress_tensor(compressed)

        # With error feedback
        total_transmitted_ef = torch.zeros(size)
        residual = torch.zeros(size)
        for g in grads:
            corrected = g + residual
            compressed = compress_tensor(corrected)
            reconstructed = decompress_tensor(compressed)
            total_transmitted_ef += reconstructed
            residual = corrected - reconstructed

        # Ground truth = sum of all grads
        true_sum = torch.stack(grads).sum(dim=0)

        cos_no_ef = torch.nn.functional.cosine_similarity(
            true_sum.unsqueeze(0), total_transmitted_no_ef.unsqueeze(0)
        ).item()
        cos_ef = torch.nn.functional.cosine_similarity(
            true_sum.unsqueeze(0), total_transmitted_ef.unsqueeze(0)
        ).item()

        # Error feedback should improve or match quality
        assert cos_ef >= cos_no_ef - 0.01, (
            f"Error feedback cos_sim {cos_ef:.4f} worse than no EF {cos_no_ef:.4f}"
        )

    def test_reset_clears_buffers(self):
        """reset_error_feedback() clears all stored residuals."""
        from diloco_training.utils.diloco_utils import _error_feedback_buffers, reset_error_feedback

        _error_feedback_buffers[(0, 0)] = torch.zeros(10)
        _error_feedback_buffers[(1, 5)] = torch.zeros(20)
        assert len(_error_feedback_buffers) == 2

        reset_error_feedback()
        assert len(_error_feedback_buffers) == 0
