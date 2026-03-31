"""
Lattice-based compression for DiLoCo pseudo-gradients.

Uses the E8P12 lattice (65,536 points in 8D) to vector-quantize parameter
differences at ~2 bits/dim, achieving ~8x compression vs fp32.

The E8 lattice provides near-optimal sphere packing in 8 dimensions.
Each 8-dimensional block is mapped to its nearest lattice point, stored as
a 16-bit index. Per-block L2 norms (fp16) preserve magnitude information.

Random rotation preprocessing (as used in TurboQuant for KV-cache compression)
was empirically evaluated and found to provide NO benefit for pseudo-gradients
(cos_sim 0.9540 without vs 0.9539 with rotation). This makes sense because
pseudo-gradients are already near-isotropic, unlike KV-cache vectors which
have structured coordinate distributions. Omitting rotation avoids O(d^2)
matrix storage and matmul overhead.

Compression pipeline:
  1. Flatten tensor, pad to multiple of 8
  2. Reshape to (N, 8) blocks
  3. Store per-block L2 norms (fp16), normalize to unit norm
  4. Scale by calibrated E8 factor, E8P12 quantize → uint16 index per block
  5. Pack: (indices, block_norms, e8_scale)

Achieves ~0.96 cosine similarity at 2 bits/dim on pseudo-gradient-like data.
"""

import os
import math
import torch
from typing import Tuple, NamedTuple


_E8P_CODESZ = 8

_CODEBOOK_DIR = os.path.join(os.path.dirname(__file__), "codebooks")
_CODEBOOK_PATH = os.path.join(_CODEBOOK_DIR, "e8p12_grid.pt")


# ---------------------------------------------------------------------------
# E8P12 Grid generation (only on first use if .pt cache is missing)
# ---------------------------------------------------------------------------


def _get_norm12() -> torch.Tensor:
    return (
        torch.tensor(
            [
                [3, 1, 1, 1, 3, 3, 3, 3], [1, 3, 1, 1, 3, 3, 3, 3],
                [1, 1, 3, 1, 3, 3, 3, 3], [1, 1, 1, 3, 3, 3, 3, 3],
                [3, 3, 3, 1, 3, 3, 1, 1], [3, 3, 3, 1, 3, 1, 3, 1],
                [3, 3, 3, 1, 1, 3, 3, 1], [3, 3, 3, 1, 3, 1, 1, 3],
                [3, 3, 3, 1, 1, 3, 1, 3], [3, 3, 3, 1, 1, 1, 3, 3],
                [3, 3, 1, 3, 3, 3, 1, 1], [3, 3, 1, 3, 3, 1, 3, 1],
                [3, 3, 1, 3, 1, 3, 3, 1], [3, 3, 1, 3, 3, 1, 1, 3],
                [3, 3, 1, 3, 1, 3, 1, 3], [3, 3, 1, 3, 1, 1, 3, 3],
                [3, 1, 3, 3, 3, 3, 1, 1], [3, 1, 3, 3, 3, 1, 3, 1],
                [3, 1, 3, 3, 1, 3, 3, 1], [3, 1, 3, 3, 3, 1, 1, 3],
                [3, 1, 3, 3, 1, 3, 1, 3], [1, 3, 3, 3, 1, 1, 3, 3],
                [1, 3, 3, 3, 3, 3, 1, 1], [1, 3, 3, 3, 3, 1, 3, 1],
                [1, 3, 3, 3, 1, 3, 3, 1], [1, 3, 3, 3, 3, 1, 1, 3],
                [1, 3, 3, 3, 1, 3, 1, 3], [1, 1, 3, 3, 1, 3, 3, 3],
                [3, 3, 1, 1, 3, 3, 3, 1],
            ]
        )
        / 2
    )


def _get_abs_grid() -> torch.Tensor:
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * 8).float() + 1 / 2
    d8m2 = d8.sum(dim=-1) % 2 == 0
    d8n = d8.norm(dim=-1) ** 2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    norm12 = _get_norm12()
    return torch.concat([d8abs, norm12], dim=0)


def _get_packed_abs_grid() -> torch.Tensor:
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * 8).float() + 1 / 2
    d8m2 = d8.sum(dim=-1) % 2 == 0
    d8n = d8.norm(dim=-1) ** 2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    norm12 = _get_norm12()
    cba = torch.concat([d8abs, norm12], dim=0)
    cba = cba[:, [0, 2, 4, 6, 1, 3, 5, 7]]
    cba[:, 7] *= 1 - 2 * (cba.sum(1) % 2)
    cba = cba * 2 + 8
    cba = cba.to(torch.int32)
    acc = cba[:, 0]
    for i in range(7):
        acc = acc | (cba[:, (i + 1)] << ((i + 1) * 4))
    return acc


def _get_full_grid_vectorized(packed_abs_grid: torch.Tensor):
    """Build the full 65,536-point E8P12 grid from packed absolute grid."""
    N = 1 << 16
    c = torch.arange(N, dtype=torch.int64)

    signs_raw = c & 255
    abs_val = (c >> 8).long()

    parity = torch.zeros(N, dtype=torch.int64)
    for i in range(8):
        parity = parity ^ ((signs_raw >> i) & 1)
    signs_raw = signs_raw ^ parity

    abs_codes = packed_abs_grid[abs_val].long()

    shuffle_map = [0, 4, 1, 5, 2, 6, 3, 7]
    synth_codebook = torch.zeros(N, 8)

    for i in range(8):
        ii = shuffle_map[i]
        coord = ((abs_codes >> (4 * ii)) & 15) - 8
        synth_codebook[:, i] = coord.float() * 0.5
        sign_bit = ((signs_raw >> ii) & 1).bool()
        synth_codebook[sign_bit, i] *= -1

    parity_mask = parity.bool()
    synth_codebook[parity_mask] -= 0.25
    synth_codebook[~parity_mask] += 0.25

    parity_idx = torch.where(parity_mask)[0].tolist()
    return synth_codebook, torch.arange(N), parity_idx


def _load_or_build_grids():
    """Load pre-serialized grids from disk, or build and cache them."""
    if os.path.exists(_CODEBOOK_PATH):
        try:
            data = torch.load(_CODEBOOK_PATH, map_location="cpu", weights_only=True)
            return (
                data["packed_abs_grid"],
                data["grid"],
                data["grid_idx"],
                data["parity_idx"],
                data["abs_grid"],
            )
        except Exception:
            pass

    packed_abs = _get_packed_abs_grid()
    grid, grid_idx, parity_idx = _get_full_grid_vectorized(packed_abs)
    abs_grid = _get_abs_grid()

    os.makedirs(_CODEBOOK_DIR, exist_ok=True)
    try:
        torch.save(
            {
                "packed_abs_grid": packed_abs,
                "grid": grid,
                "grid_idx": grid_idx,
                "parity_idx": parity_idx,
                "abs_grid": abs_grid,
            },
            _CODEBOOK_PATH,
        )
    except Exception:
        pass

    return packed_abs, grid, grid_idx, parity_idx, abs_grid


_E8P_PACKED_ABS, _E8P_GRID, _E8P_GRID_IDX, _PARITY_IDX, _E8P_ABS_GRID = (
    _load_or_build_grids()
)


# ---------------------------------------------------------------------------
# E8P12 Quantizer (core lattice encoder/decoder)
# ---------------------------------------------------------------------------


class E8P12Quantizer:
    """Encodes 8D vectors to their nearest E8P12 lattice point (16-bit index)."""

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.codesz = _E8P_CODESZ

        self.grid = _E8P_GRID.to(self.device)
        self.grid_norm = self.grid.norm(dim=-1) ** 2

        grid_part = _E8P_GRID[_PARITY_IDX] + 0.25
        mask = ((grid_part[:, :7] < 0).sum(dim=-1) <= 1) & (
            grid_part[:, :7].min(dim=-1).values >= -0.5
        )
        grid_part = grid_part[torch.where(mask)[0]]
        self.grid_part = grid_part.to(self.device)
        self.grid_part_norm = (grid_part.norm(dim=-1) ** 2).to(self.device)

        abs_grid = _E8P_ABS_GRID
        self.grid_abs_odd = (abs_grid.sum(dim=-1) % 2 == 1).to(self.device)

        abs_grid_dev = abs_grid.to(self.device)
        abs_grid_norm = (abs_grid.norm(dim=-1) ** 2).to(self.device)
        self.part_abs_map = self._round(
            grid_part.abs().to(self.device), abs_grid_dev, abs_grid_norm
        )[1]

        self.bit_map = (2 ** torch.arange(8)).to(self.device)

    def _round(self, x: torch.Tensor, grid: torch.Tensor, grid_norm: torch.Tensor):
        xqidx = (2 * x @ grid.T - grid_norm).argmax(-1)
        return grid[xqidx], xqidx

    def _fast_quantize_part(self, x: torch.Tensor, parity: bool):
        x_part = torch.abs(x)
        x_odd = torch.where((x < 0).sum(dim=-1) % 2 != 0)[0]
        x_part[x_odd, 7] = -x_part[x_odd, 7]
        mask = 1 - 2 * (x < 0).to(torch.float32)
        mask[x_odd, 7] = -mask[x_odd, 7]

        roundout, xqidx = self._round(x_part, self.grid_part, self.grid_part_norm)
        vals = roundout * mask
        err = (x - vals).norm(dim=-1)

        abs_idx = self.part_abs_map[xqidx]
        sign_mask = ((roundout < 0) ^ (mask < 0))[:, [0, 2, 4, 6, 1, 3, 5, 7]]
        sign_mask[:, 7] = sign_mask[:, 7] ^ self.grid_abs_odd[abs_idx]
        sign_mask[:, 0] = sign_mask[:, 0] ^ parity
        mask_idx = (sign_mask * self.bit_map).sum(dim=-1).int()
        idx = (abs_idx << 8) + mask_idx
        return vals, idx, err

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize 8D vectors to nearest E8P12 lattice points.

        Args:
            x: Tensor of shape (N, 8)

        Returns:
            (reconstructed_values, indices) where indices are uint16-range ints
        """
        assert x.shape[-1] == 8
        x = x.to(self.grid_part.dtype).to(self.device)

        x_plus = x + 0.25
        x_minus = x - 0.25

        plus_vals, plus_idx, plus_err = self._fast_quantize_part(x_plus, True)
        minus_vals, minus_idx, minus_err = self._fast_quantize_part(x_minus, False)

        which = plus_err < minus_err
        final_vals = torch.where(
            which.unsqueeze(-1), plus_vals - 0.25, minus_vals + 0.25
        )
        final_idx = torch.where(which, plus_idx, minus_idx)
        return final_vals, final_idx

    @torch.no_grad()
    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices back to 8D lattice points."""
        return self.grid[indices.long()]

    def to(self, device: torch.device | str):
        self.device = torch.device(device)
        self.grid = self.grid.to(self.device)
        self.grid_norm = self.grid_norm.to(self.device)
        self.grid_part = self.grid_part.to(self.device)
        self.grid_part_norm = self.grid_part_norm.to(self.device)
        self.grid_abs_odd = self.grid_abs_odd.to(self.device)
        self.part_abs_map = self.part_abs_map.to(self.device)
        self.bit_map = self.bit_map.to(self.device)
        return self


# Singleton factory — one quantizer per device
_E8_INSTANCES: dict[str, E8P12Quantizer] = {}


def get_e8_quantizer(device: torch.device | str = "cpu") -> E8P12Quantizer:
    """Return a shared E8P12Quantizer instance for the given device."""
    key = str(device)
    if key not in _E8_INSTANCES:
        _E8_INSTANCES[key] = E8P12Quantizer(device=device)
    return _E8_INSTANCES[key]


# ---------------------------------------------------------------------------
# Compressed representation
# ---------------------------------------------------------------------------


class LatticeCompressed(NamedTuple):
    """Compressed pseudo-gradient representation."""

    indices: torch.Tensor       # (n_blocks,) int32 — E8 lattice indices
    block_norms: torch.Tensor   # (n_blocks,) fp16 — normalized per-block L2 norms
    norm_scale: torch.Tensor    # scalar fp32 — max block norm (for fp16 rescaling)
    e8_scale: float             # calibrated E8 scale factor
    original_shape: torch.Size  # for reshaping on decompress
    original_numel: int         # actual element count (before padding)


# ---------------------------------------------------------------------------
# High-level compress / decompress API for arbitrary tensors
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# E8 scale calibration — find optimal input scaling for the lattice grid
# ---------------------------------------------------------------------------

_E8_SCALE_CACHE: dict[str, float] = {}


def _calibrate_e8_scale(device: torch.device | str = "cpu") -> float:
    """Find the optimal scale factor for E8P12 quantization.

    Unit-norm 8D blocks have norm 1.0, but E8 grid points have norms in
    [0.7, 4.0].  We search for the scale s such that quantize(block * s) / s
    minimizes MSE over random unit-norm blocks.

    The result is cached per device.
    """
    key = str(device)
    if key in _E8_SCALE_CACHE:
        return _E8_SCALE_CACHE[key]

    quantizer = get_e8_quantizer(device)

    # Generate random unit-norm 8D blocks for calibration
    rng = torch.Generator(device="cpu")
    rng.manual_seed(99999)
    data = torch.randn(5000, 8, generator=rng)
    data = data / data.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    data = data.to(device)

    # Coarse search
    best_scale, best_mse = 1.0, float("inf")
    for s in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]:
        rec, _ = quantizer.quantize(data * s)
        mse = ((data - rec / s) ** 2).mean().item()
        if mse < best_mse:
            best_mse, best_scale = mse, float(s)

    # Fine search
    for s in torch.linspace(best_scale * 0.7, best_scale * 1.3, 30).tolist():
        rec, _ = quantizer.quantize(data.to(device) * s)
        mse = ((data - rec / s) ** 2).mean().item()
        if mse < best_mse:
            best_mse, best_scale = mse, float(s)

    _E8_SCALE_CACHE[key] = best_scale
    return best_scale


# ---------------------------------------------------------------------------
# High-level compress / decompress API for arbitrary tensors
# ---------------------------------------------------------------------------


@torch.no_grad()
def compress_tensor(
    tensor: torch.Tensor,
    device: torch.device | str = "cpu",
) -> LatticeCompressed:
    """Compress a tensor using E8P12 lattice vector quantization.

    Achieves ~2 bits/dim (~8x compression vs fp32).

    Pipeline:
      1. Flatten and pad to multiple of 8
      2. Reshape into (N, 8) blocks
      3. Store per-block L2 norms (fp16), normalize blocks to unit norm
      4. Scale by calibrated E8 factor, quantize with E8P12 lattice
      5. Pack: uint16 indices + fp16 block norms

    Random rotation is NOT applied — empirical tests show it provides no
    benefit for pseudo-gradients (which are already near-isotropic), unlike
    KV-cache vectors where rotation is critical.  This avoids O(d^2) storage
    and matmul overhead.

    Args:
        tensor: tensor of any shape (typically a pseudo-gradient)
        device: device for the E8 quantizer

    Returns:
        LatticeCompressed namedtuple
    """
    original_shape = tensor.shape
    original_numel = tensor.numel()
    flat = tensor.detach().float().flatten()

    # Pad to multiple of 8
    remainder = flat.numel() % 8
    if remainder != 0:
        pad_size = 8 - remainder
        flat = torch.nn.functional.pad(flat, (0, pad_size), value=0.0)

    # Reshape to (n_blocks, 8)
    blocks = flat.reshape(-1, 8)

    # Per-block norms — normalize into fp16-safe range via a global scale
    block_norms = blocks.norm(dim=-1)
    norm_scale = block_norms.max().clamp(min=1e-30)
    block_norms_normalized = block_norms / norm_scale
    safe_norms = block_norms.clamp(min=1e-12)

    # Normalize to unit norm, then scale to calibrated E8 range
    e8_scale = _calibrate_e8_scale(device)
    blocks_scaled = (blocks / safe_norms.unsqueeze(-1)) * e8_scale

    # E8P12 quantize — find nearest lattice point
    quantizer = get_e8_quantizer(device)
    _, indices = quantizer.quantize(blocks_scaled.to(device))

    return LatticeCompressed(
        indices=indices.to("cpu").int(),
        block_norms=block_norms_normalized.half(),
        norm_scale=norm_scale,
        e8_scale=e8_scale,
        original_shape=original_shape,
        original_numel=original_numel,
    )


@torch.no_grad()
def decompress_tensor(
    compressed: LatticeCompressed,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Decompress a LatticeCompressed back to a tensor.

    Args:
        compressed: LatticeCompressed from compress_tensor()
        device: target device for the output tensor

    Returns:
        Reconstructed tensor with original shape
    """
    quantizer = get_e8_quantizer(device)
    e8_scale = compressed.e8_scale

    # Decode lattice points → unit-scale reconstruction
    blocks = quantizer.dequantize(compressed.indices.to(device))
    blocks = blocks / e8_scale

    # Rescale by stored per-block norms (undo fp16 normalization)
    block_norms = compressed.block_norms.float().to(device) * compressed.norm_scale.to(device)
    blocks = blocks * block_norms.unsqueeze(-1)

    # Flatten and remove padding
    flat = blocks.reshape(-1)[: compressed.original_numel]

    return flat.reshape(compressed.original_shape)


@torch.no_grad()
def compression_ratio(compressed: LatticeCompressed) -> float:
    """Compute the compression ratio achieved.

    Returns:
        ratio of original size to compressed size
    """
    original_bytes = compressed.original_numel * 4  # fp32

    # Compressed: uint16 indices + fp16 norms + fp32 global_scale
    n_blocks = compressed.indices.numel()
    compressed_bytes = (
        n_blocks * 2  # uint16 indices
        + n_blocks * 2  # fp16 block norms
        + 4  # fp32 global scale
    )

    return original_bytes / compressed_bytes


@torch.no_grad()
def roundtrip_quality(tensor: torch.Tensor, device: str = "cpu") -> dict:
    """Measure compression quality via roundtrip encode/decode.

    Returns:
        dict with mse, cosine_similarity, sqnr_db, compression_ratio
    """
    compressed = compress_tensor(tensor, device=device)
    reconstructed = decompress_tensor(compressed, device=device)

    flat_orig = tensor.float().flatten()
    flat_recon = reconstructed.float().flatten()

    mse = ((flat_orig - flat_recon) ** 2).mean().item()

    cos_sim = torch.nn.functional.cosine_similarity(
        flat_orig.unsqueeze(0), flat_recon.unsqueeze(0)
    ).item()

    signal_power = (flat_orig**2).mean().item()
    noise_power = mse
    sqnr_db = 10 * math.log10(signal_power / max(noise_power, 1e-20))

    ratio = compression_ratio(compressed)

    return {
        "mse": mse,
        "cosine_similarity": cos_sim,
        "sqnr_db": sqnr_db,
        "compression_ratio": ratio,
    }
