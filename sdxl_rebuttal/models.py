from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """ReLU SAE with an untied, unit-column decoder."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.normal_(self.decoder.weight, std=1.0 / input_dim**0.5)
        self.normalize_decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x - self.decoder_bias))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.decoder_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    @torch.no_grad()
    def remove_decoder_parallel_gradients(self) -> None:
        if self.decoder.weight.grad is None:
            return
        weight = self.decoder.weight.data
        grad = self.decoder.weight.grad
        self.decoder.weight.grad = grad - weight * (weight * grad).sum(dim=0, keepdim=True)


class AlignmentMap(nn.Module):
    """Bias-free per-spatial-token map into the SDXL mid-block H space."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def ranked_inputs(self) -> torch.Tensor:
        return self.linear.weight.detach().float().norm(dim=0).argsort(descending=True)


class FactorizedAlignmentMap(nn.Module):
    """Low-rank nonlinear token map used to regularize sparse alignment."""

    def __init__(self, input_dim: int, output_dim: int, rank: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = int(rank)
        self.input = nn.Linear(input_dim, rank, bias=False)
        self.output = nn.Linear(rank, output_dim, bias=False)
        nn.init.kaiming_uniform_(self.input.weight, a=5**0.5)
        nn.init.normal_(self.output.weight, std=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.input(x)))

    def ranked_inputs(self) -> torch.Tensor:
        return self.input.weight.detach().float().norm(dim=0).argsort(descending=True)


class AsyrpDeltaBlock(nn.Module):
    """Official Asyrp DeltaBlock architecture adapted to SDXL dimensions.

    The official iDDPM block is GroupNorm/SiLU/1x1-conv, addition of a
    projected U-Net timestep embedding, then GroupNorm/SiLU/1x1-conv.
    """

    def __init__(self, channels: int, emb_channels: int):
        super().__init__()
        self.channels = int(channels)
        self.emb_channels = int(emb_channels)
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, channels))
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, h: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        delta = self.in_layers(h)
        if temb is not None:
            emb = self.emb_layers(temb).to(delta.dtype)
            delta = delta + emb[:, :, None, None]
        return self.out_layers(delta)


@dataclass
class InterventionSpec:
    method: str
    strength: float
    topk: int | None
    normalize_delta: bool


class HSpaceIntervention(nn.Module):
    """CASL or matched dense H-space intervention applied token-wise."""

    def __init__(
        self,
        spec: InterventionSpec,
        mapper: nn.Module,
        sae: SparseAutoencoder | None = None,
    ):
        super().__init__()
        if spec.method.startswith("casl") and sae is None:
            raise ValueError("CASL requires an SAE")
        if spec.method not in {"casl", "casl_lowrank", "dense", "asyrp_cnn"}:
            raise ValueError(f"Unknown method: {spec.method}")
        self.spec = spec
        self.mapper = mapper
        self.sae = sae
        self.register_buffer("_top_indices", torch.empty(0, dtype=torch.long), persistent=False)
        if spec.method.startswith("casl") and spec.topk is not None:
            input_features = sae.latent_dim if sae is not None else 0
            count = min(int(spec.topk), input_features)
            self._top_indices = mapper.ranked_inputs()[:count]

    def _input_code(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.spec.method == "dense":
            return tokens.to(next(self.mapper.parameters()).dtype)
        assert self.sae is not None
        code = self.sae.encode(tokens.to(self.sae.encoder.weight.dtype))
        if self._top_indices.numel():
            sparse = torch.zeros_like(code)
            sparse[:, self._top_indices] = code[:, self._top_indices]
            code = sparse
        return code

    def forward(self, h: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        if self.spec.method == "asyrp_cnn":
            dtype = next(self.mapper.parameters()).dtype
            delta = self.mapper(
                h.to(dtype),
                None if temb is None else temb.to(dtype),
            )
        else:
            batch, channels, height, width = h.shape
            tokens = h.permute(0, 2, 3, 1).reshape(-1, channels)
            code = self._input_code(tokens)
            code = code.to(next(self.mapper.parameters()).dtype)
            delta = self.mapper(code).reshape(
                batch, height, width, channels
            ).permute(0, 3, 1, 2)
        if self.spec.normalize_delta:
            rms = delta.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
            delta = delta / rms.to(delta.dtype)
        return h + float(self.spec.strength) * delta.to(h.dtype)


def load_sae(path: str, device: torch.device, dtype: torch.dtype = torch.float32) -> SparseAutoencoder:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SparseAutoencoder(payload["input_dim"], payload["latent_dim"])
    model.load_state_dict(payload["state_dict"])
    model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_mapper(path: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    architecture = payload.get("architecture", "linear")
    if architecture == "asyrp_delta_block":
        model = AsyrpDeltaBlock(payload["output_dim"], payload["emb_dim"])
    elif architecture == "factorized":
        model = FactorizedAlignmentMap(
            payload["input_dim"], payload["output_dim"], payload["rank"]
        )
    else:
        model = AlignmentMap(payload["input_dim"], payload["output_dim"])
    model.load_state_dict(payload["state_dict"])
    model.to(device=device, dtype=dtype).eval()
    return model
