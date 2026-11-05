#!/usr/bin/env python3
"""
Train an SAE on pooled DiT activations saved by dit_preliminary_validation.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path("/p/realai/zhenghao/LoRA_SAE")
SAE_MODULE_ROOT = REPO_ROOT / "Asyrp_SAE"
if str(SAE_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(SAE_MODULE_ROOT))

from models.sae.sae import ACTIVATIONS_CLASSES, Autoencoder  # type: ignore


DEFAULT_ACTIVATION_FILE = (
    REPO_ROOT / "Asyrp_SAE" / "results" / "dit_activation_dump" / "block_20_pooled_activations.pt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Asyrp_SAE" / "results" / "dit_sae"


def load_activations(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        if "pooled_activations" not in payload:
            raise KeyError(f"'pooled_activations' not found in {path}")
        activations = payload["pooled_activations"]
    else:
        activations = payload

    if not isinstance(activations, torch.Tensor):
        raise TypeError(f"Expected tensor activations in {path}")
    if activations.ndim != 2:
        raise ValueError(f"Expected [N, D] activations, got shape {tuple(activations.shape)}")
    return activations.float()


def build_activation_module(name: str, topk: int):
    if name not in ACTIVATIONS_CLASSES:
        raise ValueError(f"Unsupported activation: {name}. Choices: {sorted(ACTIVATIONS_CLASSES)}")
    if name == "TopK":
        return ACTIVATIONS_CLASSES[name](k=topk)
    return ACTIVATIONS_CLASSES[name]()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAE on pooled DiT activations.")
    parser.add_argument("--activation-file", type=Path, default=DEFAULT_ACTIVATION_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latent-multiplier", type=int, default=8, help="n_latents = input_dim * latent_multiplier")
    parser.add_argument("--activation", type=str, default="ReLU", choices=sorted(ACTIVATIONS_CLASSES.keys()))
    parser.add_argument("--topk", type=int, default=64, help="Used only when activation=TopK")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1-weight", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    activations = load_activations(args.activation_file)
    n_samples, input_dim = activations.shape
    n_latents = input_dim * args.latent_multiplier

    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    activation_module = build_activation_module(args.activation, args.topk)
    model = Autoencoder(
        n_latents=n_latents,
        n_inputs=input_dim,
        activation=activation_module,
        tied=False,
        normalize=args.normalize,
        ignore_timesteps=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Loaded activations: shape={tuple(activations.shape)}")
    print(f"Training SAE: input_dim={input_dim}, n_latents={n_latents}, device={device}")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_sparse = 0.0

        for (batch,) in loader:
            batch = batch.to(device)
            _, latents, recons = model(batch)
            recon_loss = F.mse_loss(recons, batch)
            sparse_loss = latents.abs().mean()
            loss = recon_loss + args.l1_weight * sparse_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_n = batch.size(0)
            total_loss += loss.item() * batch_n
            total_recon += recon_loss.item() * batch_n
            total_sparse += sparse_loss.item() * batch_n

        denom = float(n_samples)
        print(
            f"[epoch {epoch:03d}] "
            f"loss={total_loss / denom:.6f} "
            f"recon={total_recon / denom:.6f} "
            f"sparse={total_sparse / denom:.6f}"
        )
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / denom,
                "recon": total_recon / denom,
                "sparse": total_sparse / denom,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = (
        f"dit_sae_{args.activation.lower()}_lm{args.latent_multiplier}"
        f"_l1{args.l1_weight:g}"
        f"{'_norm' if args.normalize else ''}.pt"
    )
    ckpt_path = args.output_dir / ckpt_name
    meta_path = args.output_dir / (ckpt_path.stem + ".json")
    history_path = args.output_dir / (ckpt_path.stem + "_history.json")

    torch.save(model.state_dict(), ckpt_path)
    metadata = {
        "activation_file": str(args.activation_file),
        "num_samples": n_samples,
        "input_dim": input_dim,
        "n_latents": n_latents,
        "activation": args.activation,
        "topk": args.topk if args.activation == "TopK" else None,
        "normalize": args.normalize,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "l1_weight": args.l1_weight,
        "seed": args.seed,
        "checkpoint": str(ckpt_path),
        "history_file": str(history_path),
        "final_loss": history[-1]["loss"] if history else None,
        "final_recon": history[-1]["recon"] if history else None,
        "final_sparse": history[-1]["sparse"] if history else None,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    history_path.write_text(json.dumps(history, indent=2))

    print(f"Saved SAE checkpoint to: {ckpt_path}")
    print(f"Saved training metadata to: {meta_path}")
    print(f"Saved training history to: {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
