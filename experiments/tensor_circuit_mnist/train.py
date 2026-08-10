from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from model import GroupedTensorTreeCircuit, exact_active_normalization_error, num_parameters


VARIANTS = {
    "shared-r16": {"rank": 16, "max_groups": 1, "epochs": 24, "lr": 4e-3},
    "group64-r16": {"rank": 16, "max_groups": 64, "epochs": 36, "lr": 3e-3},
    "unshared-r16": {"rank": 16, "max_groups": 2048, "epochs": 36, "lr": 2.5e-3},
    "group64-r32": {"rank": 32, "max_groups": 64, "epochs": 32, "lr": 2e-3},
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def morton_code(y: int, x: int, side: int = 32) -> int:
    out = 0
    bit = 0
    while (1 << bit) < side:
        out |= ((x >> bit) & 1) << (2 * bit)
        out |= ((y >> bit) & 1) << (2 * bit + 1)
        bit += 1
    return out


def active_positions_and_order() -> tuple[np.ndarray, np.ndarray]:
    # Center 28x28 in a 32x32 canvas. active_positions[k] is the leaf occupied
    # by canonical variable k after the canonical variables themselves are
    # sorted in Morton order.
    triples: list[tuple[int, int]] = []
    for y in range(28):
        for x in range(28):
            original = y * 28 + x
            leaf = morton_code(y + 2, x + 2)
            triples.append((leaf, original))
    triples.sort()
    positions = np.asarray([leaf for leaf, _ in triples], dtype=np.int64)
    order = np.asarray([original for _, original in triples], dtype=np.int64)
    return positions, order


def make_loader(x: Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(x), batch_size=batch_size, shuffle=shuffle,
        num_workers=0, drop_last=False,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for (x,) in loader:
            x = x.to(device)
            total += (-model.log_prob(x)).sum().item()
            count += len(x)
    return total / count


def benchmark_eval(model: nn.Module, loader: DataLoader, device: torch.device, repeats: int = 3) -> float:
    model.eval()
    with torch.inference_mode():
        for (x,) in loader:
            model.log_prob(x.to(device))
            break
    start = time.perf_counter()
    count = 0
    with torch.inference_mode():
        for _ in range(repeats):
            for (x,) in loader:
                model.log_prob(x.to(device))
                count += len(x)
    return count / (time.perf_counter() - start)


def product_mixture_nll(x: np.ndarray, weights: np.ndarray, probs: np.ndarray, batch: int = 2048) -> float:
    p = np.clip(probs.astype(np.float64), 1e-9, 1.0 - 1e-9)
    logp = np.log(p)
    log1mp = np.log1p(-p)
    logw = np.log(np.clip(weights.astype(np.float64), 1e-300, None))
    total = 0.0
    for start in range(0, len(x), batch):
        xb = x[start:start+batch].astype(np.float64, copy=False)
        score = xb @ (logp - log1mp).T + log1mp.sum(axis=1)[None, :] + logw[None, :]
        m = score.max(axis=1)
        total += (-m - np.log(np.exp(score - m[:, None]).sum(axis=1))).sum()
    return float(total / len(x))


def save_samples(samples: Tensor, inverse_order: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    samples = samples.detach().cpu().numpy()
    row_major = np.empty_like(samples)
    row_major[:, inverse_order] = samples
    images = row_major.reshape(-1, 28, 28)
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for image, ax in zip(images[:64], axes.flat):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")
    fig.tight_layout(pad=0.05)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)
    threads = max(1, os.cpu_count() or 1)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = VARIANTS[args.variant]
    rank = int(spec["rank"])

    data = np.load(args.data_dir / "static_binarized_mnist_uint8.npz")
    positions, order = active_positions_and_order()
    train_np = np.asarray(data["train"], dtype=np.uint8)[:, order]
    valid_np = np.asarray(data["valid"], dtype=np.uint8)[:, order]
    test_np = np.asarray(data["test"], dtype=np.uint8)[:, order]
    train = torch.from_numpy(train_np.astype(np.int64, copy=False))
    valid = torch.from_numpy(valid_np.astype(np.int64, copy=False))
    test = torch.from_numpy(test_np.astype(np.int64, copy=False))

    mixture = np.load(args.data_dir / f"product_mixture_r{rank}.npz")
    mix_weights = np.asarray(mixture["weights"], dtype=np.float32)
    # Mixture probabilities are stored in row-major image order.
    mix_probs = np.asarray(mixture["probabilities"], dtype=np.float32)[:, order]
    mixture_test_nll = product_mixture_nll(test_np, mix_weights, mix_probs)

    train_loader = make_loader(train, args.batch_size, True)
    val_loader = make_loader(valid, args.batch_size * 2, False)
    test_loader = make_loader(test, args.batch_size * 2, False)

    model = GroupedTensorTreeCircuit(
        1024,
        torch.from_numpy(positions),
        rank=rank,
        cp_rank=rank,
        max_groups=int(spec["max_groups"]),
    )
    model.initialize_from_product_mixture(
        torch.from_numpy(mix_probs), torch.from_numpy(mix_weights), diagonal_strength=6.0
    )
    model.to(device)

    norm_error = exact_active_normalization_error(device)
    if norm_error > 2e-5:
        raise AssertionError(f"normalization error {norm_error}")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["lr"]))
    best_val = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []
    start_wall = time.perf_counter()
    train_seconds = 0.0
    seen = 0
    stale = 0

    for epoch in range(1, int(spec["epochs"]) + 1):
        model.train()
        begin = time.perf_counter()
        total = 0.0
        count = 0
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = -model.log_prob(x).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        elapsed = time.perf_counter() - begin
        train_seconds += elapsed
        seen += count
        train_nll = total / count
        val_nll = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_nll": train_nll,
            "validation_nll": val_nll,
            "train_examples_per_second": count / elapsed,
        })
        print(
            f"[{args.variant}] epoch={epoch:02d} train={train_nll:.4f} "
            f"valid={val_nll:.4f} ex/s={count/elapsed:.1f}", flush=True,
        )
        if val_nll < best_val - 1e-4:
            best_val = val_nll
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping after {epoch} epochs", flush=True)
                break

    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    model.to(device)
    test_nll = evaluate(model, test_loader, device)
    eval_eps = benchmark_eval(model, test_loader, device)
    sample_begin = time.perf_counter()
    samples = model.sample(1024, device=device)
    sample_eps = 1024 / (time.perf_counter() - sample_begin)
    wall = time.perf_counter() - start_wall

    torch.save({"state_dict": best_state, "variant": args.variant}, args.output / "checkpoint.pt")
    save_samples(samples[:64], order, args.output / "samples.png")

    metrics = {
        "variant": args.variant,
        "protocol": "static binarized MNIST, 50000/10000/10000, exact NLL",
        "seed": args.seed,
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "cpu_count": os.cpu_count(),
            "threads": torch.get_num_threads(),
        },
        "architecture": {
            "rank": rank,
            "cp_rank": rank,
            "max_groups": int(spec["max_groups"]),
            "groups_per_level": model.groups_per_level,
            "parameters": num_parameters(model),
            "padded_leaves": 1024,
            "observed_variables": 784,
        },
        "correctness": {"normalization_absolute_error": norm_error},
        "product_mixture_test_nll": mixture_test_nll,
        "best_epoch": best_epoch,
        "best_validation_nll": best_val,
        "test_nll_nats_per_image": test_nll,
        "test_bits_per_dimension": test_nll / (784 * math.log(2.0)),
        "train_examples_per_second": seen / train_seconds,
        "eval_examples_per_second": eval_eps,
        "sample_examples_per_second": sample_eps,
        "wall_seconds": wall,
        "history": history,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (args.output / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
