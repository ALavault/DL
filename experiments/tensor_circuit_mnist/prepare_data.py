from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def fit_product_mixture(
    x: np.ndarray,
    components: int,
    iterations: int,
    seed: int,
    smoothing: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    x = x.astype(np.float32, copy=False)
    prototypes = x[rng.choice(len(x), size=components, replace=False)]
    probs = 0.04 + 0.92 * prototypes
    weights = np.full(components, 1.0 / components, dtype=np.float32)
    history: list[float] = []

    for iteration in range(1, iterations + 1):
        p = np.clip(probs, 1e-5, 1.0 - 1e-5)
        logp = np.log(p)
        log1mp = np.log1p(-p)
        # x @ (log p - log(1-p))^T plus the all-zero contribution.
        scores = x @ (logp - log1mp).T
        scores += log1mp.sum(axis=1)[None, :]
        scores += np.log(np.clip(weights, 1e-12, None))[None, :]
        lse = logsumexp(scores, axis=1)
        nll = float(-lse.mean())
        history.append(nll)

        resp = np.exp(scores - lse[:, None]).astype(np.float32, copy=False)
        mass = resp.sum(axis=0) + 1e-8
        weights = mass / mass.sum()
        probs = (resp.T @ x + smoothing) / (mass[:, None] + 2.0 * smoothing)
        print(f"[mixture-r{components}] iter {iteration:02d}/{iterations} nll={nll:.4f}", flush=True)

    return weights.astype(np.float32), probs.astype(np.float32), history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mixture-iterations", type=int, default=25)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.input)
    required = ["train_data", "valid_data", "test_data"]
    for key in required:
        if key not in raw:
            raise KeyError(f"missing {key}; found {list(raw.keys())}")
    train = np.asarray(raw["train_data"], dtype=np.uint8)
    valid = np.asarray(raw["valid_data"], dtype=np.uint8)
    test = np.asarray(raw["test_data"], dtype=np.uint8)
    expected = [(50000, 784), (10000, 784), (10000, 784)]
    for name, array, shape in zip(required, [train, valid, test], expected):
        if array.shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {array.shape}")
        if array.min() < 0 or array.max() > 1:
            raise ValueError(f"{name} is not binary")

    dataset_path = args.output / "static_binarized_mnist_uint8.npz"
    np.savez_compressed(dataset_path, train=train, valid=valid, test=test)

    mixture_summary: dict[str, object] = {}
    for rank in (16, 32):
        weights, probs, history = fit_product_mixture(
            train, rank, args.mixture_iterations, seed=1000 + rank
        )
        np.savez_compressed(
            args.output / f"product_mixture_r{rank}.npz",
            weights=weights,
            probabilities=probs,
        )
        mixture_summary[str(rank)] = {
            "iterations": args.mixture_iterations,
            "train_nll_last_e_step": history[-1],
            "history": history,
        }

    manifest = {
        "protocol": "static binarized MNIST of Salakhutdinov and Murray",
        "splits": {"train": len(train), "validation": len(valid), "test": len(test)},
        "variables": 784,
        "mixtures": mixture_summary,
    }
    (args.output / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
