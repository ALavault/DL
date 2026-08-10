from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from geometry import (
    COMPONENTS,
    GEOMETRY_COMPONENTS,
    HOMOGENEOUS_COMPONENTS,
    active_positions_and_order,
    validate_geometries,
)
from model import GroupedTensorTreeCircuit, num_parameters


PRIOR_CYCLE_R32_NLL = 100.83791817703247


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def logsumexp_np(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return (maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))).squeeze(axis)


def mixture_nll(log_prob: np.ndarray, weights: np.ndarray) -> float:
    log_weights = np.log(np.clip(weights, 1e-300, None))
    return float(-logsumexp_np(log_prob + log_weights[None, :], axis=1).mean())


def optimize_mixture_weights(
    log_prob: np.ndarray,
    max_iterations: int = 2000,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, float, int]:
    if log_prob.ndim != 2 or log_prob.shape[1] == 0:
        raise ValueError("log_prob must be [examples, components]")
    components = log_prob.shape[1]
    weights = np.full(components, 1.0 / components, dtype=np.float64)
    for iteration in range(1, max_iterations + 1):
        joint = log_prob + np.log(np.clip(weights, 1e-300, None))[None, :]
        normalizer = logsumexp_np(joint, axis=1)
        responsibilities = np.exp(joint - normalizer[:, None])
        updated = responsibilities.mean(axis=0)
        updated = np.clip(updated, 1e-12, None)
        updated /= updated.sum()
        if np.abs(updated - weights).sum() < tolerance:
            weights = updated
            break
        weights = updated
    return weights, mixture_nll(log_prob, weights), iteration


def exhaustive_best_subset(
    log_prob: np.ndarray,
    names: list[str],
    subset_size: int,
) -> dict[str, object]:
    if not 1 <= subset_size <= len(names):
        raise ValueError("invalid subset size")
    best: dict[str, object] | None = None
    for indices in itertools.combinations(range(len(names)), subset_size):
        selected = log_prob[:, indices]
        weights, nll, iterations = optimize_mixture_weights(selected)
        candidate = {
            "components": [names[i] for i in indices],
            "weights": weights.tolist(),
            "validation_nll": nll,
            "em_iterations": iterations,
        }
        if best is None or nll < float(best["validation_nll"]):
            best = candidate
    if best is None:
        raise RuntimeError("subset search failed")
    return best


def locate_component_dir(root: Path, name: str) -> Path:
    candidates = [root / name, root / f"component-{name}"]
    for candidate in candidates:
        if (candidate / "checkpoint.pt").exists():
            return candidate
    raise FileNotFoundError(f"cannot locate artifact for {name} below {root}")


def load_component(
    root: Path,
    name: str,
    device: torch.device,
) -> tuple[GroupedTensorTreeCircuit, Tensor]:
    component_dir = locate_component_dir(root, name)
    checkpoint = torch.load(component_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    geometry = str(checkpoint["geometry"])
    positions, order = active_positions_and_order(geometry)
    model = GroupedTensorTreeCircuit(
        1024,
        torch.from_numpy(positions),
        rank=int(checkpoint["rank"]),
        cp_rank=int(checkpoint["rank"]),
        max_groups=int(checkpoint["max_groups"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), torch.from_numpy(order).to(device)


class ExactTreeMixture(nn.Module):
    def __init__(
        self,
        models: list[GroupedTensorTreeCircuit],
        orders: list[Tensor],
        weights: Iterable[float],
    ) -> None:
        super().__init__()
        if len(models) == 0 or len(models) != len(orders):
            raise ValueError("models and orders must have the same non-zero length")
        self.models = nn.ModuleList(models)
        for index, order in enumerate(orders):
            self.register_buffer(f"order_{index}", order.long())
        weights_tensor = torch.as_tensor(list(weights), dtype=torch.float32)
        if weights_tensor.shape != (len(models),) or (weights_tensor <= 0).any():
            raise ValueError("weights must be strictly positive")
        weights_tensor /= weights_tensor.sum()
        self.mixture_logits = nn.Parameter(weights_tensor.log())

    @property
    def weights(self) -> Tensor:
        return torch.softmax(self.mixture_logits, dim=0)

    def component_log_prob(self, x_row_major: Tensor) -> Tensor:
        values = []
        for index, model in enumerate(self.models):
            order = getattr(self, f"order_{index}")
            values.append(model.log_prob(x_row_major.index_select(1, order)))
        return torch.stack(values, dim=1)

    def log_prob(self, x_row_major: Tensor) -> Tensor:
        component_values = self.component_log_prob(x_row_major)
        return torch.logsumexp(
            component_values + torch.log_softmax(self.mixture_logits, dim=0)[None, :],
            dim=1,
        )

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> tuple[Tensor, Tensor]:
        chosen = torch.multinomial(self.weights.to(device), n, replacement=True)
        output = torch.empty(n, 784, dtype=torch.long, device=device)
        for index, model in enumerate(self.models):
            locations = torch.nonzero(chosen == index, as_tuple=False).flatten()
            if len(locations) == 0:
                continue
            local = model.sample(len(locations), device=device)
            order = getattr(self, f"order_{index}")
            row_major = torch.empty_like(local)
            row_major[:, order] = local
            output.index_copy_(0, locations, row_major)
        return output, chosen


def exact_mixture_normalization_error(device: torch.device) -> float:
    torch.manual_seed(99)
    model_a = GroupedTensorTreeCircuit(
        8, torch.tensor([0, 2, 5, 7]), rank=3, cp_rank=3, max_groups=8
    ).to(device)
    model_b = GroupedTensorTreeCircuit(
        8, torch.tensor([1, 3, 4, 6]), rank=3, cp_rank=3, max_groups=8
    ).to(device)
    mixture = ExactTreeMixture(
        [model_a, model_b],
        [torch.tensor([0, 1, 2, 3], device=device), torch.tensor([2, 0, 3, 1], device=device)],
        [0.37, 0.63],
    ).to(device)
    states = torch.tensor(
        [[(value >> bit) & 1 for bit in range(4)] for value in range(16)],
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode():
        return abs(mixture.log_prob(states).exp().sum().item() - 1.0)


def make_loader(x: Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def per_example_log_prob(
    model: ExactTreeMixture,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for (x,) in loader:
            chunks.append(model.log_prob(x.to(device)).double().cpu().numpy())
    return np.concatenate(chunks)


def per_example_component_log_prob(
    model: ExactTreeMixture,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for (x,) in loader:
            chunks.append(model.component_log_prob(x.to(device)).double().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def save_samples(samples: Tensor, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    images = samples.detach().cpu().numpy().reshape(-1, 28, 28)
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for image, ax in zip(images[:64], axes.flat):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(pad=0.05, rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_fixed_mixture(
    validation_log_prob: np.ndarray,
    test_log_prob: np.ndarray,
    all_names: list[str],
    selection: dict[str, object],
) -> dict[str, object]:
    indices = [all_names.index(name) for name in selection["components"]]
    weights = np.asarray(selection["weights"], dtype=np.float64)
    test_nll = mixture_nll(test_log_prob[:, indices], weights)
    uniform = np.full(len(indices), 1.0 / len(indices), dtype=np.float64)
    return {
        **selection,
        "test_nll": test_nll,
        "test_bits_per_dimension": test_nll / (784 * math.log(2.0)),
        "uniform_validation_nll": mixture_nll(validation_log_prob[:, indices], uniform),
        "uniform_test_nll": mixture_nll(test_log_prob[:, indices], uniform),
        "effective_component_count": float(np.exp(-(weights * np.log(weights)).sum())),
    }


def responsibility_summary(log_prob: np.ndarray, weights: np.ndarray) -> dict[str, object]:
    joint = log_prob + np.log(np.clip(weights, 1e-300, None))[None, :]
    responsibilities = np.exp(joint - logsumexp_np(joint, axis=1)[:, None])
    hard = np.bincount(responsibilities.argmax(axis=1), minlength=log_prob.shape[1]) / len(log_prob)
    entropy = -(responsibilities * np.log(np.clip(responsibilities, 1e-300, None))).sum(axis=1)
    return {
        "mean_responsibilities": responsibilities.mean(axis=0).tolist(),
        "hard_assignment_fractions": hard.tolist(),
        "mean_responsibility_entropy": float(entropy.mean()),
        "median_responsibility_entropy": float(np.median(entropy)),
    }


def joint_finetune(
    label: str,
    selected_names: list[str],
    initial_weights: np.ndarray,
    component_root: Path,
    train: Tensor,
    valid: Tensor,
    test: Tensor,
    output: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    seed_all(seed)
    models: list[GroupedTensorTreeCircuit] = []
    orders: list[Tensor] = []
    for name in selected_names:
        model, order = load_component(component_root, name, device)
        models.append(model)
        orders.append(order)
    mixture = ExactTreeMixture(models, orders, initial_weights).to(device)
    train_loader = make_loader(train, batch_size, True)
    valid_loader = make_loader(valid, batch_size * 2, False)
    test_loader = make_loader(test, batch_size * 2, False)

    parameter_groups = [
        {
            "params": [parameter for model in mixture.models for parameter in model.parameters()],
            "lr": 3e-4,
        },
        {"params": [mixture.mixture_logits], "lr": 1e-3},
    ]
    optimizer = torch.optim.Adam(parameter_groups)
    initial_valid = float(-per_example_log_prob(mixture, valid_loader, device).mean())
    best_val = initial_valid
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in mixture.state_dict().items()}
    history: list[dict[str, object]] = []
    stale = 0
    train_seconds = 0.0
    seen = 0

    for epoch in range(1, epochs + 1):
        mixture.train()
        begin = time.perf_counter()
        total = 0.0
        count = 0
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = -mixture.log_prob(x).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite mixture loss in {label}, epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mixture.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        elapsed = time.perf_counter() - begin
        train_seconds += elapsed
        seen += count
        validation_nll = float(-per_example_log_prob(mixture, valid_loader, device).mean())
        weights = mixture.weights.detach().cpu().double().numpy()
        row = {
            "epoch": epoch,
            "train_nll": total / count,
            "validation_nll": validation_nll,
            "train_examples_per_second": count / elapsed,
            "weights": weights.tolist(),
        }
        history.append(row)
        print(
            f"[{label}] epoch={epoch:02d} train={row['train_nll']:.4f} "
            f"valid={validation_nll:.4f} weights={np.round(weights, 3).tolist()}",
            flush=True,
        )
        if validation_nll < best_val - 1e-4:
            best_val = validation_nll
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in mixture.state_dict().items()}
        else:
            stale += 1
            if stale >= 4:
                break

    mixture.load_state_dict(best_state)
    mixture.to(device)
    validation_log_prob = per_example_log_prob(mixture, valid_loader, device)
    test_log_prob = per_example_log_prob(mixture, test_loader, device)
    test_component_log_prob = per_example_component_log_prob(mixture, test_loader, device)
    weights = mixture.weights.detach().cpu().double().numpy()
    sample_start = time.perf_counter()
    samples, chosen = mixture.sample(1024, device)
    sampling_eps = 1024 / (time.perf_counter() - sample_start)
    save_samples(samples[:64], output / f"samples-{label}.png", label)
    torch.save(
        {
            "state_dict": best_state,
            "components": selected_names,
            "weights": weights,
            "label": label,
        },
        output / f"checkpoint-{label}.pt",
    )
    np.save(output / f"validation_log_prob-{label}.npy", validation_log_prob)
    np.save(output / f"test_log_prob-{label}.npy", test_log_prob)

    result = {
        "label": label,
        "components": selected_names,
        "parameters": num_parameters(mixture),
        "initial_validation_nll": initial_valid,
        "best_epoch": best_epoch,
        "best_validation_nll": float(-validation_log_prob.mean()),
        "test_nll": float(-test_log_prob.mean()),
        "test_bits_per_dimension": float(-test_log_prob.mean()) / (784 * math.log(2.0)),
        "weights": weights.tolist(),
        "effective_component_count": float(np.exp(-(weights * np.log(weights)).sum())),
        "train_examples_per_second": seen / train_seconds if train_seconds else math.nan,
        "sample_examples_per_second": sampling_eps,
        "sample_component_fractions": (
            torch.bincount(chosen.cpu(), minlength=len(selected_names)).double() / len(chosen)
        ).tolist(),
        "responsibilities": responsibility_summary(test_component_log_prob, weights),
        "history": history,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--joint-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9001)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    validate_geometries()
    seed_all(args.seed)
    threads = min(4, max(1, os.cpu_count() or 1))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_names = list(COMPONENTS)
    validation_columns = []
    test_columns = []
    component_rows: list[dict[str, object]] = []
    for name in all_names:
        component_dir = locate_component_dir(args.component_root, name)
        validation_columns.append(np.load(component_dir / "validation_log_prob.npy"))
        test_columns.append(np.load(component_dir / "test_log_prob.npy"))
        metrics = json.loads((component_dir / "metrics.json").read_text())
        component_rows.append(
            {
                "component": name,
                "geometry": metrics["geometry"],
                "seed": metrics["seed"],
                "parameters": metrics["architecture"]["parameters"],
                "validation_nll": metrics["best_validation_nll"],
                "test_nll": metrics["test_nll_nats_per_image"],
                "best_epoch": metrics["best_epoch"],
                "train_examples_per_second": metrics["train_examples_per_second"],
            }
        )
    validation_log_prob = np.stack(validation_columns, axis=1).astype(np.float64)
    test_log_prob = np.stack(test_columns, axis=1).astype(np.float64)

    component_table_path = args.output / "component_results.csv"
    with component_table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(component_rows[0]))
        writer.writeheader()
        writer.writerows(component_rows)

    nll_matrix = -validation_log_prob
    correlation = np.corrcoef(nll_matrix, rowvar=False)
    mean_abs_difference = np.abs(
        validation_log_prob[:, :, None] - validation_log_prob[:, None, :]
    ).mean(axis=0)
    np.savetxt(args.output / "validation_nll_correlation.csv", correlation, delimiter=",")
    np.savetxt(
        args.output / "validation_mean_abs_log_prob_difference.csv",
        mean_abs_difference,
        delimiter=",",
    )

    geometry_names = list(GEOMETRY_COMPONENTS)
    geometry_indices = [all_names.index(name) for name in geometry_names]
    geometry_validation = validation_log_prob[:, geometry_indices]
    search_by_size: dict[str, object] = {}
    for size in range(1, 5):
        search_by_size[str(size)] = exhaustive_best_subset(geometry_validation, geometry_names, size)

    heterogeneous_selection = search_by_size["4"]
    heterogeneous = evaluate_fixed_mixture(
        validation_log_prob,
        test_log_prob,
        all_names,
        heterogeneous_selection,
    )

    homogeneous_names = list(HOMOGENEOUS_COMPONENTS)
    homogeneous_indices = [all_names.index(name) for name in homogeneous_names]
    homogeneous_weights, homogeneous_val, homogeneous_iterations = optimize_mixture_weights(
        validation_log_prob[:, homogeneous_indices]
    )
    homogeneous_selection = {
        "components": homogeneous_names,
        "weights": homogeneous_weights.tolist(),
        "validation_nll": homogeneous_val,
        "em_iterations": homogeneous_iterations,
    }
    homogeneous = evaluate_fixed_mixture(
        validation_log_prob,
        test_log_prob,
        all_names,
        homogeneous_selection,
    )

    global_selection = exhaustive_best_subset(validation_log_prob, all_names, 4)
    global_best = evaluate_fixed_mixture(
        validation_log_prob,
        test_log_prob,
        all_names,
        global_selection,
    )

    for result in (heterogeneous, homogeneous, global_best):
        indices = [all_names.index(name) for name in result["components"]]
        result["responsibilities"] = responsibility_summary(
            test_log_prob[:, indices], np.asarray(result["weights"], dtype=np.float64)
        )

    data = np.load(args.data_dir / "static_binarized_mnist_uint8.npz")
    train = torch.from_numpy(np.asarray(data["train"], dtype=np.int64))
    valid = torch.from_numpy(np.asarray(data["valid"], dtype=np.int64))
    test = torch.from_numpy(np.asarray(data["test"], dtype=np.int64))

    heterogeneous_joint = joint_finetune(
        "heterogeneous-joint",
        list(heterogeneous["components"]),
        np.asarray(heterogeneous["weights"], dtype=np.float64),
        args.component_root,
        train,
        valid,
        test,
        args.output,
        device,
        args.joint_epochs,
        args.batch_size,
        args.seed,
    )
    homogeneous_joint = joint_finetune(
        "homogeneous-joint",
        list(homogeneous["components"]),
        np.asarray(homogeneous["weights"], dtype=np.float64),
        args.component_root,
        train,
        valid,
        test,
        args.output,
        device,
        max(6, args.joint_epochs - 2),
        args.batch_size,
        args.seed + 1,
    )

    normalization_error = exact_mixture_normalization_error(device)
    if normalization_error > 2e-5:
        raise AssertionError(f"mixture normalization error {normalization_error}")

    summary = {
        "protocol": "static binarized MNIST, exact NLL, parameter-matched four-tree mixtures",
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "cpu_count": os.cpu_count(),
            "threads": torch.get_num_threads(),
        },
        "correctness": {"mixture_normalization_absolute_error": normalization_error},
        "prior_cycle_reference": {
            "model": "single group64-r32",
            "parameters": 1030176,
            "test_nll": PRIOR_CYCLE_R32_NLL,
        },
        "components": component_rows,
        "geometry_subset_search": search_by_size,
        "heterogeneous_weight_only": heterogeneous,
        "homogeneous_weight_only": homogeneous,
        "global_best_weight_only": global_best,
        "heterogeneous_joint": heterogeneous_joint,
        "homogeneous_joint": homogeneous_joint,
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
