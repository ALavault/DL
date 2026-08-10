from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def num_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@dataclass(frozen=True)
class CircuitSpec:
    rank: int
    cp_rank: int
    max_groups: int


class GroupedTensorTreeCircuit(nn.Module):
    """Exact normalized density over active binary variables in a padded tree.

    The active 28x28 image variables are embedded into a 32x32 Morton tree.
    Inactive leaves are analytically marginalized by feeding an all-ones
    likelihood message. Internal conditionals use non-negative CP factors:

      p(z_l, z_r | z_p) = sum_s A[z_p,s] B[s,z_l] C[s,z_r].

    A is row-stochastic over s, and B/C are row-stochastic over child states.
    Parameters may be shared by an adjustable number of contiguous node groups
    at every level. max_groups=1 is the compact model; a large value gives one
    factorization per tree node.
    """

    def __init__(
        self,
        num_leaves: int,
        active_positions: Tensor,
        rank: int = 16,
        cp_rank: int | None = None,
        max_groups: int = 1,
    ) -> None:
        super().__init__()
        if num_leaves <= 0 or num_leaves & (num_leaves - 1):
            raise ValueError("num_leaves must be a positive power of two")
        active_positions = torch.as_tensor(active_positions, dtype=torch.long)
        if active_positions.ndim != 1:
            raise ValueError("active_positions must be one-dimensional")
        if len(active_positions) == 0:
            raise ValueError("at least one active variable is required")
        if active_positions.min().item() < 0 or active_positions.max().item() >= num_leaves:
            raise ValueError("active position outside padded tree")
        if torch.unique(active_positions).numel() != active_positions.numel():
            raise ValueError("active positions must be unique")
        if max_groups <= 0:
            raise ValueError("max_groups must be positive")

        self.num_leaves = int(num_leaves)
        self.num_active = int(active_positions.numel())
        self.rank = int(rank)
        self.cp_rank = int(cp_rank if cp_rank is not None else rank)
        self.max_groups = int(max_groups)
        self.num_levels = int(math.log2(num_leaves))
        self.register_buffer("active_positions", active_positions)

        # Only observed leaves need emission parameters. Padded leaves are
        # marginalized exactly and therefore do not carry decorative weights.
        self.leaf_logits = nn.Parameter(torch.empty(self.num_active, self.rank, 2))
        self.root_logits = nn.Parameter(torch.empty(self.rank))

        self.a_logits = nn.ParameterList()
        self.b_logits = nn.ParameterList()
        self.c_logits = nn.ParameterList()
        nodes = num_leaves // 2
        for level in range(self.num_levels):
            groups = min(nodes, self.max_groups)
            self.a_logits.append(nn.Parameter(torch.empty(groups, self.rank, self.cp_rank)))
            self.b_logits.append(nn.Parameter(torch.empty(groups, self.cp_rank, self.rank)))
            self.c_logits.append(nn.Parameter(torch.empty(groups, self.cp_rank, self.rank)))
            # Contiguous spatial groups in Morton order. Registering these
            # avoids rebuilding routing tensors in the hot loop.
            group_idx = torch.div(
                torch.arange(nodes, dtype=torch.long) * groups,
                nodes,
                rounding_mode="floor",
            )
            self.register_buffer(f"group_index_{level}", group_idx)
            nodes //= 2
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.leaf_logits.normal_(0.0, 0.08)
            self.root_logits.zero_()
            for params in (self.a_logits, self.b_logits, self.c_logits):
                for p in params:
                    p.normal_(0.0, 0.08)

    @property
    def groups_per_level(self) -> list[int]:
        return [p.shape[0] for p in self.a_logits]

    @torch.no_grad()
    def initialize_from_product_mixture(
        self,
        component_probabilities: Tensor,
        mixture_probabilities: Tensor,
        diagonal_strength: float = 6.0,
    ) -> None:
        expected = (self.rank, self.num_active)
        if tuple(component_probabilities.shape) != expected:
            raise ValueError(f"component_probabilities must be {expected}")
        if tuple(mixture_probabilities.shape) != (self.rank,):
            raise ValueError(f"mixture_probabilities must be {(self.rank,)}")
        if self.cp_rank < self.rank:
            raise ValueError("cp_rank must be >= rank for diagonal initialization")

        eps = 1e-5
        p = component_probabilities.clamp(eps, 1.0 - eps)
        # [R,D] -> [D,R,2]
        self.leaf_logits.copy_(
            torch.stack(((1.0 - p).log(), p.log()), dim=-1).permute(1, 0, 2)
        )
        self.root_logits.copy_(mixture_probabilities.clamp_min(eps).log())

        for level in range(self.num_levels):
            a = self.a_logits[level]
            b = self.b_logits[level]
            c = self.c_logits[level]
            a.fill_(-diagonal_strength)
            b.fill_(-diagonal_strength)
            c.fill_(-diagonal_strength)
            idx = torch.arange(self.rank, device=a.device)
            a[:, idx, idx] = diagonal_strength
            b[:, idx, idx] = diagonal_strength
            c[:, idx, idx] = diagonal_strength
            # Symmetry breaking is necessary because the exact embedded
            # mixture otherwise gives identical gradients to many factors.
            a.add_(0.005 * torch.randn_like(a))
            b.add_(0.005 * torch.randn_like(b))
            c.add_(0.005 * torch.randn_like(c))

    def _leaf_messages(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or x.shape[1] != self.num_active:
            raise ValueError(f"expected [N,{self.num_active}], got {tuple(x.shape)}")
        if x.dtype != torch.long:
            x = x.long()

        probs = F.softmax(self.leaf_logits, dim=-1)  # [D,R,2]
        table = probs.permute(0, 2, 1).reshape(self.num_active * 2, self.rank)
        offsets = torch.arange(self.num_active, device=x.device, dtype=x.dtype) * 2
        flat = (x + offsets.unsqueeze(0)).reshape(-1)
        active_like = table.index_select(0, flat).reshape(x.shape[0], self.num_active, self.rank)

        # A marginalized leaf contributes likelihood 1 for every latent state.
        leaf_like = torch.ones(
            x.shape[0], self.num_leaves, self.rank,
            device=x.device, dtype=active_like.dtype,
        )
        leaf_like = leaf_like.index_copy(1, self.active_positions, active_like)

        norm = leaf_like.sum(dim=-1).clamp_min(torch.finfo(leaf_like.dtype).tiny)
        return leaf_like / norm.unsqueeze(-1), norm.log()

    def log_prob(self, x: Tensor) -> Tensor:
        message, scale = self._leaf_messages(x)
        for level in range(self.num_levels):
            left = message[:, 0::2, :]
            right = message[:, 1::2, :]
            left_scale = scale[:, 0::2]
            right_scale = scale[:, 1::2]

            a = F.softmax(self.a_logits[level], dim=-1)
            b = F.softmax(self.b_logits[level], dim=-1)
            c = F.softmax(self.c_logits[level], dim=-1)
            groups = a.shape[0]

            if groups == 1:
                flat_rows = left.shape[0] * left.shape[1]
                lp = left.reshape(flat_rows, self.rank) @ b[0].t()
                rp = right.reshape(flat_rows, self.rank) @ c[0].t()
                parent = (lp * rp) @ a[0].t()
                parent = parent.reshape(left.shape[0], left.shape[1], self.rank)
            else:
                group_index = getattr(self, f"group_index_{level}")
                an = a.index_select(0, group_index)  # [nodes,R,S]
                bn = b.index_select(0, group_index)  # [nodes,S,R]
                cn = c.index_select(0, group_index)
                lp = torch.einsum("bnr,nsr->bns", left, bn)
                rp = torch.einsum("bnr,nsr->bns", right, cn)
                parent = torch.einsum("bns,nrs->bnr", lp * rp, an)

            norm = parent.sum(dim=-1).clamp_min(torch.finfo(parent.dtype).tiny)
            message = parent / norm.unsqueeze(-1)
            scale = left_scale + right_scale + norm.log()

        prior = F.softmax(self.root_logits, dim=-1)
        root_like = (message[:, 0, :] * prior).sum(dim=-1).clamp_min(
            torch.finfo(message.dtype).tiny
        )
        return scale[:, 0] + root_like.log()

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | str | None = None) -> Tensor:
        if device is None:
            device = self.root_logits.device
        prior = F.softmax(self.root_logits, dim=-1).to(device)
        states = torch.multinomial(prior.expand(n, -1), 1).reshape(n, 1)

        for level in reversed(range(self.num_levels)):
            a = F.softmax(self.a_logits[level], dim=-1).to(device)
            b = F.softmax(self.b_logits[level], dim=-1).to(device)
            c = F.softmax(self.c_logits[level], dim=-1).to(device)
            group_index = getattr(self, f"group_index_{level}").to(device)
            an = a.index_select(0, group_index)  # [nodes,R,S]
            bn = b.index_select(0, group_index)  # [nodes,S,R]
            cn = c.index_select(0, group_index)
            batch, nodes = states.shape

            parent_probs = an.unsqueeze(0).expand(batch, -1, -1, -1)
            parent_probs = parent_probs.gather(
                2, states[:, :, None, None].expand(-1, -1, 1, self.cp_rank)
            ).squeeze(2)
            s = torch.multinomial(
                parent_probs.reshape(-1, self.cp_rank), 1
            ).reshape(batch, nodes)

            b_probs = bn.unsqueeze(0).expand(batch, -1, -1, -1).gather(
                2, s[:, :, None, None].expand(-1, -1, 1, self.rank)
            ).squeeze(2)
            c_probs = cn.unsqueeze(0).expand(batch, -1, -1, -1).gather(
                2, s[:, :, None, None].expand(-1, -1, 1, self.rank)
            ).squeeze(2)
            left = torch.multinomial(
                b_probs.reshape(-1, self.rank), 1
            ).reshape(batch, nodes)
            right = torch.multinomial(
                c_probs.reshape(-1, self.rank), 1
            ).reshape(batch, nodes)
            states = torch.stack((left, right), dim=-1).reshape(batch, -1)

        active_states = states.index_select(1, self.active_positions.to(device))
        leaf_probs = F.softmax(self.leaf_logits, dim=-1).to(device)
        positions = torch.arange(self.num_active, device=device).unsqueeze(0)
        selected = leaf_probs[positions, active_states]
        return torch.multinomial(
            selected.reshape(-1, 2), 1
        ).reshape(n, self.num_active)


def exact_active_normalization_error(device: torch.device | str = "cpu") -> float:
    torch.manual_seed(11)
    # Four observed variables embedded among eight leaves; the remaining leaves
    # are analytically marginalized.
    model = GroupedTensorTreeCircuit(
        8, torch.tensor([0, 2, 5, 7]), rank=3, cp_rank=3, max_groups=8
    ).to(device)
    states = torch.tensor(
        [[(i >> bit) & 1 for bit in range(4)] for i in range(16)],
        dtype=torch.long,
        device=device,
    )
    return abs(model.log_prob(states).exp().sum().item() - 1.0)
