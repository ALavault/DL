from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometrySpec:
    name: str
    kind: str
    offset_y: int = 2
    offset_x: int = 2
    interleave: str = "xy"


GEOMETRIES: dict[str, GeometrySpec] = {
    "morton-center-xy": GeometrySpec("morton-center-xy", "morton", 2, 2, "xy"),
    "morton-center-yx": GeometrySpec("morton-center-yx", "morton", 2, 2, "yx"),
    "hilbert-center": GeometrySpec("hilbert-center", "hilbert", 2, 2),
    "snake-row-center": GeometrySpec("snake-row-center", "snake-row", 2, 2),
    "morton-topleft-xy": GeometrySpec("morton-topleft-xy", "morton", 0, 0, "xy"),
    "morton-bottomright-xy": GeometrySpec("morton-bottomright-xy", "morton", 4, 4, "xy"),
}


COMPONENTS: dict[str, tuple[str, int]] = {
    "morton-center-xy-s1234": ("morton-center-xy", 1234),
    "morton-center-yx-s1234": ("morton-center-yx", 1234),
    "hilbert-center-s1234": ("hilbert-center", 1234),
    "snake-row-center-s1234": ("snake-row-center", 1234),
    "morton-topleft-xy-s1234": ("morton-topleft-xy", 1234),
    "morton-bottomright-xy-s1234": ("morton-bottomright-xy", 1234),
    "morton-center-xy-s2234": ("morton-center-xy", 2234),
    "morton-center-xy-s3234": ("morton-center-xy", 3234),
    "morton-center-xy-s4234": ("morton-center-xy", 4234),
}

GEOMETRY_COMPONENTS = tuple(name for name, (_, seed) in COMPONENTS.items() if seed == 1234)
HOMOGENEOUS_COMPONENTS = (
    "morton-center-xy-s1234",
    "morton-center-xy-s2234",
    "morton-center-xy-s3234",
    "morton-center-xy-s4234",
)


def morton_code(y: int, x: int, side: int = 32, interleave: str = "xy") -> int:
    if interleave not in {"xy", "yx"}:
        raise ValueError(f"unknown Morton interleave {interleave!r}")
    out = 0
    bit = 0
    while (1 << bit) < side:
        xb = (x >> bit) & 1
        yb = (y >> bit) & 1
        if interleave == "xy":
            out |= xb << (2 * bit)
            out |= yb << (2 * bit + 1)
        else:
            out |= yb << (2 * bit)
            out |= xb << (2 * bit + 1)
        bit += 1
    return out


def _hilbert_rotate(scale: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = scale - 1 - x
            y = scale - 1 - y
        x, y = y, x
    return x, y


def hilbert_code(y: int, x: int, side: int = 32) -> int:
    if side <= 0 or side & (side - 1):
        raise ValueError("side must be a positive power of two")
    d = 0
    scale = side // 2
    xx, yy = x, y
    while scale > 0:
        rx = 1 if (xx & scale) else 0
        ry = 1 if (yy & scale) else 0
        d += scale * scale * ((3 * rx) ^ ry)
        xx, yy = _hilbert_rotate(scale, xx, yy, rx, ry)
        scale //= 2
    return d


def leaf_code(y: int, x: int, spec: GeometrySpec, side: int = 32) -> int:
    yy = y + spec.offset_y
    xx = x + spec.offset_x
    if not (0 <= yy < side and 0 <= xx < side):
        raise ValueError(f"pixel {(y, x)} maps outside the {side}x{side} canvas")
    if spec.kind == "morton":
        return morton_code(yy, xx, side=side, interleave=spec.interleave)
    if spec.kind == "hilbert":
        return hilbert_code(yy, xx, side=side)
    if spec.kind == "snake-row":
        return yy * side + (xx if yy % 2 == 0 else side - 1 - xx)
    raise ValueError(f"unknown geometry kind {spec.kind!r}")


def active_positions_and_order(geometry: str) -> tuple[np.ndarray, np.ndarray]:
    spec = GEOMETRIES[geometry]
    triples: list[tuple[int, int]] = []
    for y in range(28):
        for x in range(28):
            original = y * 28 + x
            triples.append((leaf_code(y, x, spec), original))
    triples.sort()
    positions = np.asarray([leaf for leaf, _ in triples], dtype=np.int64)
    order = np.asarray([original for _, original in triples], dtype=np.int64)
    if len(np.unique(positions)) != 784:
        raise AssertionError(f"geometry {geometry} is not injective")
    if positions.min() < 0 or positions.max() >= 1024:
        raise AssertionError(f"geometry {geometry} produced an invalid leaf")
    return positions, order


def validate_geometries() -> None:
    for name in GEOMETRIES:
        positions, order = active_positions_and_order(name)
        if sorted(order.tolist()) != list(range(784)):
            raise AssertionError(f"geometry {name} does not define a permutation")
        if len(np.unique(positions)) != len(positions):
            raise AssertionError(f"geometry {name} reuses a leaf")
