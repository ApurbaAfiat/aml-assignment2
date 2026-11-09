"""
Architecture manipulation and encoding utilities for NATS-Bench TSS.

Provides functions to:
- Parse architecture strings into graph edges.
- Convert architectures into one-hot numeric feature vectors for surrogate models (e.g., Gaussian Processes in Bayesian Optimization).
- Decode feature vectors back into architecture strings.
"""

from typing import List, Tuple
import numpy as np


# The 5 possible operations defined in NATS-Bench / NAS-Bench-201 TSS
OPERATIONS = (
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
)

OP_TO_IDX = {op: i for i, op in enumerate(OPERATIONS)}
IDX_TO_OP = {i: op for i, op in enumerate(OPERATIONS)}

NUM_EDGES = 6
NUM_OPS = len(OPERATIONS)  # 5
ENCODING_DIM = NUM_EDGES * NUM_OPS  # 6 * 5 = 30


def parse_arch(arch_str: str) -> List[Tuple[int, int, str]]:
    edges = []
    # Each '+' group corresponds to incoming edges for dst = 1, 2, 3
    groups = arch_str.strip().split("+")
    for dst, group in enumerate(groups, start=1):
        items = group.strip().strip("|").split("|")
        for src, item in enumerate(items):
            op_name = item.split("~")[0]
            edges.append((src, dst, op_name))

    if len(edges) != NUM_EDGES:
        raise ValueError(f"Expected {NUM_EDGES} edges, got {len(edges)} from '{arch_str}'")

    return edges


def arch_to_vector(arch_str: str) -> np.ndarray:
    edges = parse_arch(arch_str)
    vector = np.zeros(ENCODING_DIM, dtype=np.float32)

    for edge_idx, (_, _, op) in enumerate(edges):
        if op not in OP_TO_IDX:
            raise ValueError(f"Unknown operation '{op}' in architecture '{arch_str}'")
        op_idx = OP_TO_IDX[op]
        # Offset for this edge is edge_idx * 5
        vector[edge_idx * NUM_OPS + op_idx] = 1.0

    return vector


def vector_to_arch(vector: np.ndarray) -> str:
    flat = np.asarray(vector).ravel()
    if flat.shape[0] != ENCODING_DIM:
        raise ValueError(f"Expected vector of length {ENCODING_DIM}, got {flat.shape[0]}")

    edge_ops: List[str] = []
    for edge_idx in range(NUM_EDGES):
        edge_slice = flat[edge_idx * NUM_OPS : (edge_idx + 1) * NUM_OPS]
        chosen_op_idx = int(np.argmax(edge_slice))
        edge_ops.append(IDX_TO_OP[chosen_op_idx])

    part1 = f"|{edge_ops[0]}~0|"
    part2 = f"|{edge_ops[1]}~0|{edge_ops[2]}~1|"
    part3 = f"|{edge_ops[3]}~0|{edge_ops[4]}~1|{edge_ops[5]}~2|"

    return f"{part1}+{part2}+{part3}"