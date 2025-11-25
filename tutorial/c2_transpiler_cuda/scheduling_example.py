#!/usr/bin/env python3
"""
Threadblock Graph Scheduling Example

This script demonstrates the scheduling algorithm used by Mirage's transpiler
to order threadblock-level operations.

The algorithm uses a modified topological sort with "depth" labeling to:
1. Minimize the number of __syncthreads() calls
2. Group operators that can execute without synchronization

Key insight: Operators at the same depth can execute without sync between them.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum


class OpType(Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    MATMUL = "MATMUL"
    EXP = "EXP"
    SQUARE = "SQUARE"
    REDUCTION = "REDUCTION"
    ADD = "ADD"
    MUL = "MUL"
    DIV = "DIV"
    ACCUM = "ACCUM"


@dataclass
class Tensor:
    name: str
    producer: Optional['Operator'] = None


@dataclass
class Operator:
    name: str
    op_type: OpType
    inputs: List[Tensor] = field(default_factory=list)
    outputs: List[Tensor] = field(default_factory=list)
    forloop_dim: int = -1  # -1 means no forloop dimension

    # Scheduling metadata
    depth: int = 0
    is_fused_with_prev: bool = False
    is_fused_with_next: bool = False
    fuse_chain_idx: int = 0


def can_fuse(prev_op: Operator, curr_op: Operator) -> bool:
    """
    Check if curr_op can be fused with prev_op.

    Fusion rules (from resolve_tb_fusion.cc):
    1. curr_op must be unary elementwise or forloop_accum_no_red
    2. prev_op's output must have exactly one consumer
    3. prev_op must not be forloop_accum
    4. prev_op must not be input with forloop_dim == -1
    5. prev_op must not be reduction_max (has two outputs)
    """
    # Only fuse unary ops
    unary_ops = {OpType.EXP, OpType.SQUARE, OpType.ACCUM}
    if curr_op.op_type not in unary_ops:
        return False

    # Don't fuse with accumulators (they cross loop iterations)
    if prev_op.op_type == OpType.ACCUM:
        return False

    # Don't fuse with pre-loop inputs
    if prev_op.op_type == OpType.INPUT and prev_op.forloop_dim == -1:
        return False

    return True


def compute_depths_and_fusion(operators: List[Operator]) -> Dict[str, int]:
    """
    Compute depth for each operator using dynamic programming.

    depth[op] = max(depth[input.producer] for input in op.inputs) + 1
    unless op is fused with prev, then depth[op] = depth[prev]
    """
    # Build consumer count for fusion analysis
    consumer_count: Dict[str, int] = {}
    for op in operators:
        for inp in op.inputs:
            if inp.producer:
                key = inp.name
                consumer_count[key] = consumer_count.get(key, 0) + 1

    # First pass: determine fusion relationships
    for op in operators:
        if op.inputs and op.inputs[0].producer:
            prev_op = op.inputs[0].producer
            prev_out = prev_op.outputs[0] if prev_op.outputs else None

            if prev_out and consumer_count.get(prev_out.name, 0) == 1:
                if can_fuse(prev_op, op):
                    op.is_fused_with_prev = True
                    prev_op.is_fused_with_next = True

    # Second pass: compute depths
    chain_idx = 0
    depths = {}

    for op in operators:
        if op.op_type == OpType.INPUT:
            op.depth = 0
            op.fuse_chain_idx = chain_idx
            chain_idx += 1
        elif op.is_fused_with_prev:
            prev_op = op.inputs[0].producer
            op.depth = prev_op.depth
            op.fuse_chain_idx = prev_op.fuse_chain_idx
        else:
            max_input_depth = -1
            for inp in op.inputs:
                if inp.producer:
                    max_input_depth = max(max_input_depth, inp.producer.depth)
            op.depth = max_input_depth + 1
            op.fuse_chain_idx = chain_idx
            chain_idx += 1

        depths[op.name] = op.depth

    return depths


def schedule_operators(operators: List[Operator]) -> Tuple[List[str], int]:
    """
    Schedule operators based on their depth.

    Returns: (schedule, num_syncthreads)
    """
    # Sort by depth, then by chain index
    sorted_ops = sorted(operators, key=lambda op: (op.depth, op.fuse_chain_idx))

    schedule = []
    num_syncs = 0
    prev_depth = -1

    for op in sorted_ops:
        if op.depth > prev_depth and prev_depth >= 0:
            schedule.append("__syncthreads()")
            num_syncs += 1
        schedule.append(op.name)
        prev_depth = op.depth

    return schedule, num_syncs


def visualize_graph(operators: List[Operator]):
    """Print the operator graph structure."""
    print("\n  Operator Graph:")
    print("  " + "-"*50)

    for op in operators:
        inputs = ", ".join(inp.name for inp in op.inputs) if op.inputs else "none"
        outputs = ", ".join(out.name for out in op.outputs) if op.outputs else "none"
        fused = " [FUSED]" if op.is_fused_with_prev else ""
        print(f"  {op.name:20s} depth={op.depth} inputs=({inputs}) outputs=({outputs}){fused}")


def visualize_schedule(schedule: List[str], num_syncs: int):
    """Print the schedule with sync points."""
    print("\n  Execution Schedule:")
    print("  " + "-"*50)

    for item in schedule:
        if item == "__syncthreads()":
            print(f"  {'='*40}")
            print(f"  {item}")
            print(f"  {'='*40}")
        else:
            print(f"    {item}")

    print(f"\n  Total __syncthreads(): {num_syncs}")


def example_rmsnorm():
    """RMSNorm example: X -> square -> reduce -> sqrt -> div -> output"""
    print("\n" + "="*70)
    print("Example 1: RMSNorm Kernel")
    print("="*70)
    print("""
    RMSNorm computation:
    1. X_sq = X * X (square)
    2. mean = reduce_sum(X_sq) / N
    3. rms = sqrt(mean)
    4. output = X / rms

    Graph structure:
    INPUT(X) -> SQUARE -> REDUCTION -> SQRT -> DIV -> OUTPUT
                           ^                    ^
                           |                    |
    INPUT(X) --------------+--------------------+
    """)

    # Create tensors
    t_x = Tensor("x")
    t_x_sq = Tensor("x_sq")
    t_sum = Tensor("sum")
    t_rms = Tensor("rms")
    t_norm = Tensor("norm")

    # Create operators
    op_input = Operator("INPUT_X", OpType.INPUT, [], [t_x], forloop_dim=0)
    t_x.producer = op_input

    op_square = Operator("SQUARE", OpType.SQUARE, [t_x], [t_x_sq])
    t_x_sq.producer = op_square

    op_reduce = Operator("REDUCTION", OpType.REDUCTION, [t_x_sq], [t_sum])
    t_sum.producer = op_reduce

    op_sqrt = Operator("SQRT", OpType.SQUARE, [t_sum], [t_rms])  # Using SQUARE as proxy
    t_rms.producer = op_sqrt

    op_div = Operator("DIV", OpType.DIV, [t_x, t_rms], [t_norm])
    t_norm.producer = op_div

    op_output = Operator("OUTPUT", OpType.OUTPUT, [t_norm], [])

    operators = [op_input, op_square, op_reduce, op_sqrt, op_div, op_output]

    # Compute schedule
    depths = compute_depths_and_fusion(operators)
    schedule, num_syncs = schedule_operators(operators)

    visualize_graph(operators)
    visualize_schedule(schedule, num_syncs)


def example_attention():
    """Attention example with parallel paths."""
    print("\n" + "="*70)
    print("Example 2: Simplified Attention (Q, K, V)")
    print("="*70)
    print("""
    Attention computation:
    1. scores = Q @ K^T
    2. weights = softmax(scores)
    3. output = weights @ V

    Graph structure (simplified):
    INPUT(Q) ---------+
                      |
                      v
    INPUT(K) -----> MATMUL_QK -> EXP -> REDUCTION -> DIV -> MATMUL_OUT -> OUTPUT
                                  ^       ^           ^         ^
                                  |       |           |         |
                                  +-------+-----------+         |
    INPUT(V) -----------------------------------------------+
    """)

    # Create tensors
    t_q = Tensor("Q")
    t_k = Tensor("K")
    t_v = Tensor("V")
    t_qk = Tensor("QK")
    t_exp = Tensor("exp_QK")
    t_sum = Tensor("sum")
    t_softmax = Tensor("softmax")
    t_out = Tensor("output")

    # Create operators
    op_input_q = Operator("INPUT_Q", OpType.INPUT, [], [t_q], forloop_dim=0)
    t_q.producer = op_input_q

    op_input_k = Operator("INPUT_K", OpType.INPUT, [], [t_k], forloop_dim=0)
    t_k.producer = op_input_k

    op_input_v = Operator("INPUT_V", OpType.INPUT, [], [t_v], forloop_dim=-1)  # Static
    t_v.producer = op_input_v

    op_matmul_qk = Operator("MATMUL_QK", OpType.MATMUL, [t_q, t_k], [t_qk])
    t_qk.producer = op_matmul_qk

    op_exp = Operator("EXP", OpType.EXP, [t_qk], [t_exp])
    t_exp.producer = op_exp

    op_reduce = Operator("REDUCTION", OpType.REDUCTION, [t_exp], [t_sum])
    t_sum.producer = op_reduce

    op_div = Operator("DIV", OpType.DIV, [t_exp, t_sum], [t_softmax])
    t_softmax.producer = op_div

    op_matmul_out = Operator("MATMUL_OUT", OpType.MATMUL, [t_softmax, t_v], [t_out])
    t_out.producer = op_matmul_out

    op_output = Operator("OUTPUT", OpType.OUTPUT, [t_out], [])

    operators = [op_input_q, op_input_k, op_input_v, op_matmul_qk, op_exp,
                 op_reduce, op_div, op_matmul_out, op_output]

    depths = compute_depths_and_fusion(operators)
    schedule, num_syncs = schedule_operators(operators)

    visualize_graph(operators)
    visualize_schedule(schedule, num_syncs)


def example_fusion():
    """Example showing operator fusion."""
    print("\n" + "="*70)
    print("Example 3: Matmul with Fused Epilogue")
    print("="*70)
    print("""
    Matmul followed by unary operations that can be fused:

    INPUT(A) --+
               |
               v
    INPUT(B) -> MATMUL -> EXP -> SQUARE -> ACCUM -> OUTPUT

    With fusion: MATMUL, EXP, SQUARE, and ACCUM share the same depth!
    The values stay in registers, avoiding shared memory round-trips.
    """)

    # Create tensors
    t_a = Tensor("A")
    t_b = Tensor("B")
    t_c = Tensor("C")
    t_exp = Tensor("exp_C")
    t_sq = Tensor("sq")
    t_acc = Tensor("accum")

    # Create operators
    op_input_a = Operator("INPUT_A", OpType.INPUT, [], [t_a], forloop_dim=0)
    t_a.producer = op_input_a

    op_input_b = Operator("INPUT_B", OpType.INPUT, [], [t_b], forloop_dim=0)
    t_b.producer = op_input_b

    op_matmul = Operator("MATMUL", OpType.MATMUL, [t_a, t_b], [t_c])
    t_c.producer = op_matmul

    op_exp = Operator("EXP", OpType.EXP, [t_c], [t_exp])
    t_exp.producer = op_exp

    op_square = Operator("SQUARE", OpType.SQUARE, [t_exp], [t_sq])
    t_sq.producer = op_square

    op_accum = Operator("ACCUM", OpType.ACCUM, [t_sq], [t_acc])
    t_acc.producer = op_accum

    op_output = Operator("OUTPUT", OpType.OUTPUT, [t_acc], [])

    operators = [op_input_a, op_input_b, op_matmul, op_exp, op_square, op_accum, op_output]

    depths = compute_depths_and_fusion(operators)
    schedule, num_syncs = schedule_operators(operators)

    visualize_graph(operators)
    visualize_schedule(schedule, num_syncs)

    print("\n  Fusion chains:")
    chains = {}
    for op in operators:
        idx = op.fuse_chain_idx
        if idx not in chains:
            chains[idx] = []
        chains[idx].append(op.name)

    for idx, chain in sorted(chains.items()):
        if len(chain) > 1:
            print(f"    Chain {idx}: {' -> '.join(chain)} (FUSED)")
        else:
            print(f"    Chain {idx}: {chain[0]}")


def example_parallel_paths():
    """Example with parallel computation paths."""
    print("\n" + "="*70)
    print("Example 4: Parallel Paths (Different Schedules)")
    print("="*70)
    print("""
    A graph with two parallel computation paths:

         INPUT_A          INPUT_B
            |                |
            v                v
         SQUARE           SQUARE
            |                |
            v                v
         EXP              EXP
            |                |
            +-------+--------+
                    |
                    v
                   ADD
                    |
                    v
                 OUTPUT

    Both paths can execute in parallel at each depth level!
    """)

    t_a = Tensor("A")
    t_b = Tensor("B")
    t_sq_a = Tensor("sq_A")
    t_sq_b = Tensor("sq_B")
    t_exp_a = Tensor("exp_A")
    t_exp_b = Tensor("exp_B")
    t_add = Tensor("sum")

    op_input_a = Operator("INPUT_A", OpType.INPUT, [], [t_a], forloop_dim=0)
    t_a.producer = op_input_a

    op_input_b = Operator("INPUT_B", OpType.INPUT, [], [t_b], forloop_dim=0)
    t_b.producer = op_input_b

    op_sq_a = Operator("SQUARE_A", OpType.SQUARE, [t_a], [t_sq_a])
    t_sq_a.producer = op_sq_a

    op_sq_b = Operator("SQUARE_B", OpType.SQUARE, [t_b], [t_sq_b])
    t_sq_b.producer = op_sq_b

    op_exp_a = Operator("EXP_A", OpType.EXP, [t_sq_a], [t_exp_a])
    t_exp_a.producer = op_exp_a

    op_exp_b = Operator("EXP_B", OpType.EXP, [t_sq_b], [t_exp_b])
    t_exp_b.producer = op_exp_b

    op_add = Operator("ADD", OpType.ADD, [t_exp_a, t_exp_b], [t_add])
    t_add.producer = op_add

    op_output = Operator("OUTPUT", OpType.OUTPUT, [t_add], [])

    operators = [op_input_a, op_input_b, op_sq_a, op_sq_b, op_exp_a, op_exp_b, op_add, op_output]

    depths = compute_depths_and_fusion(operators)
    schedule, num_syncs = schedule_operators(operators)

    visualize_graph(operators)
    visualize_schedule(schedule, num_syncs)

    print("\n  Note: Operators at the same depth can execute concurrently")
    print("        (limited by available parallelism in the thread block)")


def main():
    print("""
    ================================================================
    Mirage Transpiler: Scheduling Algorithm Demonstration
    ================================================================

    The scheduler determines the execution order of threadblock operators.
    Goals:
    1. Minimize __syncthreads() calls (each sync is expensive)
    2. Enable operator fusion (keep data in registers)

    Algorithm: Modified topological sort with depth labeling
    - depth[INPUT] = 0
    - depth[fused_op] = depth[prev_op]
    - depth[op] = max(depth[inputs]) + 1

    Sync is inserted when depth increases.
    """)

    example_rmsnorm()
    example_attention()
    example_fusion()
    example_parallel_paths()

    print("\n" + "="*70)
    print("For more details, see:")
    print("  - src/transpiler/sched_tb_graph.cc")
    print("  - src/transpiler/resolve_tb_fusion.cc")
    print("="*70)


if __name__ == "__main__":
    main()
