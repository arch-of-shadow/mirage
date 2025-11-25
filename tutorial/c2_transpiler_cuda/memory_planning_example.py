#!/usr/bin/env python3
"""
Memory Planning Example - Dynamic Storage Allocation (DSA)

This script demonstrates the memory planning algorithms used by Mirage's
transpiler to allocate shared memory for STensors.

The problem: Given tensors with (size, alloc_time, free_time), find memory
addresses that minimize peak usage while ensuring no overlapping lifetimes
share the same memory.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import heapq


@dataclass
class TensorDecl:
    """Declaration of a tensor with its size and lifecycle."""
    name: str
    size: int
    alloc_time: int
    free_time: int


@dataclass
class AllocResult:
    """Result of memory allocation."""
    peak_usage: int
    addresses: Dict[str, int]


class OnlineMemoryPlanner:
    """Base class for online memory allocation algorithms."""

    def __init__(self):
        self.free_ranges: List[Tuple[int, int]] = []  # [(start, end), ...]
        self.addresses: Dict[str, int] = {}
        self.cur_peak = 0

    def select_range(self, size: int) -> Tuple[int, int]:
        """Select a free range for allocation. Override in subclasses."""
        raise NotImplementedError

    def allocate(self, tensor: TensorDecl) -> int:
        """Allocate memory for a tensor, return its address."""
        # Find a suitable free range
        suitable_ranges = [(s, e) for s, e in self.free_ranges if e - s >= tensor.size]

        if not suitable_ranges:
            # Need to extend peak usage
            last_free = 0
            if self.free_ranges and self.free_ranges[-1][1] == self.cur_peak:
                last_free = self.free_ranges[-1][1] - self.free_ranges[-1][0]
                self.free_ranges.pop()
            new_peak = self.cur_peak + (tensor.size - last_free)
            self.free_ranges.append((self.cur_peak - last_free if last_free else self.cur_peak, new_peak))
            self.cur_peak = new_peak

        # Select range using strategy
        start, end = self.select_range(tensor.size)

        # Split the selected range
        self.free_ranges = [(s, e) for s, e in self.free_ranges if not (s == start or e == start + tensor.size)]

        # Re-add remaining parts
        for s, e in list(self.free_ranges):
            pass  # Already filtered

        # Find and split the containing range
        new_free = []
        found = False
        for s, e in self.free_ranges:
            if s <= start and start + tensor.size <= e and not found:
                found = True
                if s < start:
                    new_free.append((s, start))
                if start + tensor.size < e:
                    new_free.append((start + tensor.size, e))
            else:
                new_free.append((s, e))

        if not found:
            # Must be a new allocation at the end
            new_free = [r for r in self.free_ranges if r != (start, start + tensor.size)]

        self.free_ranges = sorted(new_free)
        self.addresses[tensor.name] = start
        return start

    def free(self, tensor: TensorDecl):
        """Free memory for a tensor."""
        addr = self.addresses[tensor.name]
        new_range = (addr, addr + tensor.size)

        # Coalesce with adjacent free ranges
        merged = False
        new_free = []
        for s, e in sorted(self.free_ranges + [new_range]):
            if new_free and new_free[-1][1] >= s:
                # Merge with previous
                new_free[-1] = (new_free[-1][0], max(new_free[-1][1], e))
            else:
                new_free.append((s, e))

        self.free_ranges = new_free


class FirstFitPlanner(OnlineMemoryPlanner):
    """First-fit: Choose the first free block that fits."""

    def select_range(self, size: int) -> Tuple[int, int]:
        for s, e in sorted(self.free_ranges):
            if e - s >= size:
                return (s, s + size)
        raise RuntimeError("No suitable range found")


class BestFitPlanner(OnlineMemoryPlanner):
    """Best-fit: Choose the smallest free block that fits."""

    def select_range(self, size: int) -> Tuple[int, int]:
        best = None
        best_waste = float('inf')
        for s, e in self.free_ranges:
            waste = (e - s) - size
            if waste >= 0 and waste < best_waste:
                best = (s, s + size)
                best_waste = waste
        if best is None:
            raise RuntimeError("No suitable range found")
        return best


class WorstFitPlanner(OnlineMemoryPlanner):
    """Worst-fit: Choose the largest free block that fits."""

    def select_range(self, size: int) -> Tuple[int, int]:
        best = None
        best_waste = -1
        for s, e in self.free_ranges:
            waste = (e - s) - size
            if waste >= 0 and waste > best_waste:
                best = (s, s + size)
                best_waste = waste
        if best is None:
            raise RuntimeError("No suitable range found")
        return best


def plan_memory_simple(tensors: List[TensorDecl], planner_class) -> AllocResult:
    """
    Simplified memory planning that processes events in order.

    This is a simplified version of the algorithm in plan_stensor_memory.cc
    """
    # Create events
    events = []
    for t in tensors:
        events.append((t.alloc_time, 'alloc', t))
        events.append((t.free_time, 'free', t))

    # Sort: time first, then free before alloc (to maximize reuse)
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'free' else 1))

    planner = planner_class()

    for time, event_type, tensor in events:
        if event_type == 'alloc':
            planner.allocate(tensor)
        else:
            planner.free(tensor)

    return AllocResult(planner.cur_peak, planner.addresses)


def visualize_allocation(tensors: List[TensorDecl], result: AllocResult, title: str):
    """Print ASCII visualization of memory allocation over time."""
    max_time = max(t.free_time for t in tensors)

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  Peak memory usage: {result.peak_usage} bytes")
    print(f"{'='*60}")

    # Create a grid
    height = result.peak_usage
    width = max_time + 1

    # Scale for display
    scale_y = max(1, height // 20)
    scale_x = max(1, width // 40)

    grid_h = (height + scale_y - 1) // scale_y
    grid_w = (width + scale_x - 1) // scale_x

    grid = [[' ' for _ in range(grid_w)] for _ in range(grid_h)]

    # Fill in tensors
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    for i, tensor in enumerate(tensors):
        addr = result.addresses[tensor.name]
        char = chars[i % len(chars)]

        for t in range(tensor.alloc_time, tensor.free_time):
            for m in range(addr, addr + tensor.size):
                y = grid_h - 1 - (m // scale_y)
                x = t // scale_x
                if 0 <= y < grid_h and 0 <= x < grid_w:
                    grid[y][x] = char

    # Print grid
    print(f"\n  Memory ^")
    for row in grid:
        print(f"    {''.join(row)} |")
    print(f"    {'-' * grid_w}-> Time")

    # Print legend
    print(f"\n  Legend:")
    for i, tensor in enumerate(tensors):
        char = chars[i % len(chars)]
        addr = result.addresses[tensor.name]
        print(f"    {char}: {tensor.name} (size={tensor.size}, "
              f"life=[{tensor.alloc_time},{tensor.free_time}), addr={addr})")


def example_attention_memory():
    """
    Example: Memory allocation for a simplified attention kernel.

    This mimics the memory patterns in a threadblock computing attention:
    - Q, K, V inputs loaded at different times
    - Intermediate results (QK^T, softmax, attention)
    - Output written at the end
    """
    print("\n" + "="*70)
    print("Example: Simplified Attention Kernel Memory Planning")
    print("="*70)

    # Time units:
    # 0-10: pre-loop
    # 10-100: main loop (simplified)
    # 100-110: post-loop

    tensors = [
        # Pre-loop loaded tensors (static across iterations)
        TensorDecl("V_weight", size=1024, alloc_time=0, free_time=100),

        # Loop tensors (loaded each iteration, but here simplified)
        TensorDecl("Q_tile", size=512, alloc_time=10, free_time=50),
        TensorDecl("K_tile", size=512, alloc_time=10, free_time=40),
        TensorDecl("QK_result", size=256, alloc_time=30, free_time=60),
        TensorDecl("softmax", size=256, alloc_time=50, free_time=80),
        TensorDecl("attn_out", size=512, alloc_time=70, free_time=100),

        # Accumulator (lives across entire loop)
        TensorDecl("accumulator", size=512, alloc_time=9, free_time=105),

        # Post-loop output buffer
        TensorDecl("output", size=512, alloc_time=100, free_time=110),
    ]

    # Try all three planners
    planners = [
        ("First-Fit", FirstFitPlanner),
        ("Best-Fit", BestFitPlanner),
        ("Worst-Fit", WorstFitPlanner),
    ]

    results = []
    for name, planner_class in planners:
        result = plan_memory_simple(tensors, planner_class)
        results.append((name, result))
        visualize_allocation(tensors, result, name)

    # Summary
    print("\n" + "="*70)
    print("Summary of Memory Planning Results")
    print("="*70)
    for name, result in results:
        print(f"  {name:12s}: Peak = {result.peak_usage:4d} bytes")

    best_name, best_result = min(results, key=lambda x: x[1].peak_usage)
    print(f"\n  Winner: {best_name} (selected by transpiler)")


def example_simple_ops():
    """
    Simpler example showing the DSA problem clearly.
    """
    print("\n" + "="*70)
    print("Example: Simple Operator Memory Planning")
    print("="*70)
    print("""
    This example shows 4 tensors with overlapping lifetimes:

    Time:     0    5    10   15   20   25   30
              |----A----|
                   |----B----|
              |---------C---------|
                             |----D----|

    Without planning: Would need 4 * size = 400 bytes
    With planning: Can reuse memory!
    """)

    tensors = [
        TensorDecl("A", size=100, alloc_time=0, free_time=10),
        TensorDecl("B", size=100, alloc_time=5, free_time=15),
        TensorDecl("C", size=100, alloc_time=0, free_time=20),
        TensorDecl("D", size=100, alloc_time=15, free_time=25),
    ]

    for name, planner_class in [("First-Fit", FirstFitPlanner), ("Best-Fit", BestFitPlanner)]:
        result = plan_memory_simple(tensors, planner_class)
        visualize_allocation(tensors, result, name)


if __name__ == "__main__":
    print("""
    ================================================================
    Mirage Transpiler: Memory Planning Algorithm Demonstration
    ================================================================

    The transpiler uses Dynamic Storage Allocation (DSA) to minimize
    shared memory usage. This is an NP-complete problem, so we use
    heuristics: First-Fit, Best-Fit, and Worst-Fit.

    Key insight: Tensors with non-overlapping lifetimes can share memory!
    """)

    example_simple_ops()
    example_attention_memory()

    print("\n" + "="*70)
    print("For more details, see: src/transpiler/plan_stensor_memory.cc")
    print("="*70)
