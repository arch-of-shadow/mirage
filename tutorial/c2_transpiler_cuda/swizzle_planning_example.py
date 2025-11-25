#!/usr/bin/env python3
"""
Swizzle Planning Example

This script demonstrates the swizzling algorithms used by Mirage's transpiler
to avoid shared memory bank conflicts.

Shared memory has 32 banks. When threads in a warp access the same bank,
accesses are serialized (bank conflict). Swizzling reorders the memory layout
to spread accesses across different banks.

Two methods:
1. XOR-based: new_addr = old_addr XOR row (for power-of-2 dimensions)
2. Shift-based: new_addr = old_addr + row * shift (for non-power-of-2)
"""

import numpy as np
from typing import Tuple, List


NUM_BANKS = 32  # CUDA shared memory has 32 banks
BANK_WIDTH = 4  # Each bank is 4 bytes wide


def get_bank(addr: int, element_size: int = 2) -> int:
    """
    Get the bank number for a given byte address.

    Bank = (addr / 4) % 32

    For half precision (2 bytes), every 2 elements share a bank.
    """
    byte_addr = addr * element_size
    return (byte_addr // BANK_WIDTH) % NUM_BANKS


def visualize_bank_access(matrix: np.ndarray, name: str, element_size: int = 2):
    """Visualize which bank each element maps to."""
    rows, cols = matrix.shape
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Shape: {rows}x{cols}, Element size: {element_size} bytes")
    print(f"{'='*60}")

    print("\n  Memory addresses (linear offset):")
    for r in range(min(rows, 8)):
        row_str = "  "
        for c in range(min(cols, 16)):
            row_str += f"{matrix[r, c]:3d} "
        if cols > 16:
            row_str += "..."
        print(row_str)
    if rows > 8:
        print("  ...")

    print("\n  Bank numbers:")
    for r in range(min(rows, 8)):
        row_str = "  "
        banks_in_row = set()
        for c in range(min(cols, 16)):
            bank = get_bank(matrix[r, c], element_size)
            banks_in_row.add(bank)
            row_str += f"{bank:3d} "
        if cols > 16:
            row_str += "..."
        # Highlight conflicts
        conflicts = cols - len(banks_in_row) if cols <= 16 else "?"
        row_str += f"  (unique banks: {len(banks_in_row)}, conflicts: {conflicts})"
        print(row_str)
    if rows > 8:
        print("  ...")


def row_major_layout(rows: int, cols: int) -> np.ndarray:
    """Create a row-major layout matrix (address = row * cols + col)."""
    matrix = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            matrix[r, c] = r * cols + c
    return matrix


def xor_swizzle(matrix: np.ndarray, B: int, M: int, S: int) -> np.ndarray:
    """
    Apply XOR-based swizzling.

    CuTe Swizzle<B, M, S> means:
    - B: number of bits for the base (chunk size in terms of banks)
    - M: number of bits to skip (for element size)
    - S: number of bits for the stride

    The XOR is: new_addr = old_addr XOR (row << shift_amount)

    For simplicity, we use: new_addr = old_addr XOR (row * xor_stride)
    where xor_stride depends on B, M, S parameters.
    """
    rows, cols = matrix.shape
    result = np.zeros_like(matrix)

    # Simplified XOR swizzle: XOR the column index with the row index
    # This is a conceptual demonstration
    for r in range(rows):
        for c in range(cols):
            # XOR the address within each row
            old_addr = matrix[r, c]
            row_contribution = r & ((1 << B) - 1)  # Mask to B bits
            xor_mask = row_contribution << M
            # Apply XOR to the column portion
            base = (r * cols)
            offset = c ^ (row_contribution)
            result[r, c] = base + offset
    return result


def xor_swizzle_simple(rows: int, cols: int) -> np.ndarray:
    """
    Simplified XOR swizzle: new_col = col XOR row

    This demonstrates the basic principle.
    """
    matrix = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            new_c = c ^ (r % cols)  # XOR column with row (mod cols for wrap)
            matrix[r, c] = r * cols + new_c
    return matrix


def shift_swizzle(rows: int, cols: int, shift: int) -> np.ndarray:
    """
    Apply shift-based swizzling.

    new_addr = old_addr + row * shift
    = row * (cols + shift) + col

    This effectively increases the stride of each row.
    """
    new_stride = cols + shift
    matrix = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            matrix[r, c] = r * new_stride + c
    return matrix


def count_bank_conflicts(matrix: np.ndarray, warp_rows: int = 1, warp_cols: int = 32,
                         element_size: int = 2) -> int:
    """
    Count bank conflicts for a warp access pattern.

    Assumes threads in a warp access consecutive elements in a row.
    """
    rows, cols = matrix.shape
    total_conflicts = 0

    for r in range(0, rows, warp_rows):
        for c in range(0, cols, warp_cols):
            # Get banks accessed by this warp
            banks = set()
            accesses_per_bank = {}
            for wr in range(warp_rows):
                for wc in range(min(warp_cols, cols - c)):
                    if r + wr < rows:
                        addr = matrix[r + wr, c + wc]
                        bank = get_bank(addr, element_size)
                        if bank not in accesses_per_bank:
                            accesses_per_bank[bank] = 0
                        accesses_per_bank[bank] += 1
                        banks.add(bank)

            # Conflicts = total accesses - unique banks
            num_accesses = sum(accesses_per_bank.values())
            conflicts = num_accesses - len(banks)
            total_conflicts += conflicts

    return total_conflicts


def example_8x8_matrix():
    """Example with a small 8x8 matrix to clearly show swizzling."""
    print("\n" + "="*70)
    print("Example 1: 8x8 Matrix (Power of 2)")
    print("="*70)
    print("""
    An 8x8 matrix of half precision (2 bytes each).
    With row-major layout, threads loading row elements may conflict.

    Original: Each row has the same bank pattern (0,0,1,1,2,2,3,3,...)
    XOR swizzled: Each row has a different bank pattern
    """)

    rows, cols = 8, 8

    # Original row-major
    original = row_major_layout(rows, cols)
    visualize_bank_access(original, "Original Row-Major Layout")

    # XOR swizzled
    swizzled = xor_swizzle_simple(rows, cols)
    visualize_bank_access(swizzled, "XOR Swizzled Layout")

    # Count conflicts for ldmatrix-like access (8 threads, each loads 8 elements)
    print("\n  Bank conflict analysis (simulated ldmatrix):")
    print(f"    Original:  {count_bank_conflicts(original, 8, 1)} conflicts")
    print(f"    Swizzled:  {count_bank_conflicts(swizzled, 8, 1)} conflicts")


def example_16x16_matrix():
    """Example with 16x16 matrix for matmul tile."""
    print("\n" + "="*70)
    print("Example 2: 16x16 Matrix (MMA Tile)")
    print("="*70)
    print("""
    A 16x16 tile commonly used in tensor core operations.
    ldmatrix loads 8 rows at a time, with each thread providing an address.
    """)

    rows, cols = 16, 16

    original = row_major_layout(rows, cols)
    swizzled = xor_swizzle_simple(rows, cols)

    print("\n  Original layout (first 8x8 corner):")
    print("  Addresses:    ", end="")
    for c in range(8):
        print(f"{original[0, c]:3d}", end=" ")
    print()
    print("  Banks:        ", end="")
    for c in range(8):
        print(f"{get_bank(original[0, c]):3d}", end=" ")
    print()

    print("\n  Swizzled layout (first 8x8 corner):")
    print("  Addresses:    ", end="")
    for c in range(8):
        print(f"{swizzled[0, c]:3d}", end=" ")
    print()
    print("  Banks:        ", end="")
    for c in range(8):
        print(f"{get_bank(swizzled[0, c]):3d}", end=" ")
    print()

    print(f"\n  Total conflicts (16 row accesses):")
    print(f"    Original:  {count_bank_conflicts(original, 1, 16)}")
    print(f"    Swizzled:  {count_bank_conflicts(swizzled, 1, 16)}")


def example_non_power_of_2():
    """Example with non-power-of-2 dimension (requires shift swizzle)."""
    print("\n" + "="*70)
    print("Example 3: 8x12 Matrix (Non-Power of 2)")
    print("="*70)
    print("""
    When the innermost dimension is not a power of 2, XOR swizzle doesn't work.
    Instead, we use shift-based swizzling:

    new_addr = row * (cols + shift) + col

    We find the smallest shift such that gcd(cols + shift, 32) = 1
    """)

    rows, cols = 8, 12

    original = row_major_layout(rows, cols)
    visualize_bank_access(original, f"Original Row-Major {rows}x{cols}")

    # Find optimal shift
    import math
    shift = 0
    while math.gcd(cols + shift, NUM_BANKS) != 1:
        shift += 1
    print(f"\n  Finding shift: gcd({cols} + shift, 32) = 1")
    print(f"  Optimal shift = {shift} (new stride = {cols + shift})")

    shifted = shift_swizzle(rows, cols, shift)
    visualize_bank_access(shifted, f"Shift Swizzled (shift={shift})")

    print(f"\n  Bank conflicts:")
    print(f"    Original:  {count_bank_conflicts(original, 1, cols)}")
    print(f"    Shifted:   {count_bank_conflicts(shifted, 1, cols)}")
    print(f"\n  Memory overhead: {rows * shift} elements ({rows * shift * 2} bytes)")


def example_chunk_size():
    """Example showing how chunk size affects swizzling."""
    print("\n" + "="*70)
    print("Example 4: Understanding Chunk Size")
    print("="*70)
    print("""
    When performing 128-bit (16-byte) copies, each "chunk" is 8 half elements.
    The swizzle must preserve chunk contiguity.

    For ldmatrix, each thread loads 8 elements (128 bits).
    These 8 elements must remain contiguous after swizzling.
    """)

    rows, cols = 8, 16  # 8 rows, each with 2 chunks of 8 elements

    print("\n  Chunks in row-major layout:")
    print("  Row 0: [chunk0: elems 0-7] [chunk1: elems 8-15]")
    print("  Row 1: [chunk0: elems 16-23] [chunk1: elems 24-31]")
    print("  ...")

    print("\n  After XOR swizzling (B=1, M=3, S=4):")
    print("  Row 0: [chunk0: banks 0-3] [chunk1: banks 4-7] - no change")
    print("  Row 1: [chunk0: banks 8-11 XOR 1] [chunk1: ...] - shifted")
    print("\n  Each chunk stays contiguous, but starts at different bank!")


def main():
    print("""
    ================================================================
    Mirage Transpiler: Swizzle Planning Algorithm Demonstration
    ================================================================

    Bank conflicts occur when multiple threads in a warp access the same
    shared memory bank. The GPU serializes conflicting accesses.

    Shared memory: 32 banks, each 4 bytes wide
    Half precision: 2 bytes per element, so 2 elements per bank

    Swizzling reorganizes memory layout to avoid conflicts.
    """)

    example_8x8_matrix()
    example_16x16_matrix()
    example_non_power_of_2()
    example_chunk_size()

    print("\n" + "="*70)
    print("For more details, see:")
    print("  - src/transpiler/plan_tb_swizzle.cc")
    print("  - CuTe Swizzle documentation")
    print("="*70)


if __name__ == "__main__":
    main()
