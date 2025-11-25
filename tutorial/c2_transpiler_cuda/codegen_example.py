#!/usr/bin/env python3
"""
Code Generation Example

This script demonstrates how Mirage's transpiler generates CUDA code from
a kernel graph. It shows the structure of generated code for different
operator types.

Note: This is a conceptual demonstration. Actual code generation uses
the C++ transpiler in src/transpiler/.
"""


def generated_kernel_level_matmul():
    """Show generated code for kernel-level matmul using cuBLAS."""
    print("""
    ================================================================
    Example 1: Kernel-Level Matmul (cuBLAS)
    ================================================================

    For standalone matmul at the kernel level, Mirage calls cuBLAS:

    Input Graph:
    ```
    A = graph.new_input(dims=(M, K), dtype=float16)
    B = graph.new_input(dims=(K, N), dtype=float16)
    C = graph.matmul(A, B)
    ```

    Generated Code (transpiler_kn.cc:372-434):
    ```cpp
    {
      // OP type: kn_matmul_op
      half_t *dtensor10000000 = (half_t*)input_tensors.at(0);  // A
      half_t *dtensor10000001 = (half_t*)input_tensors.at(1);  // B
      half_t *dtensor10000002 = (half_t*)output_tensors.at(0); // C

      kn::gemm<CUBLAS_COMPUTE_16F>(
        dtensor10000002,  // output C
        dtensor10000000,  // input A
        dtensor10000001,  // input B
        M, N, K,          // dimensions
        K, 1,             // strides for A (row-major: stride_row=K, stride_col=1)
        N, 1,             // strides for B
        N, 1,             // strides for C
        1,                // batch_size
        M*K, K*N, M*N     // batch strides (not used for batch=1)
      );
    }
    ```

    Key observations:
    - Uses runtime kn::gemm which wraps cuBLAS
    - Layout info (strides) passed to handle non-standard layouts
    - Supports batched matmul via batch_size and batch_strides
    """)


def generated_elementwise():
    """Show generated code for elementwise operations."""
    print("""
    ================================================================
    Example 2: Kernel-Level Elementwise Operations
    ================================================================

    For elementwise ops, Mirage generates CuTe-based kernels:

    Input Graph:
    ```
    X = graph.new_input(dims=(M, N), dtype=float16)
    Y = graph.exp(X)
    ```

    Generated Code (transpiler_kn.cc:437-485):
    ```cpp
    {
      // OP type: kn_exp_op
      half_t *dtensor10000000 = (half_t*)input_tensors.at(0);
      half_t *dtensor10000001 = (half_t*)output_tensors.at(0);

      // Layout moves innermost dim to last for coalesced access
      using kernel = kn::ElementUnaryKernel<
        half_t,
        kn::ElementUnaryOpType::EXP,
        Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, Int<1>>>,  // input layout
        Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, Int<1>>>   // output layout
      >;

      kernel::run(dtensor10000001, dtensor10000000);
    }
    ```

    The runtime kernel (in runtime/kernel/element_unary.h) handles:
    - Thread indexing
    - Memory coalescing
    - Vectorized loads/stores when possible
    """)


def generated_custom_kernel_structure():
    """Show the structure of a generated custom kernel."""
    print("""
    ================================================================
    Example 3: Custom Kernel Structure (KN_CUSTOMIZED_OP)
    ================================================================

    For complex fused operations, Mirage generates complete CUDA kernels.

    Input (Python):
    ```python
    kgraph = mi.new_kernel_graph()
    X = kgraph.new_input(dims=(batch, seq, hidden))
    W = kgraph.new_input(dims=(hidden,))

    # Create threadblock graph for RMSNorm
    bgraph = mi.new_threadblock_graph(grid_dim, block_dim, forloop_range=seq)
    x = bgraph.new_input(dtensor=X, input_map=(-1, 0, -1), forloop_dim=1)
    w = bgraph.new_input(dtensor=W, input_map=(-1, -1, -1), forloop_dim=-1)
    x_sq = bgraph.square(x)
    x_sum = bgraph.reduction(x_sq, dim=2)
    x_rms = bgraph.sqrt(x_sum)
    x_norm = bgraph.div(x, x_rms)
    y = bgraph.mul(x_norm, w)
    bgraph.new_output(y)

    Y = kgraph.customized(bgraph)
    ```

    Generated CUDA Kernel (transpiler_tb.cc):
    ```cpp
    __global__ void __launch_bounds__(256) custom_kernel_0(
        half* __restrict__ dtensor10000002_ptr,      // output Y
        half const* __restrict__ dtensor10000000_ptr, // input X
        half const* __restrict__ dtensor10000001_ptr  // input W
    ) {
      //---------------------------------------------------------------
      // Setup
      //---------------------------------------------------------------
      int thread_idx = threadIdx.x;
      static constexpr int NUM_THREADS = 256;

      extern __shared__ char buf[];

      // STensor pointers (addresses from memory planner)
      half *stensor0_ptr = (half*)(buf + 0);      // x tile
      half *stensor1_ptr = (half*)(buf + 1024);   // w
      half *stensor2_ptr = (half*)(buf + 2048);   // x_sq
      half *stensor3_ptr = (half*)(buf + 3072);   // x_sum
      half *stensor4_ptr = (half*)(buf + 3136);   // x_rms
      half *stensor5_ptr = (half*)(buf + 3200);   // x_norm
      half *stensor6_ptr = (half*)(buf + 4224);   // y

      // Clear shared memory (for matmul padding)
      *((uint128_t*)buf) = 0ul;

      //---------------------------------------------------------------
      // G->S Copy Atoms (define copy patterns)
      //---------------------------------------------------------------
      const half *dtensor10000000_tile_ptr =
          dtensor10000000_ptr + blockIdx.x * TILE_SIZE;

      using STensor0InputAtom = tb::InputChunkedSyncCopy<
          half,
          Layout<Shape<Int<16>, Int<64>>, Stride<Int<64>, Int<1>>>,
          DTensor0TileLayout,
          NUM_THREADS
      >;

      //---------------------------------------------------------------
      // Pre-loop: Load static inputs
      //---------------------------------------------------------------
      STensor1InputAtom::run(stensor1_ptr, dtensor10000001_tile_ptr, thread_idx);
      __syncthreads();

      //---------------------------------------------------------------
      // Main loop
      //---------------------------------------------------------------
      for (int for_idx = 0; for_idx < 64; for_idx++) {
        // Load X tile for this iteration
        STensor0InputAtom::run(
            stensor0_ptr,
            dtensor10000000_tile_ptr + for_idx * 64 * 64,  // stride * tile_size
            thread_idx
        );
        __syncthreads();

        // Square
        {
          using Kernel = tb::ElementUnaryKernel<
              half,
              tb::ElementUnaryOpType::SQUARE,
              OutLayout, InLayout,
              NUM_THREADS,
              tb::EpilogueStore<half>  // No fusion
          >;
          Kernel::run(stensor2_ptr, stensor0_ptr, thread_idx, 1.0f, scalars);
        }
        __syncthreads();

        // Reduction (sum)
        {
          using Kernel = tb::ReductionKernel<
              half,
              OutLayout, InLayout,
              2,  // reduction dimension
              NUM_THREADS,
              tb::EpilogueStore<half>
          >;
          Kernel::run(stensor3_ptr, stensor2_ptr, thread_idx, scalars);
        }
        __syncthreads();

        // Sqrt
        {
          using Kernel = tb::ElementUnaryKernel<half, SQRT, ...>;
          Kernel::run(stensor4_ptr, stensor3_ptr, thread_idx, 1.0f, scalars);
        }
        __syncthreads();

        // Div (X / RMS)
        {
          using Kernel = tb::ElementBinaryKernel<half, DIV, ...>;
          Kernel::run(stensor5_ptr, stensor0_ptr, stensor4_ptr, thread_idx, scalars);
        }
        __syncthreads();

        // Mul (norm * W)
        {
          using Kernel = tb::ElementBinaryKernel<half, MUL, ...>;
          Kernel::run(stensor6_ptr, stensor5_ptr, stensor1_ptr, thread_idx, scalars);
        }
      }
      __syncthreads();

      //---------------------------------------------------------------
      // Epilogue: Write output
      //---------------------------------------------------------------
      STensor6OutputAtom::run(dtensor10000002_tile_ptr, stensor6_ptr, thread_idx);
    }
    ```
    """)


def generated_fused_matmul():
    """Show code generation for matmul with fused epilogue."""
    print("""
    ================================================================
    Example 4: Matmul with Fused Epilogue
    ================================================================

    When matmul is followed by fusable operations, the transpiler
    generates an epilogue chain:

    Graph: A @ B -> exp -> accumulator

    Instead of:
    ```cpp
    // Without fusion (3 shared memory round-trips)
    Matmul::run(C, A, B);        // Write C to smem
    __syncthreads();
    Exp::run(exp_C, C);          // Read C, write exp_C to smem
    __syncthreads();
    Accum::run(acc, exp_C);      // Read exp_C, accumulate
    ```

    With fusion:
    ```cpp
    // Fused (data stays in registers)
    using Epilogue = tb::EpilogueExp<
        half,
        tb::EpilogueStoreAccum<half>  // Nested: exp then accumulate
    >;

    using Matmul0Kernel = tb::Matmul<
        half,
        SM80_16x8x16_F16F16F16F16_TN,  // MMA atom
        Layout<Shape<Int<2>, Int<2>, _1>>,  // Thread layout for MMA
        true,   // use ldmatrix
        false,  // don't use stmatrix (accumulating to smem)
        LayoutA, LayoutB, LayoutC,
        LayoutAAligned, LayoutBAligned,
        NUM_THREADS,
        1,      // num_exps_before_store = 1
        true    // is_store_accum
    >;

    // Accumulator in register file (no smem allocation needed)
    auto matmul_accum = Matmul0Kernel::get_mma_rC(thread_idx);

    for (int for_idx = 0; for_idx < K_TILES; for_idx++) {
      // Load A and B tiles...
      __syncthreads();

      // Fused matmul + exp + accumulate
      // Data flows: A,B -> registers -> MMA -> exp(in regs) -> acc (in regs)
      Matmul0Kernel::run(matmul_accum, stensor_A, stensor_B, buf, thread_idx);
    }
    __syncthreads();

    // Write accumulated result to shared memory (only once, at the end)
    Matmul0Kernel::write_back_mma_rC(stensor_output, matmul_accum, thread_idx);
    ```

    Benefits:
    - Eliminates 2 shared memory round-trips per iteration
    - Accumulator stays in registers across loop iterations
    - Only one write to shared memory at the end
    """)


def generated_async_copy():
    """Show code generation with async copy (software pipelining)."""
    print("""
    ================================================================
    Example 5: Async Copy with Software Pipelining
    ================================================================

    For inputs with forloop_dim, Mirage can use async copies to overlap
    data movement with computation:

    Without pipelining:
    ```cpp
    for (int i = 0; i < N; i++) {
      // Serialized: wait for copy, then compute
      sync_copy(stensor, dtensor + i * stride);  // Wait for this
      __syncthreads();
      compute(stensor);
    }
    ```

    With pipelining (cp.async):
    ```cpp
    // Double buffering with async copies
    half *stensor_ptr = buf;
    half *stensor_async_copy_buf = buf + TILE_SIZE;

    // Prime the pipeline: issue first async copy
    async_copy(stensor_async_copy_buf, dtensor);
    cp_async_fence();

    for (int i = 0; i < N; i++) {
      // Issue next copy while processing current
      if (i + 1 != N) {
        async_copy(stensor_ptr, dtensor + (i+1) * stride);
      }
      cp_async_fence();

      // Wait for previous copy (now in stensor_async_copy_buf)
      cp_async_wait<1>();

      // Swap buffers
      SWAP(stensor_ptr, stensor_async_copy_buf);

      // Compute on the data that just arrived
      __syncthreads();
      compute(stensor_ptr);
    }
    ```

    The transpiler generates this pattern when:
    - Input op has forloop_dim >= 0
    - is_pipelined_input = true in TBSchedOpMeta
    - Using InputChunkedAsyncCopy instead of InputChunkedSyncCopy
    """)


def generated_layout_resolution():
    """Show how layout resolution affects generated code."""
    print("""
    ================================================================
    Example 6: Layout Resolution Impact
    ================================================================

    The Z3 solver decides tensor layouts. This affects generated code:

    Case 1: Innermost dim matches between DTensor and STensor
    → Use chunked copy (128-bit loads)
    ```cpp
    using STensor0InputAtom = tb::InputChunkedSyncCopy<
        half,
        STensorLayout,    // Same innermost as DTensor
        DTensorLayout,
        NUM_THREADS
    >;
    ```

    Case 2: Innermost dims differ
    → Use non-chunked copy (element by element)
    ```cpp
    using STensor0InputAtom = tb::InputNonChunkedSyncCopy<
        half,
        STensorLayout,    // Different innermost
        DTensorLayout,
        NUM_THREADS
    >;
    ```

    Case 3: For matmul with ldmatrix
    → Requires specific innermost dim
    ```cpp
    // Z3 constraint: innermost must be in last 2 dims
    opt.add(d_is_innermost[tensor.guid][num_dims - 1] ||
            d_is_innermost[tensor.guid][num_dims - 2]);
    ```

    The cost model penalizes suboptimal layouts:
    - No wide copy: +4000 cost
    - No ldmatrix: +10000 cost
    - No cp.async: +20000 cost
    """)


def main():
    print("""
    ================================================================
    Mirage Transpiler: Code Generation Examples
    ================================================================

    This tutorial shows the structure of CUDA code generated by Mirage's
    transpiler for different operation types and optimization patterns.

    The transpiler (src/transpiler/) generates code that uses the runtime
    library (include/mirage/transpiler/runtime/) for efficient execution.
    """)

    generated_kernel_level_matmul()
    generated_elementwise()
    generated_custom_kernel_structure()
    generated_fused_matmul()
    generated_async_copy()
    generated_layout_resolution()

    print("""
    ================================================================
    Summary: Code Generation Pipeline
    ================================================================

    1. Kernel-level operators:
       - Matmul → cuBLAS call
       - Elementwise → CuTe kernel with layouts
       - Custom → Full CUDA kernel generation

    2. Custom kernel structure:
       - Setup: thread idx, shared memory pointers
       - Copy atoms: define G↔S copy patterns
       - Pre-loop: load static inputs
       - Main loop: execute TB operators in scheduled order
       - Epilogue: write outputs

    3. Key optimizations in generated code:
       - Operator fusion via epilogues
       - Register-file accumulators
       - Async copy with software pipelining
       - Swizzled layouts for bank conflict avoidance

    For the actual implementation, see:
      - src/transpiler/transpiler_kn.cc (kernel-level)
      - src/transpiler/transpiler_tb.cc (threadblock-level)
      - include/mirage/transpiler/runtime/ (runtime library)
    """)


if __name__ == "__main__":
    main()
