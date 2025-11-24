# Chapter 1: Architecture and Internals

In this chapter, we will dive deep into the architecture of the Mirage project. We will explore the **MuGraph** intermediate representation, the **Search Algorithms** that drive optimization, and the overall **Compiler Pipeline**. We will use a concrete example—optimizing an **RMSNorm followed by a Linear layer**—to illustrate these concepts.

## 1. The Running Example: RMSNorm + Linear

To understand how Mirage works, let's consider a common pattern in Large Language Models (LLMs): an RMSNorm layer followed by a Linear layer (Matrix Multiplication).

Mathematically, this looks like:
$$ Y = \text{RMSNorm}(X) \times W $$

where, RMSNorm is:

$$y_i = \frac{ x_i * g_i }{ \sqrt{\frac{1}{n} \sum_{i=1}^{n}{x_i^2}} }$$

In a standard deep learning framework (like PyTorch), this is executed as two separate kernels:
1.  **Kernel 1**: Reads $X$, computes RMSNorm, writes intermediate result to global memory.
2.  **Kernel 2**: Reads intermediate result, performs Matrix Multiplication with $W$, writes $Y$.

**Mirage's Goal**: Automatically discover a fused kernel that is faster. For example, it might discover that we can perform the division of RMSNorm *after* the matrix multiplication accumulation in some cases, or simply fuse them to reduce memory traffic.

## 2. Core Components: MuGraph

Mirage uses a multi-level graph representation called **MuGraph** to specify the execution of a tensor program on GPUs. Unlike standard computation graphs (like in PyTorch or TensorFlow) that only represent data dependencies, MuGraph explicitly represents things together: 1. operator graph at hierarchies; 2. loop schedule; 3. memory allocation. 

[Multi-Level Graph Representation](../../docs/source/mugraph.rst) gives a very clear description of MuGraph. Below we introduce some related source code to help developers.

### 2.1 Kernel Graph (`mirage::kernel::Graph`)

The top level is the **Kernel Graph**. It represents computations at the grid level.
*   **Nodes**: `KNOperator` (Kernel Operators). These represent operations running on the entire grid.
*   **Edges**: `DTensor` (Device Tensors). These represent data stored in **Global Memory** (DRAM).

**Why is it needed?** It defines the boundaries of GPU kernels and manages global memory allocation.

**Code Snippet (`include/mirage/kernel/graph.h`)**:
```cpp
class Graph {
public:
  // ...
  // Create a new input tensor (resides in Global Memory)
  DTensor new_input(std::vector<int> const &dims, ...);

  // Create a standard operator (e.g., Matmul)
  DTensor matmul(DTensor const &A, DTensor const &B);

  // Create a "Customized" operator defined by a Thread Block Graph
  // This is where kernel fusion happens!
  std::vector<DTensor> customized(std::vector<DTensor> const &inputs,
                                  mirage::threadblock::Graph const &_graph);
  
  std::vector<mirage::kernel::KNOperator *> operators;
  // ...
};
```

### 2.2 Thread Block Graph (`mirage::threadblock::Graph`)

The second level is the **Thread Block Graph**. It represents computations within a single CUDA thread block.
*   **Nodes**: `TBOperator` (Thread Block Operators). These represent operations like loading data, computing matmul on a tile, or reducing values.
*   **Edges**: `STensor` (Shared Memory Tensors). These represent data stored in **Shared Memory** (SRAM).

**Why is it needed?** It allows Mirage to model and optimize data movement between Global Memory and Shared Memory, which is critical for performance (e.g., tiling for MatMul).

**Code Snippet (`src/threadblock/graph.h`)**:
```cpp
class Graph {
public:
  // ...
  // Create an input operator (Loads data from Global -> Shared Memory)
  STensor new_input(mirage::kernel::DTensor const &dtensor, ...);

  // Perform computation in Shared Memory
  STensor matmul(STensor const &A, STensor const &B);
  STensor reduction(STensor const &A, int dim);

  // Mark output (Stores data from Shared -> Global Memory)
  mirage::kernel::DTensor mark_output(STensor const &stensor, ...);

  std::vector<mirage::threadblock::TBOperator *> operators;
  // ...
};
```

### 2.3 Putting it together (The Example)

We can inspect the graph structure using the following script:

#### Creating from C++

Refer to [cpp_examples](../../cpp_examples/)

```cpp
// TODO
```

#### Creating from Python

Let's create a simple kernel graph and visualize it:

```python
python3 tutorial/c1_architecture/kernel_graph.py
```

This will generate a DOT file [`rmsnorm.dot`](tutorial/c1_architecture/rmsnorm.dot) and a PNG file [`rmsnorm.png`](tutorial/c1_architecture/rmsnorm.png) in the current directory.

We can manually create a hierarchical one:

> Refer to [demo/reference_mugraphs/rms_norm.py](demo/reference_mugraphs/rms_norm.py)

```python
python3 tutorial/c1_architecture/hierarchical_graph.py
```

This generates a DOT file [`rmsnorm_hierarchical.dot`](tutorial/c1_architecture/rmsnorm_hierarchical.dot) and a PNG file [`rmsnorm_hierarchical.png`](tutorial/c1_architecture/rmsnorm_hierarchical.png).


These visualization products help us understand the graph structure.

## 3. Search Algorithms

Mirage doesn't just "compile" code; it **searches** for the best implementation. The search engine is located in `src/search`.

### 3.0 Cython Entry

In `include/mirage/search/search_c.h`, declare the cython entry point:

```cpp
int cython_search(mirage::kernel::Graph const *input_graph,
                  char const *backend,
                  int max_num_graphs,
                  mirage::kernel::Graph **new_graphs,
                  std::vector<MInt3> imap_to_explore,
                  std::vector<MInt3> omap_to_explore,
                  std::vector<MDim3> grid_dim_to_explore,
                  std::vector<MDim3> block_dim_to_explore,
                  std::vector<int> fmap_to_explore,
                  std::vector<int> frange_to_explore,
                  char const *filename,
                  bool verbose,
                  char const *default_config,
                  bool is_formal_verified);
```

I think here is the right place to describe the search's parameters:
- `input_graph`: The input kernel graph.
- `backend`: The backend to use (e.g., "cuda").
- `max_num_graphs`: The maximum number of graphs to generate.
- `new_graphs`: The output graphs.
- `imap_to_explore`: `imap` specifies how the input tensor is partitioned into sub-tensors for individual blocks.
- `omap_to_explore`: `omap` specifies how the outputs of all blocks are concatenated to construct the final output of the kernel operator.
- `grid_dim_to_explore`: specify the number of blocks along the x, y, and z dimensions.
- `block_dim_to_explore`: specify the organization of threads within the block.
- `fmap_to_explore`: specify which part of the input tensor to load in each iteration.
- `frange_to_explore`: TODO
- `filename`: The filename to save the search results.
- `verbose`: Whether to print verbose output.
- `default_config`: TODO
- `is_formal_verified`: TODO.   

It is implemented in `src/search/search_c.cc`.

```cpp
int cython_search(...) {
  // 1. If a checkpoint file exists, load graphs from it instead of searching
  if (filename) {
    //..
    new_graphs[num] = new kernel::Graph();
    from_json(graph, *new_graphs[num]);
  }

  {
    // 2. Initialize Search Configuration
    search::GeneratorConfig config =
        search::GeneratorConfig::get_default_config();
    //..

    // 3. Set Exploration Parameters (Search Space Pruning)
    // These vectors limit the search to specific configurations if provided.
    // .. imap, omap, grid_dim, block_dim, fmap, frange

    // 4. Run the Search
    // Instantiate the Generator
    search::KernelGraphGenerator gen(
        *input_graph, config, result_filename, verbose);
    
    // Start the search process
    gen.generate_kernel_graphs();

    // 5. Collect and Return Results
    //..
  }
}
```

We'll dive into `KernelGraphGenerator` in the next section.

### 3.1 Backtracking Search

The core algorithm is a backtracking search that incrementally builds the MuGraph. It starts with an empty graph (containing only input tensors) and tries to append operators until the graph produces the desired output.

**Code Snippet (`src/search/search.cc`, `include/mirage/search/search.h`)**:


#### Preprocess

`KernelGraphGenerator::KernelGraphGenerator(..)` calls `preprocess`:

```cpp
void KernelGraphGenerator::preprocess(kernel::Graph const &computation_graph) {
  // 1. Get input attributes: enumerate computation_graph.operators, check op_type == KN_INPUT_OP, and store in computation_graph_input_attrs
  //..

  // 2. Get possible abstract expressions, showing the kernel graph's computation tasks.
  //    They will be used for pruning illegal graphs during search.
  //    
  abstract_expr_eval(computation_graph, computation_graph_exprs);

  // 3. Some ”range“ things. I'm not sure what they are for.
  // ..

  // 4. Get output expressions (abstract)
  for (kernel::KNOperator *op : computation_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      computation_graph_output_exprs.push_back(
          computation_graph_exprs.at(op->input_tensors[0].guid));
    }
  }
  // 4.5  Initalize final expressions
  //      THIS CONSTRUCTS AN E-GRAPH
  for (auto const &final_expr : computation_graph_output_exprs) {
    initialize_final_expr(final_expr);
    // ⤷ get_egraph(expr->to_egg().c_str()) // This build an e-graph!
  }
  
  // 5. Initialize verifier, formal / probabilistic
  // ..
}
```

#### Generate Kernel Graphs

```cpp
void KernelGraphGenerator::generate_kernel_graphs() {
  
  // Create a SearchContext
  SearchContext c;
  c.level = SearchLevel::LV_KERNEL;
  c.kn_graph = std::make_shared<kernel::Graph>();

  std::vector<SerializedSearchContext> verified_graphs;

  // This is the main backtracking search function
  generate_next_operator(
      c,
      [this](SearchContext const &c) {
        return c.level == SearchLevel::LV_KERNEL &&
                this->verify(*c.kn_graph);
      },
      verified_graphs,
      /*search_depth=*/0,
      /*is_a_new_thread_start=*/true);


  save_results();
}
```


```cpp
void KernelGraphGenerator::generate_next_operator(SearchContext &c, ...) {
    // 1. Check if we found a valid graph
    if (verify(c)) {
        verified_graphs.push_back(SerializedSearchContext(c));
        return;
    }

    // 2. TODO: Something about abstract expr
    std::unordered_map<type::GuidType, std::shared_ptr<AbstractExpr const>> algebraic_expr;
    abstract_expr_eval(*c.kn_graph, algebraic_expr);
    if (c.tb_graph) {
      abstract_expr_eval(*c.tb_graph, algebraic_expr);
    }

    // 2.5 Lambda: check if adding an operator yields a valid abstract expression
    //     What is valid? check_abstract_expr checks if the expression is an subexpr to any final exprs
  
    auto infer_and_check_abstract_expr = [&](auto const &input_tensors,
                                           auto op_type) {
      std::vector<std::shared_ptr<AbstractExpr const>> input_exprs =
          vector_map(input_tensors,
                    [&](auto const &t) { return algebraic_expr.at(t.guid); });
      std::shared_ptr<AbstractExpr const> expr =
          get_abstract_expr(op_type, input_tensors, input_exprs);
      return check_abstract_expr(expr);
    };


    // 2. Try adding a Kernel Operator (KNOperator)
    for (type::KNOperatorType op_type : dim_strategy.get_knop_cand()) {
        // Case K1: finish and verify the current graph
        if (op_type != type::KNOperatorType::KN_CUSTOMIZED_OP) {
          // Case K2: generate a pre-defined kernel operator

          // .. Some checking
          // Add to the graph
          KNOperator *new_op = create_op(*c.kn_graph, op_type, input_tensors);
          if (new_op) {
            c.kn_graph->operators.push_back(new_op);
            // Search deeper
            generate_next_operator(
                c, verify, verified_graphs, search_depth + 1);
          }
          // Backtrack
          while (c.kn_graph->operators.back() != old_last_op) {
            delete c.kn_graph->operators.back();
            c.kn_graph->operators.pop_back();
          }
        } else {
          // Case K3: generate a graph-def kernel operator
          // .. Some checking
          
          // Enumerate grid_dim, block_dim, input_map, forloop_dim, forloop_range
          for (dim3 grid_dim : dim_strategy.get_grid_dim_cand(..)) {
            for (dim3 block_dim :
                 dim_strategy.get_block_dim_cand(..)) {
              for (std::vector<int3> const &input_map :
                   dim_strategy.get_input_map_cand(..)) {
                for (std::vector<int> const &forloop_dim :
                     dim_strategy.get_forloop_dim_cand(..)) {
                  for (int forloop_range :
                       dim_strategy.get_forloop_range_cand(..)) {
                    c.tb_graph = std::make_shared<threadblock::Graph>(
                        grid_dim,
                        block_dim,
                        forloop_range,
                        config.reduction_dimx);
                    
                    // Create input tensors
                    // ..

                    c.level = SearchLevel::LV_THREADBLOCK;
                    // Search deeper
                    generate_next_operator(
                        c, verify, verified_graphs, search_depth + 1);
                    
                    // Backtrack
                    c.level = SearchLevel::LV_KERNEL;
                    c.tb_graph = nullptr;
                  }
                }
              }
            }
          }
        }

    }
    
    // 3. Try adding a Customized Operator (Thread Block Graph Search)
    //    I think this looks very similar to the Kernel Search, except no graph-def kernels
    // ...
}
```

### 3.2 Dimensionality Strategy (`DimStrategy`)

To make the search efficient, Mirage uses `DimStrategy` to propose only "promising" candidates. It decides:
*   Which operators to try next (`get_knop_cand`).
*   How to split tensors across thread blocks (Grid Dimensions).
*   How to tile tensors within a block (Block Dimensions).

**Code Snippet (`src/search/dim_strategy.cc`)**:
```cpp
std::vector<type::KNOperatorType> DimStrategy::get_knop_cand() {
  // Returns a list of operators to explore (e.g., MATMUL, EXP, ADD), random_shuffled
  return config.knop_to_explore;
}

std::vector<dim3> DimStrategy::get_grid_dim_cand(std::vector<DTensor> const &tensors) {
    // Proposes grid dimensions (e.g., {128, 1, 1}, {256, 1, 1}) based on input shapes
    // ...
}
```

### 3.3 Pruning and Verification

The search space is huge. Mirage uses **Abstract Interpretation** to prune invalid branches early. It maintains an abstract state (e.g., "this tensor represents $X \times W$") and checks if adding an operator gives an subexpr of any final expressions.

The core function here is `check_abstract_expr`:

```cpp
bool KernelGraphGenerator::check_abstract_expr(std::shared_ptr<AbstractExpr const> expr, ..) {
  for (auto const &final_expr : computation_graph_output_exprs) {
    if (subexpr_to_final_expr(expr)) {
      return true;
    }
  }
  return false;
}

std::vector<bool> subexpr_to_final_expr(
    std::vector<std::shared_ptr<AbstractExpr const>> const &exprs) {
  // The `egg_equiv` functions use e-graph's pattern matching to get the expressions that appear in the e-graph, indicating equivalance to any sub-expr of final expressions.
  bool *results_in_raw_array =
      egg_equiv(exprs_c_str, static_cast<int>(exprs.size()));
  // ..
}
```

For implementation of `egg_equiv`, refer to `src/search/abstract_expr/abstract_subexpr/src/lib.rs`. [`cpp_examples/egg_tests.cc`](../../cpp_examples/egg_tests.cc) provides an example of how the e-graph things work here, run the script to try:

```bash
// THIS REQUIRES libcuda.so
```

Once a complete graph is generated, the **Verifier** (`src/search/verification`) checks if it is functionally equivalent to the user's specification.

TODO: Add verification details.

## 4. Compiler Pipeline

When you run `mi.superoptimize(graph)` (defined in [`python/mirage/kernel.py`](../../python/mirage/kernel.py)), the following pipeline executes:

1.  **Input Specification**: You define the target computation (e.g., RMSNorm + Linear) using the Mirage Python API.
2.  **Search (`src/search`)**:
    *   `KernelGraphGenerator` initializes the search.
    *   It explores the space of Kernel Graphs and Thread Block Graphs.
    *   It uses `DimStrategy` to guide the search and `AbstractExpr` to prune.
    *   Valid graphs are verified and collected.
3.  **Transpilation (`src/transpiler`)**:
    *   The best discovered MuGraph is passed to the Transpiler.
    *   **`transpiler_kn.cc`**: Generates the host code (kernel launch).
    *   **`transpiler_tb.cc`**: Generates the device code (CUDA/Triton) for the Thread Block Graph. It handles complex details like:
        *   `get_dtensor_ptr`: Resolving global memory pointers.
        *   `generate_tma_code_hopper`: Generating Tensor Memory Accelerator (TMA) instructions for NVIDIA Hopper GPUs.
4.  **Code Generation & Execution**:
    *   The transpiler outputs a `.cu` file.
    *   NVCC compiles this file into a binary.
    *   Mirage loads and executes the binary.

For details,

```python
class KNGraph:
  def superoptimize(
    self,
    imaps: list = None,
    omaps: list = None,
    griddims: list = None,
    blockdims: list = None,
    fmaps: list = None,
    franges: list = None,
    verbose: bool = False,
    config: str = None,
    backend: str = "cuda",
    warmup_iters: int = 16,
    profile_iters: int = 1000,
    use_graph_dataset: bool = True,
    use_cached_graphs: bool = True,
    save_codes: bool = False,
    is_formal_verified: bool = False,
  ):
    # Some checkpoint handling
    # ..

    # Search for mugraphs
    cygraphs = search(
        self.cygraph,
        backend=backend,
        imaps=imaps,
        omaps=omaps,
        griddims=griddims,
        blockdims=blockdims,
        fmaps=fmaps,
        franges=franges,
        previous_checkpoint=previous_checkpoint,
        verbose=verbose,
        default_config=config,
        is_formal_verified=is_formal_verified,
    )
    all_graphs = [KNGraph(g) for g in cygraphs]
    print("Finished search, discovering {} mugraphs ...".format(len(all_graphs)))
    
    # Backend-specific profiling and selection
    if backend == "cuda":
        # profile and use the best graph
        # ..
        g.compile(
            async_=True,
            inputs=input_tensors,
            pipeline_stages=pipeline_stages,
            num_warp_groups=num_warp_groups,
        )
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        for _ in range(profile_iters):
            g(inputs=input_tensors)
        ender.record()
        torch.cuda.synchronize()
        perf = starter.elapsed_time(ender) / profile_iters
        print("muGraph {}: profiled performance (ms) = {}".format(idx, perf))
        if perf < best_perf:
            best_graph, best_perf = g, perf
        return best_graph
    elif backend == "nki":
        # ..
    elif backend == "triton":
        # ..
        return best_graph
    else:
        assert False, "Unsupported backend"
        return None
```


For running on a non-GPU machine, you can use the `FAKE_GPU` option in `config.cmake`:

```
python3 tutorial/c1_architecture/run_search.py
```

But to my suprise, it gives:
```
Mirage::DeviceMemoryManager: gpu_id(0) num_gpus(1)========== Search Configuration ==========
  max num threadblock graph op: 9
  max num kernel_graph op: 5
  max num threadblock graphs: 1
  max num threadblock graph inputs: 3
  max num threadblock graph outputs: 2
  search_thread: 16
  imaps to explore:
  imap combs to explore:
  omaps to explore:
  grid dims to explore:
  block dims to explore:
  fmaps to explore:
  franges to explore:4 16 64 
num_thread = 16
num_tasks = 0 tasks901, Random tests: 404, Valid mugraphs: 0, Time: 7.874953

[Search] Second step finished. Time elapsed: 7.921125sec
[Search] Total states explored: 53934
[Search] Random tests performed: 404
[Serach] Valid kernel graphs explored: 0
Finished search, discovering 0 mugraphs ...
```

## Summary

*   **MuGraph** captures the GPU memory hierarchy (Global vs. Shared).
*   **Search** explores different ways to map computation to this hierarchy (Tiling, Fusion).
*   **Transpiler** converts the best graph into highly optimized CUDA code.

By modifying `DimStrategy` or adding new operators to `Graph` and `Transpiler`, you can extend Mirage to support new hardware features or operations.
