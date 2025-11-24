# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Mirage Persistent Kernel (MPK)** is a compiler and runtime system that automatically transforms multi-GPU LLM inference into high-performance megakernels. It uses a multi-level graph representation (MuGraph) to optimize tensor programs by fusing operations and reducing memory traffic.

## Building and Installation

### Quick Installation (Development)
```bash
# Clone with submodules
git clone --recursive --branch mpk https://www.github.com/mirage-project/mirage
cd mirage

# Install from source (editable mode)
pip install -e . -v

# Set MIRAGE_HOME environment variable
export MIRAGE_HOME=$(pwd)
```

### Build Process Details
The build process involves several steps:
1. **Rust libraries**: Two Rust libraries are built automatically during setup:
   - `abstract_subexpr`: Built to `build/abstract_subexpr/release/`
   - `formal_verifier`: Built to `build/formal_verifier/release/`
2. **C++ runtime**: CMake builds the Mirage runtime library (`libmirage_runtime.a`)
3. **Python bindings**: Cython extensions are compiled and linked

### Standalone C++ Build
```bash
# Build Z3 dependency
cd deps/z3
mkdir build && cd build
cmake ..
make -j

# Set Z3_DIR
export Z3_DIR=$MIRAGE_HOME/deps/z3/build

# Build Mirage
cd $MIRAGE_HOME
mkdir build && cd build
cmake ..
make -j
make install
```

### Configuration
Edit `config.cmake` to control build options:
- `USE_CUDA ON/OFF`: Enable CUDA backend (default: ON)
- `USE_NKI ON/OFF`: Enable NKI backend (default: OFF)
- `BUILD_CPP_EXAMPLES ON/OFF`: Build C++ examples (default: OFF)

## Testing

### Python Tests
```bash
# Run CI test suite
tests/ci-tests/run_python_tests.sh before-installation
tests/ci-tests/run_python_tests.sh after-installation

# Run specific pytest tests
pytest tests/python/test_tensor_program.py
```

### Runtime Tests
```bash
# Test runtime kernels (various GPU architectures)
python tests/runtime_python/blackwell/sm100_linear/test_matmul_mpk.py
python tests/runtime_python/hopper/test_*.py
```

## Running Demos

### Basic Demo (Qwen3-8B)
```bash
# Run with native kernels (Triton/FlashInfer)
python demo/qwen3/demo.py

# Run with Mirage-compiled megakernel
python demo/qwen3/demo.py --use-mirage

# Enable profiling
python demo/qwen3/demo.py --use-mirage --profiling
```

### Other Demos
```bash
# LLaMA3-8B
python demo/demo_llama3-8b.py

# Group Query Attention
python demo/demo_group_query_attention.py

# RMSNorm
python demo/demo_rms_norm.py

# LoRA
python demo/demo_lora.py
```

## Code Architecture

### Multi-Level Graph (MuGraph)
Mirage uses a hierarchical graph representation with three levels:

1. **Kernel Graph** (`mirage::kernel::Graph`):
   - Represents operations at GPU grid level
   - Nodes: `KNOperator` (kernel operators)
   - Edges: `DTensor` (device tensors in global memory/DRAM)
   - Location: `src/kernel/graph.cc`, `include/mirage/kernel/graph.h`

2. **Thread Block Graph** (`mirage::threadblock::Graph`):
   - Represents operations within a CUDA thread block
   - Nodes: `TBOperator` (thread block operators)
   - Edges: `STensor` (shared memory tensors/SRAM)
   - Location: `src/threadblock/`, `include/mirage/threadblock/`

3. **Instruction Graph** (lower level):
   - Represents warp-level and thread-level instructions

### Key Components

#### Python API (`python/mirage/`)
- `persistent_kernel.py`: Main `PersistentKernel` API for megakernel compilation
- `kernel.py`: Standard kernel graph API (`KNGraph`)
- `threadblock.py`: Thread block graph API (`TBGraph`)
- `core.py` (Cython): Python bindings to C++ core

#### C++ Core (`src/`)
- `kernel/`: Kernel graph operators (matmul, reduction, RMSNorm, etc.)
- `threadblock/`: Thread block operators and scheduling
- `search/`: Search algorithms for optimization
  - `search.cc`: Main superoptimization search
  - `dim_strategy.cc`: Dimension tiling strategies
  - `abstract_expr/`: Rust-based abstract expression handling
- `transpiler/`: Code generation for CUDA
  - `transpiler_kn.cc`: Kernel-level code generation
  - `transpiler_tb*.cc`: Thread block code generation (generic, Hopper, Blackwell)
- `base/`: Data types and layouts

#### Headers (`include/mirage/`)
- `kernel/`: Kernel graph interface
- `threadblock/`: Thread block graph interface
- `transpiler/runtime/`: Runtime headers for generated kernels
- `persistent_kernel/`: Persistent kernel runtime

### Search and Optimization
The search component (`src/search/search.cc`) implements superoptimization:
- Explores different graph transformations
- Uses equivalence checking (Z3 solver)
- Employs cost models to select best implementations

### Code Generation
The transpiler (`src/transpiler/`) generates CUDA code:
- Supports multiple GPU architectures (Hopper SM90, Blackwell SM100)
- Handles memory planning (shared memory, global memory)
- Generates thread block swizzling patterns

## Common Development Patterns

### Creating a Custom Kernel Graph
```python
import mirage as mi

# Create kernel graph
graph = mi.new_kernel_graph()

# Add input tensors
X = graph.new_input(dims=(batch_size, hidden_size), dtype=mi.float16)
W = graph.new_input(dims=(hidden_size, output_size), dtype=mi.float16)

# Add operators
Y = graph.matmul(X, W)

# Optimize and generate code
optimized_graph = graph.superoptimize()
```

### Creating a Persistent Kernel
```python
import mirage as mi

mpk = mi.PersistentKernel(
    world_size=num_gpus,
    mpi_rank=rank,
    num_workers=96,
    num_local_schedulers=48,
    num_remote_schedulers=0,
    meta_tensors=[step, tokens],
    profiler_tensor=profiler_tensor,
)

# Attach inputs
x = mpk.attach_input(torch_tensor=input_tensor, name="input")

# Define computation
y = mpk.rmsnorm_linear_layer(input=x, weight_norm=w_norm,
                               weight_linear=w_linear, output=output,
                               grid_dim=(96, 1, 1), block_dim=(128, 1, 1))

# Compile and run
mpk.compile()
mpk()  # Execute megakernel
```

### Adding a New Operator
1. Define operator in kernel graph: `src/kernel/your_op.cc`
2. Add thread block implementation: `src/threadblock/your_op.cc`
3. Update transpiler to generate code: `src/transpiler/transpiler_tb.cc`
4. Add Python binding: `python/mirage/_cython/core.pyx`

## Code Formatting
```bash
# Format C++ code (uses clang-format)
./scripts/format.sh
```

## Docker Usage
```bash
# Run Docker container
./docker/run_docker.sh mlso/mirage

# Build custom Docker image
docker build -f docker/Dockerfile -t mirage-custom .
```

## Important Notes

- **GPU Architecture**: The codebase supports CUDA compute capabilities 75, 80, 86, 89, 90 (set in CMakeLists.txt)
- **Branching**: Main development branch is `mpk`, not `main`
- **Dependencies**: Requires CUDA 11.0+, CUDNN 8.0+, CMake 3.24+, Python 3.8+
- **Memory Management**: Be careful with shared memory limits when designing thread block graphs
- **Grid/Block Dimensions**: Should be multiples of SM count for optimal performance
- **Profiling**: Use `--profiling` flag to visualize task execution timelines

## Tutorials
See `tutorial/` directory for in-depth guides:
- `c1_architecture/`: MuGraph architecture, search algorithms, compiler pipeline
