import mirage as mi
import pathlib as pl

CURRENT_DIR = pl.Path(__file__).parent
if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(16, 4096), dtype=mi.float16)
    W = graph.new_input(dims=(4096, 6144), dtype=mi.float16)
    D = graph.rms_norm(X, normalized_shape=(4096,))
    O = graph.matmul(D, W)
    graph.mark_output(O)
    vis_dir = CURRENT_DIR / "searched_graphs"
    optimized_graph = graph.superoptimize(config="mlp", use_graph_dataset=False, use_cached_graphs=False, graph_visualization_dir=vis_dir)