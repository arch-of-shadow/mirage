import mirage as mi
import os


BATCH_SIZE = 1
HIDDEN_DIM = 4096
OUTPUT_DIM = 6144

if __name__ == "__main__":
    graph = mi.new_kernel_graph()

    X = graph.new_input(dims=(2 * BATCH_SIZE, HIDDEN_DIM), dtype=mi.float16)
    W = graph.new_input(dims=(HIDDEN_DIM, OUTPUT_DIM), dtype=mi.float16)
    D = graph.rms_norm(X, normalized_shape=(HIDDEN_DIM,))
    O = graph.matmul(D, W)
    graph.mark_output(O)

    # We don't do any superoptimization for now
    # Just visualize the graph
    visualize_path = os.path.join(os.path.dirname(__file__), "rmsnorm")
    graph.visualize(visualize_path)