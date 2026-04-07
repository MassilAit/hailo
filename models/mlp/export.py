"""
Export MLP models to ONNX (compatible with Hailo SDK onnx==1.16.0).
Uses onnx directly (no torch) to avoid protobuf/onnxscript conflicts.

Architecture: Linear -> ReLU -> ... -> Linear
Input:  [1, INPUT_DIM]  float32
Output: [1, OUTPUT_DIM] float32
"""

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper
import os


def make_linear_relu(prefix, in_dim, out_dim, nodes, initializers):
    """Add a Linear + ReLU block to nodes/initializers. Returns output tensor name."""
    W = np.random.randn(out_dim, in_dim).astype(np.float32) * (2 / in_dim) ** 0.5
    b = np.zeros(out_dim, dtype=np.float32)

    W_name = f"{prefix}_weight"
    b_name = f"{prefix}_bias"
    mm_out = f"{prefix}_mm"
    add_out = f"{prefix}_add"
    relu_out = f"{prefix}_relu"

    initializers.append(numpy_helper.from_array(W, W_name))
    initializers.append(numpy_helper.from_array(b, b_name))

    nodes.append(helper.make_node("Gemm", [mm_out.replace("mm", "in"), W_name, b_name], [add_out],
                                  name=f"{prefix}_gemm", transB=1))
    nodes.append(helper.make_node("Relu", [add_out], [relu_out], name=f"{prefix}_relu"))
    return relu_out


def make_linear(prefix, in_name, in_dim, out_dim, nodes, initializers):
    W = np.random.randn(out_dim, in_dim).astype(np.float32) * (2 / in_dim) ** 0.5
    b = np.zeros(out_dim, dtype=np.float32)
    W_name = f"{prefix}_weight"
    b_name = f"{prefix}_bias"
    out_name = f"{prefix}_out"
    initializers.append(numpy_helper.from_array(W, W_name))
    initializers.append(numpy_helper.from_array(b, b_name))
    nodes.append(helper.make_node("Gemm", [in_name, W_name, b_name], [out_name],
                                  name=f"{prefix}_gemm", transB=1))
    return out_name


def build_mlp_onnx(name, input_dim, hidden_dim, output_dim, num_layers):
    nodes = []
    initializers = []
    dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

    current = "input"
    for i in range(len(dims) - 1):
        in_d = dims[i]
        out_d = dims[i + 1]
        W = np.random.randn(out_d, in_d).astype(np.float32) * (2 / in_d) ** 0.5
        b = np.zeros(out_d, dtype=np.float32)
        W_name = f"layer{i}_W"
        b_name = f"layer{i}_b"
        gemm_out = f"layer{i}_gemm"
        initializers.append(numpy_helper.from_array(W, W_name))
        initializers.append(numpy_helper.from_array(b, b_name))
        nodes.append(helper.make_node("Gemm", [current, W_name, b_name], [gemm_out],
                                      name=f"layer{i}_gemm", transB=1))
        if i < len(dims) - 2:
            relu_out = f"layer{i}_relu"
            nodes.append(helper.make_node("Relu", [gemm_out], [relu_out], name=f"layer{i}_relu"))
            current = relu_out
        else:
            current = gemm_out

    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, input_dim])],
        [helper.make_tensor_value_info(current,  TensorProto.FLOAT, [1, output_dim])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7
    # rename output to "output"
    graph.output[0].name = "output"
    nodes[-1].output[0] = "output"
    onnx.checker.check_model(model)
    return model


CONFIGS = {
    "mlp_small":  dict(input_dim=64,  hidden_dim=128,  output_dim=16, num_layers=3),
    "mlp_medium": dict(input_dim=128, hidden_dim=256,  output_dim=32, num_layers=4),
    "mlp_large":  dict(input_dim=256, hidden_dim=512,  output_dim=64, num_layers=5),
}

if __name__ == "__main__":
    os.makedirs("onnx", exist_ok=True)
    np.random.seed(42)
    for name, cfg in CONFIGS.items():
        model = build_mlp_onnx(name, **cfg)
        path = f"onnx/{name}.onnx"
        onnx.save(model, path)
        print(f"Exported {name}: input={cfg['input_dim']} hidden={cfg['hidden_dim']} "
              f"output={cfg['output_dim']} layers={cfg['num_layers']} -> {path}")
