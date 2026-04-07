"""
Export TCN (Temporal Convolutional Network) models to ONNX.

Conv1D is implemented as Conv2D with kernel [1, K] — Hailo's native format.
Input shape: [1, in_channels, 1, seq_len]  (NCHW, H=1, W=seq_len)

Architecture per layer:
    Conv2D(kernel=[1,K], padding=[0, K//2]) -> ReLU
    + residual projection if channels change

Final:
    GlobalAvgPool -> Flatten -> Linear(output_dim)
"""

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper
import os


def add_conv_relu(nodes, inits, inp, out, in_c, out_c, kernel, layer_idx):
    """Conv2D [1, K] with same-padding along W, then ReLU."""
    W = np.random.randn(out_c, in_c, 1, kernel).astype(np.float32) * (2 / (in_c * kernel)) ** 0.5
    b = np.zeros(out_c, dtype=np.float32)
    W_name = f"tcn_{layer_idx}_W"
    b_name = f"tcn_{layer_idx}_b"
    conv_out = f"tcn_{layer_idx}_conv"
    relu_out = out

    inits += [numpy_helper.from_array(W, W_name), numpy_helper.from_array(b, b_name)]
    nodes.append(helper.make_node(
        "Conv", [inp, W_name, b_name], [conv_out],
        name=f"tcn_{layer_idx}_conv",
        kernel_shape=[1, kernel],
        pads=[0, kernel // 2, 0, kernel // 2],  # pad W only, keep seq_len unchanged
    ))
    nodes.append(helper.make_node("Relu", [conv_out], [relu_out], name=f"tcn_{layer_idx}_relu"))


def add_residual_proj(nodes, inits, inp, out, in_c, out_c, layer_idx):
    """1x1 conv to match channels for residual connection."""
    W = np.random.randn(out_c, in_c, 1, 1).astype(np.float32) * (2 / in_c) ** 0.5
    b = np.zeros(out_c, dtype=np.float32)
    W_name = f"tcn_{layer_idx}_proj_W"
    b_name = f"tcn_{layer_idx}_proj_b"
    inits += [numpy_helper.from_array(W, W_name), numpy_helper.from_array(b, b_name)]
    nodes.append(helper.make_node(
        "Conv", [inp, W_name, b_name], [out],
        name=f"tcn_{layer_idx}_proj",
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
    ))


def build_tcn_onnx(name, in_channels, hidden, out_dim, kernel, n_layers, seq_len):
    nodes, inits = [], []

    current = "input"
    for i in range(n_layers):
        in_c  = in_channels if i == 0 else hidden
        out_c = hidden
        relu_out = f"tcn_{i}_relu"

        add_conv_relu(nodes, inits, current, relu_out, in_c, out_c, kernel, i)

        # residual: add skip connection (project if channels differ)
        add_out = f"tcn_{i}_add"
        if in_c != out_c:
            proj_out = f"tcn_{i}_proj"
            add_residual_proj(nodes, inits, current, proj_out, in_c, out_c, i)
            nodes.append(helper.make_node("Add", [relu_out, proj_out], [add_out], name=f"tcn_{i}_add"))
        else:
            nodes.append(helper.make_node("Add", [relu_out, current], [add_out], name=f"tcn_{i}_add"))

        current = add_out

    # Global average pool over W (time): [1, C, 1, T] -> [1, C, 1, 1]
    pool_out = "gap_out"
    nodes.append(helper.make_node(
        "GlobalAveragePool", [current], [pool_out], name="gap"
    ))

    # Flatten: [1, C, 1, 1] -> [1, C]
    flat_out = "flat_out"
    shape_tensor = numpy_helper.from_array(np.array([1, hidden], dtype=np.int64), "flat_shape")
    inits.append(shape_tensor)
    nodes.append(helper.make_node("Reshape", [pool_out, "flat_shape"], [flat_out], name="flatten"))

    # Linear head
    W_h = np.random.randn(out_dim, hidden).astype(np.float32) * (2 / hidden) ** 0.5
    b_h = np.zeros(out_dim, dtype=np.float32)
    inits += [numpy_helper.from_array(W_h, "head_W"), numpy_helper.from_array(b_h, "head_b")]
    nodes.append(helper.make_node(
        "Gemm", [flat_out, "head_W", "head_b"], ["output"],
        name="head", transB=1
    ))

    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, in_channels, 1, seq_len])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, out_dim])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


# in_channels = features per timestep
# seq_len     = number of timesteps
CONFIGS = {
    "tcn_small":  dict(in_channels=32, hidden=64,  out_dim=16, kernel=3, n_layers=4, seq_len=64),
    "tcn_medium": dict(in_channels=64, hidden=128, out_dim=32, kernel=3, n_layers=6, seq_len=64),
    "tcn_large":  dict(in_channels=64, hidden=256, out_dim=64, kernel=3, n_layers=8, seq_len=128),
}

if __name__ == "__main__":
    os.makedirs("onnx", exist_ok=True)
    np.random.seed(42)
    for name, cfg in CONFIGS.items():
        model = build_tcn_onnx(name, **cfg)
        path = f"onnx/{name}.onnx"
        onnx.save(model, path)
        print(f"Exported {name}: in={cfg['in_channels']} hidden={cfg['hidden']} "
              f"seq={cfg['seq_len']} layers={cfg['n_layers']} -> {path}")
