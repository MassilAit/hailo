"""
Export LSTM model to ONNX for CPU benchmark only.

NOTE: LSTM is NOT compatible with Hailo 8L for sequences > 4 timesteps.
The Hailo compiler unrolls LSTM over time steps, creating a sequential chain
that cannot be partitioned across its parallel clusters.
-> Run benchmark/run_cpu.py only for this model.

Input shape:  [1, seq_len, input_size]
Output shape: [1, output_dim]

Architecture:
    Transpose([1, T, F] -> [T, 1, F])
    LSTM (1 or 2 layers)
    Take last hidden state Y_h -> Squeeze -> Linear
"""

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper
import os


def lstm_weights(input_size, hidden_size, seed_offset=0):
    np.random.seed(42 + seed_offset)
    k = 1.0 / hidden_size ** 0.5
    W = np.random.uniform(-k, k, (1, 4 * hidden_size, input_size)).astype(np.float32)
    R = np.random.uniform(-k, k, (1, 4 * hidden_size, hidden_size)).astype(np.float32)
    B = np.zeros((1, 8 * hidden_size), dtype=np.float32)
    return W, R, B


def add_lstm_layer(nodes, inits, inp, prefix, input_size, hidden_size, seq_len, batch=1):
    W, R, B = lstm_weights(input_size, hidden_size, seed_offset=len(inits))
    W_name, R_name, B_name = f"{prefix}_W", f"{prefix}_R", f"{prefix}_B"
    seq_name, h0_name, c0_name = f"{prefix}_seq", f"{prefix}_h0", f"{prefix}_c0"

    inits += [
        numpy_helper.from_array(W, W_name),
        numpy_helper.from_array(R, R_name),
        numpy_helper.from_array(B, B_name),
        numpy_helper.from_array(np.array([seq_len] * batch, dtype=np.int32), seq_name),
        numpy_helper.from_array(np.zeros((1, batch, hidden_size), dtype=np.float32), h0_name),
        numpy_helper.from_array(np.zeros((1, batch, hidden_size), dtype=np.float32), c0_name),
    ]

    Y, Y_h, Y_c = f"{prefix}_Y", f"{prefix}_Yh", f"{prefix}_Yc"
    nodes.append(helper.make_node(
        "LSTM", [inp, W_name, R_name, B_name, seq_name, h0_name, c0_name],
        [Y, Y_h, Y_c], name=f"{prefix}_lstm",
        hidden_size=hidden_size, direction="forward",
    ))
    return Y, Y_h


def build_lstm_onnx(name, input_size, hidden_size, output_dim, seq_len, n_layers):
    nodes, inits = [], []
    batch = 1

    nodes.append(helper.make_node(
        "Transpose", ["input"], ["input_t"],
        name="input_transpose", perm=[1, 0, 2]
    ))

    current_inp, current_size = "input_t", input_size
    for i in range(n_layers):
        Y, Y_h = add_lstm_layer(nodes, inits, current_inp, f"lstm{i}",
                                 current_size, hidden_size, seq_len, batch)
        if i < n_layers - 1:
            sq = f"lstm{i}_Y_sq"
            nodes.append(helper.make_node("Squeeze", [Y], [sq],
                                          name=f"lstm{i}_squeeze", axes=[1]))
            current_inp, current_size = sq, hidden_size

    # Y_h [1, batch, hidden] -> [batch, hidden]
    nodes.append(helper.make_node("Squeeze", [Y_h], ["last_h"],
                                  name="squeeze_yh", axes=[0]))

    W_h = np.random.randn(output_dim, hidden_size).astype(np.float32) * (2 / hidden_size) ** 0.5
    b_h = np.zeros(output_dim, dtype=np.float32)
    inits += [numpy_helper.from_array(W_h, "head_W"), numpy_helper.from_array(b_h, "head_b")]
    nodes.append(helper.make_node("Gemm", ["last_h", "head_W", "head_b"], ["output"],
                                  name="head", transB=1))

    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, [batch, seq_len, input_size])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, output_dim])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    return model


CONFIGS = {
    "lstm_small":  dict(input_size=32, hidden_size=64,  output_dim=16, seq_len=32, n_layers=1),
    "lstm_medium": dict(input_size=64, hidden_size=128, output_dim=32, seq_len=64, n_layers=1),
    "lstm_large":  dict(input_size=64, hidden_size=256, output_dim=64, seq_len=64, n_layers=2),
}

if __name__ == "__main__":
    os.makedirs("onnx", exist_ok=True)
    for name, cfg in CONFIGS.items():
        model = build_lstm_onnx(name, **cfg)
        path = f"onnx/{name}.onnx"
        onnx.save(model, path)
        print(f"Exported {name}: input={cfg['input_size']} hidden={cfg['hidden_size']} "
              f"seq={cfg['seq_len']} layers={cfg['n_layers']} -> {path}")
