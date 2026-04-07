"""
CPU latency benchmark using ONNX Runtime.

Install on Raspberry Pi:
    pip install onnxruntime numpy

Run:
    python run_cpu.py --onnx ../models/mlp/onnx/mlp_small.onnx --runs 200
"""

import argparse
import time
import numpy as np
import onnxruntime as ort


def benchmark(onnx_path, n_runs=200, warmup=20):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    inp = sess.get_inputs()[0]
    input_name = inp.name
    input_shape = inp.shape          # e.g. [1, 64]
    input_shape = [d if isinstance(d, int) else 1 for d in input_shape]

    dummy = np.random.randn(*input_shape).astype(np.float32)

    # warmup
    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    # measure
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {input_name: dummy})
    elapsed = time.perf_counter() - t0

    lat_ms = elapsed / n_runs * 1000
    tput   = n_runs / elapsed

    print(f"model     : {onnx_path}")
    print(f"runs      : {n_runs}")
    print(f"latency   : {lat_ms:.3f} ms")
    print(f"throughput: {tput:.1f} inf/s")
    return lat_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()
    benchmark(args.onnx, args.runs)
