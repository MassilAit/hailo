"""
CPU latency benchmark for the TFT ONNX model.

Run:
    python benchmark/run_tft_cpu.py
    python benchmark/run_tft_cpu.py --runs 500
"""

import argparse
import time
import numpy as np
import onnxruntime as ort

HEF_PATH = "models/tft/tft_duree_fin_cluster0.onnx"

# Fixed from model inspection
BATCH         = 1
ENC_SEQ       = 12
DEC_SEQ       = 1
N_CAT         = 7
N_CONT        = 19


def make_dummy_inputs():
    return {
        "encoder_cat":     np.zeros((BATCH, ENC_SEQ, N_CAT),  dtype=np.int64),
        "encoder_cont":    np.random.randn(BATCH, ENC_SEQ, N_CONT).astype(np.float32),
        "decoder_cat":     np.zeros((BATCH, DEC_SEQ, N_CAT),  dtype=np.int64),
        "decoder_cont":    np.random.randn(BATCH, DEC_SEQ, N_CONT).astype(np.float32),
        "encoder_lengths": np.array([ENC_SEQ] * BATCH, dtype=np.int64),
        "decoder_lengths": np.array([DEC_SEQ] * BATCH, dtype=np.int64),
        "target_scale":    np.ones((BATCH, 2), dtype=np.float32),
    }


def benchmark(n_runs=200, warmup=20):
    sess = ort.InferenceSession(HEF_PATH, providers=["CPUExecutionProvider"])
    dummy = make_dummy_inputs()

    for _ in range(warmup):
        sess.run(None, dummy)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, dummy)
    elapsed = time.perf_counter() - t0

    lat_ms = elapsed / n_runs * 1000
    tput   = n_runs / elapsed
    out    = sess.run(None, dummy)[0]

    print(f"model     : {HEF_PATH}")
    print(f"runs      : {n_runs}")
    print(f"latency   : {lat_ms:.3f} ms")
    print(f"throughput: {tput:.1f} inf/s")
    print(f"output    : shape={list(out.shape)}  (batch, horizon, quantiles)")
    return lat_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()
    benchmark(args.runs)
