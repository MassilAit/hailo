"""
Run all benchmarks (CPU + Hailo) and print a summary table.

Usage:
    python benchmark/run_all.py
    python benchmark/run_all.py --runs 500
    python benchmark/run_all.py --no-hailo   # CPU only (no Hailo device)
"""

import argparse
import os
import time
import numpy as np
import onnxruntime as ort

MODELS = {
    "mlp": [
        ("mlp_small",  "models/mlp/onnx/mlp_small.onnx",  "models/mlp/hef/mlp_small.hef"),
        ("mlp_medium", "models/mlp/onnx/mlp_medium.onnx", "models/mlp/hef/mlp_medium.hef"),
        ("mlp_large",  "models/mlp/onnx/mlp_large.onnx",  "models/mlp/hef/mlp_large.hef"),
    ],
    "tcn": [
        ("tcn_small",  "models/tcn/onnx/tcn_small.onnx",  "models/tcn/hef/tcn_small.hef"),
        ("tcn_medium", "models/tcn/onnx/tcn_medium.onnx", "models/tcn/hef/tcn_medium.hef"),
        ("tcn_large",  "models/tcn/onnx/tcn_large.onnx",  "models/tcn/hef/tcn_large.hef"),
    ],
    "lstm (cpu only)": [
        ("lstm_small",  "models/lstm/onnx/lstm_small.onnx",  None),
        ("lstm_medium", "models/lstm/onnx/lstm_medium.onnx", None),
        ("lstm_large",  "models/lstm/onnx/lstm_large.onnx",  None),
    ],
}


def bench_cpu(onnx_path, n_runs, warmup):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    dummy = np.random.randn(*shape).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {inp.name: dummy})
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {inp.name: dummy})
    return (time.perf_counter() - t0) / n_runs * 1000  # ms


def bench_hailo(hef_path, n_runs, warmup):
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface, FormatType,
        InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams,
    )
    hef = HEF(hef_path)
    target = VDevice()
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    ng = network_groups[0]
    ng_params = ng.create_params()
    in_params  = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

    with ng.activate(ng_params):
        with InferVStreams(ng, in_params, out_params) as pipeline:
            info  = hef.get_input_vstream_infos()[0]
            dummy = {info.name: np.random.randn(*info.shape).astype(np.float32)}
            for _ in range(warmup):
                pipeline.infer(dummy)
            t0 = time.perf_counter()
            for _ in range(n_runs):
                pipeline.infer(dummy)
            return (time.perf_counter() - t0) / n_runs * 1000  # ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",     type=int, default=200)
    parser.add_argument("--warmup",   type=int, default=20)
    parser.add_argument("--no-hailo", action="store_true", help="Skip Hailo benchmarks")
    args = parser.parse_args()

    results = []  # (family, name, cpu_ms, hailo_ms or None)

    for family, entries in MODELS.items():
        for name, onnx_path, hef_path in entries:
            cpu_ms, hailo_ms = None, None

            if os.path.exists(onnx_path):
                print(f"  CPU  {name}...", end=" ", flush=True)
                try:
                    cpu_ms = bench_cpu(onnx_path, args.runs, args.warmup)
                    print(f"{cpu_ms:.3f} ms")
                except Exception as e:
                    print(f"ERROR: {e}")
            else:
                print(f"  CPU  {name}: ONNX not found, skipping")

            if hef_path and not args.no_hailo:
                if os.path.exists(hef_path):
                    print(f"  Hailo {name}...", end=" ", flush=True)
                    try:
                        hailo_ms = bench_hailo(hef_path, args.runs, args.warmup)
                        print(f"{hailo_ms:.3f} ms")
                    except Exception as e:
                        print(f"ERROR: {e}")
                else:
                    print(f"  Hailo {name}: HEF not found, skipping")

            results.append((family, name, cpu_ms, hailo_ms))

    # Summary table
    col = 14
    print()
    print("=" * 62)
    print(f"{'Model':<20} {'CPU (ms)':>10} {'Hailo (ms)':>12} {'Speedup':>10}")
    print("-" * 62)
    current_family = None
    for family, name, cpu_ms, hailo_ms in results:
        if family != current_family:
            print(f"  [{family}]")
            current_family = family
        cpu_s   = f"{cpu_ms:.3f}"   if cpu_ms   is not None else "—"
        hailo_s = f"{hailo_ms:.3f}" if hailo_ms is not None else "—"
        if cpu_ms and hailo_ms:
            speedup = f"{cpu_ms / hailo_ms:.1f}x"
        elif hailo_ms is None and cpu_ms:
            speedup = "cpu only"
        else:
            speedup = "—"
        print(f"  {name:<18} {cpu_s:>10} {hailo_s:>12} {speedup:>10}")
    print("=" * 62)


if __name__ == "__main__":
    main()
