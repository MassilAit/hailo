"""
Hailo 8L latency benchmark using HailoRT Python API.

Install on Raspberry Pi (with Hailo PCIe/M.2 driver):
    pip install hailo-platform   # included in HailoRT package

Run:
    python run_hailo.py --hef ../models/mlp/hef/mlp_small.hef --runs 200
"""

import argparse
import time
import numpy as np

try:
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface, FormatType,
        InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams,
    )
except ImportError:
    raise SystemExit(
        "hailo-platform not found. "
        "Install HailoRT on the Raspberry Pi and run: pip install hailo-platform"
    )


def benchmark(hef_path, n_runs=200, warmup=20):
    hef    = HEF(hef_path)
    target = VDevice()

    configure_params = ConfigureParams.create_from_hef(
        hef, interface=HailoStreamInterface.PCIe
    )
    network_groups = target.configure(hef, configure_params)
    ng             = network_groups[0]
    ng_params      = ng.create_params()

    in_params  = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

    with ng.activate(ng_params):
        with InferVStreams(ng, in_params, out_params) as infer_pipeline:
            input_info = hef.get_input_vstream_infos()[0]
            shape      = [d for d in input_info.shape]
            name       = input_info.name

            dummy = {name: np.random.randn(*shape).astype(np.float32)}

            # warmup
            for _ in range(warmup):
                infer_pipeline.infer(dummy)

            # measure
            t0 = time.perf_counter()
            for _ in range(n_runs):
                infer_pipeline.infer(dummy)
            elapsed = time.perf_counter() - t0

    lat_ms = elapsed / n_runs * 1000
    tput   = n_runs / elapsed

    print(f"model     : {hef_path}")
    print(f"runs      : {n_runs}")
    print(f"latency   : {lat_ms:.3f} ms")
    print(f"throughput: {tput:.1f} inf/s")
    return lat_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", required=True)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()
    benchmark(args.hef, args.runs)
