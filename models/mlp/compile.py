"""
Hailo DFC pipeline: ONNX -> HAR (quantized) -> HEF

Requires: hailo_venv
Run from models/mlp/:
    python compile.py --model mlp_small
    python compile.py --model mlp_medium
    python compile.py --model mlp_large
    python compile.py --all
"""

import argparse
import os
import numpy as np
from hailo_sdk_client import ClientRunner

CONFIGS = {
    "mlp_small":  dict(input_dim=64),
    "mlp_medium": dict(input_dim=128),
    "mlp_large":  dict(input_dim=256),
}


def compile_model(name, cfg):
    os.makedirs("har", exist_ok=True)
    os.makedirs("hef", exist_ok=True)

    print(f"\n[{name}] 1/3 parsing ONNX")
    runner = ClientRunner(hw_arch="hailo8l")
    runner.translate_onnx_model(
        f"onnx/{name}.onnx",
        name,
        net_input_shapes={"input": [1, cfg["input_dim"]]},
    )
    runner.save_har(f"har/{name}.har")

    print(f"[{name}] 2/3 calibrating (int8)")
    calib = np.random.randn(64, cfg["input_dim"]).astype(np.float32)
    runner.optimize(calib)
    runner.save_har(f"har/{name}_quantized.har")

    print(f"[{name}] 3/3 compiling HEF")
    hef = runner.compile()
    with open(f"hef/{name}.hef", "wb") as f:
        f.write(hef)

    print(f"[{name}] done -> hef/{name}.hef")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=list(CONFIGS))
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    targets = list(CONFIGS) if args.all else [args.model]
    for name in targets:
        compile_model(name, CONFIGS[name])
