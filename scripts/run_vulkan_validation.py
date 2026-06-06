#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import os
import subprocess
import sys


OUTPUT_TAIL_LINES = 160
TERMINATION_TIMEOUT_SECONDS = 5.0

VALIDATION_MARKERS = (
    "Validation Error",
    "VUID-",
    "ERROR / SPEC",
)

MISSING_LAYER_MARKERS = (
    "VK_LAYER_KHRONOS_validation",
    "not found",
    "Failed to find layer",
    "doesn't exist",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command under Vulkan validation layers and fail on validation errors."
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("expected command after --")

    env = os.environ.copy()
    env["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"

    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_tail: deque[str] = deque(maxlen=OUTPUT_TAIL_LINES)
    has_validation_error = False

    assert process.stdout is not None
    for line in process.stdout:
        output_tail.append(line)
        if any(marker in line for marker in VALIDATION_MARKERS):
            has_validation_error = True
            process.terminate()
            break

    try:
        return_code = process.wait(timeout=TERMINATION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        return_code = process.wait()

    output = "".join(output_tail)
    missing_layer = (
        "VK_LAYER_KHRONOS_validation" in output
        and any(marker in output for marker in MISSING_LAYER_MARKERS)
    )

    if return_code != 0 or has_validation_error or missing_layer:
        if output:
            print(output, end="")
        if missing_layer:
            print(
                "VK_LAYER_KHRONOS_validation is required for BlockLab validation tests.",
                file=sys.stderr,
            )
        if has_validation_error:
            print("Vulkan validation reported an error.", file=sys.stderr)
        return return_code if return_code != 0 and not has_validation_error else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
