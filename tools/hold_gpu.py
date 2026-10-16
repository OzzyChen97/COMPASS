#!/usr/bin/env python
"""Reserve CUDA memory until the process receives SIGINT or SIGTERM."""

import argparse
import signal
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-gb", type=float, default=70.0)
    parser.add_argument("--compute-size", type=int, default=4096)
    parser.add_argument("--compute-seconds", type=float, default=0.35)
    parser.add_argument("--cycle-seconds", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    reserve_bytes = int(args.memory_gb * 1024**3)
    allocation = torch.empty(reserve_bytes, dtype=torch.uint8, device=device)
    matrix_a = torch.randn(
        args.compute_size, args.compute_size, dtype=torch.float16, device=device)
    matrix_b = torch.randn_like(matrix_a)
    torch.cuda.synchronize(device)

    print(
        f"Holding {args.memory_gb:.1f} GiB on {torch.cuda.get_device_name(device)} "
        f"({device}); compute duty cycle "
        f"{100 * args.compute_seconds / args.cycle_seconds:.0f}%",
        flush=True,
    )
    while running:
        cycle_started = time.monotonic()
        while running and time.monotonic() - cycle_started < args.compute_seconds:
            matrix_a = torch.mm(matrix_a, matrix_b)
            matrix_a.mul_(1e-3)
            torch.cuda.synchronize(device)
        remaining = args.cycle_seconds - (time.monotonic() - cycle_started)
        if remaining > 0:
            time.sleep(remaining)

    del matrix_a, matrix_b, allocation
    torch.cuda.empty_cache()
    print("GPU memory released", flush=True)


if __name__ == "__main__":
    main()
