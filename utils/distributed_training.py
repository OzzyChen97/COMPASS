"""Small torchrun helpers shared by baseline training entrypoints."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


class NoOpLogger:
    def info(self, *args, **kwargs):
        pass


def setup_distributed():
    if "RANK" not in os.environ:
        torch.cuda.set_device(0)
        return False, 0, 0, 1, torch.device("cuda:0")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    timeout_seconds = int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_SECONDS", "7200"))
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=timeout_seconds),
    )
    return True, rank, local_rank, world_size, torch.device("cuda", local_rank)


def barrier(distributed):
    if distributed:
        dist.barrier()


def reduce_sums(values, device, distributed):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def cleanup(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()
