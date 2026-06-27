"""
Helpers for distributed training.
"""

import os

import torch
import torch.distributed as dist

LOCAL_RANK = int(os.environ.get("LOCAL_RANK"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE"))
WORLD_RANK = int(os.environ.get("RANK"))

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    torch.cuda.set_device(LOCAL_RANK)
    torch.tensor([0.0], device=f"cuda:{LOCAL_RANK}")
    backend = "gloo" if not torch.cuda.is_available() else "nccl"
    dist.init_process_group(backend=backend)

    print(f"[Rank {dist.get_rank()}] using GPU {LOCAL_RANK}: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    dist.barrier(device_ids=[LOCAL_RANK])

def dev():
    """
    Get the device to use for torch.distributed.
    """

    return torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")
