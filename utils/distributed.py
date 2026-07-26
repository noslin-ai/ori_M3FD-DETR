"""分布式训练工具 — DDP 初始化和辅助函数。

支持单机多卡训练 (DistributedDataParallel)。

启动方式:
    torchrun --nproc_per_node=4 train.py
"""

import os
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    """检查分布式是否可用且已初始化。"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """获取总进程数。"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """获取当前进程 rank。"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    """获取本地 rank。"""
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process():
    """检查是否是主进程 (rank == 0)。"""
    return get_rank() == 0


def init_distributed_mode():
    """初始化分布式训练环境。

    从环境变量中读取 RANK, WORLD_SIZE, LOCAL_RANK 等，
    调用 dist.init_process_group。

    Returns:
        dict: {"rank": int, "world_size": int, "local_rank": int}
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
        torch.cuda.set_device(local_rank)

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
    }


def reduce_dict(input_dict, average=True):
    """对字典中的值进行 all-reduce。

    Args:
        input_dict: 值均为 tensor 的字典
        average: 是否求平均

    Returns:
        dict: reduce 后的字典
    """
    if not is_dist_avail_and_initialized():
        return input_dict

    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []
        for k, v in sorted(input_dict.items()):
            names.append(k)
            values.append(v)

        values = torch.stack(values, dim=0)
        dist.all_reduce(values)

        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict


def cleanup_distributed():
    """清理分布式环境。"""
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()
