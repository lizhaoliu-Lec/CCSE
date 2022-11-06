from detectron2.utils.comm import get_world_size
import torch.distributed as dist


def all_reduce(data, op="sum"):
    def op_map(op):
        op_dict = {
            "SUM": dist.ReduceOp.SUM,
            "MAX": dist.ReduceOp.MAX,
            "MIN": dist.ReduceOp.MIN,
            "BAND": dist.ReduceOp.BAND,
            "BOR": dist.ReduceOp.BOR,
            "BXOR": dist.ReduceOp.BXOR,
            "PRODUCT": dist.ReduceOp.PRODUCT,
        }
        return op_dict[op]

    world_size = get_world_size()
    if world_size > 1:
        reduced_data = data.clone()
        dist.all_reduce(reduced_data, op=op_map(op.upper()))
        return reduced_data
    return data
