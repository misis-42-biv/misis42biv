import torch
import torch.nn.functional as F


def stack_pad(items: list[torch.Tensor], pad_value):
    max_len = max(x.shape[0] for x in items)
    items = [F.pad(x, (max_len - x.shape[0], 0), value=pad_value) for x in items]
    return torch.stack(items, dim=0)
