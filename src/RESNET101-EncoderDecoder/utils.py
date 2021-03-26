from typing import List, Optional

import torch
import random
import numpy as np
import torch.distributed as dist
from torch import Tensor

import json
import os

MAX_DIM = 299

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def load_selfies(filename):
    idx2selfies, selfies2idx = {}, {}
    idx = 0
    with open(filename) as f:
        for l in f:
            l = l.strip()
            idx2selfies[idx] = l
            selfies2idx[l] = idx
            idx += 1
    return idx2selfies, selfies2idx

def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim
    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)
    return image

def read_json(file_name):
    with open(file_name) as handle:
        out = json.load(handle)
    return out

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = [3, MAX_DIM, MAX_DIM]
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0