import torch
from typing import List, Tuple, Union
from detectron2.layers import cat, permute_to_N_HWA_K

class levelwiseTensor:
    """
    """

    def __init__(self, tensors: List[torch.Tensor], K: int):
        """
        """
        device = tensors[0].device if isinstance(tensors[0], torch.Tensor) else torch.device("cpu")
        tensors = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in tensors]
        
        self.tensors = [permute_to_N_HWA_K(x, K) for x in self.tensors]

    def level_first_list(self):
        return  self.tensors

    def image_first_tensor(self):
        return cat(self.tensors, dim=1)