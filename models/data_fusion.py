import torch
from torch.nn.functional import pad
from typing import Tuple, Union

def concat(features: list[torch.Tensor],perform__pad: bool, padding: Union[Tuple,None]) -> torch.Tensor:
    if perform__pad:
        return pad(torch.cat(tensors=features, dim=1, ).type(torch.FloatTensor), padding)
    else:
        return torch.cat(tensors=features, dim=1,).type(torch.FloatTensor)