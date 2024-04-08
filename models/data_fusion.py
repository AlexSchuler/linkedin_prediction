from __future__ import annotations

from typing import Union, tuple

import torch
from torch.nn.functional import pad


def concat(features: list[torch.Tensor], perform__pad: bool, padding: Union[tuple, None]) -> torch.Tensor:
    if perform__pad:
        return pad(
            torch.cat(
                tensors=features,
                dim=1,
            ).type(torch.FloatTensor),
            padding,
        )
    return torch.cat(
        tensors=features,
        dim=1,
    ).type(torch.FloatTensor)
