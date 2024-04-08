from pathlib import Path
from typing import tuple

import torch


def parse_constraints(
    path: Path, num_classes: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    with open(path) as f:
        positive_antecedents = []
        negative_antecedents = []
        consequents = []
        for line in f:
            split = line.split()
            assert split[1] == '<-'
            positive_antecedent = torch.zeros(num_classes, device=device, dtype=dtype)
            negative_antecedent = torch.zeros(num_classes, device=device, dtype=dtype)
            consequent = torch.zeros(num_classes, device=device, dtype=dtype)
            consequent[int(split[0])] = 1
            for consequent in split[2:]:
                if 'n' in consequent:
                    negative_antecedent[int(consequent[1:])] = 1
                else:
                    positive_antecedent[int(consequent)] = 1
            negative_antecedents.append(negative_antecedent)
            positive_antecedents.append(positive_antecedent)
            consequents.append(consequent)
    return torch.stack(positive_antecedents), torch.stack(negative_antecedents), torch.stack(consequents).T
