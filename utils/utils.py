import os
import random
from functools import partial, update_wrapper
from importlib import import_module
from pathlib import Path
from typing import Union

import hydra
import numpy as np
import pandas as pd
import polars as pl
import torch

def set_device_agnostic():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device=device)

def set_random_state(seed: int, deterministic: Union[bool, None])-> None:
    """Sets randomness for numpy, pythons random libary, torch and CUDA.
    Using deterministic can degrade performance of torch.
    """
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True

def save(model: torch.nn.Module, optimizer: torch.optim, scheduler: torch.optim, path: Path, epoch: int= 0):
    torch.save(model.state_dict(), str(path)+f'/{epoch}.model')
    torch.save(optimizer.state_dict(), str(path)+f'/{epoch}.optimizer')
    torch.save(scheduler.state_dict(), str(path)+f'/{epoch}.scheduler')


def instantiate(config, *args, is_func: bool=False, **kwargs):
    """Wrapper function for hydra.utils.instantiate.
    1. return None if config.class is None
    2. return function handle if is_func is True
    """
    assert "_target_" in config, "Config should have '_target_' for class instantiation."
    target = config["_target_"]
    if target is None:
        return None
    if is_func:
        # get function handle
        modulename, funcname = target.rsplit(".", 1)
        mod = import_module(modulename)
        func = getattr(mod, funcname)

        # make partial function with arguments given in config, code
        kwargs.update({k: v for k, v in config.items() if k != "_target_"})
        partial_func = partial(func, *args, **kwargs)

        # update original function's __name__ and __doc__ to partial function
        update_wrapper(partial_func, func)
        return partial_func
    return hydra.utils.instantiate(config, *args, **kwargs)


def load_dataset_list(
        path: str,
        file_pattern: str,
) -> list[Path, None, None]:
    folder_path = Path(path)
    return list(folder_path.glob("*." + file_pattern))


def load_dataset(
        path: Path,
        backend: str = "pandas",
) -> Union[pd.DataFrame, pl.DataFrame, None]:
    dataset = None
    file_extension = path.suffix[1:]
    match [file_extension, backend]:
        case ["csv", "pandas"]:
            dataset = pd.read_csv(path, index_col=False, )
        case ["csv", "polars"]:
            dataset = pl.read_csv(path)
        case ["parquet", "pandas"]:
            dataset = pd.read_parquet(path)
        case ["parquet", "polars"]:
            dataset = pl.read_parquet(path)
    return dataset
