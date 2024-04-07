import pandas as pd
import torch
from torch.utils.data import Dataset
from preprocessing import Preprocessor
from typing import Tuple


class PartialDataset(Dataset):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        dataset_type: str,
        preprocessor: Preprocessor,
    ):
        self.X = X
        self.y = y
        self.dataset_type = dataset_type
        self.preprocessor = preprocessor
        self.onh_encodings = None
        self.ord_encodings = None
        self.num_features = None
        self._prepare_dataset()

    def _prepare_dataset(self) -> None:
        self.onh_encodings, self.ord_encodings = self.preprocessor.construct_encodings(
            self.X
        )
        self.num_features = self.preprocessor.preprocess_numerical(
            self.X, self.dataset_type
        )
        to_drop = (
            self.preprocessor.ordinal_columns
            + self.preprocessor.numerical_columns
            + self.preprocessor.one_hot_columns
        )
        self.X = self.X.drop(to_drop,axis=1)

    def __getitem__(self, idx) -> Tuple[list[torch.Tensor, torch.Tensor, torch.Tensor], list[dict], torch.Tensor]:
        num_features = torch.tensor(self.num_features[idx])
        onh_features = torch.tensor(self.onh_encodings[idx])
        ord_features = torch.tensor(self.ord_encodings[idx])
        labels = torch.tensor(self.y.iloc[[idx]].values).type(torch.float32)
        textblocks =self.X.iloc[idx].tolist()
        return ([num_features, onh_features, ord_features], textblocks, labels)

    def __len__(self):
        return self.X.shape[0]
