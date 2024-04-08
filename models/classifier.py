from pathlib import Path
from typing import tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_

from config import (ClassifierConf, CriterionConf, ModelConf, OptimizerConf,
                    SchedulerConf)
from utils.constraints import parse_constraints
from utils.utils import instantiate


class MultiLabelClassifier:
    def __init__(
        self,
        criterion_config: CriterionConf,
        optimizer_config: OptimizerConf,
        scheduler_config: SchedulerConf,
        model_config: ModelConf,
        batch_size: int,
        constraint_file: Path,
    ) -> None:
        super().__init__()
        self.mlp = MLP(model_config.classifier)
        self.criterion = instantiate(criterion_config.classifier)
        self.optimizer = instantiate(optimizer_config.classifier, params=self.mlp.parameters())
        self.scheduler = instantiate(scheduler_config.classifier, optimizer=self.optimizer)
        self.batch_size = batch_size
        self.output_dim = model_config.classifier.output_dim
        self.positive_antecedents = None
        self.negative_antecedents = None
        self.R_batch = None
        self._setup_constraints(
            constraint_file=constraint_file, batch_size=batch_size, output_dim=model_config.classifier.output_dim
        )
        self.num_constraints = self.positive_antecedents.shape[1]

    def _setup_constraints(self, constraint_file: Path, batch_size: int, output_dim: int) -> None:
        positive_antecedents, negative_antecedents, consequents = parse_constraints(
            path=constraint_file, num_classes=output_dim
        )
        num_positive_antecedents = positive_antecedents.shape[0]
        num_negative_antecedents = positive_antecedents.shape[0]
        self.positive_antecedents = positive_antecedents.unsqueeze(0).expand(
            batch_size, num_positive_antecedents, output_dim
        )
        self.negative_antecedents = negative_antecedents.unsqueeze(0).expand(
            batch_size, num_negative_antecedents, output_dim
        )
        R = torch.cat((torch.eye(output_dim).unsqueeze(0), consequents.unsqueeze(0)), dim=2)
        self.R_batch = R.expand(batch_size, output_dim, output_dim + num_positive_antecedents)

    def construct_implicant_vectors(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        # expand is used instead of repeat to avoid excess memory usage, extend returns a view without additional memory overhead
        # number of positive constraints = number of negative constraints due to parsing, missing constraints are filled with zeros
        y_pred = prediction.unsqueeze(1).expand(self.batch_size, self.num_constraints, self.output_dim)
        y = ground_truth.unsqueeze(1).expand(self.batch_size, self.num_constraints, self.output_dim)
        p_min, _ = torch.min(self.positive_antecedents * y_pred * y + (1 - self.positive_antecedents), dim=2)
        n_min, _ = torch.min(
            self.negative_antecedents * (1 - y_pred) * (1 - y) + (1 - self.negative_antecedents), dim=2
        )
        positive_vector = torch.min(p_min, n_min)
        p_min, _ = torch.min(
            self.positive_antecedents * y_pred * (1 - y)
            + (1 - self.positive_antecedents)
            + self.positive_antecedents * y,
            dim=2,
        )
        n_min, _ = torch.min(
            self.negative_antecedents * (1 - y_pred) * y
            + (1 - self.negative_antecedents)
            + self.negative_antecedents * (1 - y),
            dim=2,
        )
        negative_vector = torch.min(p_min, n_min)
        return positive_vector, negative_vector

    def calculate_constraint_output(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        v_positive, v_negative = self.construct_implicant_vectors(prediction=y_pred, ground_truth=y)
        V_positive = (
            torch.cat((y_pred, v_positive), dim=1)
            .unsqueeze(1)
            .expand(self.batch_size, self.output_dim, self.output_dim + self.num_constraints)
        )
        V_negative = (
            torch.cat((y_pred, v_negative), dim=1)
            .unsqueeze(1)
            .expand(self.batch_size, self.output_dim, self.output_dim + self.num_constraints)
        )
        Y = y.unsqueeze(1).expand(self.batch_size, self.num_constraints, self.output_dim)
        IY_T = torch.cat(
            (
                torch.eye(self.output_dim).unsqueeze(0).expand(self.batch_size, self.output_dim, self.output_dim),
                torch.transpose(Y, 1, 2),
            ),
            dim=2,
        ).expand(self.batch_size, self.output_dim, self.output_dim + self.num_constraints)
        constraint_out, _ = torch.max(
            (self.R_batch * V_positive * IY_T + (self.R_batch * V_negative) * (1 - IY_T)), dim=2
        )
        return constraint_out

    def training_step(self, X_train: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        y_pred = self.mlp(X_train)
        constraint_out = self.calculate_constraint_output(y_pred=y_pred, y=y_train)
        loss = self.criterion(constraint_out, y_train)
        loss.backward()
        clip_grad_norm_(self.mlp.parameters(), max_norm=2.0, norm_type=2)
        self.optimizer.step()
        return loss

    def make_prediction(self, X: torch.Tensor) -> torch.Tensor:
        # ToDo: update to account for constraint module during inference time
        logit_prediction = self.mlp(X)
        return F.sigmoid(logit_prediction)


class MLP(nn.Module):
    def __init__(
        self,
        layer_conf: dict,
    ) -> None:
        super().__init__()
        self.mlp = _construct_layers(layer_conf)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mlp(X)


def _construct_layers(
    layer_conf: ClassifierConf,
) -> nn.Sequential:
    """Constructs a series of linear, batchnorm, activation and optionally dropout layers.
    Last layer returns logits.

    Args:
    ----
        layer_conf: input_dim: int, output_dim:int, hidden_layers:int, hidden_dim:
                    dropout_probability: Union[None, float],
                    activation_func: torch.nn.Module
                    norm: torch.nn._NormBase

    """
    layers = []
    layers.extend([
        nn.Linear(in_features=layer_conf.input_dim, out_features=layer_conf.hidden_dim),
        instantiate(layer_conf.activation_func),
        instantiate(layer_conf.norm, num_features=layer_conf.hidden_dim),
    ])
    if 'dropout_probability' in layer_conf:
        layers.append(nn.Dropout(layer_conf.dropout_probability))
    for i in range(layer_conf.hidden_layers):
        layers.extend([
            nn.Linear(in_features=layer_conf.hidden_dim, out_features=layer_conf.hidden_dim),
            instantiate(layer_conf.activation_func),
            instantiate(layer_conf.norm, num_features=layer_conf.hidden_dim),
        ])
        if 'dropout_probability' in layer_conf:
            layers.append(nn.Dropout(layer_conf.dropout_probability))
    layers.extend(nn.Linear(layer_conf.hidden_dim, layer_conf.output_dim), nn.Sigmoid())
    return nn.Sequential(*layers)
