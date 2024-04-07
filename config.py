from dataclasses import dataclass


@dataclass
class RandomState:
    state: int


@dataclass
class FolderConf:
    path: str
    file_pattern: str
    save_path: str
    constraint_file: str


@dataclass
class ClassifierConf:
    input_dim: int
    output_dim:int
    hidden_layers:int
    hidden_dim: int
    dropout_probability: float
    activation_func: dict[str]
    norm: dict[str]


@dataclass
class ClassifierOptimizer:
    _target_: str
    momentum: float
    lr: float


@dataclass
class ClassifierCriterion:
    _target_: str


@dataclass
class ClassifierScheduler:
    _target_: str
    gamma: float


@dataclass
class ModelConf:
    classifier_conf: ClassifierConf


@dataclass
class ClassifierExperiementConf:
    epochs: int
    batch_size: int
    num_workers: int


@dataclass
class ExperiementConf:
    classifier_experiementconf: ClassifierExperiementConf


@dataclass
class PreprocessorConf:
    numerical_columns: list[str]
    ordinal_columns: list[str]
    ordinal_categories: list[str]
    one_hot_columns: list[str]


@dataclass
class CrossValidationConf:
    _target_: str
    n_splits: int
    test_size: float


@dataclass
class OptimizerConf:
    classifier_optimizer: ClassifierOptimizer    


@dataclass
class CriterionConf:
    classifier_criterion: ClassifierCriterion


@dataclass
class SchedulerConf:
    classifier_scheduler: ClassifierScheduler


@dataclass
class MLConfig:
    models: ModelConf
    optimizer: OptimizerConf
    criterion: CriterionConf
    scheduler: SchedulerConf
    folder_config: FolderConf
    random_state: RandomState
    experiments: ExperiementConf
    preprocessor_config: PreprocessorConf
    cross_validation: CrossValidationConf
