import os
from pathlib import Path

import hydra
import wandb
from experiments.experiments import Experiment
from hydra.core.config_store import ConfigStore

from config import MLConfig
from models.classifier import MultiLabelClassifier
from utils.utils import set_random_state


@hydra.main(config_path='conf', config_name='config', version_base='1.3')
def main(cfg):
    os.environ['TOKENIZERS_PARALLELISM'] = (
        'false'  # Set to prevent deadlock in tokenizer class when using more than 1 worker in dataloader
    )
    set_random_state(cfg.reproducibility.random_state, deterministic=False)
    cs = ConfigStore.instance()
    cs.store(name='config', node=MLConfig)
    with wandb.init(project='linkedin_prediction'):
        mlc = MultiLabelClassifier(
            model_config=cfg.models,
            criterion_config=cfg.criterion,
            optimizer_config=cfg.optimizer,
            scheduler_config=cfg.scheduler,
            batch_size=cfg.experiments.classifier.batch_size,
            constraint_file=Path(cfg.paths.constraint_file),
        )
        experiment = Experiment(
            mlc=mlc,
            multilabel_stratified_shuffle_split_config=cfg.cross_validation,
            experiment_conf=cfg.experiments,
            preprocessor_conf=cfg.preprocessing,
            random_state=cfg.reproducibility.random_state,
            folder_conf=cfg.paths,
            classifier_input_dim=cfg.models.classifier.input_dim,
            classifier_output_dim=cfg.models.classifier.output_dim,
        )
        experiment.run()


if __name__ == '__main__':
    main()
