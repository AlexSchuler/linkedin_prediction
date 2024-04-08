from typing import tuple

import torch
import wandb
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertConfig

from dataset.dataset import PartialDataset
from models.classifier import MultiLabelClassifier
from models.custom_bert import CustomBertModel
from models.data_fusion import concat


class ClassifierRunner:
    def __init__(
        self,
        mlc: MultiLabelClassifier,
        batch_size: int,
        num_workers: int,
        input_dim: int,
    ):
        self.mlc = mlc
        self.bert_model = 'bert-base-multilingual-uncased'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classifier_input_dim = input_dim
        self.bert, self.tokenizer = self._setup_nlp(self.bert_model)
        self.example_cnt = 0  # number of examples seen

    def train(self, partial_dataset: PartialDataset, epoch: int) -> None:
        self.mlc.mlp.train(mode=True)
        batch_cnt = 0
        train_dataloader = DataLoader(
            dataset=partial_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        for encoded_features, textblocks, y in train_dataloader:
            tokenized_textblocks = self._tokenize_textblocks(textblocks)
            embedded_textblocks = [
                self.bert(textblock) for textblock in tokenized_textblocks
            ]
            X = concat(
                encoded_features + embedded_textblocks,
                perform__pad=True,
                padding=(0, self.classifier_input_dim - _get_dim(tensors=encoded_features + embedded_textblocks)),
            )
            loss = self.mlc.training_step(X_train=X, y_train=y.squeeze(1))
            if ((batch_cnt) % 5) == 0:
                wandb.log({'Epoch': epoch, 'Loss': loss.data}, step=self.example_cnt)
            batch_cnt = batch_cnt + 1
            self.example_cnt = self.example_cnt + len(X)
            print(f'Loss for batch {loss.data}')

    def evaluate(self, validation_dataset: PartialDataset, threshold: float = 0.5) -> tuple:
        self.mlc.mlp.eval()
        validation_dataloader = DataLoader(
            dataset=validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )
        y_pred, y_val = [], []
        with torch.no_grad():
            for encoded_features, textblocks, y in validation_dataloader:
                tokenized_textblocks = self._tokenize_textblocks(textblocks)
                embedded_textblocks = [
                    self.bert(textblock) for textblock in tokenized_textblocks
                ]
                X = concat(
                    encoded_features + embedded_textblocks,
                    perform__pad=True,
                    padding=(0, self.classifier_input_dim - _get_dim(tensors=encoded_features + embedded_textblocks)),
                )
                pred = self.mlc.make_prediction(X=X)
                y_pred.append((pred > threshold).int())
                y_val.append(y.squeeze(1))
        y_val, y_pred = torch.cat(y_val), torch.cat(y_pred)
        f1_weighted = f1_score(y_true=y_val, y_pred=y_pred, average='weighted')
        f1_micro = f1_score(y_true=y_val, y_pred=y_pred, average='micro')
        hamming = hamming_loss(y_true=y_val, y_pred=y_pred)
        roc_auc = roc_auc_score(y_true=y_val, y_score=y_pred, average='weighted')
        return f1_weighted, f1_micro, hamming, roc_auc

    def _setup_nlp(self, bert_model: str) -> tuple[CustomBertModel, AutoTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        bert_config = BertConfig(
            vocab_size=tokenizer.vocab_size,
        )
        bert = CustomBertModel(bert_config=bert_config)
        return (bert, tokenizer)

    def _tokenize_textblocks(self, textblocks: list) -> torch.Tensor:
        return [
            self.tokenizer(
                seq,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt',
            )
            for seq in textblocks
        ]


def _get_dim(tensors: list[torch.Tensor]) -> int:
    return sum(t.shape[1] for t in tensors)
