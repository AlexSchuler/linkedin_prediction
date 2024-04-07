from transformers import BertForSequenceClassification, BertConfig, BatchEncoding
import torch
from typing import Optional


class CustomBertModel(BertForSequenceClassification):
    def __init__(self, bert_config: Optional[BertConfig] = None) -> None:
        if not bert_config:
            bert_config = BertConfig()
        super().__init__(config = bert_config)

    def forward(self, batch_encoding: BatchEncoding) -> torch.Tensor:
        # batch encoding shape of tensors [batch, 1, sequence length]
        input_ids = batch_encoding['input_ids'].squeeze(1)
        token_type_ids = batch_encoding['token_type_ids'].squeeze(1)
        attention_mask = batch_encoding['attention_mask'].squeeze(1)
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return output[1]
