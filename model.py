
import torch.nn as nn
from transformers import BertForTokenClassification

class NERBert(nn.Module):
    def __init__(self,
                 outputs=2,
                 ):
        super().__init__()

        self.bert_encoder = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=outputs)

        print(self.bert_encoder)

    def forward(self,
                src,
                mask,
                label,
                ):

        output = self.bert_encoder(input_ids=src,
                                   attention_mask=mask,
                                   labels=label,
                                   return_dict=False
                                   )

        return output