
from datasets import load_dataset

from dataloader import dataloader, ner_collate_fn

import torch
from functools import partial

from model import NERBert

from transformers import BertTokenizerFast
# load the data
dataset = load_dataset("conll2003")

# create dataloader for different splits
train_data = dataloader(dataset['train'])
valid_data = dataloader(dataset['validation'])
test_data = dataloader(dataset['test'])

tag_to_num = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

num_to_tag = {}

for key, value in tag_to_num.items():
    num_to_tag[value] = key

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

ner_model = NERBert(len(tag_to_num))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, collate_fn=partial(ner_collate_fn, tokenizer=tokenizer), shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32, collate_fn=partial(ner_collate_fn, tokenizer=tokenizer), shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, collate_fn=partial(ner_collate_fn, tokenizer=tokenizer), shuffle=False)

from train import train, evaluate_acc

#define hyperparameters
LR = 1e-5

optimizer = torch.optim.Adam(ner_model.parameters(), lr=LR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ner_model = ner_model.to(device)

for iters in range(15):
    print(iters)
    loss = train(ner_model.to(device), train_loader, optimizer, device)
    print(loss)
    val_accuracy = evaluate_acc(ner_model, valid_loader, device)
    print(val_accuracy)

test_accuracy = evaluate_acc(ner_model, test_loader, device)
print(test_accuracy)