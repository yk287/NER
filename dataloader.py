



from torch.utils.data import Dataset
import torch

class dataloader(Dataset):

    def __init__(self, dataset):

        self.dataset = dataset

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        return self.dataset[index]


def ner_collate_fn(batch, tokenizer):
    # get the max len so that we can pad
    max_len = 0
    sents = []
    ners = []

    # find the max len of the batch
    for sen in batch:
        s = ' '.join(sen['tokens'])

        sents.append(s)
        ners.append(sen['ner_tags'])

        if len(sen['tokens']) > max_len:
            max_len = len(sen['tokens'])

    # tokenize the inputs
    tokenized = tokenizer(sents, padding="max_length", max_length=max_len, truncation=True)

    # pad the labels
    temp = []
    for v in ners:
        pad = max_len - len(v)
        temp.append(v + [-100] * pad) # apparently huggingface ignores labels with value -100
        #https://discuss.huggingface.co/t/how-to-structure-labels-for-token-classification/1216

    return torch.tensor(tokenized['input_ids']), torch.tensor(tokenized['attention_mask']), torch.tensor(temp)