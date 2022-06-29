
import torch
import torch.optim

def train(model,
          dataloader,
          optimizer,
          device,
          scheduler=None):

    model.train()

    epoch_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        sentences, masks, labels = batch[0], batch[1], batch[2]

        output = model(sentences.to(device), masks.to(device), labels.to(device))

        loss = output[0] # get the loss
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


# calculate the prediction accuracy on the provided dataloader
def evaluate_acc(model,
                 dataloader,
                 device):
    model.eval()

    with torch.no_grad():
        total_correct = 0
        total = 0
        for i, batch in enumerate(dataloader):
            sentences, masks, labels = batch[0], batch[1], batch[2]

            output = model(sentences.to(device), masks.to(device), labels.to(device))

            logits = output[1][labels != -100] # get the non-padded logits
            # code above reduces the dimension of data by one since each sentence can have different length.

            output_class = torch.argmax(logits, dim=1)

            labels = labels[labels != -100] # get the non-padded classes

            matches = torch.where(output_class == labels.to(device), 1, 0)

            total_correct += torch.sum(matches)

            total += len(matches)

    return total_correct / total