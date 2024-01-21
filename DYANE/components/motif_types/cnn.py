import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn import Conv2d
from torch.nn import CrossEntropyLoss
from torch.nn import Embedding, ModuleList, Dropout
from torch.nn import Linear
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

from DYANE.components.motif_types.data.process_motifs_data import helper_get_data_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MotifTypeCNN(torch.nn.Module):
    # def __init__(self, num_nodes, vector_size,
    def __init__(self, pretrained_embeddings, vector_size,
                 n_filters, filter_sizes, output_dim):
        super(MotifTypeCNN, self).__init__()

        # Embeddings "lookup table"
        self.embedding = Embedding.from_pretrained(pretrained_embeddings,
                                                   freeze=True)

        # Specify convolutions with filters of different sizes (fs)
        # in_channels - Number of channels in the input image
        # out_channels - Number of channels produced by the convolution
        # kernel_size - Size of the convolving kernel (height, width)
        self.convs = ModuleList([Conv2d(in_channels=1,
                                        out_channels=n_filters,
                                        kernel_size=(fs, vector_size))
                                 for fs in filter_sizes])

        # Add a fully connected layer for final predictions
        self.linear = Linear(len(filter_sizes) * n_filters, output_dim)

        # Drop some of the nodes to increase robustness in training
        # self.dropout = Dropout(dropout)
        self.dropout = Dropout()

    def forward(self, x):
        # Get *node* embeddings and format them for convolutions
        x = self.embedding(x)
        x = x.to(torch.float)
        x = x.unsqueeze(1)

        # torch.permute(x, (2, 0, 1))

        # Perform convolutions and apply activation functions
        conved = [F.relu(conv(x)).squeeze(3)
                  for conv in self.convs]

        # Pooling layer to reduce dimensionality
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]

        # Dropout layer
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.linear(cat).squeeze(1)

    def update_embeddings(self, new_embeddings):
        # Update embeddings "lookup table"
        self.embedding = Embedding.from_pretrained(new_embeddings,
                                                   freeze=True)


def train(model, iterator, optimizer, criterion, labels):
    print("|- Training...")
    sys.stdout.flush()

    epoch_loss = 0
    epoch_acc = 0
    epoch_auc = 0

    model.train()

    for batch in tqdm(iterator, file=sys.stdout, desc="| |- Train batches"):
        x = batch['x'].to(device)
        y = batch['y'].to(device).squeeze(1)

        predictions = model(x).to(torch.float)

        loss = criterion(predictions, y)

        # https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
        # Reset the gradient to not use them in multiple passes
        optimizer.zero_grad()
        # Backprop
        loss.backward()
        # Weights optimizing
        optimizer.step()

        predictions = F.softmax(predictions, dim=-1)
        acc = accuracy(torch.argmax(predictions, dim=1), y)
        preds = predictions.clone().detach().cpu().numpy()
        target = y.clone().cpu().numpy()
        auc = roc_auc_score(target, preds, multi_class='ovo', average='macro', labels=labels)

        # Record accuracy and loss
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_auc += auc

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_auc / len(iterator)


# @torch.no_grad()  # No need to backprop in eval
def test(model, iterator, criterion, labels):
    '''Evaluate model performance. '''
    print("|- Evaluating...")
    sys.stdout.flush()

    epoch_loss = 0
    epoch_acc = 0
    epoch_auc = 0

    # Turn off dropout while evaluating
    model.eval()

    # No need to backprop in eval
    with torch.no_grad():
        for batch in tqdm(iterator, file=sys.stdout, desc="| |- Eval. batches"):
            x = batch['x'].to(device)
            y = batch['y'].to(device).squeeze(1)

            predictions = model(x).to(torch.float)
            loss = criterion(predictions, y)

            predictions = F.softmax(predictions, dim=-1)
            acc = accuracy(torch.argmax(predictions, dim=1), y)
            preds = predictions.detach().cpu().numpy()
            target = y.cpu().numpy()
            auc = roc_auc_score(target, preds, multi_class='ovo', average='macro', labels=labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_auc += auc

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_auc / len(iterator)


# Helper functions
def accuracy(preds, y):
    """ Return accuracy per batch. """
    correct = (preds == y).float()
    return correct.sum() / len(correct)


def epoch_time(start_time, end_time):
    '''Track training time. '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def learn_cnn(model_name, motifs_dataset, pretrained_embeddings, num_epochs, params_dir):
    # initialize pre-trained model (node embeddings mapped, ie ordered by index)
    embedding_len = pretrained_embeddings.size()[1]
    pretrained_embeddings = pretrained_embeddings.to(device)
    print(f'|- Embedding dimensions = {embedding_len:,}')
    sys.stdout.flush()

    # Number of filters
    n_filters = 100
    # "N-grams" that we want to analyze using filters
    filter_sizes = [1, 2, 3]

    # Num. classes
    output_dim = motifs_dataset.num_classes

    # CNN only
    model = MotifTypeCNN(pretrained_embeddings, embedding_len,
                         n_filters, filter_sizes,
                         output_dim).to(device)

    train_subset = Subset(motifs_dataset, motifs_dataset.train_mask)
    test_subset = Subset(motifs_dataset, motifs_dataset.test_mask)
    labels = np.array(motifs_dataset.classes)  # NOTE: should be sorted

    train_loader = helper_get_data_loader(train_subset, motifs_dataset.y[motifs_dataset.train_mask].tolist())
    test_loader = helper_get_data_loader(test_subset, motifs_dataset.y[motifs_dataset.test_mask].tolist())

    optimizer = Adam(model.parameters())
    optimizer.zero_grad()

    # Loss function
    criterion = CrossEntropyLoss()
    criterion = criterion.to(device)

    # Training loop
    best_valid_loss = float('inf')
    val_loss = []
    val_acc = []
    val_auc = []

    tr_loss = []
    tr_acc = []
    tr_auc = []

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1:>2}')
        sys.stdout.flush()

        # Calculate training time
        start_time = time.time()

        # Get epoch losses and accuracies
        train_loss, train_acc, train_auc = train(model, train_loader, optimizer, criterion, labels)
        valid_loss, valid_acc, valid_auc = test(model, test_loader, criterion, labels)
        sys.stdout.flush()

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Save training metrics
        val_loss.append(valid_loss)
        val_acc.append(valid_acc)
        val_auc.append(valid_auc)

        tr_loss.append(train_loss)
        tr_acc.append(train_acc)
        tr_auc.append(train_auc)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # torch.save(model.state_dict(), 'CNN-model.pt')
            fname = os.path.join(params_dir, f'best_params-transformer_model-version_{motifs_dataset.version}.pt')
            torch.save(model.state_dict(), fname)

        print(f'|- Epoch {epoch + 1:2} - Time: {epoch_mins}m {epoch_secs}s')
        print(f'| |- Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:<5.2f}% | Train AUC: {train_auc:.3f}')
        print(f'| |- Val. Loss:  {valid_loss:.3f} | Val.  Acc: {valid_acc * 100:<5.2f}% | Val.  AUC: {valid_auc:.3f}')
        sys.stdout.flush()

    results = dict(val_loss=val_loss, val_acc=val_acc, val_auc=val_auc,
                   tr_loss=tr_loss, tr_acc=tr_acc, tr_auc=tr_auc)

    return model, results


@torch.no_grad()
def inference_motif_types(model, x):
    # Turn off dropout
    model.eval()

    x = x.to(device)
    predictions = model(x).to(torch.float)
    predictions = F.softmax(predictions, dim=-1)

    return predictions


@torch.no_grad()
def batch_inference_motif_types(model, batch_iterator: DataLoader):
    # Turn off dropout
    model.eval()

    preds = []
    for batch in batch_iterator:
        batch = batch[0].to(device)
        predictions = model(batch).to(torch.float)
        predictions = F.softmax(predictions, dim=-1)
        preds.append(predictions)

    return torch.cat(preds)
