import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import GConvLSTM
from torch_geometric.loader import DataLoader
from DYANE.components.evaluation.metrics import eval_accuracy, eval_spearmans, eval_kl_div


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NodeRolesGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NodeRolesGCN, self).__init__()
        self.channels = [64, 64, 32, 16]  # moved here from global
        self.conv1 = GCNConv(in_channels=in_channels, out_channels=self.channels[0])
        self.conv2 = GCNConv(in_channels=self.channels[1], out_channels=self.channels[2])
        self.lstm = GConvLSTM(in_channels=self.channels[2], out_channels=self.channels[3], K=2)
        self.linear = Linear(in_features=self.channels[3], out_features=out_channels)
        self.h = None
        self.c = None

    def reset_h_c(self):
        self.h = None
        self.c = None

    def forward(self, x, edge_index, edge_attr):
        # (1)
        x = self.conv1(x, edge_index, edge_attr)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # (2)
        x = self.conv2(x, edge_index, edge_attr)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # (3)
        h, c = self.lstm(x, edge_index, edge_attr, self.h, self.c)
        self.h, self.c = h.detach().clone(), c.detach().clone()

        # (4)
        x = F.relu(h)
        x = self.linear(x)

        return x


def train_gcn(model, loss_name, loss_func, optimizer, dataset, train_mask, num_epochs):
    print("Training...")
    sys.stdout.flush()

    train_loader = DataLoader(dataset, batch_size=1)

    model.train()

    losses = []
    accs = []
    for epoch in range(num_epochs):
        print(f"|- Epoch {epoch:02d}")

        # reset state matrices for ea/epoch
        model.reset_h_c()

        epoch_losses = []
        epoch_accs = []
        for time, snapshot in enumerate(train_loader):
            print(f"|  |- Timestep t={time}")
            sys.stdout.flush()

            snapshot.edge_attr = snapshot.edge_attr.type(torch.float)
            snapshot = snapshot.to(device, 'x', 'y', 'edge_index', 'edge_attr')

            out = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

            y = snapshot.y[train_mask]
            out = out[train_mask]
            y_hat = F.softmax(out, dim=1)  # needed for BCELoss & acc

            # refactor: cross-entropy loss (no need for softmax before)
            if loss_name == 'categorical_loss':
                loss = loss_func(out, y)
            else:
                loss = loss_func(y_hat, y)

            loss.backward()
            epoch_losses.append(np.float(loss))

            optimizer.step()
            optimizer.zero_grad()

            tmp = (y_hat.argmax(dim=-1) == y.argmax(dim=-1))
            acc = int(tmp.sum()) / len(tmp)
            epoch_accs.append(acc)

            print(f'|  |  |  Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
            sys.stdout.flush()

        # save training loss
        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accs)

        print(f'|  |- AVERAGE Loss: {avg_loss:.4f}, Train: {avg_acc:.4f}')
        sys.stdout.flush()

        losses.append(epoch_losses)
        accs.append(epoch_accs)

    return losses, accs


@torch.no_grad()
def test_gcn(model, dataset, train_mask, test_mask, num_classes, h=None, c=None):
    print("Evaluating...")
    sys.stdout.flush()

    model.eval()

    loader = DataLoader(dataset, batch_size=1)

    accuracy = []
    spearmanr = []
    kl_div = []
    for time, snapshot in enumerate(loader):
        print(f"|  |- Timestep t={time}")
        sys.stdout.flush()

        # TODO: typecast
        snapshot.edge_attr = snapshot.edge_attr.type(torch.float)
        snapshot = snapshot.to(device, 'x', 'y', 'edge_index', 'edge_attr')

        out = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        y_hat = F.softmax(out, dim=1).detach()  # needed for BCELoss & acc
        y = snapshot.y.detach()

        accs = []
        ranks = []
        divs = []
        for mask in [train_mask, test_mask]:
            y_hat_masked = y_hat[mask]
            y_masked = y[mask]

            y_pred = F.one_hot(y_hat_masked.argmax(dim=-1), num_classes).cpu().numpy()
            y_true = F.one_hot(y_masked.argmax(dim=-1), num_classes).cpu().numpy()
            y_hat_masked = y_hat_masked.cpu().numpy()
            y_masked = y_masked.cpu().numpy()

            accs.append(eval_accuracy(y_true, y_pred))
            ranks.append(eval_spearmans(y_masked, y_hat_masked)[0])
            divs.append(eval_kl_div(y_masked, y_hat_masked))

        accuracy.append(accs)
        spearmanr.append(ranks)
        kl_div.append(divs)
        print(f'|  |  |  Accuracy: train={accs[0]:.4f}, test={accs[1]:.4f}')
        print(f'|  |  |  Ranking:  train={ranks[0]:.4f}, test={ranks[1]:.4f}')
        print(f'|  |  |  KL Div:   train={divs[0]:.4f}, test={divs[1]:.4f}')
        sys.stdout.flush()

    return accuracy, spearmanr, kl_div


@torch.no_grad()
def inference_gcn(model, dataset):
    print("Inference...")
    sys.stdout.flush()

    model.eval()

    loader = DataLoader(dataset, batch_size=1)
    for time, snapshot in enumerate(loader):
        print(f"|- Timestep t={time}")
        sys.stdout.flush()

        snapshot.edge_attr = snapshot.edge_attr.type(torch.float)
        snapshot = snapshot.to(device, 'x', 'edge_index', 'edge_attr')

        out = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

    embeddings = out
    return embeddings


def get_train_mask(num_nodes, data, fold=None):
    if fold > 1:
        train_mask = data.train_masks[0][fold]
    else:
        train_mask = data.train_masks[0]
    torch_mask = torch.zeros(num_nodes, dtype=torch.bool)
    torch_mask[train_mask] = True
    return torch_mask


def get_test_mask(num_nodes, data, fold=None):
    if fold > 1:
        test_mask = data.test_masks[0][fold]
    else:
        test_mask = data.test_masks[0]
    torch_mask = torch.zeros(num_nodes, dtype=torch.bool)
    torch_mask[test_mask] = True
    return torch_mask


def learn_gcn(my_dataset, loss_name, fold, percent_train_idx=0, num_epochs=10):
    model = NodeRolesGCN(my_dataset.num_node_features, len(my_dataset.roles)).to(device)
    print("|- Instantiated Node Roles GCN class")
    sys.stdout.flush()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # refactor: cross-entropy loss (check train_gcn)
    if loss_name == 'categorical_loss':
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        loss_func = torch.nn.BCELoss()

    num_nodes = my_dataset.num_nodes
    num_classes = len(my_dataset.roles)

    # get training and test set masks
    train_mask = get_train_mask(num_nodes, my_dataset[0], fold)
    test_mask = get_test_mask(num_nodes, my_dataset[0], fold)

    # train model
    train_loss, train_acc = train_gcn(model, loss_name, loss_func, optimizer, my_dataset, train_mask, num_epochs)

    # test model
    accuracy, spearmanr, kl_div = test_gcn(model, my_dataset, train_mask, test_mask, num_classes)

    results = {'training': {'loss': train_loss, 'accuracy': train_acc},
               'testing': {'accuracy': accuracy, 'ranking': spearmanr, 'KL div.': kl_div}}

    return model, results
