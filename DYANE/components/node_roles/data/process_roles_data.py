import os
import sys

import numpy as np
import torch
from scipy.sparse import identity
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected
from torch_geometric.utils.convert import from_scipy_sparse_matrix


class NodeRolesDataLarge(Dataset):
    def __init__(self, root, data_params, num_folds, transform=None, pre_transform=None, pre_filter=None):
        self.roles = data_params['roles']
        self.adj_data = data_params['adj_data']
        self.timesteps = data_params['timesteps']
        self.num_timesteps = len(data_params['timesteps'])
        self.node_labels = data_params['node_labels']
        self.num_nodes = len(data_params['node_labels'])
        self.num_folds = num_folds
        self.gcn_version = data_params['gcn_version']
        self.node_labels_timesteps = data_params['node_labels_timesteps']

        # super().__init__(root, transform, pre_transform, pre_filter)
        super(NodeRolesDataLarge, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # NOTE: gcn version is in pytorch dir
        return [f"data_{idx}.pt" for idx in range(self.num_timesteps)]

    def download(self):
        pass

    def process(self):
        # Node features (identity matrix) - static
        x_feats = identity(self.num_nodes, dtype=int).tocoo()
        values = torch.FloatTensor(x_feats.data)
        indices = torch.tensor(np.vstack((x_feats.row, x_feats.col)))
        x = torch.sparse_coo_tensor(indices, values, torch.Size(x_feats.shape))

        # Node labels - static
        y_true = self.node_labels
        y = torch.FloatTensor(y_true)

        # Train/test split
        test_masks = []
        train_masks = []
        if self.num_folds > 1:
            print("Creating masks for k-fold splits...")
            sys.stdout.flush()
            kf = KFold(n_splits=self.num_folds, random_state=42, shuffle=True)
            for train_index, test_index in kf.split(y_true):
                test_masks.append(test_index)
                train_masks.append(train_index)
            print("|- Done!")
        else:
            print("Creating masks for train/test split...")
            sys.stdout.flush()
            X_train, X_test, y_train, y_test = train_test_split(list(range(self.num_nodes)), y_true,
                                                                test_size=0.10, random_state=42)
            train_masks.append(X_train)
            test_masks.append(X_test)

            print("|- Done!")
            sys.stdout.flush()

        print("Creating dictionary with masks...")
        sys.stdout.flush()
        extra_args = dict(train_masks=np.array(train_masks)
                          ,test_masks=np.array(test_masks)
                          )
        print("|- Done!")

        print("Creating edge list tensors per timestep...")
        sys.stdout.flush()
        for idx, t in enumerate(self.timesteps):
            print(f"|- Timestep t={t}")
            sys.stdout.flush()

            edge_index, edge_weight = from_scipy_sparse_matrix(self.adj_data[t])
            edge_index, edge_weight = to_undirected(edge_index, edge_attr=edge_weight, reduce="add")

            print(f"|  |- Creating Data object for t={t}...")
            sys.stdout.flush()
            data = Data(x=x,
                        edge_index=edge_index,
                        edge_attr=edge_weight,
                        y=y,
                        **extra_args)

            print(f"|  |- Torch save data & slices for t={t}...")
            sys.stdout.flush()
            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
        print("|- Done!")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data


def get_node_roles_dataset(pytorch_data_dir, data_params, num_cv_folds=1):
    my_dataset = NodeRolesDataLarge(pytorch_data_dir, data_params, num_cv_folds)
    return my_dataset
