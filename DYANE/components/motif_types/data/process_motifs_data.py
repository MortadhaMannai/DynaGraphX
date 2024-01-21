import sys
from typing import Iterator, Sequence
from itertools import chain

import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

from DYANE.components.helpers.utils import *


# https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab
class MotifTypesDataset(Dataset):
    def __init__(self, root_dir, data_params, node_map, transform=None):
        self.root_dir = root_dir
        self.version = data_params['version']
        self.test_prop = data_params['test_prop']
        self.x, self.y, self.w, self.train_mask, self.test_mask = self.process(data_params, node_map)
        self.classes = sorted(torch.unique(self.y).tolist())
        self.num_classes = len(self.classes)

    @property
    def processed_file_names(self):
        # return "data_motifs.pt"
        return f"data_motifs--version_{self.version}.pt"

    def process(self, data_params, node_map):
        processed_data_file = os.path.join(self.root_dir, self.processed_file_names)

        if not os.path.exists(processed_data_file):
            logging.info(f"Processing dataset: motif types (version: {self.version})")
            sys.stdout.flush()

            # contains 3 nodes in motif, and the motif type
            logging.info("|- Read labels csv...")
            sys.stdout.flush()
            labeled_df = read_dataframe(data_params['labeled_data_csv'])

            # map node indexes for embeddings
            logging.info("|- Map node indexes...")
            sys.stdout.flush()
            triplets = labeled_df[['u', 'v', 'w']]

            logging.info(f'triplets.head(1):\n{triplets.head(1)}')
            logging.info(f'labeled_df columns = {triplets.columns}')

            # set x, y, w
            logging.info("|- Creating tensors...")
            sys.stdout.flush()
            x = torch.LongTensor(triplets.iloc[:, :].values)  # triplet
            y = torch.LongTensor(labeled_df.iloc[:, -2:-1].values)  # motif type
            w = torch.FloatTensor(labeled_df.iloc[:, -1].values)  # last column (weight)

            # Train/test splits
            logging.info("Creating masks for train/test split...")
            sys.stdout.flush()
            train_mask, test_mask, y_train, y_test = train_test_split(np.arange(len(y)),
                                                                      y,
                                                                      test_size=self.test_prop,
                                                                      stratify=y,
                                                                      random_state=RANDOM_SEED)  # refactor: random seed
            num_classes = len(torch.unique(y).tolist())
            assert len(np.unique(y_train)) == num_classes
            assert len(np.unique(y_test)) == num_classes

            train_mask = np.array(train_mask)
            test_mask = np.array(test_mask)

            # save processed file
            logging.info("|- Creating dictionary with masks...")
            sys.stdout.flush()
            torch.save(dict(x=x, y=y, w=w,
                            train_mask=train_mask,
                            test_mask=test_mask), processed_data_file)
        else:
            # print("Reading processed dataset: motif types")
            logging.info(f"Reading processed dataset: motif types (version: {self.version})")
            sys.stdout.flush()

            # load processed file
            data = torch.load(processed_data_file)

            # set x, y, w
            x = data['x']
            y = data['y']
            w = data['w']
            train_mask = data['train_mask']
            test_mask = data['test_mask']

        logging.info("|- Done!")
        sys.stdout.flush()

        return x, y, w, train_mask, test_mask

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return dict(x=self.x[idx], y=self.y[idx])


class NodeEmbeddings:
    def __init__(self, root_dir, data_params, transform=None):
        self.root_dir = root_dir
        self.embeddings, self.node_map = self.process(data_params)

    @property
    def processed_file_names(self):
        return ["data_embeddings.pt"]

    def process(self, data_params):
        processed_data_file = os.path.join(self.root_dir, self.processed_file_names[0])

        if not os.path.exists(processed_data_file):
            logging.info("Processing dataset: node embeddings")
            sys.stdout.flush()

            # read embedding dataframes - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
            emb_df = read_dataframe(data_params['node_embeddings_csv'])

            # embeddings - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
            embeddings = torch.from_numpy(emb_df.drop(columns=['node']).to_numpy())
            # save embeddings
            torch.save(embeddings, processed_data_file)

        else:
            logging.info("Reading processed dataset: node embeddings")
            sys.stdout.flush()

            # load processed file
            embeddings = torch.load(processed_data_file)

        node_map = read_node_map(data_params['node_map_file'])

        return embeddings, node_map


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_motif_types_dataset(pytorch_data_dir, data_params, num_cv_folds=1):
    node_data = NodeEmbeddings(pytorch_data_dir, data_params)
    # refactor: don't need node_map
    motif_dataset = MotifTypesDataset(pytorch_data_dir, data_params, node_data.node_map)
    return motif_dataset, node_data


def helper_get_batch_size_and_splits(subset_size: int,
                                     batch_size: int = 1_000,
                                     split_size: int = 5) -> Tuple[int, int]:
    batch_size = min([batch_size, int(subset_size / split_size)])
    num_splits = int(np.ceil(subset_size / batch_size))
    return batch_size, num_splits


def helper_get_data_loader(subset, y):
    subset_size = len(subset)
    batch_size, num_splits = helper_get_batch_size_and_splits(subset_size)
    sampler = StratifiedSampler(labels=y, num_splits=num_splits)
    loader = DataLoader(dataset=subset,
                        shuffle=False,
                        sampler=sampler,
                        batch_size=batch_size,
                        pin_memory=True)
    return loader


class StratifiedSampler(Sampler[int]):
    def __init__(self, labels: Sequence[int], num_splits: int) -> None:
        # super().__init__()
        self.num_splits = num_splits
        self.num_samples = len(labels)
        skf = StratifiedKFold(n_splits=self.num_splits,
                              random_state=RANDOM_SEED,  # refactor: random seed
                              shuffle=True)
        tmp = [test_idx for _, test_idx in skf.split(list(range(self.num_samples)), labels)]
        self.indices = np.array(list(chain(*tmp)))
        # logging.info(f'self.indices.shape = {self.indices.shape} <------------------------------')

    def __iter__(self) -> Iterator[int]:
        for i in range(self.num_samples):
            yield self.indices[i]

    def __len__(self) -> int:
        return self.num_samples
