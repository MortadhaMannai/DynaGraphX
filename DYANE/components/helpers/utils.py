import itertools
import math
import os
import time
from typing import Any, Iterable, List, Union, Tuple, Type
import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix, csgraph
from tqdm import tqdm


RANDOM_SEED = int(''.join(list(str(ord(x)) for x in 'YHLQMDLG')))
CHUNKS = 6


def make_df_node_map_index(nodes_df: pd.DataFrame,
                           col_id: str,
                           is_sorted: bool = False) -> pd.Index:
    """
    Returns pandas node map
    :param node_df: dataframe that uses original node ids
    :param col_id: name of column with original node id
    :param is_sorted: is dataframe already sorted by the col_id
    :return: node_map
    """
    # NOTE: already sorted when reading nodes_csv
    # if not is_sorted:
    #     df = node_df.sort_values(by=[col_id], ascending=True).reset_index(drop=True)
    #     # drop -> Do not try to insert index into dataframe columns.
    #     # (this resets the index to the default integer index)
    # else:
    #     df = node_df

    return pd.Index(nodes_df[col_id])


def df_apply_node_mapping(df: pd.DataFrame,
                          node_map: pd.Index,
                          cols: List[str]) -> pd.DataFrame:
    # ID -> (row) Index map
    return df[cols].applymap(lambda x: node_map.get_loc(x))


def df_unapply_node_mapping(df: pd.DataFrame,
                            node_map: pd.Index,
                            cols: List[str]) -> pd.DataFrame:
    # ID -> (row) Index map
    return df[cols].applymap(lambda x: node_map[x])


def read_node_map(node_map_file: Union[str, os.PathLike]) -> pd.Index:
    return pd.Index(pd.read_pickle(node_map_file, compression='gzip'))


def write_node_map(node_map: pd.Index, node_map_file: Union[str, os.PathLike]):
    node_map.to_series().to_pickle(node_map_file, compression='gzip')


def read_dataframe(csv: Union[str, os.PathLike],
                   index_col: Union[bool, int, str] = False,
                   sep: str = '|',
                   use_header: bool = True,
                   escapechar: Union[bool, str] = None):
    if use_header:
        df = pd.read_csv(csv, sep=sep, index_col=index_col, escapechar=escapechar)
    else:
        df = pd.read_csv(csv, sep=sep, index_col=index_col, header=None, escapechar=escapechar)

    if not index_col and type(index_col) != int:
        unknown_index = 'Unnamed: 0'
        if unknown_index in df.columns:
            df = df[df.columns.drop(unknown_index)]
    return df


def write_dataframe(df: pd.DataFrame,
                    csv: Union[str, os.PathLike],
                    index: Union[bool, int, str] = False,
                    header: Union[bool, List[str]] = True,
                    sep: str = '|',
                    escapechar: Union[bool, str] = None):
    # noinspection PyTypeChecker
    df.to_csv(csv, sep=sep, index=index, header=header, escapechar=escapechar)


# https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html
def get_sparse_adjacency_matrix(df: pd.DataFrame, num_nodes: int) -> csr_matrix:
    return get_graph_matrix(df, num_nodes)


def get_graph_matrix(edges_df: pd.DataFrame, num_nodes: int) -> csr_matrix:
    # create sparse matrix
    row = edges_df['source'].to_numpy()
    col = edges_df['target'].to_numpy()
    weight = edges_df['weight'].to_numpy()
    csr_adj_matrix = csr_matrix((weight, (row, col)), shape=(num_nodes, num_nodes))
    return csr_adj_matrix


def get_largest_connected_component_matrix(edges_df: pd.DataFrame, num_nodes: int):
    lcc_df = get_largest_connected_component_df(edges_df, num_nodes)
    return get_graph_matrix(lcc_df, num_nodes)


def get_largest_connected_component_df(edges_df: pd.DataFrame, num_nodes: int) -> pd.DataFrame:
    # create sparse matrix
    csr_adj_matrix = get_graph_matrix(edges_df, num_nodes)

    # find connected components
    n_components, component_list = csgraph.connected_components(csgraph=csr_adj_matrix,
                                                                directed=False,
                                                                return_labels=True)
    largest_component_id = np.argmax(np.bincount(component_list))

    # get nodes in LCC
    arr_nodes = np.arange(num_nodes)
    lcc_node = list(arr_nodes[np.nonzero(component_list == largest_component_id)])

    # create dataframe LCC
    lcc_df = edges_df[(edges_df['source'].isin(lcc_node)) &
                      (edges_df['target'].isin(lcc_node))]
    return lcc_df


def transform_indices(indices: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    # logging.info("V3: Transform indices")
    logging.info("Sort indices")
    indices = np.sort(indices, axis=0)
    # logging.info("Transform indices")
    num_indices = len(indices)
    prev = 0
    with np.nditer(indices, op_flags=['readwrite']) as it:
        # for x in it:
        for x in tqdm(it, total=num_indices, desc='Transform indices'):
            y = x + 1
            x[...] = x - prev
            prev = y
    return indices


def sample_indices(num: int, sample_size: int) -> npt.NDArray[np.int_]:
    logging.info(f'Sample {sample_size:,} indices from {num:,}')
    return np.random.default_rng(RANDOM_SEED).choice(num, sample_size, replace=False)


def nth(iterable: Iterable[Any], n: int, default: Any = None) -> Any:
    "Returns the nth item or a default value"
    return next(itertools.islice(iterable, n, n + 1), default)


def take(n: int, iterable: Iterable[Any]) -> List[Any]:
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))


def take_2d_arr(n: int, d: int, iterable: Iterable[Tuple[Any]], dtype: Type) -> npt.ArrayLike:
    return np.fromiter(yield_2d_elements(itertools.islice(iterable, n)),
                       dtype=dtype, count=n * d).reshape((-1, d))


def yield_2d_elements(iterable_2d: Iterable[Tuple[Any]]) -> Any:
    for row in iterable_2d:
        for elem in sorted(row):
            yield elem


def log_time():
    start_time = time.time()
    return lambda x: "{3}[{0}h {1}m {2:.1f}s]".format(math.floor((time.time() - start_time) // 60 // 60),
                                                      math.floor((time.time() - start_time) // 60 -
                                                                 ((time.time() - start_time) // 60 // 60) * 60),
                                                      (time.time() - start_time) % 60, x)
