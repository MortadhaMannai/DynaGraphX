import logging
from typing import Union, Dict, List
import igraph
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from DYANE.components.helpers.utils import get_graph_matrix
from DYANE.components.learning.learning_utils import get_edges_t

node_metric_names = ['activity rate',
                     'local clustering',
                     'closeness',
                     'connected component size',
                     'degrees'
                     ]


def get_node_metrics(edges_df: pd.DataFrame,
                     nodes_df: pd.DataFrame,
                     num_nodes: int,
                     timesteps: List[int],
                     unweighted: bool = False) -> Dict[str, Dict[int,
                                                                 Union[Union[int, float],
                                                                       List[Union[int, float]]]]]:
    # NOTE: edges_df must have edge weights already
    logging.info("Get node behavior metrics")

    # refactor: all_nodes (num_nodes / nodes_df)
    # (nodes_df) node_map must be applied before calling get_node_metrics()
    all_nodes = list(range(num_nodes))
    # all_nodes_df = pd.DataFrame({'node': all_nodes})

    all_metrics = {m: {n: [] for n in all_nodes}
                   for m in node_metric_names}

    for t in tqdm(timesteps, desc="Timesteps"):
        edges_t_df = get_edges_t(edges_df, t, unweighted=unweighted)

        # ignore empty timestep 0
        if edges_t_df.shape[0] > 0:
            g_t = get_graph_matrix(edges_t_df, num_nodes)

            # get graph g_t
            if unweighted:
                g_t = igraph.Graph.Adjacency(g_t, mode='undirected')
            else:
                g_t = igraph.Graph.Weighted_Adjacency(g_t, mode='undirected')
            assert g_t.vcount() == num_nodes

            degrees_t = degree(g_t)
            local_clustering_t = local_clustering_coeff(g_t)
            conn_comp_size_t = connected_component_size(g_t, all_nodes)
            closeness_t = closeness_centrality(g_t)

            # save metrics
            for v in all_nodes:
                # Node closeness centrality
                all_metrics['closeness'][v].append(closeness_t[v])

                # Node connected component size
                if v in conn_comp_size_t:
                    all_metrics['connected component size'][v].append(conn_comp_size_t[v])
                else:
                    logging.warning(f'Node {v} not found in conn_comp_size_t for t={t}')

                # Node degrees
                all_metrics['degrees'][v].append(degrees_t[v])

                # Node local clustering coefficient
                all_metrics['local clustering'][v].append(local_clustering_t[v])

    all_metrics['activity rate'] = activity_rate(edges_df, all_nodes, len(timesteps))

    return all_metrics


def get_node_edges(edges_df: pd.DataFrame, node: int):
    return edges_df[(edges_df['source'] == node) | (edges_df['target'] == node)]


def activity_rate(edges_df: pd.DataFrame, nodes: npt.NDArray[np.int_], num_timesteps: int) -> Dict[int, float]:
    rates = {}
    for v in nodes:
        edges_v = get_node_edges(edges_df, v)
        rates[v] = edges_v['timestep'].nunique() / num_timesteps
    return rates


def closeness_centrality(g: igraph.Graph) -> List[float]:
    closeness = g.closeness()
    closeness = np.nan_to_num(closeness).tolist()  # replace NaNs with zeroes
    return closeness


def connected_component_size(g: igraph.Graph, nodes: npt.NDArray[np.int_]) -> Dict[int, int]:
    nodes_component_size_t = {v: 0 for v in nodes}
    components = g.components()
    for comp in components:
        size_component = len(comp)
        for v in comp:
            nodes_component_size_t[v] = size_component
    return nodes_component_size_t


def degree(g: igraph.Graph) -> List[int]:
    return g.degree(loops=False)


def local_clustering_coeff(g: igraph.Graph) -> List[float]:
    return g.transitivity_local_undirected(mode='zero')
