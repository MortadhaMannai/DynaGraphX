import logging
from typing import Union, Dict, List

import igraph
import pandas as pd
from tqdm import tqdm

from DYANE.components.helpers.utils import get_graph_matrix, get_largest_connected_component_df
from DYANE.components.learning.learning_utils import get_edges_t

metric_names = ['density',
                'local clustering',
                'global clustering',
                'avg. path length',
                's-metric'
                ]


def get_structure_metrics(edges_df: pd.DataFrame,
                          num_nodes: int,
                          timesteps: List[int],
                          unweighted: bool = False) -> Dict[str, List[Union[int, float]]]:
    # NOTE: edges_df must have edge weights already
    logging.info("Get graph structure metrics")

    all_metrics = {m: [] for m in metric_names}
    for t in tqdm(timesteps, desc="Timesteps"):
        edges_t_df = get_edges_t(edges_df, t, unweighted=unweighted)

        if edges_t_df.shape[0] > 0:
            # get graph g_t
            g_t = get_graph_matrix(edges_t_df, num_nodes)

            if unweighted:
                g_t = igraph.Graph.Adjacency(g_t, mode='undirected')
            else:
                g_t = igraph.Graph.Weighted_Adjacency(g_t, mode='undirected')
            assert g_t.vcount() == num_nodes

            # calc. metrics
            all_metrics['density'].append(density(g_t))
            all_metrics['local clustering'].append(local_clustering_coeff(g_t))
            all_metrics['global clustering'].append(global_clustering_coeff(g_t))

            # get LCC g_t
            lcc_df = get_largest_connected_component_df(edges_t_df, num_nodes)
            # NOTE: edges df will be weighted or unweighted already
            lcc_g_t = get_graph_matrix(lcc_df, num_nodes)
            if unweighted:
                lcc_g_t = igraph.Graph.Adjacency(lcc_g_t, mode='undirected')
            else:
                lcc_g_t = igraph.Graph.Weighted_Adjacency(lcc_g_t, mode='undirected')

            all_metrics['avg. path length'].append(avg_path_length(lcc_g_t))
            all_metrics['s-metric'].append(s_metric(g_t))

    return all_metrics


def get_aggregate_structure_metrics(edges_df: pd.DataFrame,
                                    num_nodes: int,
                                    unweighted: bool = False):
    all_metrics = {m: None for m in metric_names}

    g = get_graph_matrix(edges_df, num_nodes)
    if unweighted:
        g = igraph.Graph.Adjacency(g, mode='undirected')
    else:
        g = igraph.Graph.Weighted_Adjacency(g, mode='undirected')
    assert g.vcount() == num_nodes

    all_metrics['density'] = density(g)
    all_metrics['local clustering'] = local_clustering_coeff(g)
    all_metrics['global clustering'] = global_clustering_coeff(g)
    all_metrics['s-metric'] = s_metric(g)

    # get LCC for average path length
    lcc_df = get_largest_connected_component_df(edges_df, num_nodes)
    # NOTE: edges df will be weighted or unweighted already
    lcc_g = get_graph_matrix(lcc_df, num_nodes)
    if unweighted:
        lcc_g = igraph.Graph.Adjacency(lcc_g, mode='undirected')
    else:
        lcc_g = igraph.Graph.Weighted_Adjacency(lcc_g, mode='undirected')

    all_metrics['avg. path length'] = avg_path_length(lcc_g)

    return all_metrics


def avg_path_length(g: igraph.Graph) -> Union[int, float]:
    return g.average_path_length(directed=False, unconn=True)


def density(g: igraph.Graph) -> Union[int, float]:
    return g.density()


def global_clustering_coeff(g: igraph.Graph) -> float:
    return g.transitivity_undirected()


def local_clustering_coeff(g: igraph.Graph) -> float:
    return g.transitivity_avglocal_undirected(mode='nan')


def s_metric(g: igraph.Graph) -> int:
    sum_total = 0
    for edge in g.es:
        i, j = edge.tuple
        sum_total += g.degree(i) * g.degree(j)
    return sum_total
