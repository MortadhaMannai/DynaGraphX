from typing import FrozenSet, Iterator, Set

import more_itertools
from scipy import stats
from scipy.special import comb

from DYANE.components.helpers.file_utils import *
from DYANE.components.helpers.utils import *

roles = ['equal3', 'hub', 'spoke', 'equal2', 'outlier']
distributions = {'Exponential': stats.expon,
                 'Gamma': stats.gamma,
                 'Logistic': stats.genlogistic,
                 'Normal': stats.norm,
                 'Log-Uniform': stats.loguniform,
                 'Beta': stats.beta
                 }


def get_nodes_data(nodes_csv: Union[str, os.PathLike]) -> pd.DataFrame:
    nodes_df = pd.read_csv(nodes_csv, sep='|',
                           header=0, names=['node', 'timestep'])
    nodes_df = nodes_df.sort_values(by=['node']).reset_index(drop=True)
    return nodes_df


def get_num_nodes(nodes_df: pd.DataFrame) -> int:
    return nodes_df['node'].nunique()


def get_active_nodes_t(nodes_df: pd.DataFrame,
                       t: int) -> List[int]:
    active_nodes_t = nodes_df[nodes_df['timestep'] <= t]['node'].tolist()
    return active_nodes_t


def get_edges_data(edges_csv: Union[str, os.PathLike]) -> pd.DataFrame:
    edges_df = read_dataframe(edges_csv)
    if 'weight' not in edges_df.columns:
        # get weighted edges
        edges_df = get_edge_counts(edges_df)
    # NOTE: removed any self-loops in data pre-processing
    num_before = edges_df.shape[0]
    edges_df = edges_df[edges_df['source'] != edges_df['target']]
    num_after = edges_df.shape[0]
    if num_before > num_after:
        logging.info(f'Removed loops in edges csv')
    return edges_df


def get_edge_counts(edges_df: pd.DataFrame) -> pd.DataFrame:
    edges_df = pd.DataFrame(edges_df.groupby(edges_df.columns.tolist(), as_index=False).size())
    edges_df.rename(columns={'size': 'weight'}, inplace=True)
    return edges_df


def get_timesteps_df(edges_df: pd.DataFrame):
    return sorted(edges_df['timestep'].unique())


def get_edges_t(edges_df: pd.DataFrame,
                t: int,
                unweighted: bool = False) -> pd.DataFrame:
    edges_t_df = edges_df[edges_df['timestep'] == t][['source', 'target', 'weight']]
    if unweighted:
        edges_t_df['weight'] = 1
    return edges_t_df


def get_node_adj(nodes_df: pd.DataFrame,
                 edges_df: pd.DataFrame,
                 timesteps: List[int]) -> Dict[int, Dict[int, Set[int]]]:
    # init graph adjacency "list"
    node_adj = {t: {a: set([])  # includes non-active nodes
                    for a in nodes_df['node'].unique()}
                for t in timesteps}

    for t in tqdm(timesteps, desc='Node adjacency'):
        # get edges at time t
        edges_t_df = get_edges_t(edges_df, t)  # duplicates already taken care of with groupby

        for i, e in edges_t_df.iterrows():
            source = e['source']
            target = e['target']
            if source != target:
                node_adj[t][source].add(target)
                node_adj[t][target].add(source)

    return node_adj


def get_motif_edges(edges_t_df: pd.DataFrame,
                    motif: FrozenSet[int]) -> pd.DataFrame:
    m_edges_df = edges_t_df[(edges_t_df['source'].isin(list(motif))) &
                            (edges_t_df['target'].isin(list(motif)))]
    return m_edges_df


def get_w_triangles(u: int,
                    v: int,
                    neighbors_u: Set[int],
                    neighbors_v: Set[int]) -> Set[int]:
    # https://docs.python.org/3/reference/expressions.html#operator-precedence
    # subtraction precedes Bitwise AND
    return (neighbors_u & neighbors_v) - {u, v}


def get_w_wedges(u: int,
                 v: int,
                 neighbors_u: Set[int],
                 neighbors_v: Set[int]) -> Set[int]:
    # https://docs.python.org/3/reference/expressions.html#operator-precedence
    # subtraction precedes Bitwise XOR
    return (neighbors_u ^ neighbors_v) - {u, v}


def get_w_1edges(active_nodes_t: Set[int],
                 neighbors_u: Set[int],
                 neighbors_v: Set[int]) -> Set[int]:
    return active_nodes_t - neighbors_u - neighbors_v


def get_num_triangles(u: int,
                      v: int,
                      neighbors_u: Set[int],
                      neighbors_v: Set[int]) -> int:
    return len(get_w_triangles(u, v, neighbors_u, neighbors_v))


def get_num_wedges(u: int,
                   v: int,
                   neighbors_u: Set[int],
                   neighbors_v: Set[int]) -> int:
    return len(get_w_wedges(u, v, neighbors_u, neighbors_v))


def get_num_1edges(active_nodes_t: Set[int],
                   neighbors_u: Set[int],
                   neighbors_v: Set[int]) -> int:
    return len(get_w_1edges(active_nodes_t, neighbors_u, neighbors_v))


def update_edge_count(cnt_rem: Union[int, float],
                      num_motif: int) -> Union[int, float]:
    return max([0, cnt_rem - num_motif])


def get_edge_count_motif(cnt_rem: Union[int, float],
                         num_motif: int) -> Union[int, float]:
    return min([cnt_rem, num_motif])


def get_triangle_motif_edge_weight(num_triangles: int,
                                   cnt_rem: Union[int, float]) -> Union[int, float]:
    return get_edge_count_motif(num_triangles, cnt_rem) / num_triangles


def get_wedge_motif_edge_weight(num_wedges: int, cnt_rem: Union[int, float], u: int, v: int,
                                neighbors_u: Set[int], neighbors_v: Set[int]) -> Union[int, float]:
    num_triangles = get_num_triangles(u, v, neighbors_u, neighbors_v)
    cnt_rem = update_edge_count(cnt_rem, num_triangles)
    return get_edge_count_motif(cnt_rem, num_wedges) / num_wedges


def get_1edge_motif_edge_weight(num_1edges: int, cnt_rem: Union[int, float], u: int, v: int,
                                neighbors_u: Set[int], neighbors_v: Set[int]) -> Union[int, float]:
    num_triangles = get_num_triangles(u, v, neighbors_u, neighbors_v)
    num_wedges = get_num_wedges(u, v, neighbors_u, neighbors_v)
    cnt_rem = update_edge_count(cnt_rem, num_triangles + num_wedges)
    return get_edge_count_motif(cnt_rem, num_1edges) / num_1edges


def get_hub_spoke(u: int, v: int, w: int, neighbors_u: Set[int], neighbors_v: Set[int]) -> (int, int):
    if w in neighbors_u:
        # hub, spoke
        return u, v
    elif w in neighbors_v:
        # hub, spoke
        return v, u


def get_roles_weight_motif(cnt_rem: Union[int, float], num_motif) -> Union[int, float]:
    # calc edge count
    edge_count_motif = get_edge_count_motif(cnt_rem, num_motif)
    # calculate weight
    weight_motif = edge_count_motif / num_motif
    return weight_motif


def calc_node_roles_distr(role_counts_df: pd.DataFrame) -> pd.DataFrame:
    df = role_counts_df.div(role_counts_df.sum(axis=1), axis=0)
    df = df.fillna(0)
    return df


def update_motifs_found(u: int, v: int, w_list: Union[Set[int], List[int]], num_edges: int, t: int,
                        motifs: Set[FrozenSet[int]],
                        motif_types: Dict[FrozenSet[int], int], motif_timesteps: Dict[FrozenSet[int], List[int]]) -> \
        Tuple[Set[FrozenSet[int]], Dict[FrozenSet[int], int], Dict[FrozenSet[int], List[int]]]:
    for w in w_list:
        m = frozenset({u, v, w})
        if m not in motifs:
            motifs.add(m)
            motif_types[m] = num_edges
            motif_timesteps[m] = [t]
        elif num_edges > motif_types[m]:
            motif_types[m] = num_edges
            motif_timesteps[m] = [t]
        elif num_edges == motif_types[m]:
            if t not in motif_timesteps[m]:
                motif_timesteps[m].append(t)
    return motifs, motif_types, motif_timesteps


def filter_w_1edges(u: int, v: int, w_1edges: Set[int],
                    motifs: Set[FrozenSet[int]], motif_types: Dict[FrozenSet[int], int]) -> List[int]:
    return [w for w in w_1edges
            if (frozenset({u, v, w}) not in motifs or
                motif_types[frozenset({u, v, w})] <= 1)]  # filter wedges, triangles


def get_some_empty_examples(motifs: Set[FrozenSet[int]],
                            nodes_df: pd.DataFrame,
                            num_samples: int) -> List[List[int]]:
    logging.info('Get some empty examples')

    nodes = nodes_df['node'].tolist()
    num_nodes = len(nodes)
    possible = int(comb(num_nodes, 3))

    sampled = set()
    it = 1
    tries = 0
    while len(sampled) < num_samples and tries < 10:
        indices = sample_indices(num=possible, sample_size=num_samples * 2)
        for idx in tqdm(indices, desc=f'Combinations (iter #{it})'):
            if len(sampled) < num_samples:
                example = more_itertools.nth_combination(nodes, 3, idx)
                m = frozenset(example)

                if m not in motifs and m not in sampled:
                    sampled.add(m)

        logging.info(f'len(sampled) = {len(sampled)}')
        it += 1
        tries += 1

    logging.info(f'Finished (sampled {len(sampled):,})')
    return [list(m) for m in sampled]


def subsample_motifs(motifs_i: List[List[int]], num_i: int, sample_size: int) -> npt.NDArray:
    motifs_i = np.array(motifs_i)
    if num_i > sample_size:
        sample = sample_indices(num_i, sample_size)
        return motifs_i[sample]
    else:
        return motifs_i


def subsample_1edge_motifs(motifs_1edge: Dict[FrozenSet[int], Set[int]], num_i: int, sample_size: int) -> npt.NDArray:
    logging.info(f"V1: Sample {sample_size:,} from {num_i:,} 1-edge motifs")
    sample = sample_indices(num_i, sample_size)
    return np.array(slice_1edge_motifs(motifs_1edge, sample))


def subsample_1edge_motifs_v2(motifs_i: List[FrozenSet[int]], num_i: int, sample_size: int,
                              params_data: Dict[str, Any]) -> npt.NDArray:
    logging.info(f"V2: Sample {sample_size:,} from {num_i:,} 1-edge motifs")
    sample = sample_indices(num_i, sample_size)
    return np.array(slice_1edge_motifs_v2(motifs_i, sample, params_data))


def slice_1edge_motifs(motifs_1edge: Dict[FrozenSet[int], Set[int]], indices: npt.NDArray[np.int_]) -> List[List[int]]:
    logging.info("V1: Create iterable chain")
    iterable = itertools.chain.from_iterable((prep_triplets_sorted(prep_uvw(motifs_1edge[u_v], u_v))
                                              for u_v in motifs_1edge))
    logging.info("V1: Slice indices")
    return slice_indices_1edge(indices, iterable)


def slice_1edge_motifs_v2(motifs_1edge: List[FrozenSet[int]], indices: npt.NDArray[np.int_],
                          params_data: Dict[str, Any]) -> List[List[int]]:
    logging.info("V2: Create iterable chain")
    iterable = itertools.chain.from_iterable((prep_triplets_sorted(prep_uvw_v2(u_v, params_data))
                                              for u_v in tqdm(motifs_1edge, desc="(u,v) edges")))
    return [list(m) for m in slice_indices_1edge(indices, iterable)
            if m is not None]


def make_triplet(x: Tuple[int, FrozenSet[int]]) -> FrozenSet[int]:
    w, (u, v) = x
    return frozenset({u, v, w})


def make_triplet_sorted(x: Tuple[int, FrozenSet[int]]) -> List[int]:
    w, (u, v) = x
    return sorted([u, v, w])


def prep_triplets_sorted(uvw_iterable: Iterator[Tuple[int, FrozenSet[int]]]) -> Iterator[List[int]]:
    return (make_triplet_sorted(uvw) for uvw in uvw_iterable)


def prep_uvw(w_list: Set[int], u_v: FrozenSet[int]) -> Iterator[Tuple[int, FrozenSet[int]]]:
    return zip(w_list, itertools.repeat(u_v))


def prep_uvw_v2(u_v: FrozenSet[int], params_data: Dict[str, Any]) -> Iterator[Tuple[int, FrozenSet[int]]]:
    u, v = sorted(list(u_v))
    w_list = msgpack_load(os.path.join(params_data['tmp_files_dir'], f"motifs_1edge--({u},{v}).msg"))
    return zip(w_list, itertools.repeat(u_v))


def slice_indices(indices: npt.NDArray[np.int_], iterable: Iterator[List[int]]) -> List[List[int]]:
    return [element for i, element in enumerate(iterable) if i in indices]


def slice_indices_1edge(indices: npt.NDArray[np.int_], iterable: Iterator[List[int]]) -> List[List[int]]:
    print()
    logging.info("V3: Slice indices")
    indices = transform_indices(indices)
    logging.info("V3: Get indices from generator")
    return [nth(iterable, idx) for idx in tqdm(indices)]


def fit_arrival_time_distribution(node_arrival_times: List[int]) -> Dict[str, Dict['str', Any]]:
    # distributions to learn
    arrivals_distribution = {d: {'parameters': None} for d in distributions.keys()}
    # fit each distribution
    for d in arrivals_distribution.keys():
        arrivals_distribution[d]['parameters'] = distributions[d].fit(node_arrival_times)

    return arrivals_distribution


def fit_rates_distributions(rates_motifs: Dict[int, List[float]],
                            m_types: List[int]) -> Dict[str, Dict[str, Any]]:
    """
    Learn motif types inter-arrival rates
    :param rates_motifs: rates for each motif type
    :param m_types: motif types
    :return:
    """
    # distributions to learn
    rates_distributions = {d: {'parameters': {i: None for i in m_types}} for d in distributions.keys()}
    # fit each distribution
    for d in rates_distributions.keys():
        logging.info(f'Fitting {d} distribution...')
        for i in m_types:
            rates_distributions[d]['parameters'][i] = distributions[d].fit(rates_motifs[i])
    return rates_distributions
