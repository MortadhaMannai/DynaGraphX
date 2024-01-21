import random
import sys
from itertools import chain
from itertools import combinations
from itertools import product
from typing import Set, FrozenSet

import dask
import more_itertools
import scipy.stats
import torch
from scipy.special import comb
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from tqdm.dask import TqdmCallback

from DYANE.components.helpers.file_utils import *
from DYANE.components.helpers.utils import *
from DYANE.components.learning.learning_utils import roles, distributions
from DYANE.components.motif_types.cnn import batch_inference_motif_types


# REFACTOR: get rid of unused functions

def read_param(file: Union[str, os.PathLike]) -> Union[Any, pd.DataFrame]:
    if file.endswith('.msg'):
        return msgpack_load(file)
    elif file.endswith('.csv'):
        # TODO: do I need indexed columns for any of the params?
        # return pd.read_csv(file, sep='|', index_col='node')
        return read_dataframe(file)


def get_gen_dir_t(gen_dir: Union[str, os.PathLike], t: int) -> Union[str, os.PathLike]:
    gen_dir_t = os.path.join(gen_dir, f't_{t + 1}')
    check_dir_exists(gen_dir_t)
    return gen_dir_t


def get_gen_topic_dir_t(gen_dir: Union[str, os.PathLike],
                        params_data: Dict[str, Any]) -> Union[str, os.PathLike]:
    content_t_dir = os.path.join(gen_dir, params_data['content_filename'])
    check_dir_exists(content_t_dir)
    return content_t_dir


def update_role_counts(role_counts_df: pd.DataFrame,
                       nodes: Union[int, List[int]],
                       role: str,
                       motif_timesteps: List[int]) -> pd.DataFrame:
    """
    Update node role counts

    :param role_counts_df: node role counts
    :param nodes: nodes to update counts for
    :param role: role to update count
    :param motif_timesteps: timesteps motif appears in
    """
    # NOTE: `.loc` updates by ref
    role_counts_df.loc[nodes, role] -= len(motif_timesteps)
    role_counts_df = role_counts_df.fillna(0)
    # check for negative counts, set them to zero
    role_counts_df[(role_counts_df < 0)] = 0
    return role_counts_df


def update_role_distr_df(role_counts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Update node role probabilities

    :param role_counts_df: node role counts
    :return: updated node role probabilities
    """
    df = role_counts_df.div(role_counts_df.sum(axis=1), axis=0)
    df = df.fillna(0)
    df[(df < 0)] = 0
    return df


def update_roles_assigned(roles_assigned_df: pd.DataFrame,
                          nodes: Union[int, List[int]],
                          role: str):
    # NOTE: `.loc` updates by ref
    roles_assigned_df.loc[nodes, role] += 1


def save_role_counts_t(role_counts_df: pd.DataFrame, role_counts_file: Union[str, os.PathLike]):
    # save a backup for last timestep generated
    write_dataframe(role_counts_df, role_counts_file)


def read_role_counts_t(role_counts_file: Union[str, os.PathLike]) -> pd.DataFrame:
    # read a backup for last timestep generated
    return read_dataframe(role_counts_file)


def init_roles_assigned_counts(num_nodes: int) -> pd.DataFrame:
    roles_assigned_counts_df = pd.DataFrame({'node': list(range(num_nodes))})
    roles_assigned_counts_df = roles_assigned_counts_df.assign(**{r: 0 for r in roles})
    return roles_assigned_counts_df


def save_roles_assigned_counts(roles_assigned_counts_df: pd.DataFrame, gen_dir: Union[str, os.PathLike]):
    roles_assigned_counts_file = os.path.join(gen_dir, f'roles_assigned_counts.csv')
    write_dataframe(roles_assigned_counts_df, roles_assigned_counts_file)


def helper_sample_roles(role_distr_df: pd.DataFrame, role: str, nodes: Union[int, List[int]]):
    p = role_distr_df.loc[nodes, role].to_numpy()
    # TODO: RANDOM_SEED
    if any(p):
        role_node = np.random.default_rng().choice(nodes, p=p / sum(p))
    else:
        role_node = np.random.default_rng().choice(nodes)
    return role_node, [v for v in nodes if v != role_node]


def helper_get_node_with_role(roles_motifs: Dict[FrozenSet[int], Dict[int, str]],
                              role: str,
                              motif: FrozenSet[int]) -> Tuple[List[int], List[int]]:
    role_node = [node for node in motif if roles_motifs[motif][node] == role]
    return role_node, [v for v in motif if v != role_node]


def get_node_avg_content_embeddings(params_data: Dict[str, Any],
                                    node_map: pd.Index,
                                    nodes: List[int] = None) -> pd.DataFrame:
    averages_df = read_dataframe(params_data['node_avg_content_embeddings_csv'])
    averages_df['node'] = df_apply_node_mapping(averages_df, node_map, ['node'])
    if nodes:
        averages_df = helper_filter_node_embeddings(averages_df, nodes)
    return averages_df


def get_node_var_content_embeddings(params_data: Dict[str, Any],
                                    node_map: pd.Index,
                                    nodes: List[int] = None) -> pd.DataFrame:
    variance_df = read_dataframe(params_data['node_var_content_embeddings_csv'])
    variance_df['node'] = df_apply_node_mapping(variance_df, node_map, ['node'])
    if nodes:
        variance_df = helper_filter_node_embeddings(variance_df, nodes)
    return variance_df


def helper_get_node_roles_distr(params_data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    role_distr_df = read_dataframe(params_data['node_role_distr_csv'])
    role_counts_df = read_dataframe(params_data['node_role_counts_csv'])
    return role_distr_df, role_counts_df


def helper_get_roles_meta_model(model_path: Union[str, os.PathLike]) -> Union[RidgeCV, KernelRidge]:
    return pickle_load(model_path)


def helper_get_motif_types_model(model_path: Union[str, os.PathLike]) -> Type[torch.nn.Module]:
    cnn_model = torch.load(model_path)
    cnn_model.eval()
    return cnn_model


def helper_filter_node_embeddings(node_embeddings_df: pd.DataFrame,
                                  subset_nodes: List[int]) -> pd.DataFrame:
    return node_embeddings_df[node_embeddings_df['node'].isin(subset_nodes)]


def helper_get_node_influencers(motif: FrozenSet[int],
                                roles_motif: Dict[int, str]) -> List[int]:
    nodes = []
    for node in motif:
        if roles_motif[node] != 'outlier':
            # outlier node should not influence
            nodes.append(node)
            if roles_motif[node] == 'hub':
                # hub node should have double chance to influence than spokes
                nodes.append(node)
    return nodes


def helper_calc_motif_type_probs(motif_tensors: torch.LongTensor,
                                 motifs_cnn: Type[torch.nn.Module]) -> torch.LongTensor:
    # logging.info('Create DataLoader for motif tensors')
    batch_loader = DataLoader(dataset=TensorDataset(motif_tensors),
                              shuffle=False, batch_size=10_000)
    # estimate probabilities
    p = batch_inference_motif_types(motifs_cnn, batch_loader)

    return p.detach().cpu()


def write_probabilities_dataframe(motif_probabilities: npt.NDArray,
                                  probs_csv: Union[str, os.PathLike]):
    probs_df = pd.DataFrame(motif_probabilities, columns=['u', 'v', 'w',
                                                          'P(empty)',
                                                          'P(1-edge)',
                                                          'P(wedge)',
                                                          'P(triangle)'])
    write_dataframe(probs_df, probs_csv)


def arr_set_diff(a: npt.NDArray,
                 b: npt.NDArray) -> npt.NDArray:
    """Return items in a that are not in b."""
    if a.shape > (1, 0) and b.shape > (1, 0):
        mask = (a[:, None] == b)
        if isinstance(mask, np.ndarray):
            return a[~mask.all(-1).any(-1)]
        else:
            return a
    else:
        return a


def arr_concat(a: npt.NDArray,
               b: npt.NDArray) -> npt.NDArray:
    """Concatenate arrays a and b."""
    if a.shape > (1, 0) and b.shape > (1, 0):
        return np.vstack((a, b))
    elif b.shape > (1, 0):
        return b
    else:
        return a


def arr_sort_desc(arr: npt.NDArray, idx: int) -> npt.NDArray:
    """Sort array in descending order by index given."""
    return arr[arr[:, idx].argsort()][::-1]


def take_top(a: npt.NDArray,
             b: npt.NDArray,
             n: int,
             idx: int) -> npt.NDArray:
    """Concatenate a and b into c. Take (upto) top n rows."""
    # concatenate arrays
    c = arr_concat(a, b)
    # sort by idx
    c = arr_sort_desc(c, idx)
    # return top
    top_n = min([n, len(c)])
    return c[:top_n]


def tensor_concat(a: Union[torch.Tensor, torch.LongTensor, torch.FloatTensor],
                  b: Union[torch.Tensor, torch.LongTensor, torch.FloatTensor],
                  dim=0):
    """Concatenate tensors a and b."""
    return torch.cat([a, b], dim=dim)


def tensor_sort_desc(tensor: Union[torch.Tensor, torch.LongTensor, torch.FloatTensor],
                     idx: int):
    """Return tensor sorted in descending order by index given."""
    return tensor[:, idx].argsort(dim=0, descending=True)


def tensor_argsort_desc(tensor: Union[torch.Tensor, torch.LongTensor, torch.FloatTensor],
                        idx: int):
    """Return indices of tensor sorted in descending order by index given."""
    return tensor[:, idx].argsort(dim=0, descending=True)  # indices


def helper_sample_motifs_batches_dyane(motifs_cnn: Type[torch.nn.Module],
                                       triplets: Iterable[Tuple[int]],
                                       num_triplets: int,
                                       num_appear_types: Dict[int, int],
                                       total_expected: Dict[int, int],
                                       m_types: List[int],
                                       use_stash: bool,
                                       ) -> Tuple[Set[FrozenSet[int]], Dict[FrozenSet[int], int], Dict[int, int], List[List[int]]]:
    logging.info(f'V10: Batch probabilities, sample from all probs.')
    motif_tensors = []
    motif_probs = []

    total_appear = sum(num_appear_types.values())
    batch_size = max([total_appear, min([num_triplets, 100_000])])
    remaining = num_triplets

    # keep track of current batch
    batch_num = 0
    with tqdm(total=num_triplets, file=sys.stdout, desc='Triplets') as pbar:
        while remaining > 0:
            # update progress
            progress = min([batch_size, remaining])
            # pbar.update(progress)
            pbar.refresh()
            remaining -= progress
            batch_num += 1

            # make batch tensors
            triplet_batch = take_2d_arr(n=progress, d=3, iterable=triplets, dtype=int)
            triplet_batch = torch.LongTensor(triplet_batch)
            motif_tensors.append(triplet_batch)

            # estimate probabilities for motif types
            p = helper_calc_motif_type_probs(triplet_batch, motifs_cnn)
            # append probabilities
            motif_probs.append(p)

    logging.info('Finished probabilities')

    # concatenate list of tensors
    motif_probs = torch.cat(motif_probs, dim=0)
    motif_tensors = torch.cat(motif_tensors, dim=0)

    # init buckets
    buckets = {i: np.array([]) for i in m_types}
    stashed = {i: np.array([]) for i in m_types}
    num_sampled = {i: 0 for i in m_types}
    logging.info('Sample from Multinomial')
    max_size_sampling = (2 ** 24) - 1
    for i in m_types:
        idx = i
        logging.info(f'i={i}: Sample {num_appear_types[i]:,} motifs '
                     f'(from {torch.count_nonzero(motif_probs[:, idx]):,} non-zero)')
        if num_appear_types[i] > 0:
            # sample motifs with estimated probabilities
            # num. triplets available
            num_avail = min([motif_tensors.size(dim=0), num_triplets])
            if num_avail > max_size_sampling:
                logging.info(f"Creating reservoir for i={i}")

                # initialize reservoir
                num_reservoir_chunks = int(np.ceil(num_avail / max_size_sampling))
                reservoir_indices = torch.LongTensor(range(0, max_size_sampling))

                start_left = max_size_sampling
                end_left = min([start_left + max_size_sampling, num_avail])

                for chunk_idx in range(num_reservoir_chunks - 1):
                    logging.info(f"> Chunk #{chunk_idx}")
                    leftover_indices = torch.LongTensor(range(start_left, end_left))

                    # sample num. triplets that will make it to reservoir from Binomial distr.
                    num_left = leftover_indices.size(dim=0)
                    num_repl = scipy.stats.binom.rvs(n=num_left, p=0.5, size=1)[0]

                    logging.info(f">> {num_avail = :>10,}")
                    logging.info(f">> {num_left  = :>10,}")
                    logging.info(f">> {num_repl  = :>10,}")

                    num_keep = max_size_sampling - num_repl  # num. currently in reservoir - num. to add to reservoir
                    logging.info(f">> {num_keep  = :>10,}")

                    # sample which triplets will be added to reservoir
                    leftover_indices = sample_multinomial_tensor(leftover_indices, dim=0, num_samples=num_repl)

                    # sample which triplets will stay in reservoir
                    reservoir_indices = sample_multinomial_tensor(reservoir_indices, dim=0,
                                                                  num_samples=num_keep)

                    # update reservoir
                    # NOTE: these indices map to the motif_tensors indices
                    reservoir_indices = torch.cat((reservoir_indices, leftover_indices), dim=0)

                    # go to next reservoir chunk
                    start_left = end_left
                    end_left = min([start_left + max_size_sampling, num_avail])

                reservoir_probs = torch.index_select(input=motif_probs, dim=0, index=reservoir_indices)

                # sample the motifs from reservoir
                motif_indices = sample_multinomial_tensor(reservoir_indices, dim=0,
                                                          num_samples=num_appear_types[i],
                                                          probs=reservoir_probs[:, idx])
            else:
                # sample the motifs
                motif_indices = sample_multinomial_tensor_indices(motif_tensors, dim=0,
                                                                  num_samples=num_appear_types[i],
                                                                  probs=motif_probs[:, idx])

            motifs_i = torch.index_select(input=motif_tensors, dim=0, index=motif_indices)
            buckets[i] = motifs_i.numpy()
            num_sampled[i] = len(buckets[i])

            # OPTIMIZE: for larger datasets
            # get rid of motifs already sampled
            motif_tensors = tensor_remove(motif_tensors, motif_indices, dim=0)
            motif_probs = tensor_remove(motif_probs, motif_indices, dim=0)

    num_sampled = {i: len(buckets[i]) for i in m_types}
    sys.stdout.flush()
    logging.info('Motifs sampled:')
    for i in m_types:
        print(f'\t\t\ttype {i} = {num_sampled[i]:>15,}')
    sys.stdout.flush()

    # build stash after sampling
    if use_stash and len(motif_probs) > 0:
        for i in m_types:
            # num. to keep top as stash
            stash_size = min([total_expected[i] - num_sampled[i], len(motif_probs)])
            logging.info(f'i={i}: Stashing {stash_size:,} motifs...')
            if stash_size > 0 and len(motif_probs) > 0:
                # sort by probability
                stash_indices = tensor_argsort_desc(motif_probs, i)[:stash_size]
                # update stashed triplets (keep top probs. i, unsampled triplets)
                stashed[i] = torch.index_select(input=motif_tensors, dim=0, index=stash_indices).numpy()
                # keep unstashed triplets for next i
                if i > 1:  # update tensors
                    motif_tensors = tensor_remove(motif_tensors, stash_indices, dim=0)
                    motif_probs = tensor_remove(motif_probs, stash_indices, dim=0)
        # don't need tensors anymore
        del motif_tensors
        del motif_probs

        # no need to slice stashes (only has nodes motif, no probs)
        tmp = [stashed[i] for i in m_types if len(stashed[i]) > 0]
        if len(tmp) > 1:
            stashed = np.concatenate(tmp).tolist()
        elif len(tmp) == 1:
            stashed = tmp[0]
        else:
            stashed = []
        del tmp
        # noinspection PyTypeChecker
        logging.info(f'Num. triplets stashed = {len(stashed):,}')

    motifs = list(np.concatenate([buckets[i] for i in m_types if num_sampled[i] > 0]))

    # logging.info('Save motifs sampled')
    motifs_sampled = set([])
    motif_types_sampled = {}
    for i in m_types:
        for idx in range(num_sampled[i]):
            motif = frozenset(motifs.pop(0))  # NOTE: need to pop position 0 (default is -1)
            motifs_sampled.add(motif)
            motif_types_sampled[motif] = i

    # noinspection PyTypeChecker
    return motifs_sampled, motif_types_sampled, num_sampled, stashed


def helper_get_all_nums_active_triplets(num_new_nodes: int, num_old_nodes: int) -> Tuple[int, int, int]:
    num_3_new = int(comb(num_new_nodes, 3, repetition=False))
    if num_old_nodes > 0:
        num_new2_old1 = int(num_new_nodes * ((num_new_nodes - 1) / 2) * num_old_nodes)
        num_new1_old2 = int(num_old_nodes * ((num_old_nodes - 1) / 2) * num_new_nodes)
        return num_3_new, num_new2_old1, num_new1_old2
    else:
        return num_3_new, 0, 0


def helper_get_active_triplets(new_nodes: List[int], old_nodes: List[int]) -> Iterable[Tuple[int]]:
    # (1) 3 new nodes
    # -------------------------------------------------------*
    triplets = combinations(new_nodes, 3)

    if len(old_nodes) > 0:
        # (2) 2 new + 1 old
        # -------------------------------------------------------*
        new2_old1 = ((u, v, w)
                     for ((u, v), w) in product(combinations(new_nodes, 2), old_nodes)
                     if w not in [u, v])

        triplets = chain(triplets, new2_old1)

        # (3) 1 new + 2 old
        # -------------------------------------------------------*
        new1_old2 = ((u, v, w)
                     for ((u, v), w) in product(combinations(old_nodes, 2), new_nodes)
                     if w not in [u, v])

        triplets = chain(triplets, new1_old2)
    return triplets


def get_triplet_batch(new_nodes, old_nodes, indices):
    num_3_new, num_new2_old1, num_new1_old2 = helper_get_all_nums_active_triplets(len(new_nodes), len(old_nodes))
    lim_1 = num_3_new
    lim_2 = lim_1 + num_new2_old1
    lim_3 = lim_2 + num_new1_old2

    triplet_batch = []
    # for idx in tqdm(indices, desc='triplet_batch'):
    for idx in indices:
        if idx < lim_1:
            triplet_batch.append(more_itertools.nth_combination(new_nodes, 3, idx))
        elif idx < lim_2:  # FIXME: nth_prod repeats 2 nodes
            idx = idx - lim_1
            uvw = more_itertools.nth_product(idx, new_nodes, new_nodes, old_nodes)
            if len(set(uvw)) == 3:
                triplet_batch.append(uvw)
        elif idx < lim_3:  # FIXME: nth_prod repeats 2 nodes
            idx = idx - lim_2
            uvw = more_itertools.nth_product(idx, new_nodes, old_nodes, old_nodes)
            if len(set(uvw)) == 3:
                triplet_batch.append(uvw)
    return triplet_batch


def helper_gen_triplet_nth(new_nodes: List[int], old_nodes: List[int], indices: npt.NDArray[np.int_]):
    triplets = helper_get_active_triplets(new_nodes, old_nodes)
    triplet_batch = []
    for idx in indices:
        triplet_batch.append(nth(triplets, idx))
    return triplet_batch


def helper_sample_motif_timesteps(num_timesteps: int,
                                  exp_counts: Dict[int, int],
                                  m_types: List[int],
                                  rates_distribution: Dict[str, Dict[str, Any]],
                                  t: int,
                                  delta: int,
                                  gen_dir_t: Union[str, os.PathLike],
                                  params_data: Dict[str, Any]
                                  ) -> Tuple[Dict[int, List[int]], Dict[int, List[float]], Dict[int, int]]:
    logging.info(f'Sample timesteps new motifs t={t + 1}')

    tmp_timesteps_data_file = os.path.join(gen_dir_t, f'tmp_timesteps_data--t_{t + 1}.msg')
    if not file_exists(tmp_timesteps_data_file):
        # Initialization
        motif_type_timesteps = {i: [] for i in m_types}
        motif_sampled_rates = {i: [] for i in m_types}
        motif_adjusted_rates = {i: [] for i in m_types}
        inter_arrivals_diffs = {i: [] for i in m_types}
        num_appear_types = {i: 0 for i in m_types}
        num_timesteps_remaining = num_timesteps - t

        # Get distribution to sample rates from
        rates_dname = params_data['motif_rates_distr']

        # Get distribution to sample inter-arrivals from
        time_dname = params_data['motif_times_distr']
        use_binomial = params_data['use_binomial']
        # resample_rates = params_data['motif_rates_resampling']

        for i in m_types:  # motif type i
            if exp_counts[i] > 0:  # check expected count
                num_sample_i = exp_counts[i]  # init num. to sample with expected count

                # if resample_rates:
                #     # try a max num. of times to resample rates for motifs that didn't appear
                #     num_tries_i = params_data['num_resampling_tries']
                # else:
                num_tries_i = 1

                while num_sample_i > 0 and num_tries_i > 0:

                    # sample inter-arrival rates for all expected motifs
                    dist_params = distributions[rates_dname].rvs(*rates_distribution[rates_dname]['parameters'][i],
                                                                 size=num_sample_i,
                                                                 # random_state=params_data['random_seed']))
                                                                 random_state=None)
                    if time_dname == 'Exponential':
                        dist_params = np.absolute(dist_params)

                    for p in tqdm(dist_params, desc=f'Motifs i={i}'):
                        if time_dname == 'Exponential':
                            # p is the lambda param here
                            timesteps_sampled = helper_sample_exponential(p, num_timesteps, num_timesteps_remaining,
                                                                          t, delta)
                            # FEATURE: track how many times Binomial stops inter-arrivals
                            r_diff = np.NaN
                            p_b = np.NaN  # OPTIMIZE: set to None and add an if before saving?
                        else:
                            # if rates_dname == 'Beta':  # refactor: this would be redundant
                            alpha = rates_distribution[rates_dname]['parameters'][i][0]
                            beta = rates_distribution[rates_dname]['parameters'][i][1]

                            if t >= delta:
                                n = num_timesteps_remaining  # (i.e., num_timesteps - t)
                                t_n = t - delta
                            else:
                                n = num_timesteps - delta
                                t_n = 0

                            x = p * n
                            p_b = (x + alpha) / (n + alpha + beta)  # posterior mean estimator (p MLE)

                            # use geometric
                            timesteps_sampled, r_diff = helper_sample_geometric(p_b, num_timesteps, n, t_n, delta,
                                                                                use_binomial)

                        # Check motif appears
                        if len(timesteps_sampled) > 0:
                            num_appear_types[i] += 1
                            motif_type_timesteps[i].append(list(timesteps_sampled))
                            motif_sampled_rates[i].append(p)
                            motif_adjusted_rates[i].append(p_b)
                            # FEATURE: track how many times Binomial stops inter-arrivals
                            inter_arrivals_diffs[i].append(r_diff)

                    # update num. to sample with num. motifs that didn't appear
                    num_sample_i = exp_counts[i] - num_appear_types[i]
                    num_tries_i -= 1  # update num. times to try again
            else:
                logging.info(f'Motifs i={i}: expected count is 0')

        # save data for timesteps sampled
        tmp_timesteps_data = {'motif_type_timesteps': motif_type_timesteps,
                              'inter_arrivals_diffs': inter_arrivals_diffs,
                              'motif_sampled_rates': motif_sampled_rates,
                              'motif_adjusted_rates': motif_adjusted_rates,
                              'num_appear_types': num_appear_types}
        msgpack_save(tmp_timesteps_data_file, tmp_timesteps_data)
    else:
        tmp_timesteps_data = msgpack_load(tmp_timesteps_data_file)
        motif_type_timesteps = tmp_timesteps_data['motif_type_timesteps']
        motif_sampled_rates = tmp_timesteps_data['motif_sampled_rates']
        num_appear_types = tmp_timesteps_data['num_appear_types']

    # FIXME:Num. motifs appear sometimes higher than num. expected for that motif type
    logging.info('Motifs appear:')
    for i in m_types:
        print(f'\t\t\ttype {i} = {num_appear_types[i]:>15,}')
    sys.stdout.flush()

    return motif_type_timesteps, motif_sampled_rates, num_appear_types


def helper_sample_exponential(l: float,
                              num_timesteps: int,
                              num_timesteps_remaining: int,
                              t: int,
                              delta: int) -> Set[int]:
    # initialize timesteps for motif j
    timesteps_sampled = set([])

    # num. inter-arrival times to sample
    n = num_timesteps_remaining

    # sample inter-arrival times from exponential
    inter_arrivals = np.random.exponential(1 / l, size=n)

    # NOTE: only want to keep from delta onwards
    curr_time = t  # first timestep motif could appear
    for next_time in inter_arrivals:  # next inter-arrival time
        x = int(np.ceil(curr_time + next_time))
        y = x - 1
        while curr_time < num_timesteps:
            # next timestep it will show in
            timestep = curr_time + y
            if delta <= timestep < num_timesteps:
                timesteps_sampled.add(timestep)  # save timestep
            # update current timestep
            curr_time = curr_time + x

    return timesteps_sampled


def helper_sample_geometric(p: float,
                            num_timesteps: int,
                            n: int,
                            t_n: int,
                            delta: int,
                            use_binomial: bool = True) -> Tuple[Set[int], int]:
    # initialize timesteps for motif j
    timesteps_sampled = set([])

    if p < 0:
        p = abs(p)

    if use_binomial:
        # sample num. times the motif will appear (r) from Binomial
        r = scipy.stats.binom.rvs(n=n, p=p, size=1)[0]
    else:
        r = n

    # FEATURE: track how many times Binomial stops inter-arrivals
    q = n
    inter_arrivals = scipy.stats.geom.rvs(p=p, size=q)
    cnt = 0

    if r > 0:
        # FEATURE: track how many times Binomial stops inter-arrivals
        # # sample r inter-arrival times from Geometric
        # inter_arrivals = scipy.stats.geom.rvs(p=p, size=r)

        # NOTE: taking care of delta with t_n in func. call
        curr_time = t_n + delta  # first timestep motif could appear
        for x in inter_arrivals:  # num. trials needed for success, x in {1,2,3,...}
            y = x - 1  # num. failures for next success, y in {0,1,2,...}
            if curr_time < num_timesteps:
                # next timestep it will show in
                timestep = curr_time + y
                # if delta <= timestep < num_timesteps:
                if timestep < num_timesteps:
                    # FEATURE: track how many times Binomial stops inter-arrivals
                    if cnt <= r:
                        timesteps_sampled.add(timestep)  # save timestep
                    cnt += 1
                # update current timestep
                curr_time = curr_time + x

    if use_binomial:
        r_diff = cnt - r
    else:
        r_diff = np.NaN

    return timesteps_sampled, r_diff


def take_from_tensor(tensor: Union[torch.Tensor, torch.LongTensor],
                     tensor_filter: Union[torch.Tensor, torch.LongTensor],
                     mask_filter: int,
                     n: int
                     ) -> Tuple[Union[torch.Tensor, torch.LongTensor], int]:
    tensor_mask = (tensor_filter == mask_filter)
    total = int(tensor_mask.sum())
    if total > 0:
        # get indices of match
        indices = tensor_mask.nonzero().squeeze()
        if total > n:
            # subsample indices
            indices = random_choice_tensor(indices, dim=0, num_samples=n)
        tensor = torch.index_select(tensor, dim=0, index=indices)
    return tensor, min([n, total])


def process_motifs_array_to_tensors(motifs: Any, num_motifs: int) -> torch.LongTensor:
    # NOTE: input param type union tuple, or set, or list....
    logging.info('Create motif tmp array')
    motifs = np.fromiter(yield_2d_elements(motifs),
                         dtype=int,
                         count=num_motifs * 3).reshape((-1, 3))  # FIXME: 1.7PiB memory
    logging.info('Create motif tensors')
    return torch.LongTensor(motifs)


def process_tensor_to_motifs_set(motifs: torch.Tensor) -> Set[FrozenSet[int]]:
    return set([frozenset(m) for m in motifs.numpy()])


def process_df_to_tensor(df: pd.DataFrame, cols_drop: List[str]) -> torch.Tensor:
    return torch.from_numpy(df[df.columns.drop(cols_drop)].to_numpy())


def tensor_remove(tensor: Union[torch.Tensor, torch.LongTensor, torch.FloatTensor],
                  indices: Union[torch.Tensor, torch.LongTensor],
                  dim: int):
    # mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask = torch.ones(tensor.size(dim=dim), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


def random_choice_tensor(input: Union[torch.Tensor, torch.LongTensor],
                         dim: int,
                         num_samples: int,
                         probs: Union[torch.Tensor, torch.FloatTensor] = None
                         ) -> Union[torch.Tensor, torch.LongTensor]:
    if probs is None:
        probs = torch.ones(input.size(dim=dim)).to(torch.float)  # default: equal probabilities
    sampled_indices = torch.multinomial(probs, num_samples=num_samples, replacement=False)
    return torch.index_select(input=input, dim=dim, index=sampled_indices)


# https://github.com/pytorch/pytorch/issues/11931#issuecomment-625882503
def sample_multinomial_tensor_indices(input: Union[torch.Tensor, torch.LongTensor],
                                      dim: int,
                                      num_samples: int,
                                      probs: Union[torch.Tensor, torch.FloatTensor] = None
                                      ) -> Union[torch.Tensor, torch.LongTensor]:
    if probs is None:
        probs = torch.ones(input.size(dim=dim)).to(torch.float)  # default: equal probabilities
    return torch.multinomial(probs, num_samples=num_samples, replacement=False)


# https://github.com/pytorch/pytorch/issues/11931#issuecomment-625882503
def sample_multinomial_tensor(input: Union[torch.Tensor, torch.LongTensor],
                              dim: int,
                              num_samples: int,
                              probs: Union[torch.Tensor, torch.FloatTensor] = None
                              ) -> Union[torch.Tensor, torch.LongTensor]:
    sampled_indices = sample_multinomial_tensor_indices(input, dim, num_samples, probs)
    logging.info(f'sampled_indices.shape = {sampled_indices.shape}')
    return torch.index_select(input=input, dim=dim, index=sampled_indices)


# https://github.com/pytorch/pytorch/issues/30968#issuecomment-859084590
def sample_categorical_tensor(p: Union[torch.Tensor, torch.LongTensor]
                              ) -> Union[torch.Tensor, torch.LongTensor]:
    """
    Sample from multiple categorical distributions simultaneously.
    (PyTorch implementation)

    :param p: probabilities
    :return: sampled categories
    """
    return (p.cumsum(-1) >= torch.rand(p.shape[:-1])[..., None]).byte().argmax(-1)
