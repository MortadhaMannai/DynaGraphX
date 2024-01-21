import errno

from DYANE.components.learning.learning_utils import *
from DYANE.components.learning.motif_embeddings import *
from DYANE.components.node_roles.data.process_roles_data import *


def learn_node_arrival_rates(nodes_df: pd.DataFrame,
                             timesteps: List[int],
                             params_data: Dict[str, Any]):
    """
    Estimate node arrival rates

    :param nodes_df: nodes and their 1st active timestep in observed graph
    :param params_data: paths for learned parameters
    """
    logging.info('Learn node arrival rates')

    node_arrival_rates_file = params_data['node_arrival_rates_file']
    if not gz_file_exists(node_arrival_rates_file):
        # map timesteps (e.g. years) to start at 0
        timesteps_map = params_data['timesteps_map']  # NOTE: moved to learn_parameters()
        df = nodes_df[['timestep']].applymap(lambda t: timesteps_map[t])
        node_arrival_times = df['timestep'].tolist()

        # fit distribution to arrival times
        arrivals_distribution = fit_arrival_time_distribution(node_arrival_times)

        # save to file
        msgpack_save(node_arrival_rates_file, arrivals_distribution)


def get_node_role_counts(nodes_df: pd.DataFrame,
                         edges_df: pd.DataFrame,
                         node_adj: Dict[int, Dict[int, Set[int]]],
                         timesteps: List[int],
                         params_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Estimate node role counts

    :param nodes_df: node data in observed graph
    :param edges_df: graph edges
    :param node_adj: node adjacency list
    :param timesteps: graph timesteps
    :param params_data: paths for learned parameters
    :return: node role counts
    """
    logging.info('Estimate node role counts')

    role_counts_csv = params_data['node_role_counts_csv']
    if not file_exists(role_counts_csv):
        # initialize role counts
        role_counts_df = nodes_df[['node']].copy()
        role_counts_df = role_counts_df.assign(**{r: 0 for r in roles})
        role_counts_df = role_counts_df.set_index('node')

        timestep_csvs = params_data['node_role_counts_timestep_csvs']

        for t_idx, t in tqdm(enumerate(timesteps), desc='Timesteps'):
            if not file_exists(timestep_csvs[t_idx]):
                # get active nodes at timestep t
                active_nodes_t = set(get_active_nodes_t(nodes_df, t))
                # get edges at timestep t
                edges_t_df = get_edges_t(edges_df, t)

                # NOTE: assumes no duplicate edges (as in undirected graphs)
                for idx, edge in tqdm(edges_t_df.iterrows(), desc=f'Edges t={t}', total=edges_t_df.shape[0]):
                    u = edge['source']
                    v = edge['target']
                    cnt_rem = edge['weight']

                    edge_pair = {u, v}
                    if len(edge_pair) == 2:  # sanity check to avoid self-loops
                        # get neighbors at time t
                        neighbors_u = node_adj[t][u]
                        neighbors_v = node_adj[t][v]

                        # triangles - - - - - - - - - - - - - - - - - - - - - - - - *
                        # w_triangles = get_triangles(u, v, neighbors_u, neighbors_v)
                        # num_triangles = len(w_triangles)
                        num_triangles = get_num_triangles(u, v, neighbors_u, neighbors_v)
                        if num_triangles > 0:
                            # calculate weight
                            weight_triangle = get_roles_weight_motif(cnt_rem, num_triangles)
                            # update remaining edge count
                            cnt_rem = update_edge_count(cnt_rem, num_triangles)
                            # equal3 counted twice per triangle
                            role_counts_df.loc[[u, v], 'equal3'] += (weight_triangle / 2)

                        # wedges - - - - - - - - - - - - - - - - - - - - - - - - - *
                        w_wedges = get_w_wedges(u, v, neighbors_u, neighbors_v)
                        num_wedges = len(w_wedges)
                        if cnt_rem > 0 and num_wedges > 0:
                            # calculate weight
                            weight_wedge = get_roles_weight_motif(cnt_rem, num_wedges)
                            # update remaining edge count
                            cnt_rem = update_edge_count(cnt_rem, num_wedges)
                            for w in w_wedges:
                                # find hub and spoke roles in triplet
                                hub, spoke = get_hub_spoke(u, v, w, neighbors_u, neighbors_v)
                                # hub counted twice per wedge
                                role_counts_df.loc[hub, 'hub'] += (weight_wedge / 2)
                                # spoke counted once (total) per wedge
                                role_counts_df.loc[spoke, 'spoke'] += weight_wedge

                        # 1-edges - - - - - - - - - - - - - - - - - - - - - - - - - *
                        w_1edges = get_w_1edges(active_nodes_t, neighbors_u, neighbors_v)
                        num_1edges = len(w_1edges)
                        if cnt_rem > 0 and num_1edges > 0:
                            # calculate weight
                            weight_1edge = get_roles_weight_motif(cnt_rem, num_1edges)
                            # equal2 sums to edge count
                            role_counts_df.loc[[u, v], 'equal2'] += get_edge_count_motif(cnt_rem, num_1edges)
                            # outlier should be fraction
                            role_counts_df.loc[list(w_1edges), 'outlier'] += weight_1edge

                # save accumulated role counts upto current timestep t
                write_dataframe(role_counts_df, timestep_csvs[t_idx])
            else:
                role_counts_df = read_dataframe(timestep_csvs[t_idx])

        # save to file
        write_dataframe(role_counts_df, role_counts_csv)
    else:
        # read from file
        role_counts_df = read_dataframe(role_counts_csv, index_col='node')

    return role_counts_df


# Runtime: see above
def learn_node_roles_distribution(nodes_df: pd.DataFrame,
                                  edges_df: pd.DataFrame,
                                  node_adj: Dict[int, Dict[int, Set[int]]],
                                  timesteps: List[int],
                                  params_data: Dict[str, Any]):
    """
    Estimate node role probabilities

    :param nodes_df: node data in observed graph
    :param edges_df: graph edges
    :param node_adj: node adjacency list
    :param timesteps: graph timesteps
    :param params_data: paths for learned parameters
    """
    logging.info('Learn node roles distribution')

    node_roles_distr_csv = params_data['node_role_distr_csv']
    if not file_exists(node_roles_distr_csv):
        # get node role counts
        node_roles_df = get_node_role_counts(nodes_df, edges_df, node_adj, timesteps, params_data)
        # calculate node role probabilities
        node_roles_df = calc_node_roles_distr(node_roles_df)
        # save to file
        write_dataframe(node_roles_df, node_roles_distr_csv)


def get_motifs_graph(nodes_df: pd.DataFrame,
                     edges_df: pd.DataFrame,
                     node_adj: Dict[int, Dict[int, Set[int]]],
                     timesteps: List[int],
                     params_data: Dict[str, Any]) -> Tuple[
    Set[FrozenSet[int]], Dict[FrozenSet, int], Dict[FrozenSet[int], List[int]]]:
    """
    Get the motifs in the input graph

    :param nodes_df: node data in observed graph
    :param edges_df: graph edges
    :param node_adj: node adjacency list
    :param timesteps: graph timesteps
    :param params_data: paths for learned parameters
    :return: motifs, types and timesteps
    """
    logging.info('Get motifs in graph')

    motifs_file = params_data['motifs_file']
    types_file = params_data['motif_types_file']
    timesteps_file = params_data['motif_timesteps_file']

    if not file_exists(motifs_file) or not file_exists(types_file):
        # init motifs
        motifs = set([])
        motif_types = dict()
        motif_timesteps = dict()

        for t in tqdm(timesteps):
            # get active nodes at timestep t
            active_nodes_t = set(get_active_nodes_t(nodes_df, t))

            # get edges at timestep t
            edges_t_df = get_edges_t(edges_df, t)

            # NOTE: assumes no duplicate edges (as in undirected graphs)
            for idx, edge in tqdm(edges_t_df.iterrows(), desc=f'Edges t={t}', total=edges_t_df.shape[0]):
                u = edge['source']
                v = edge['target']
                cnt_rem = edge['weight']

                edge_pair = {u, v}
                if len(edge_pair) == 2:  # sanity check to avoid self-loops
                    # get neighbors at time t
                    neighbors_u = node_adj[t][u]
                    neighbors_v = node_adj[t][v]

                    # triangles - - - - - - - - - - - - - - - - - - - - - - - - *
                    w_triangles = get_w_triangles(u, v, neighbors_u, neighbors_v)
                    num_triangles = len(w_triangles)
                    if num_triangles > 0:
                        num_edges = 3
                        motifs, motif_types, motif_timesteps = update_motifs_found(u, v, w_triangles, num_edges, t,
                                                                                   motifs, motif_types, motif_timesteps)
                        # update remaining edge count
                        cnt_rem = update_edge_count(cnt_rem, num_triangles)

                    # wedges - - - - - - - - - - - - - - - - - - - - - - - - - *
                    w_wedges = get_w_wedges(u, v, neighbors_u, neighbors_v)
                    num_wedges = len(w_wedges)
                    if cnt_rem > 0 and num_wedges > 0:
                        num_edges = 2
                        motifs, motif_types, motif_timesteps = update_motifs_found(u, v, w_wedges, num_edges, t,
                                                                                   motifs, motif_types, motif_timesteps)
                        # update remaining edge count
                        cnt_rem = update_edge_count(cnt_rem, num_wedges)

                    # 1-edges - - - - - - - - - - - - - - - - - - - - - - - - - *
                    if cnt_rem > 0:
                        w_1edges = get_w_1edges(active_nodes_t, neighbors_u, neighbors_v)
                        if len(w_1edges) > 0:
                            # filter for w for other motif types (triangle, wedge)
                            w_1edges = filter_w_1edges(u, v, w_1edges, motifs, motif_types)
                            num_1edges = len(w_1edges)
                            if num_1edges > 0:
                                num_edges = 1
                                motifs, motif_types, motif_timesteps = update_motifs_found(u, v, w_1edges, num_edges, t,
                                                                                           motifs, motif_types,
                                                                                           motif_timesteps)

        # save to files
        msgpack_save(motifs_file, motifs)
        msgpack_save(types_file, motif_types)
        msgpack_save(timesteps_file, motif_timesteps)
    else:
        # read from files
        motifs = msgpack_load(motifs_file)
        motif_types = msgpack_load(types_file)
        motif_timesteps = msgpack_load(timesteps_file)
    return motifs, motif_types, motif_timesteps


def get_motif_edge_counts(motifs: Set[FrozenSet[int]], motif_types: Dict[FrozenSet[int], int],
                          motif_timesteps: Dict[FrozenSet[int], List[int]],
                          nodes_df: pd.DataFrame, edges_df: pd.DataFrame, node_adj: Dict[int, Dict[int, Set[int]]],
                          timesteps: List[int], params_data: Dict[str, Any]) -> \
        Dict[FrozenSet[int], Union[Union[int, float], Dict[int, Union[int, float]]]]:
    logging.info('Estimate motif edge counts')

    counts_file = params_data['motif_counts_file']
    # OPTIMIZE: included here instead of get_motif_timestep_counts()
    timestep_counts_file = params_data['motif_timestep_counts_file']
    if not file_exists(counts_file):
        # init
        motif_counts = {m: 0 for m in motifs}
        timestep_count_data = {t: {i: 0 for i in [3, 2, 1]}
                               for t in timesteps}

        for t in tqdm(timesteps, desc='Timesteps'):
            # get active nodes at timestep t
            active_nodes_t = set(get_active_nodes_t(nodes_df, t))
            timestep_count_data[t]['num_active_nodes_t'] = len(active_nodes_t)

            # get edges at timestep t
            edges_t_df = get_edges_t(edges_df, t)

            # get motifs that show in timestep t
            motifs_t = [m for m in motifs if t in motif_timesteps[m]]
            for m in tqdm(motifs_t, desc=f'Motifs t={t}'):
                i = motif_types[m]
                num_1edges = 0  # init

                # NOTE: assumes no duplicate edges (as in undirected graphs)
                m_edges = get_motif_edges(edges_t_df, m)

                for idx, edge in m_edges.iterrows():
                    u = edge['source']
                    v = edge['target']
                    cnt_rem = edge['weight']

                    # get neighbors at time t
                    neighbors_u = node_adj[t][u]
                    neighbors_v = node_adj[t][v]

                    if i == 3:  # triangle
                        num_triangles = get_num_triangles(u, v, neighbors_u, neighbors_v)
                        if num_triangles > 0:
                            weight = get_triangle_motif_edge_weight(num_triangles, cnt_rem)
                            motif_counts[m] += weight / i
                    elif i == 2:  # wedge
                        num_wedges = get_num_wedges(u, v, neighbors_u, neighbors_v)
                        if num_wedges > 0:
                            # OPTIMIZE: pass num_triangles
                            weight = get_wedge_motif_edge_weight(num_wedges, cnt_rem,
                                                                 u, v, neighbors_u, neighbors_v)
                            motif_counts[m] += weight / i
                    elif i == 1:  # 1-edge
                        w_1edges = get_w_1edges(active_nodes_t, neighbors_u, neighbors_v)
                        if len(w_1edges) > 0:
                            # filter for w for other motif types (triangle, wedge)
                            w_1edges = filter_w_1edges(u, v, w_1edges, motifs, motif_types)
                            num_1edges = len(w_1edges)
                            if num_1edges > 0:
                                # OPTIMIZE: pass num_triangles and num_wedges
                                weight = get_1edge_motif_edge_weight(num_1edges, cnt_rem,
                                                                     u, v, neighbors_u, neighbors_v)
                                if weight > 0:
                                    motif_counts[m] += weight / i

                    timestep_count_data[t][i] += num_1edges

        # save to file
        msgpack_save(counts_file, motif_counts)
        msgpack_save(timestep_counts_file, timestep_count_data)
    else:
        # read from file
        motif_counts = msgpack_load(counts_file)
    return motif_counts


def learn_motif_interarrival_rates(motifs: Set[FrozenSet[int]], motif_types: Dict[FrozenSet[int], int],
                                   motif_timesteps: Dict[FrozenSet[int], List[int]],
                                   nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                                   node_adj: Dict[int, Dict[int, Set[int]]],
                                   timesteps: List[int], params_data: Dict[str, Any]):
    """
    Estimate inter-arrival rates per motif type

    :param motifs: motifs
    :param motif_types: motif types
    :param motif_timesteps: motif timesteps in
    :param nodes_df: node data in observed graph
    :param edges_df: graph edges
    :param node_adj: node adjacency list
    :param timesteps: graph timesteps
    :param params_data: paths for learned parameters
    """
    logging.info('Learn motif inter-arrival rates')

    motif_interarrivals_file = params_data['motif_interarrivals_file']
    if not file_exists(motif_interarrivals_file):
        num_timesteps = len(timesteps)
        motif_counts = get_motif_edge_counts(motifs, motif_types, motif_timesteps,
                                             nodes_df, edges_df, node_adj, timesteps, params_data)

        # distribution of rates per motif type
        m_types = [3, 2, 1]
        rates_motifs = {i: [] for i in m_types}  # init
        for m in tqdm(motifs, desc='Motifs'):
            i = motif_types[m]
            rates_motifs[i].append(motif_counts.pop(m) / num_timesteps)

        # fit distributions
        rates_distributions = fit_rates_distributions(rates_motifs, m_types)
        del rates_motifs  # free up memory

        # save to file
        msgpack_save(motif_interarrivals_file, rates_distributions)


def learn_motif_proportions(motifs: Set[FrozenSet[int]],
                            motif_types: Dict[FrozenSet[int], int],
                            num_nodes: int,
                            params_data: Dict[str, Any]) -> Dict[int, int]:
    """
    Estimate proportions of each motif type

    :param motifs: motifs
    :param motif_types: motif types
    :param nodes_df: node data in observed graph
    :param params_data: paths for learned parameters
    :return: num motifs
    :rtype: dict
    """
    logging.info('Learn motif proportions')

    motif_prop_counts_file = params_data['motif_prop_counts_file']
    motif_proportions_file = params_data['motif_proportions_file']

    if not file_exists(motif_proportions_file):
        logging.info(f'Estimate num. possible motifs ({num_nodes:,} nodes)')
        possible = int(comb(num_nodes, 3))

        if not file_exists(motif_prop_counts_file):
            logging.info('Count motifs ea/type')
            num_motifs = {i: 0 for i in range(4)}
            for m in tqdm(motifs, desc='Motifs'):
                num_motifs[motif_types[m]] += 1

            logging.info('Estimate empty motif count')
            num_motifs[0] = possible - sum([num_motifs[i] for i in range(1, 4)])
            # save num motifs
            msgpack_save(motif_prop_counts_file, num_motifs)
        else:
            logging.info('Read counts motifs ea/type')
            # read num motifs
            num_motifs = msgpack_load(motif_prop_counts_file)
            logging.info(f"num motifs = {num_motifs}")

        # save proportions
        proportions = {i: num_motifs[i] / possible for i in range(4)}
        msgpack_save(motif_proportions_file, proportions)
    else:
        # read num motifs
        num_motifs = msgpack_load(motif_prop_counts_file)

    return num_motifs


def learn_parameters(dataset_dir: Union[str, os.PathLike], dataset_info: Dict[str, Any]):
    """
    Learn parameters from input graph

    :param dataset_dir: dataset directory
    :param dataset_info: dataset information
    """

    # Path for parameters and tmp files
    params_data = get_directories_parameters(dataset_dir, dataset_info)

    # added to params_data in get_params_filepaths()
    content_option = params_data['node_content_embedding_option']  # refactor: delete

    # use node timesteps of when user accounts created (delta)
    params_data['use_timesteps_nodes'] = dataset_info['use_timesteps_nodes']
    params_data['timesteps_nodes'] = dataset_info['timesteps_nodes']
    if dataset_info['use_timesteps_nodes']:
        use_timesteps_nodes = '-use_timesteps_nodes'
    else:
        use_timesteps_nodes = f'-all_distrs_nodes'

    logging.info(f'Learn parameters ({content_option} content embeddings, {use_timesteps_nodes})')

    saved_params_fpath = os.path.join(params_data['params_dir'],  # ext = add Beta and Log-Uniform distrs.
                                      f'model_params_ext-embedding_{content_option}{use_timesteps_nodes}'
                                      f'.msg')
    params_data['params_file'] = saved_params_fpath

    logging.info(f'Params. fpath = {saved_params_fpath}')

    if file_exists(saved_params_fpath) or gz_file_exists(saved_params_fpath):
        logging.info("Read model parameters")
        params_data = msgpack_load(saved_params_fpath)
    else:
        logging.info('Learning the model parameters...')
        logging.info('Get input graph data')
        # get node data
        nodes_df = get_nodes_data(dataset_info['nodes_csv'])
        num_nodes = get_num_nodes(nodes_df)
        params_data['num_nodes'] = num_nodes

        # get pandas df node index (node_map)
        if not file_exists(params_data['node_map_file']):
            # NOTE: is_sorted param isn't being used
            node_map = make_df_node_map_index(nodes_df, 'node', is_sorted=False)
            write_node_map(node_map, params_data['node_map_file'])
        else:
            node_map = read_node_map(params_data['node_map_file'])

        # get edges data
        edges_df = get_edges_data(dataset_info['edges_csv'])

        # FEATURE: applied node_map at beginning of learn_parameters()
        # apply node_map here to nodes_df and edges_df
        map_cols = ['node']
        nodes_df[map_cols] = df_apply_node_mapping(nodes_df, node_map, map_cols)
        map_cols = ['source', 'target']
        edges_df[map_cols] = df_apply_node_mapping(edges_df, node_map, map_cols)

        # get timesteps
        timesteps = sorted(dataset_info['timesteps'])

        # map timesteps (e.g. years) to start at 0
        if 'timesteps_all' in dataset_info:
            timesteps_map = {t: i for i, t, in enumerate(sorted(dataset_info['timesteps_all']))}
        else:
            timesteps_map = {t: i for i, t, in enumerate(timesteps)}

        params_data['timesteps'] = timesteps
        params_data['timesteps_map'] = timesteps_map
        params_data['num_timesteps'] = len(timesteps)
        params_data['length_timestep'] = dataset_info['length_timestep']

        if params_data['use_timesteps_nodes'] and 'timesteps_nodes' in params_data:
            timesteps_nodes = params_data['timesteps_nodes']
        else:
            timesteps_nodes = timesteps

        # get node adjacency dict
        if not file_exists(params_data['node_adj_file']):
            node_adj = get_node_adj(nodes_df, edges_df, timesteps)
            msgpack_save(params_data['node_adj_file'], node_adj)
        else:
            node_adj = msgpack_load(params_data['node_adj_file'])

        # Learn parameters - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *

        # (1) Learn node arrival rates - - - - - - - - - - - - - - - - - - *
        learn_node_arrival_rates(nodes_df, timesteps_nodes, params_data)

        # (2) Learn node roles distribution (and embeddings?) - - - - - - - *
        learn_node_roles_distribution(nodes_df, edges_df, node_adj, timesteps, params_data)
        # learn node role embeddings
        train_node_role_embeddings(edges_df, nodes_df, num_nodes, timesteps, params_data)
        # OPTIMIZE: move train_node_role_embeddings inside learn_node_roles_distribution()?

        # Get motifs in graph - - - - *
        motifs, motif_types, motif_timesteps = get_motifs_graph(nodes_df, edges_df, node_adj,
                                                                timesteps, params_data)

        # refactor: check where need the counts now
        motif_counts = get_motif_edge_counts(motifs, motif_types, motif_timesteps,
                                             nodes_df, edges_df, node_adj, timesteps, params_data)

        # (3) Learn motif type inter-arrival rates - - - - - - - - - - - - *
        # weighted counts for rates
        learn_motif_interarrival_rates(motifs, motif_types, motif_timesteps,
                                       nodes_df, edges_df, node_adj, timesteps, params_data)
        # free up memory
        del edges_df
        del motif_timesteps
        del node_adj

        # (4) Learn motif proportions - - - - - - - - - - - - - - - - - *
        # FIXME: Look at way I learned motif proportions, weighted counts
        # num. motifs found ea/type
        num_motifs = learn_motif_proportions(motifs, motif_types, num_nodes, params_data)

        # (4) Learn motif type probabilities CNN - - - - - - - - - - - - *
        # LOCAL: normally called in train_motif_types_model()
        calc_node_avg_var_content_embeddings(params_data)

        # FEATURE: get node content embeddings
        #  use a function passed by param (so I can have different for Twitter dataset)

        # Prep data for motif type CNN
        sample_size = 100_000  # OPTIMIZE: read from params
        params_data['motif_type_training_csv'] = get_motif_type_training_data(motifs, motif_types, nodes_df,
                                                                              num_motifs, sample_size, params_data)
        # free up memory
        del nodes_df
        del motifs
        del motif_types

        # learn motif type CNN
        params_data['motif_types_model'] = train_motif_types_model(sample_size, params_data)

        # save all csv and msg filepaths
        logging.info("Save model parameters")
        msgpack_save(saved_params_fpath, params_data)

    print(f"Params PATH = {saved_params_fpath}")
    return params_data


def get_dataset(dataset_dir: Union[str, os.PathLike]) -> Tuple[Union[str, os.PathLike], Dict[str, Any]]:
    """
    Get dataset directory, info, and graph.

    :param dataset_dir: dataset directory
    :return: dataset directory and info
    """
    # Validate dataset directory
    if not os.path.exists(dataset_dir):
        logging.error(f'Dataset directory not found.\n{dataset_dir}')
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dataset_dir)

    # Get dataset info
    dataset_info_file = os.path.join(dataset_dir, 'dataset_info.msg')

    if not gz_file_exists(dataset_info_file):
        logging.error(f'Dataset info file not found in directory.\n{dataset_info_file}')
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dataset_info_file)

    try:
        dataset_info = msgpack_load(dataset_info_file)
    except Exception as e:
        logging.error(f'Could not read dataset info file.\n{dataset_info_file}')
        raise e

    return dataset_dir, dataset_info


def get_params_filepaths(params_dir: Union[str, os.PathLike], tmp_files_dir: Union[str, os.PathLike],
                         dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    # refactor: remove param keywords not needed/used anymore
    emb = dataset_info['node_content_embedding_option']
    timesteps = dataset_info['timesteps_all'] if 'timesteps_all' in dataset_info else dataset_info['timesteps']
    motif_rates_version = f"version_{dataset_info['motif_rates']}"  # refactor
    params_data = {'nodes_csv': dataset_info['nodes_csv'],
                   'edges_csv': dataset_info['edges_csv'],
                   'node_map_file': os.path.join(params_dir, 'node_map.pkl.gzip'),
                   'node_adj_file': os.path.join(params_dir, 'node_adj.msg'),
                   'node_arrival_rates_file': os.path.join(params_dir,
                                                           'node_arrival_rates-all_distrs.msg'),
                   'node_role_counts_csv': os.path.join(params_dir, 'node_role_counts.csv'),
                   'node_role_counts_timestep_csvs': [os.path.join(tmp_files_dir, f'node_role_counts-t_{t}.csv')
                                                      for t in timesteps],
                   'node_role_distr_csv': os.path.join(params_dir, 'node_role_distr.csv'),
                   'node_role_embeddings_csv': os.path.join(params_dir, f'node_role_embeddings-gcn.csv'),
                   'node_role_embeddings_meta_model': os.path.join(params_dir,
                                                                   f'node_role_embeddings-meta_model.sav'),
                   'node_roles_gcn_params_file': os.path.join(params_dir, f'data_params-node_roles-gcn.msg'),
                   'node_content_embedding_option': emb,
                   'node_embeddings_csv': os.path.join(params_dir, f'node_embeddings-embedding_{emb}-gcn.csv'),
                   'node_avg_content_embeddings_csv': os.path.join(params_dir, 'node_avg_content_embeddings.csv'),
                   'node_var_content_embeddings_csv': os.path.join(params_dir, 'node_var_content_embeddings.csv'),
                   'node_ctr_content_embeddings_csv': os.path.join(params_dir, 'node_ctr_content_embeddings.csv'),
                   'all_content_embeddings_csv': dataset_info['all_content_embeddings_csv'],
                   'node_content_embed_all_csv': dataset_info['node_content_embed_all_csv'],
                   'content_authored_csv': dataset_info['content_authored_csv'],
                   'motifs_model_name': dataset_info['motifs_model_name'],
                   'motifs_file': os.path.join(params_dir, 'motifs.msg'),
                   'motif_types_file': os.path.join(params_dir, 'motif_types.msg'),
                   # motifs timesteps in
                   'motif_timesteps_file': os.path.join(params_dir, 'motif_timesteps.msg'),
                   # motif edge counts
                   'motif_counts_file': os.path.join(params_dir, 'motif_counts.msg'),
                   # timestep_count_data, from motif edge weights calc
                   'motif_timestep_counts_file': os.path.join(params_dir, 'motif_timestep_counts.msg'),
                   # motif inter-arrival rates
                   'motif_interarrivals_file': os.path.join(params_dir,  # ext = add Beta and Log-Uniform distrs.
                                                            f'motif_interarrivals-all_distrs_ext-{motif_rates_version}.msg'),
                   # motif proportion counts
                   'motif_prop_counts_file': os.path.join(params_dir,
                                                          f'motif_prop_counts.msg'),

                   'motif_proportions_file': os.path.join(params_dir,
                                                          f'motif_proportions.msg'),
                   }
    return params_data


def get_directories_parameters(dataset_dir: Union[str, os.PathLike],
                               dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the directories for parameters and temp save files.
    If they don't exist, create the directories.

    :param dataset_dir: dataset directory
    :param dataset_info: dataset info
    :return: parameters directory, tmp files directory, pytorch directory, params filepaths
    """
    # refactor: remove param keywords (for path) not needed/used anymore

    # Directory to save parameters
    print(dataset_dir)
    if str(dataset_dir).startswith('/Users/gzenotor/Development/datasets/'):
        # local
        params_dir = os.path.join(dataset_dir, f'learned_parameters_v2-local')
    else:
        # MSU
        params_dir = os.path.join(dataset_dir, f'learned_parameters_v2-msu')
    # FIXME
    if 'arnetminer_tiny_updated' in dataset_dir:
        params_dir = os.path.join(dataset_dir, f'learned_parameters-msu')

    if dataset_info['empty_cnn']:
        params_dir = os.path.join(params_dir, 'cnn_empty_examples')

    check_dir_exists(params_dir)

    # Directory to save temp files
    tmp_files_dir = os.path.join(params_dir, 'tmp_save_files')
    check_dir_exists(tmp_files_dir)

    # Directory to save temp files
    pytorch_dir = os.path.join(params_dir, 'pytorch_dataset')
    check_dir_exists(pytorch_dir)

    params_data = get_params_filepaths(params_dir, tmp_files_dir, dataset_info)
    # OPTIMIZE: send dirs in same dictionary

    params_data['dataset_dir'] = dataset_dir
    params_data['params_dir'] = params_dir
    params_data['tmp_files_dir'] = tmp_files_dir
    params_data['pytorch_dir'] = pytorch_dir

    return params_data


if __name__ == '__main__':
    if len(sys.argv) == 2:
        try:
            # Learn parameters
            # TODO: pass dataset_info
            _ = learn_parameters(*get_dataset(dataset_dir=sys.argv[1]))
        except Exception as e:
            logging.error('Learn parameters failed!')
            raise e
    else:
        logging.error('Required parameters: dataset directory.')

    if __package__ is None:
        testpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.append(testpath)
