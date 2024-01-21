import errno

from DYANE.components.learning.motif_embeddings import update_node_embeddings
from DYANE.components.sampling.sample_topics import sample_motif_topics
from DYANE.components.sampling.sampling_utils import *


def get_active_nodes(num_timesteps: int,
                     length_timestep: int,
                     num_nodes: int,
                     gen_dir: Union[str, os.PathLike],
                     params_data: Dict[str, Any]
                     ) -> Dict[int, List[int]]:
    """
    Get active nodes.

    :param num_timesteps: number of timesteps
    :param length_timestep: length of timestep
    :param num_nodes: number of nodes
    :param node_rate: node arrival rate
    :param gen_dir: directory for graph generation
    :return: active nodes per timestep
    """
    logging.info('Get active nodes')
    active_nodes_file = os.path.join(gen_dir, f'active_nodes_gen.msg')
    if not gz_file_exists(active_nodes_file):
        last_timestep = 0
        V_A = {t: [] for t in range(num_timesteps)}  # initialize V_A

        # Get distribution to sample rates from
        dname = params_data['node_rates_distr']
        arrivals_distribution = msgpack_load(params_data['node_arrival_rates_file'])
        X = np.absolute(distributions[dname].rvs(*arrivals_distribution[dname]['parameters'],
                                                 size=num_nodes,
                                                 # random_state=params_data['random_seed']))
                                                 random_state=None))

        for v in range(num_nodes):
            t = int(np.floor(X[v] / length_timestep))
            if 0 <= t <= num_timesteps:
                if t > last_timestep:
                    last_timestep = t
                for j in range(t, num_timesteps):
                    V_A[j].append(v)  # add to nodes that are active at time t

        active_nodes_data = {'V_A': V_A, 'last_timestep': last_timestep}
        msgpack_save(active_nodes_file, active_nodes_data)

    else:
        logging.info('Read active nodes from saved file')
        active_nodes_data = msgpack_load(active_nodes_file)
        V_A = active_nodes_data['V_A']

    return V_A


def get_active_triplets(new_nodes: List[int], old_nodes: List[int], t: int
                        ) -> Tuple[Iterable[Tuple[int]], int]:
    logging.info(f'Calculate new active triplets t={t + 1}')
    num_old = len(old_nodes)

    # (1) 3 new nodes
    # -------------------------------------------------------*
    triplets = combinations(new_nodes, 3)
    num_triplets = int(comb(len(new_nodes), 3, repetition=False))

    if num_old > 0:
        # (2) 2 new + 1 old
        # -------------------------------------------------------*
        new2_old1 = ((u, v, w)
                     for ((u, v), w) in product(combinations(new_nodes, 2), old_nodes)
                     if w not in [u, v])

        triplets = chain(triplets, new2_old1)
        num_triplets += len(new_nodes) * ((len(new_nodes) - 1) / 2) * len(old_nodes)

        # (3) 1 new + 2 old
        # -------------------------------------------------------*
        new1_old2 = ((u, v, w)
                     for ((u, v), w) in product(combinations(old_nodes, 2), new_nodes)
                     if w not in [u, v])
        # OPTIMIZE: make helper function for duplicated code
        triplets = chain(triplets, new1_old2)
        num_triplets += len(old_nodes) * ((len(old_nodes) - 1) / 2) * len(new_nodes)
        num_triplets = int(num_triplets)
    logging.info(f"num. new active triplets = {num_triplets:,}")
    return triplets, num_triplets


def sample_motifs(new_nodes: List[int],
                  old_nodes: List[int],
                  node_map: pd.Index,
                  motif_props: Dict[int, float],
                  total_expected: Dict[int, int],
                  motifs_cnn: Type[torch.nn.Module],
                  role_counts_df: pd.DataFrame,
                  roles_meta_model: Union[RidgeCV, KernelRidge],
                  rates_distribution: Dict[str, Dict[str, Any]],
                  t: int,
                  delta: int,
                  num_timesteps: int,
                  stashed: Set[FrozenSet[int]],
                  params_data: Dict[str, Any],
                  gen_dir: Union[str, os.PathLike]) \
        -> Tuple[Dict[str, Any], pd.DataFrame]:
    gen_dir_t = get_gen_dir_t(gen_dir, t)
    motifs_t_file = os.path.join(gen_dir_t, f'motif_data--t_{t + 1}.msg')
    role_counts_t_file = os.path.join(gen_dir_t, f'node_role_counts--t_{t + 1}.csv')

    tmp_dir_t = os.path.join(gen_dir_t, 'tmp_save_files')
    check_dir_exists(tmp_dir_t)

    content_t_dir = get_gen_topic_dir_t(gen_dir, params_data)
    content_t_file = os.path.join(content_t_dir, f'motif_content--t_{t + 1}.csv')
    logging.info(f'Content path:\n{content_t_file}')

    logging.info(f'Sample motifs t={t + 1}')
    if not file_exists(motifs_t_file):
        m_types = [3, 2, 1]
        triplets, num_triplets = get_active_triplets(new_nodes, old_nodes, t)

        # Calc expected counts for motif types
        logging.info('Calculate expected counts')
        exp_counts = {i: int(np.round(motif_props[i] * num_triplets))
                      for i in m_types}
        for i in m_types:
            print(f'\t\t\ttype {i} exp. = {exp_counts[i]:>15,}')
        sys.stdout.flush()

        if params_data['stash_unsampled'] and len(stashed) > 0:
            logging.info('Chaining triplets...')
            triplets = chain(triplets, (s for s in stashed))
            num_triplets += len(stashed)

        # Note: node role & content embeddings already in trained CNN mapping

        # Sample motif rates and timesteps - - - - - - - - - - - - - - - - - - - - - *
        sampled_timesteps_t, sampled_rates_t, num_appear_t = helper_sample_motif_timesteps(
            num_timesteps, exp_counts,
            m_types, rates_distribution,
            t, delta,
            gen_dir_t, params_data)

        num_motifs_show = sum(num_appear_t.values())
        if num_motifs_show > 0:

            # Sample motif types - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
            # DYANE sampling
            motifs_t_show, motif_types_t, num_sampled_t, stash_t = helper_sample_motifs_batches_dyane(motifs_cnn,
                                                                                                      triplets,
                                                                                                      num_triplets,
                                                                                                      num_appear_t,
                                                                                                      total_expected,
                                                                                                      m_types,
                                                                                                      params_data[
                                                                                                          'stash_unsampled']
                                                                                                      )

            # Sample motif rates and timesteps - - - - - - - - - - - - - - - - - - - - - *
            motif_timesteps_t, motif_rates_t = sample_motif_timesteps(motifs_t_show, motif_types_t,
                                                                      sampled_rates_t, sampled_timesteps_t, t)

            # Sample node roles - - - - - - - - - - - - - - - - - - - - - - - - - - - *
            roles_motifs_t, role_counts_df = sample_node_roles(motifs_t_show, motif_types_t,
                                                               motif_timesteps_t,
                                                               role_counts_df, t)

            if roles_meta_model is not None:
                # update role embeddings
                update_node_embeddings(roles_meta_model, role_counts_df, motifs_cnn, params_data)

            logging.info(f'{num_motifs_show:,} = ')
            sample_motif_topics(motifs_t_show, roles_motifs_t, motif_timesteps_t,
                                old_nodes + new_nodes, node_map, t, content_t_file, params_data)
            save_role_counts_t(role_counts_df, role_counts_t_file)

            motifs_t_data = {
                'motifs_t_show': motifs_t_show,
                'motif_types_t': motif_types_t,
                'motif_timesteps_t': motif_timesteps_t,
                'motif_rates_t': motif_rates_t,
                'roles_motifs': roles_motifs_t,
                'role_counts_t_file': role_counts_t_file,
                'topics_file': content_t_file,
                'exp_counts_t': exp_counts,
                'num_appear_t': num_appear_t,
                'num_sampled_t': num_sampled_t,
                'stashed_t': stash_t
            }
            msgpack_save(motifs_t_file, motifs_t_data)
        else:
            # logging.info(f"num_motifs_show = {num_motifs_show:,} <------- NO MOTIFS SHOW FOR t={t + 1}!!!!!")
            logging.info(f'{num_motifs_show:,} = ')
            motifs_t_data = {
                'motifs_t_show': set([]),
                'motif_types_t': {},
                'motif_timesteps_t': {},
                'motif_rates_t': {},
                'roles_motifs': {},
                'role_counts_t_file': '',
                'topics_file': '',
                'exp_counts_t': {},
                'num_appear_t': {},
                'num_sampled_t': {},
                'stashed_t': set([])
            }

    else:
        logging.info(f'Read save file for motifs t={t + 1}')
        motifs_t_data = msgpack_load(motifs_t_file)
        role_counts_df.update(read_role_counts_t(role_counts_t_file))

    return motifs_t_data, role_counts_df


def sample_node_roles(motifs: Set[FrozenSet[int]],
                      motif_type: Union[int, Dict[FrozenSet[int], int]],
                      motif_timesteps: Dict[FrozenSet[int], List[int]],
                      role_counts_df: pd.DataFrame,
                      t: int
                      ) -> Tuple[Dict[FrozenSet[int], Dict[int, str]], pd.DataFrame]:
    """
    Sample node roles for motifs

    :param motifs: motifs
    :param motif_type: motif type
    :param motif_timesteps: timesteps motifs appear in
    :param role_counts_df: node role counts
    :return: node roles in motifs
    """
    logging.info(f"Sample node roles for motifs t={t + 1}")

    roles_motifs = {motif: {node: None for node in motif}
                    for motif in motifs}

    for motif in motifs:
        # Calc. roles prob. distr. with latest counts
        role_distr_df = update_role_distr_df(role_counts_df)

        u, v, w = motif

        # get motif type
        if isinstance(motif_type, int):
            i = motif_type
        else:
            i = motif_type[motif]

        if i == 3:  # triangle
            # update counts for distr
            role_counts_df = update_role_counts(role_counts_df, [u, v, w], 'equal3',
                                                motif_timesteps[motif])

            # save assigned roles
            for eq3 in motif:
                roles_motifs[motif][eq3] = 'equal3'

        elif i == 2:  # wedge
            # sample hub role
            hub, spokes = helper_sample_roles(role_distr_df, 'hub', [u, v, w])

            # update counts for distr
            role_counts_df = update_role_counts(role_counts_df, hub, 'hub',
                                                motif_timesteps[motif])
            role_counts_df = update_role_counts(role_counts_df, spokes, 'spoke',
                                                motif_timesteps[motif])

            # save assigned roles
            roles_motifs[motif][hub] = 'hub'
            for spoke in spokes:
                roles_motifs[motif][spoke] = 'spoke'

        elif i == 1:  # 1-edge
            # sample outlier role
            outlier, equal2 = helper_sample_roles(role_distr_df, 'outlier', [u, v, w])

            # update counts for distr
            role_counts_df = update_role_counts(role_counts_df, outlier, 'outlier',
                                                motif_timesteps[motif])
            role_counts_df = update_role_counts(role_counts_df, equal2, 'equal2',
                                                motif_timesteps[motif])

            # save assigned roles
            roles_motifs[motif][outlier] = 'outlier'
            for eq2 in equal2:
                roles_motifs[motif][eq2] = 'equal2'

    return roles_motifs, role_counts_df


def get_motif_edges(motifs: Set[FrozenSet[int]],
                    motif_types: Dict[FrozenSet[int], int],
                    roles_motifs: Dict[FrozenSet[int], Dict[int, str]],
                    roles_assigned_counts_df: pd.DataFrame,
                    t: int,
                    gen_dir: Union[str, os.PathLike]) \
        -> Tuple[Dict[FrozenSet[int], List[Tuple[int]]], pd.DataFrame]:
    """
    Get motif edges

    :param motifs: motifs
    :param motif_types: motif types
    :param roles_motifs: node roles
    :param roles_assigned_counts_df: node roles assigned counts
    :param t: timestep
    :param gen_dir: directory for generated graph
    :return: motif edges
    """
    logging.info('Get motif edges')
    gen_dir_t = os.path.join(gen_dir, f't_{t + 1}')
    motif_edges_file = os.path.join(gen_dir_t, f'motif_edges_t_{t + 1}.msg')

    if not file_exists(motif_edges_file):
        motif_edges = {}
        logging.info(f'Sample edges for motifs t={t + 1}')
        # for motif in motifs:
        for motif in tqdm(motifs, desc='Motifs'):
            u, v, w = motif
            motif_type = motif_types[motif]

            if motif_type == 3:  # triangle
                # update roles assigned
                update_roles_assigned(roles_assigned_counts_df, [u, v, w], 'equal3')
                # save edges
                motif_edges[motif] = list(combinations([u, v, w], 2))

            elif motif_type == 2:  # wedge
                # get hub and spokes
                hub, spokes = helper_get_node_with_role(roles_motifs, 'hub', motif)
                # update roles assigned
                update_roles_assigned(roles_assigned_counts_df, hub, 'hub')
                update_roles_assigned(roles_assigned_counts_df, spokes, 'spoke')
                # save edges
                motif_edges[motif] = list(product(hub, spokes))  # FIXED: [hub]

            elif motif_type == 1:  # 1-edge
                # get outlier and equal2
                outlier, equal2 = helper_get_node_with_role(roles_motifs, 'outlier', motif)  # FIXED: 'hub'
                # update roles assigned
                update_roles_assigned(roles_assigned_counts_df, outlier, 'outlier')
                update_roles_assigned(roles_assigned_counts_df, equal2, 'equal2')
                # save edges
                motif_edges[motif] = [tuple(equal2)]

        msgpack_save(motif_edges_file, motif_edges)
    else:
        logging.info(f'Read edges for motifs t={t + 1} from save file')
        motif_edges = msgpack_load(motif_edges_file)

    return motif_edges, roles_assigned_counts_df


def sample_motif_timesteps(motifs: Set[FrozenSet[int]],
                           motif_type: Dict[FrozenSet[int], int],
                           motif_sampled_rates: Dict[int, List[float]],
                           motif_sampled_timesteps: Dict[int, List[int]],
                           t: int
                           ) -> Tuple[Dict[FrozenSet[int], List[int]], Dict[FrozenSet[int], float]]:
    # Initialization
    motif_timesteps = {m: set([]) for m in motifs}
    motif_rates = {m: None for m in motifs}

    logging.info(f'Save timesteps new motifs t={t + 1}')
    for motif in tqdm(motifs, desc='Motifs'):
        i = motif_type[motif]
        motif_timesteps[motif] = motif_sampled_timesteps[i].pop()
        if len(motif_sampled_rates) > 0 and len(motif_sampled_rates[i]) > 0:
            motif_rates[motif] = motif_sampled_rates[i].pop()

    return motif_timesteps, motif_rates


def generate_dynamic_graph(num_timesteps: int,
                           num_nodes: int,
                           active_nodes: Dict[int, List[int]],
                           motif_props: Dict[int, float],
                           motifs_cnn: Type[torch.nn.Module],
                           node_map: pd.Index,
                           role_counts_df: pd.DataFrame,
                           roles_meta_model: Union[RidgeCV, KernelRidge],
                           rates_distribution: Dict[str, Dict[str, Any]],
                           model_params: Dict[str, Any],
                           gen_data_dir: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Generate dynamic graph.

    :param num_timesteps: number of timesteps to generate
    :param num_nodes: number of nodes
    :param active_nodes: active nodes each timesteo
    :param motif_props: motif type proportions
    :param motifs_cnn: motif types model
    :param node_map: node id mapping
    :param role_counts_df: node role counts
    :param roles_meta_model: node roles meta-model
    :param rates_distribution: motif type inter-arrival rates
    :param model_params: model parameters
    :param gen_data_dir: directory for generated graph
    :return: generated graph data
    """
    logging.info('Generate dynamic graph')
    time_code = log_time()

    # Initialization
    motifs = set([])
    motifs_show = set([])
    motif_types = {}
    motif_edges = {}
    motif_timesteps = {}
    motif_rates = {}

    roles_assigned_counts_df = init_roles_assigned_counts(num_nodes)

    num_possible = int(comb(num_nodes, 3, repetition=False))
    total_remaining = {i: int(num_possible * motif_props[i]) for i in [3, 2, 1]}

    if model_params['ignore_delta_node_timesteps']:
        # generate same num. timesteps as observed, ignoring delta
        delta = model_params['timestep_delta']
    else:  # i.e., don't ignore delta
        # generate "warm-up" timesteps, using delta
        delta = 0

    # NOTE: still need to use in order to generate extra timesteps
    num_timesteps += model_params['timestep_delta']

    prev_nodes = []
    stashed = []
    for t in range(num_timesteps):
        logging.info(f'Generate timestep t={t + 1} - - - - - - - - - - - - - - *')
        # refactor baseline_cht
        # if t in active_nodes:  # check for new node arrivals
        #     new_nodes = list(set(active_nodes[t]) - set(prev_nodes))
        # else:
        #     new_nodes = []
        new_nodes = list(set(active_nodes[t]) - set(prev_nodes))
        num_new_nodes = len(new_nodes)
        logging.info(f'num. new nodes = {num_new_nodes:,}')
        if num_new_nodes > 0:
            gen_dir_t = os.path.join(gen_data_dir, f't_{t + 1}')
            check_dir_exists(gen_dir_t)

            # Sample motifs
            motifs_t_data, role_counts_df = sample_motifs(new_nodes,
                                                          prev_nodes,
                                                          node_map,
                                                          motif_props,
                                                          total_remaining,
                                                          motifs_cnn,
                                                          role_counts_df,
                                                          roles_meta_model,
                                                          rates_distribution,
                                                          t,
                                                          delta,
                                                          num_timesteps,
                                                          stashed,
                                                          model_params,
                                                          gen_data_dir)

            if motifs_t_data['num_sampled_t']:
                total_remaining = {i: total_remaining[i] - motifs_t_data['num_sampled_t'][i]
                                   for i in [3, 2, 1]}

            # Parse data
            motifs_t_show = motifs_t_data['motifs_t_show']
            motif_types_t = motifs_t_data['motif_types_t']
            motif_timesteps_t = motifs_t_data['motif_timesteps_t']
            motif_rates_t = motifs_t_data['motif_rates_t']
            roles_t = motifs_t_data['roles_motifs']
            stashed_t = motifs_t_data['stashed_t']

            # Get motif edges
            motif_edges_t, roles_assigned_counts_df = get_motif_edges(motifs_t_show, motif_types_t, roles_t,
                                                                      roles_assigned_counts_df,
                                                                      t, gen_data_dir)

            # Update
            motifs = motifs | motifs_t_show  # NOTE: not saving motifs that don't show
            motifs_show = motifs_show | motifs_t_show
            motif_types = {**motif_types, **motif_types_t}
            if model_params['stash_unsampled']:
                stashed = stashed_t
            motif_timesteps = {**motif_timesteps, **motif_timesteps_t}
            motif_rates = {**motif_rates, **motif_rates_t}
            motif_edges = {**motif_edges, **motif_edges_t}

            # Next round
            prev_nodes = active_nodes[t]

    total_time = time_code('')
    logging.info(f'Finished generating graph! {total_time}')

    # Create dictionary with the generated graph data
    save_roles_assigned_counts(roles_assigned_counts_df, gen_data_dir)

    gen_data = {'num_nodes': num_nodes,
                'active_nodes': active_nodes,
                'motifs': motifs,
                'motifs_show': motifs_show,
                'motif_types': motif_types,
                'motif_edges': motif_edges,
                'num_timesteps': num_timesteps,
                'motif_timesteps': motif_timesteps,
                'motif_rates': motif_rates}

    return gen_data


def get_generated_graph_data(gen_data_dir: Union[str, os.PathLike],
                             model_params: Dict[str, Any]
                             ) -> Dict[str, Any]:
    """
    Generate graph data.

    :param gen_data_dir: generated graph data directory
    :param num_timesteps: number of timesteps to generate
    :param model_params: model parameters
    :return: generated graph data
    """
    time_code = log_time()

    gen_data_file = os.path.join(gen_data_dir, 'gen_data.msg')

    if not file_exists(gen_data_file):
        logging.info('Get model parameters')

        num_nodes = model_params['num_nodes']
        num_timesteps = model_params['num_timesteps']
        length_timestep = model_params['length_timestep']

        # motif type proportions
        motif_props = read_param(model_params['motif_proportions_file'])

        # Get node map
        node_map = read_node_map(model_params['node_map_file'])

        # Get node role probabilities
        role_distr_df, role_counts_df = helper_get_node_roles_distr(model_params)

        # Get inter-arrival rates
        motif_type_rates_distribution = read_param(model_params['motif_interarrivals_file'])

        # Get active nodes
        num_node_timesteps = len(model_params['timesteps_nodes'])
        active_nodes = get_active_nodes(num_node_timesteps, length_timestep, num_nodes, gen_data_dir, model_params)

        # motif types CNN
        motifs_cnn = helper_get_motif_types_model(model_params['motif_types_model'])

        # node roles meta-model
        roles_meta_model = helper_get_roles_meta_model(model_params['node_role_embeddings_meta_model'])

        # Generate dynamic graph
        gen_data = generate_dynamic_graph(num_timesteps,
                                          num_nodes,
                                          active_nodes,
                                          motif_props,
                                          motifs_cnn,
                                          node_map,
                                          role_counts_df,
                                          roles_meta_model,
                                          motif_type_rates_distribution,
                                          model_params,
                                          gen_data_dir)

        # save filepath for the parameters used
        gen_data['params_file'] = model_params['params_file']

        msgpack_save(gen_data_file, gen_data)
    else:
        logging.info('Read generated data')
        gen_data = msgpack_load(gen_data_file)

    # Generated graph results dict
    gen_graph_file = os.path.join(gen_data_dir, 'generated_graph.msg')
    if not file_exists(gen_graph_file):
        # Graph nodes
        nodes_csv = os.path.join(gen_data_dir, 'nodes.csv')
        if not file_exists(nodes_csv):
            construct_nodes_csv(gen_data['active_nodes'],
                                nodes_csv)

        # Graph edges
        edges_csv = os.path.join(gen_data_dir, 'edges.csv')
        if not file_exists(edges_csv):
            construct_edges_csv(gen_data['motifs_show'],
                                gen_data['motif_edges'],
                                gen_data['motif_timesteps'],
                                edges_csv)

        # Motif content
        content_dir = os.path.join(gen_data_dir, model_params['content_filename'])
        all_motif_embeddings_csv = os.path.join(content_dir, 'all_motif_embeddings.csv')
        if not file_exists(all_motif_embeddings_csv):
            construct_content_csv(content_dir, all_motif_embeddings_csv, gen_data['num_timesteps'])

        gen_graph_data = {'file_data': {'gen_data_dir': gen_data_dir,
                                        'edges_csv': edges_csv,
                                        'nodes_csv': nodes_csv,
                                        'all_motif_embeddings_csv': all_motif_embeddings_csv
                                        },
                          'timesteps': list(range(model_params['timestep_delta'], gen_data['num_timesteps'])),
                          'num_nodes': model_params['num_nodes']}
        msgpack_save(gen_graph_file, gen_graph_data)
        logging.info('Saved graph files')
    else:
        gen_graph_data = msgpack_load(gen_graph_file)

    total_time = time_code('')
    logging.info(f'\nFinished experiment {total_time}\n')

    return gen_graph_data


def construct_nodes_csv(active_nodes: Dict[int, List[int]],
                        nodes_csv: Union[str, os.PathLike]):
    nodes_dict = {'node': [], 'timestep': []}
    for t in active_nodes:
        for v in active_nodes[t]:
            if v not in nodes_dict['node']:
                nodes_dict['node'].append(v)
                nodes_dict['timestep'].append(t)
    nodes_df = pd.DataFrame(nodes_dict)
    write_dataframe(nodes_df, nodes_csv)


def construct_edges_csv(motifs_show: Set[FrozenSet[int]],
                        motif_edges: Dict[FrozenSet[int], List[Tuple[int]]],
                        motif_timesteps: Dict[FrozenSet[int], List[int]],
                        edges_csv: Union[str, os.PathLike]):
    logging.info('Construct edges CSV')
    edges_dict = {'source': [], 'target': [], 'timestep': []}
    for motif in tqdm(motifs_show, desc='Motifs'):
        if motif in motif_edges and motif in motif_timesteps:
            for edge in motif_edges[motif]:
                for t in motif_timesteps[motif]:
                    edges_dict['source'].append(edge[0])
                    edges_dict['target'].append(edge[1])
                    edges_dict['timestep'].append(t)
    edges_df = pd.DataFrame(edges_dict)
    write_dataframe(edges_df, edges_csv)


def construct_content_csv(content_dir: Union[str, os.PathLike],
                          all_motif_embeddings_csv: Union[str, os.PathLike],
                          num_timesteps: int):
    logging.info('Construct content CSV')

    content_CSVs = []
    for t in range(num_timesteps):
        csv_fpath = os.path.join(content_dir, f'motif_content--t_{t + 1}.csv')
        if file_exists(csv_fpath):
            content_CSVs.append(csv_fpath)
        else:
            logging.info(f'File not found:\n{csv_fpath}')
    embedding_df = [read_dataframe(csv_fpath) for csv_fpath in content_CSVs]

    embedding_df = pd.concat(embedding_df, ignore_index=True)
    write_dataframe(embedding_df, all_motif_embeddings_csv)


def get_directories_generated_graph(model_params: Dict[str, Any]) -> Union[str, os.PathLike]:
    """
    Get directories for model parameters and generated graph.

    :param dataset_dir: dataset directory
    :return: directories for model parameters and generated graph
    """
    # refactor: get_directories_generated_graph
    params_dir = model_params['params_dir']
    exp_date = model_params['exp_date']

    # refactor: keep paper version (average) in generate_graph -- node_content
    content_option = model_params['node_content_embedding_option']  # 'average'

    node_timesteps = f'/{model_params["node_rates_distr"]}_node_timesteps'
    if not model_params['ignore_delta_node_timesteps']:  # i.e., don't ignore delta
        node_timesteps += '/use_delta_rev'  # use delta to generate more timesteps

    motif_timesteps = f'/{model_params["motif_rates_distr"]}_motif_timesteps'
    if model_params["motif_rates_distr"] == 'Beta':
        motif_timesteps += f"/use_binomial_{'yes' if model_params['use_binomial'] else 'no'}"

    update_role_embeddings = '/update_role_embeddings' if model_params['update_role_embeddings'] else ''
    use_timesteps = '/use_timesteps' if model_params['use_timesteps'] else ''

    # Directory to save generated graph
    gen_options = f'/DYANE/embedding_{content_option}{update_role_embeddings}' \
                  f'{use_timesteps}{motif_timesteps}{node_timesteps}'
    graph_name = f'{exp_date}'
    gen_data_dir = os.path.join(params_dir,
                                f'generated_graph{gen_options}/{graph_name}')

    logging.info(f'Generated graph dir = {gen_data_dir}')
    check_dir_exists(gen_data_dir)

    return gen_data_dir


def get_parameters(params_dir: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Get model parameters

    :param params_dir: parameters directory
    :return: model parameters
    """
    logging.info('Read model parameters')

    if not os.path.exists(params_dir):
        logging.error(f'Dataset parameters directory not found.\n{params_dir}')
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), params_dir)

    model_params_file = os.path.join(params_dir, 'model_params.msg')
    model_params = msgpack_load(model_params_file)

    return model_params


def dyane_generate_graph(dataset_dir: Union[str, os.PathLike], model_params: Dict[str, Any]
                         ) -> Dict[str, Any]:
    # ) -> Tuple[Union[str, os.PathLike], Union[str, os.PathLike]]:
    """
    Run graph generation

    :param dataset_dir: dataset directory
    :param num_timesteps: number of timesteps to generate
    """
    # Get directories for model parameters and generated graph
    gen_data_dir = get_directories_generated_graph(model_params)

    # # Get model parameters
    # model_params = get_parameters(params_dir)

    # Generate graph
    gen_data_results = get_generated_graph_data(gen_data_dir, model_params)

    return gen_data_results


if __name__ == '__main__':
    import os

    if len(sys.argv) == 3:
        if __package__ is None:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        try:
            dataset_dir = sys.argv[1]
            params_dir = os.path.join(dataset_dir, 'learned_parameters')
            model_params = get_parameters(params_dir)
            dyane_generate_graph(dataset_dir, model_params)
        except Exception as e:
            logging.error('Graph generation failed!')
            raise e
    else:
        logging.error('Required parameters: dataset path and number of timesteps to generate.')
