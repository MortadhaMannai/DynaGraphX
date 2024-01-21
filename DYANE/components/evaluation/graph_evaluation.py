from scipy.stats import ks_2samp

from DYANE.components.evaluation.attribute_metrics import *
from DYANE.components.evaluation.content_metrics import calc_FBD
from DYANE.components.evaluation.metrics import ks2d2s
from DYANE.components.evaluation.node_metrics import get_node_metrics
from DYANE.components.evaluation.node_metrics import node_metric_names as node_metrics
from DYANE.components.evaluation.structure_metrics import get_structure_metrics
from DYANE.components.evaluation.structure_metrics import metric_names as struct_metrics
from DYANE.components.helpers.file_utils import *
from DYANE.components.helpers.utils import *
from DYANE.components.learning.learning_utils import get_nodes_data, get_edges_data, get_edge_counts, get_timesteps_df

fbd_metric = 'Frechert BERT Distance'
content_metrics_paper = ['CKA: dcorr', fbd_metric]


def get_graph_statistics(edges_df: pd.DataFrame,
                         nodes_df: pd.DataFrame,
                         num_nodes: int,
                         timesteps: List[int],
                         file_data: Dict[str, str],
                         unweighted: bool = False):
    if unweighted:
        structure_metrics_file = file_data['structure_metrics_file-unweighted']
        node_metrics_file = file_data['node_metrics_file-unweighted']
    else:
        structure_metrics_file = file_data['structure_metrics_file']
        node_metrics_file = file_data['node_metrics_file']

    # Structure metrics
    if not file_exists(structure_metrics_file):
        structure_metrics = get_structure_metrics(edges_df, num_nodes, timesteps,
                                                  unweighted=unweighted)
        msgpack_save(structure_metrics_file, structure_metrics)
    else:
        structure_metrics = msgpack_load(structure_metrics_file)

    # Node-aligned metrics
    if not file_exists(node_metrics_file):
        node_metrics = get_node_metrics(edges_df, nodes_df, num_nodes, timesteps,
                                        unweighted=unweighted)
        msgpack_save(node_metrics_file, node_metrics)
    else:
        node_metrics = msgpack_load(node_metrics_file)

    return structure_metrics, node_metrics


def get_edge_df(csv_path):
    edges_df = get_edges_data(csv_path)
    if 'weight' not in edges_df.columns:
        edges_df = get_edge_counts(edges_df)
    return edges_df


def get_node_df(csv_path):
    return get_nodes_data(csv_path)


def compare_embeddings(nodes: List[int],
                       content_authored_df: pd.DataFrame,
                       content_embeddings_df: pd.DataFrame,
                       motif_embeddings_df: pd.DataFrame,
                       ) -> Dict[str, Dict[int, float]]:
    embedding_distances = {metric: dict() for metric in content_metrics_paper}

    for authorId in tqdm(nodes, desc='Nodes (content)'):
        # (1) Get content from input graph
        # get author's papers (already sorted by year)
        v_contentIds = content_authored_df[content_authored_df['node'] == authorId]['contentId'].tolist()
        # get those papers' embeddings
        observed_emb = content_embeddings_df[content_embeddings_df['contentId'].isin(v_contentIds)]
        observed_emb = observed_emb[observed_emb.columns.drop('contentId')].to_numpy()
        # get author's embeddings average and covariance
        observed_avg_emb = np.mean(observed_emb, axis=0)
        observed_cov_emb = np.cov(observed_emb, rowvar=False)

        # (2) Get content embeddings sampled for an author
        sampled_emb = motif_embeddings_df[(motif_embeddings_df['u'] == authorId) |
                                          (motif_embeddings_df['v'] == authorId) |
                                          (motif_embeddings_df['w'] == authorId)]
        sampled_emb = sampled_emb[sampled_emb.columns.drop(['u', 'v', 'w', 'timestep'])].to_numpy()
        # get mean and covariance
        sampled_avg_emb = np.mean(sampled_emb, axis=0)
        sampled_cov_emb = np.cov(sampled_emb, rowvar=False)

        # (3) CKA - - - - - - - - - - - - - - - - - - - *
        # dist correlation
        metric = content_metrics_paper[0]  # 'CKA: dcorr'
        embedding_distances[metric][authorId] = dcorr(observed_emb, sampled_emb)

        # (7) Frechert BERT Distance - - - - - - - - - - - - *
        metric = fbd_metric
        if sampled_emb.shape[0] > 1 and observed_avg_emb.shape[0] > 1:
            embedding_distances[metric][authorId] = calc_FBD(observed_avg_emb, observed_cov_emb,
                                                             sampled_avg_emb, sampled_cov_emb)
        else:
            embedding_distances[metric][authorId] = None

    return embedding_distances


def get_embeddings_distance(observed_data: Dict[str, Any],
                            node_map: pd.Index,
                            generated_data: Dict[str, Any],
                            gen_timesteps: List[int]):
    logging.info('(3) Node Content - - - - - - - - - - *')
    content_metrics_file = generated_data['file_data']['content_metrics_file']
    if not file_exists(content_metrics_file):
        content_avg = {m: None for m in content_metrics_paper}
        content_med = {m: None for m in content_metrics_paper}
        embedding_distances = None

        # (1) Observed content - - - - - - - - - - *
        logging.info('Get observed content')
        # get authored data
        obs_authored_df = read_dataframe(observed_data['file_data']['content_authored_csv'])
        obs_authored_df.rename(columns={'personId:STARTID': 'node', 'paperId:ENDID': 'contentId',
                                        'author': 'node', 'post_id': 'contentId'},
                               inplace=True)
        # get nodes from node map
        authors = list(node_map)
        obs_authored_df = obs_authored_df[obs_authored_df['node'].isin(authors)]
        # apply node map
        obs_authored_df['node'] = df_apply_node_mapping(obs_authored_df, node_map, ['node'])
        # get content data
        obs_embeddings_df = read_dataframe(observed_data['file_data']['all_content_embeddings_csv'])
        # rename Arnetminer column names
        obs_embeddings_df.rename(columns={'paperId:ID': 'contentId',
                                          'post_id': 'contentId'},
                                 inplace=True)

        # (2) Generated content - - - - - - - - - - *
        logging.info('Get generated content')
        motif_embeddings_df = read_dataframe(generated_data['file_data']['all_motif_embeddings_csv'])
        if not generated_data['ignore_delta_node_timesteps'] and not generated_data['use_all_timesteps_gen']:
            # Use timesteps saved (exclude delta)
            motif_embeddings_df = motif_embeddings_df[motif_embeddings_df['timestep'].isin(gen_timesteps)]

        # get authors in common
        authors_in_obs = np.unique(obs_authored_df['node'].values)
        authors_in_gen = np.unique(motif_embeddings_df[['u', 'v', 'w']].values.flatten())
        common_authors = np.intersect1d(authors_in_obs, authors_in_gen, assume_unique=True, return_indices=False)
        del authors_in_obs
        del authors_in_gen

        if len(common_authors) > 0:
            logging.info('Get embedding distances')
            embedding_distances = compare_embeddings(common_authors.tolist(),
                                                     obs_authored_df,
                                                     obs_embeddings_df,
                                                     motif_embeddings_df)

            for metric in content_metrics_paper:
                metric_scores = [embedding_distances[metric][node] for node in embedding_distances[metric]
                                 if embedding_distances[metric][node] is not None]
                # average distance
                avg_metric = np.mean(metric_scores)
                content_avg[metric] = avg_metric
                logging.info(f'Avg. {metric} = {avg_metric:.2f}')
                # median distance
                med_metric = np.median(metric_scores)
                content_med[metric] = med_metric
                logging.info(f'Med. {metric} = {med_metric:.2f}')

        content_eval = {'averages': content_avg, 'medians': content_med, 'distance': embedding_distances}
        msgpack_save(content_metrics_file, content_eval)

    else:
        content_eval = msgpack_load(content_metrics_file)
    return content_eval


def get_obs_metrics(observed_data: Dict[str, Any],
                    node_map: pd.Index,
                    unweighted: bool = False):
    logging.info("Get observed network's statistics")

    # Get observed network's edges
    obs_edges_df = get_edge_df(observed_data['file_data']['edges_csv'])

    # Get observed network's nodes
    obs_nodes_df = get_node_df(observed_data['file_data']['nodes_csv'])

    # TODO: try with loc to suppress SettingWithCopyWarning:
    # (1) Edges
    node_cols = ['source', 'target']
    # Apply node map to node id's
    obs_edges_df[node_cols] = df_apply_node_mapping(obs_edges_df, node_map, cols=node_cols)

    # (2) Nodes
    node_cols = ['node']
    # Apply node map to node id's
    obs_nodes_df[node_cols] = df_apply_node_mapping(obs_nodes_df, node_map, cols=node_cols)

    # (3) Timesteps
    obs_timesteps = get_timesteps_df(obs_edges_df)

    # Get observed network's statistics
    obs_metrics = get_graph_statistics(obs_edges_df,
                                       obs_nodes_df,
                                       obs_nodes_df.shape[0],
                                       obs_timesteps,
                                       observed_data['file_data'],
                                       unweighted=unweighted)
    return obs_metrics


def get_gen_metrics(generated_data: Dict[str, Any],
                    unweighted: bool = False):
    logging.info("Get generated network's statistics")

    # Get generated network's edges
    gen_edges_df = get_edge_df(generated_data['file_data']['edges_csv'])

    # Get observed network's nodes
    gen_nodes_df = get_node_df(generated_data['file_data']['nodes_csv'])

    # Get timesteps
    if not generated_data['ignore_delta_node_timesteps'] and generated_data['use_all_timesteps_gen']:
        # Use all timesteps generated (include delta)
        gen_timesteps = get_timesteps_df(gen_edges_df)
    else:
        # Use timesteps saved (exclude delta)
        gen_timesteps = generated_data['timesteps']

    # Get generated network's statistics
    gen_metrics = get_graph_statistics(gen_edges_df,
                                       gen_nodes_df,
                                       generated_data['num_nodes'],
                                       gen_timesteps,
                                       generated_data['file_data'],
                                       unweighted=unweighted)
    return gen_metrics, gen_timesteps


def get_ks_structure(obs_structure_metrics: Dict[str, List[Union[int, float]]],
                     gen_structure_metrics: Dict[str, List[Union[int, float]]]):
    logging.info('(1) Graph Structure - - - - - - - - - *')
    struct_ks_avg = []
    struct_ks_all = {m: [] for m in struct_metrics}
    for metric in struct_metrics:
        obs_data = [d for d in obs_structure_metrics[metric]
                    if not np.isnan(d)]
        gen_data = [d for d in gen_structure_metrics[metric]
                    if not np.isnan(d)]

        dist, pval = ks_2samp(obs_data, gen_data)

        struct_ks_avg.append(dist)
        struct_ks_all[metric] = dist
        logging.info(f'KS {metric} = {dist:.2f}')
    logging.info(f'Structure Avg. KS = {np.mean(struct_ks_avg):.2f} ***')
    return struct_ks_avg, struct_ks_all


def get_ks_nodes(obs_node_metrics: Dict[str, Dict[int, List[Union[int, float]]]],
                 gen_node_metrics: Dict[str, Dict[int, List[Union[int, float]]]]):
    logging.info('(2) Node Behavior - - - - - - - - - - *')
    node_ks_avg = []
    node_ks_all = {m: [] for m in node_metrics}
    for metric in node_metrics:
        if metric == 'activity rate':
            obs_data = [obs_node_metrics[metric][node] for node in obs_node_metrics[metric]
                        if not np.isnan(obs_node_metrics[metric][node])]
            gen_data = [gen_node_metrics[metric][node] for node in gen_node_metrics[metric]
                        if not np.isnan(gen_node_metrics[metric][node])]

            # KS test
            dist, pval = ks_2samp(obs_data, gen_data)

            # Save results
            node_ks_avg.append(dist)
            node_ks_all[metric] = dist
        else:
            # IQR observed data
            obs_metric = {node: [val for val in obs_node_metrics[metric][node]
                                 if not np.isnan(val)]
                          for node in obs_node_metrics[metric]}
            obs_25 = [np.percentile(obs_metric[node], 25, interpolation='midpoint')
                      for node in obs_metric
                      if len(obs_metric[node]) > 0]
            obs_75 = [np.percentile(obs_node_metrics[metric][node], 75, interpolation='midpoint')
                      for node in obs_node_metrics[metric]
                      if len(obs_metric[node]) > 0]

            # IQR generated data
            gen_metric = {node: [val for val in gen_node_metrics[metric][node]
                                 if not np.isnan(val)]
                          for node in gen_node_metrics[metric]}
            gen_25 = [np.percentile(gen_metric[node], 25, interpolation='midpoint')
                      for node in gen_metric
                      if len(gen_metric[node]) > 0]
            gen_75 = [np.percentile(gen_metric[node], 75, interpolation='midpoint')
                      for node in gen_metric
                      if len(gen_metric[node]) > 0]

            # sanity check for NaN values
            arrs = [gen_25, gen_75, obs_25, obs_75]
            sanity_check = []
            for a in arrs:
                sanity_check.append(len(a) == 0)
                sanity_check.append(any(np.isnan(a)))
                sanity_check.append(any(np.isinf(a)))

            if not any(sanity_check):
                # KS test
                pval, dist = ks2d2s(np.array(gen_25), np.array(gen_75),
                                    np.array(obs_25), np.array(obs_75),
                                    extra=True)
                # Save results
                node_ks_avg.append(dist)
                node_ks_all[metric] = dist
            else:
                dist = None
                logging.warning(f'Could not calculate KS for {metric} !!!')

        if dist:
            logging.info(f'KS {metric} = {dist:.2f}')
    logging.info(f'Node Avg. KS = {np.mean(node_ks_avg):.2f} ***')
    return node_ks_avg, node_ks_all


def compare_attributed_dynamic_networks(observed_data: Dict[str, Any],
                                        generated_data: Dict[str, Any],
                                        param_files: Union[Dict[str, Any], None] = None,
                                        unweighted: bool = False,
                                        eval_content: bool = True,
                                        overwrite_results: bool = False,
                                        tmp_ignore_content: bool = True
                                        ) -> Union[str, os.PathLike]:
    logging.info('Evaluate generated network results')

    eval_results_dir = generated_data['file_data']['eval_results_dir']
    full_eval_version = generated_data['file_data']['full_eval_version']
    if unweighted:
        eval_results_file = os.path.join(eval_results_dir, f'eval_results{full_eval_version}-unweighted.msg')
        weight_flag = 'unweighted'
    else:
        eval_results_file = os.path.join(eval_results_dir, f'eval_results{full_eval_version}.msg')
        weight_flag = 'weighted'

    if not file_exists(eval_results_file) or overwrite_results:
        # Observed network - - - - - - - - - - - - - - - - - - - - - - - - - - *
        # Get node map
        node_map = read_node_map(observed_data['file_data']['node_map_file'])
        # Get metrics
        obs_structure_metrics, obs_node_metrics = get_obs_metrics(observed_data, node_map, unweighted)

        # Generated network - - - - - - - - - - - - - - - - - - - - - - - - - - *
        # Get metrics and timesteps
        (gen_structure_metrics, gen_node_metrics), gen_timesteps = get_gen_metrics(generated_data, unweighted)

        logging.info(f'Comparing networks ({weight_flag}) - - - - - - - - - - - - - - - - - - - - - - *')
        # (1) Structure comparison, incl. KS-test
        struct_ks_avg, struct_ks_all = get_ks_structure(obs_structure_metrics, gen_structure_metrics)

        # (2) Node comparison, incl. 2-dim KS-test
        node_ks_avg, node_ks_all = get_ks_nodes(obs_node_metrics, gen_node_metrics)

        # (3) Content comparison (embeddings)
        if eval_content and not tmp_ignore_content:
            content_eval = get_embeddings_distance(observed_data,
                                                   node_map,
                                                   generated_data,
                                                   gen_timesteps)
            content_avg = content_eval['averages']
            content_med = content_eval['medians']
            content_dist = content_eval['distance']
        else:
            content_avg = None
            content_med = None
            content_dist = None

        # Dict[str, Dict[str, Union[None, float, Dict[str, List[Union[float, int]]]]]]
        eval_results = {'structure': {'average': np.mean(struct_ks_avg), 'all': struct_ks_all},
                        'node-aligned': {'average': np.mean(node_ks_avg), 'all': node_ks_all},
                        'content': {'averages': content_avg, 'medians': content_med, 'distance': content_dist}}

        msgpack_save(eval_results_file, eval_results)

    return eval_results_file
