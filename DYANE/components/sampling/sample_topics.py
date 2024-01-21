from DYANE.components.sampling.sampling_utils import *


def sample_motif_topics(motifs: Set[FrozenSet[int]],
                        motif_roles: Dict[FrozenSet[int], Dict[int, str]],
                        motif_timesteps: Dict[FrozenSet[int], List[int]],
                        active_nodes_t: List[int],
                        node_map: pd.Index,
                        t: int,
                        content_t_file: str,
                        params_data: Dict[str, Any]):
    logging.info(f'Sample motif topics t={t + 1}')
    if not file_exists(content_t_file):
        average_df = get_node_avg_content_embeddings(params_data, node_map, active_nodes_t)
        variance_df = get_node_var_content_embeddings(params_data, node_map, active_nodes_t)

        use_seed = params_data['topics']['use_seed']
        new_seed = params_data['topics']['new_seed']
        average_cov = params_data['topics']['average_cov']

        motif_topics_df = []
        for motif in motifs:
            if len(motif_timesteps[motif]) > 0:
                init_embedding, seed_node = initialize_topics(motif, motif_roles[motif],
                                                              average_df,
                                                              use_seed)

                motif_embeddings = []
                for t in motif_timesteps[motif]:
                    if new_seed:
                        seed_node = None
                    embedding = sample_new_embedding(motif, motif_roles, init_embedding, variance_df,
                                                     use_seed, seed_node, average_cov)
                    motif_embeddings.append(embedding.numpy())

                # add motif embeddings to list
                u, v, w = sorted(list(motif))
                num_t = len(motif_timesteps[motif])
                emb_df = pd.DataFrame({'u': [u] * num_t, 'v': [v] * num_t, 'w': [w] * num_t})
                emb_df['timestep'] = motif_timesteps[motif]
                emb_df = pd.concat([emb_df, pd.DataFrame(np.array(motif_embeddings))], axis=1)
                motif_topics_df.append(emb_df)

        if len(motif_topics_df) > 0:
            # save motif topics for new motifs sampled in timestep t
            if len(motif_topics_df) > 1:
                motif_topics_df = pd.concat(motif_topics_df, ignore_index=True)

            if type(motif_topics_df) is pd.DataFrame:
                write_dataframe(motif_topics_df, content_t_file)
    else:
        logging.info(f'Topics already sampled!')


def initialize_topics(motif: FrozenSet[int],
                      roles_motif: Dict[int, str],
                      node_embeddings_df: pd.DataFrame,
                      use_seed=False) -> Tuple[torch.Tensor, int]:
    # get initial embedding
    if use_seed:  # use a seed node
        init_embedding, seed_node = init_seed_node(motif, roles_motif, node_embeddings_df)
    else:  # average the embeddings
        init_embedding = init_average_triplet_embeddings(motif, roles_motif, node_embeddings_df)
        seed_node = None
    return init_embedding, seed_node


def init_average_triplet_embeddings(motif: FrozenSet[int],
                                    node_roles_motif: Dict[int, str],
                                    node_embeddings_df: pd.DataFrame) -> torch.Tensor:
    # filter outlier nodes and double chance to hub nodes
    nodes = helper_get_node_influencers(motif, node_roles_motif)
    # get the nodes' embeddings
    embeddings_df = helper_filter_node_embeddings(node_embeddings_df, nodes)
    # create torch tensors
    node_tensors = torch.from_numpy(embeddings_df[embeddings_df.columns.drop(['node'])].to_numpy())
    # average tensors and return embedding
    return node_tensors.mean(dim=0)


def init_seed_node(motif: FrozenSet[int],
                   node_roles_motif: Dict[int, str],
                   node_embeddings_df: pd.DataFrame) -> Tuple[torch.Tensor, int]:
    # filter outlier nodes and double chance to hub nodes
    nodes = helper_get_node_influencers(motif, node_roles_motif)
    # pick node to influence content
    seed_node = np.random.choice(nodes)
    # get the node's embedding
    embeddings_df = helper_filter_node_embeddings(node_embeddings_df, [seed_node])
    # get embedding numpy
    embedding = torch.from_numpy(embeddings_df[embeddings_df.columns.drop(['node'])].to_numpy())
    return embedding, seed_node


def sample_new_embedding(motif: FrozenSet[int],
                         roles_motif: Dict[FrozenSet[int], Dict[int, str]],
                         init_embedding: torch.Tensor,
                         variance_df: pd.DataFrame,
                         use_seed=False, seed_node=None,
                         average_cov=False) -> torch.Tensor:
    # refactor: keep version paper
    if use_seed:
        embedding = perturb_from_seed_node(motif, roles_motif[motif],
                                           init_embedding, variance_df,
                                           seed_node)
    else:  # DYANE paper version
        embedding = perturb_from_edge_nodes(motif, roles_motif[motif],
                                            init_embedding, variance_df,
                                            average_cov)
    return embedding


def perturb_from_seed_node(motif: FrozenSet[int],
                           node_roles_motif: Dict[int, str],
                           init_embedding: torch.Tensor,
                           variance_df: pd.DataFrame,
                           seed_node=None) -> torch.Tensor:
    if not seed_node:
        # filter outlier nodes and double chance to hub nodes
        nodes = helper_get_node_influencers(motif, node_roles_motif)
        # pick node to influence content
        seed_node = np.random.choice(nodes)
    # perturb embedding
    var_df = variance_df[variance_df['node'] == seed_node]
    var_tensor = process_df_to_tensor(var_df, ['node'])
    embedding = add_gaussian_noise(init_embedding, var_tensor)
    return embedding


def perturb_from_edge_nodes(motif: FrozenSet[int],
                            node_roles_motif: Dict[int, str],
                            init_embedding: torch.Tensor,
                            variance_df: pd.DataFrame,
                            average_cov=True) -> torch.Tensor:
    # filter out outlier node, if any
    edge_nodes = list(set(helper_get_node_influencers(motif, node_roles_motif)))
    if average_cov:  # average covariances & perturb embedding
        var_df = variance_df[variance_df['node'].isin(edge_nodes)]
        var_tensor = process_df_to_tensor(var_df, ['node'])
        var_tensor = var_tensor.mean(dim=0)
        embedding = add_gaussian_noise(init_embedding, var_tensor)
    else:  # perturb & average embeddings
        embeddings = []
        for node in edge_nodes:
            var_df = variance_df[variance_df['node'] == node]
            var_tensor = process_df_to_tensor(var_df, ['node'])
            embeddings.append(add_gaussian_noise(init_embedding, var_tensor))
        embedding = torch.cat(embeddings).mean(dim=0)
    return embedding


def add_gaussian_noise(embedding: torch.Tensor,
                       var: torch.Tensor) -> torch.Tensor:
    # calculate noise for perturbing
    mean = torch.zeros_like(embedding)
    std = torch.sqrt(var)
    std = torch.nan_to_num(std)
    noise = torch.normal(mean, std)
    # return the perturbed embedding
    return embedding + noise
