from bertopic import BERTopic

from DYANE.components.helpers.content_embeddings import get_document_text
from DYANE.components.helpers.file_utils import *
from DYANE.components.helpers.utils import *


def fit_topic_model(obs_content_df: pd.DataFrame,
                    obs_embeddings_df: pd.DataFrame,
                    eval_results_dir: Union[str, os.PathLike],
                    title_col: str = '',
                    text_col: str = '',
                    visualize: bool = False) -> BERTopic:
    # get title and text columns (if not specified)
    if not text_col:
        text_col = [c for c in obs_content_df.columns if 'text' in c][0]
    if not title_col:
        if 'title' in obs_content_df.columns:
            title_col = 'title'

    # get documents (text)
    logging.info("> Format document text")
    docs = get_document_text(obs_content_df, title_col, text_col, sep_token=' \n ')

    # get embeddings
    logging.info("> Convert embeddings df to numpy")
    obs_embeddings = obs_embeddings_df[obs_embeddings_df.columns.drop('contentId')].to_numpy()

    # get topic model
    logging.info("> Fit BERTopic")
    topic_model = BERTopic().fit(documents=docs, embeddings=obs_embeddings)

    if visualize:
        # documents
        logging.info('> Visualize documents')
        viz_docs(topic_model, docs, obs_embeddings, eval_results_dir)
        # topics
        logging.info('> Visualize topics')
        fig = topic_model.visualize_topics()
        fig.write_html(os.path.join(eval_results_dir, 'visualize_topics.html'))

    return topic_model


def viz_docs(topic_model: BERTopic,
             docs: List[Union[str, int]],
             embeddings: npt.NDArray,
             eval_results_dir: Union[str, os.PathLike],
             subset: str = ''):
    fig = topic_model.visualize_documents(docs, embeddings=embeddings,
                                          hide_annotations=True,
                                          hide_document_hover=True)
    fig.write_html(os.path.join(eval_results_dir, f'visualize_documents{subset}.html'))


def get_obs_content_data(observed_data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # (1) Observed content - - - - - - - - - - *
    logging.info('Get observed content')
    # get authored data
    obs_authored_df = read_dataframe(observed_data['file_data']['content_authored_csv'])
    obs_authored_df.rename(columns={'personId:STARTID': 'node', 'paperId:ENDID': 'contentId',
                                    'author': 'node', 'post_id': 'contentId'},
                           inplace=True)
    # get node map
    node_map = read_node_map(observed_data['file_data']['node_map_file'])
    authors = list(node_map)
    # filter any nodes that weren't in training data subset
    obs_authored_df = obs_authored_df[obs_authored_df['node'].isin(authors)]
    # apply node map
    obs_authored_df['node'] = df_apply_node_mapping(obs_authored_df, node_map, ['node'])
    # get content data
    obs_embeddings_df = read_dataframe(observed_data['file_data']['all_content_embeddings_csv'])
    obs_embeddings_df.rename(columns={'paperId:ID': 'contentId',
                                      'post_id': 'contentId'},
                             inplace=True)
    obs_content_df = read_dataframe(observed_data['file_data']['all_content_csv'])

    return obs_authored_df, obs_content_df, obs_embeddings_df


def get_gen_content_data(generated_data: Dict[str, Any],
                         authors_in_obs: List[int] = None) -> Tuple[pd.DataFrame, npt.NDArray]:
    # (2) Generated content - - - - - - - - - - *
    logging.info('Get generated content')
    motif_embeddings_df = read_dataframe(generated_data['file_data']['all_motif_embeddings_csv'])

    # Evaluation timestep filtering
    if not generated_data['ignore_delta_node_timesteps'] and not generated_data['use_all_timesteps_gen']:
        # Use timesteps saved (exclude delta)
        gen_timesteps = generated_data['timesteps']
        # Use timesteps saved (exclude delta)
        motif_embeddings_df = motif_embeddings_df[motif_embeddings_df['timestep'].isin(gen_timesteps)]

    # find authors seen in generated data
    authors_in_gen = np.unique(motif_embeddings_df[['u', 'v', 'w']].values.flatten())
    if len(authors_in_obs) > 0:
        authors_seen = np.intersect1d(authors_in_obs, authors_in_gen, assume_unique=True, return_indices=False)
        authors = authors_seen
        if len(authors_seen) > 0:
            # filter authors
            motif_embeddings_df = motif_embeddings_df[(motif_embeddings_df['u'].isin(authors_seen)) &
                                                      (motif_embeddings_df['v'].isin(authors_seen)) &
                                                      (motif_embeddings_df['w'].isin(authors_seen))]
    else:
        authors = np.array([])  # no authors in common

    return motif_embeddings_df, authors


def get_author_embeddings(authorId: int, obs_authored_df: pd.DataFrame,
                          obs_embeddings_df: pd.DataFrame,
                          motif_embeddings_df: pd.DataFrame):
    # (1) Get content from input graph
    # get author's papers (already sorted by year)
    v_contentIds = obs_authored_df[obs_authored_df['node'] == authorId]['contentId'].tolist()
    # get those papers' embeddings
    observed_emb = obs_embeddings_df[obs_embeddings_df['contentId'].isin(v_contentIds)]
    observed_emb = observed_emb[observed_emb.columns.drop('contentId')].to_numpy()

    # (2) Get content embeddings sampled for an author
    sampled_emb = motif_embeddings_df[(motif_embeddings_df['u'] == authorId) |
                                      (motif_embeddings_df['v'] == authorId) |
                                      (motif_embeddings_df['w'] == authorId)]
    sampled_emb = sampled_emb[sampled_emb.columns.drop(['u', 'v', 'w', 'timestep'])].to_numpy()

    return observed_emb, sampled_emb


def get_author_topics(authorId: int,
                      topic_model: BERTopic,
                      observed_emb: npt.NDArray,
                      sampled_emb: npt.NDArray,
                      eval_results_dir: Union[str, os.PathLike],
                      visualize: bool = False):
    obs_ids = [str(authorId)] * observed_emb.shape[0]
    gen_ids = [str(authorId)] * sampled_emb.shape[0]

    # (3) Get predicted topics
    obs_topics = topic_model.transform(obs_ids, observed_emb)
    gen_topics = topic_model.transform(gen_ids, sampled_emb)

    if visualize:
        viz_docs(topic_model, obs_ids, observed_emb, eval_results_dir, subset=f'-{authorId}-observed')
        viz_docs(topic_model, gen_ids, sampled_emb, eval_results_dir, subset=f'-{authorId}-sampled')

    return obs_topics, gen_topics


def get_topics_data(topic_model: BERTopic,
                    obs_authored_df: pd.DataFrame,
                    obs_embeddings_df: pd.DataFrame,
                    motif_embeddings_df: pd.DataFrame,
                    authors: List[int],
                    eval_results_dir: Union[str, os.PathLike],
                    visualize: bool = False) -> Dict[str, Dict[str, Dict[int, Any]]]:
    # get topics
    topics_data = {'all_topics': topic_model.get_topics(),
                   'author_topics': {'observed': {},
                                     'generated': {}}}

    for authorId in tqdm(authors, desc='Nodes (topics)'):
        # get author's embeddings
        observed_emb, sampled_emb = get_author_embeddings(authorId,
                                                          obs_authored_df,
                                                          obs_embeddings_df,
                                                          motif_embeddings_df)

        obs_topics, gen_topics = get_author_topics(authorId, topic_model,
                                                   observed_emb, sampled_emb,
                                                   eval_results_dir, visualize)

        topics_data['author_topics']['observed'][authorId] = obs_topics
        topics_data['author_topics']['generated'][authorId] = gen_topics

    # save topic model first
    topic_model_file = os.path.join(eval_results_dir, 'topic_model')
    topic_model.save(topic_model_file, save_embedding_model=True)

    return topics_data


def find_topics(observed_data: Dict[str, Any],
                generated_data: Dict[str, Any],
                eval_results_dir: Union[str, os.PathLike],
                title_col: str = '',
                text_col: str = '',
                visualize: bool = False,
                overwrite_results: bool = False,
                tmp_ignore_content: bool = True
                ):
    topics_file = generated_data['file_data']['content_topics_file']
    if (not file_exists(topics_file) or overwrite_results) and not tmp_ignore_content:
        # get observed content and embeddings
        logging.info("Get observed content and embeddings")
        obs_authored_df, obs_content_df, obs_embeddings_df = get_obs_content_data(observed_data)
        authors = np.unique(obs_authored_df['node'].values)

        # get topic model
        logging.info("Fit topic model")
        topic_model = fit_topic_model(obs_content_df, obs_embeddings_df, eval_results_dir,
                                      title_col, text_col, visualize)

        # filter generated content by authors in common
        logging.info("Get generated embeddings")
        # noinspection PyTypeChecker
        motif_embeddings_df, authors = get_gen_content_data(generated_data, authors)
        obs_authored_df = obs_authored_df[obs_authored_df['node'].isin(authors)]
        logging.info(f"> {obs_authored_df.shape = }")

        logging.info("Get topics data")
        topics_data = get_topics_data(topic_model,
                                      obs_authored_df, obs_embeddings_df,
                                      motif_embeddings_df, authors,
                                      eval_results_dir, visualize)

        msgpack_save(topics_file, topics_data)
