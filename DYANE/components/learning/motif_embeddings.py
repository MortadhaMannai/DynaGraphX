import logging
from typing import FrozenSet, List, Set, Type

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from DYANE.components.helpers.file_utils import *
from DYANE.components.helpers.utils import get_sparse_adjacency_matrix, read_dataframe, write_dataframe, \
    read_node_map, df_apply_node_mapping
from DYANE.components.learning.learning_utils import roles, subsample_motifs, get_nodes_data, calc_node_roles_distr, \
    get_edges_t, get_some_empty_examples
from DYANE.components.motif_types.cnn import learn_cnn
from DYANE.components.motif_types.data.process_motifs_data import get_motif_types_dataset
from DYANE.components.node_roles.data.process_roles_data import get_node_roles_dataset
from DYANE.components.node_roles.gcn import inference_gcn
from DYANE.components.node_roles.gcn import learn_gcn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_node_role_embeddings(edges_df: pd.DataFrame,
                               nodes_df: pd.DataFrame,
                               num_nodes: int,
                               timesteps: List[int],
                               param_files: Dict[str, Any]):
    logging.info("Get node role embeddings")
    role_embeddings_df = None  # Initialize

    # check if node_embeddings_csv file already created
    role_embeddings_csv = param_files['node_role_embeddings_csv']
    if not file_exists(role_embeddings_csv):
        # load dataset (need for inference() anyway, to make the csv)
        # get node roles gcn model parameters
        params_fpath = param_files['node_roles_gcn_params_file']
        if not file_exists(params_fpath):
            data_params = get_params_node_roles_gcn(edges_df, nodes_df, num_nodes, timesteps, param_files)
            logging.info('(get parameters done)')
            msgpack_save(params_fpath, data_params)
        else:
            data_params = msgpack_load(params_fpath)

        num_cv_folds = 1
        pytorch_data_dir = os.path.join(param_files['pytorch_dir'],
                                        f'node_roles/gcn/{num_cv_folds}-fold')
        check_dir_exists(pytorch_data_dir)
        roles_dataset = get_node_roles_dataset(pytorch_data_dir, data_params, num_cv_folds)

        # model version
        loss = 'categorical_loss'
        num_epochs = 200

        # check if model already saved (don't learn again)
        version = f'{loss}-gcn-epochs_{num_epochs}'
        model_fpath = os.path.join(pytorch_data_dir, f'gcn_model-{version}.pt')

        if not file_exists(model_fpath):
            gcn_model, eval_results = learn_gcn(roles_dataset, loss, num_cv_folds, percent_train_idx=0,
                                                num_epochs=num_epochs)
            torch.save(gcn_model, model_fpath)
        else:
            gcn_model = torch.load(model_fpath)

        # get embeddings
        node_role_embeddings = inference_gcn(gcn_model, roles_dataset)
        role_embeddings_df = build_node_role_embeddings_df(node_role_embeddings, nodes_df['node'].to_list())
        # save to csv
        write_dataframe(role_embeddings_df, role_embeddings_csv)

    # check if metamodel already learned
    if not file_exists(param_files['node_role_embeddings_meta_model']):
        logging.info("Train meta-model for node role embeddings")

        # get node role embeddings
        if role_embeddings_df is None:
            role_embeddings_df = read_dataframe(role_embeddings_csv)

        # get node role counts
        counts_df = read_dataframe(param_files['node_role_counts_csv'])

        X = counts_df[roles]  # input features
        y = role_embeddings_df[roles]  # target variable (embedding vector)

        score_threshold = 0.75
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
        logging.info('(1) Ridge Regression (RR)')
        # fit model, parameter search with cross-validation
        rr_model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
        rr_model.fit(X, y)
        score = rr_model.score(X, y)
        logging.info(f'Model score: {score:.3f}')

        # save model
        if score >= score_threshold:
            logging.info(f'Saving model (score >= {score_threshold})')
            pickle_save(param_files['node_role_embeddings_meta_model'], rr_model)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
        logging.info('(2) Kernel Ridge Regression (KRR)')
        krr_model = GridSearchCV(
            KernelRidge(kernel="rbf", gamma=0.1),
            param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)},
        )
        krr_model.fit(X, y)
        logging.info(f'Model score: {krr_model.best_score_:.3f}')

        # save model
        if krr_model.best_score_ >= score_threshold and krr_model.best_score_ > score:
            logging.info(f'Saving model (score >= {score_threshold} and > RR score)')
            pickle_save(param_files['node_role_embeddings_meta_model'], krr_model)

        if krr_model.best_score_ < score_threshold and score < score_threshold:
            logging.warning(f'Meta model for Node Role Embeddings is less than score threshold ({score_threshold})')
            # if neither model is better than threshold, pick the one that sucks less
            if score >= krr_model.best_score_:
                logging.info(f'Saving RR model (score = {score:.3f})')
                pickle_save(param_files['node_role_embeddings_meta_model'], rr_model)
            else:
                logging.info(f'Saving KRR model (score = {krr_model.best_score_:.3f})')
                pickle_save(param_files['node_role_embeddings_meta_model'], krr_model)


def get_params_node_roles_gcn(edges_df: pd.DataFrame,
                              nodes_df: pd.DataFrame,
                              num_nodes: int,
                              timesteps: List[int],
                              param_files: Dict[str, Any]) -> Dict[str, Any]:
    logging.info("Get parameters for node roles GCN")
    adj_data = {}
    for t in timesteps:
        edges_t_df = pd.DataFrame(get_edges_t(edges_df, t))
        adj_data[t] = get_sparse_adjacency_matrix(edges_t_df, num_nodes)

    nodes = nodes_df['node'].to_list()  # NOTE: already sorted by node
    roles_df = read_dataframe(param_files['node_role_distr_csv'])
    node_labels = roles_df[roles].to_numpy()  # already in same order as node_map

    node_labels_timesteps = []
    timestep_csvs = param_files['node_role_counts_timestep_csvs']
    for t_idx, t in tqdm(enumerate(timesteps), desc='Timestep labels'):
        # read counts
        roles_df = read_dataframe(timestep_csvs[t_idx])
        # calc. probabilities
        roles_df = calc_node_roles_distr(roles_df)
        # append to labels per timestep
        node_labels_timesteps.append(roles_df[roles].to_numpy())

    data_params = {'roles': roles,
                   'adj_data': adj_data,
                   'timesteps': timesteps,
                   'node_labels': node_labels,
                   'node_labels_timesteps': node_labels_timesteps,
                   'nodes': nodes,
                   'gcn_version': param_files['gcn_version']
                   }
    return data_params


def train_motif_types_model(
        sample_size: int,
        param_files: Dict[str, Any]) -> Union[str, os.PathLike]:
    logging.info("Get motif type model")
    # OPTIMIZE: pass by params
    num_cv_folds = 1
    test_prop = 0.2
    version = f'training_balanced_{sample_size * 4}-validation_{test_prop}'
    num_epochs = 300

    pytorch_data_dir = os.path.join(param_files['pytorch_dir'],
                                    f"motif_types/{param_files['node_content_embedding_option']}/{num_cv_folds}-fold")
    check_dir_exists(pytorch_data_dir)

    # refactor motifs_model_name / cnn_model_path ?
    model_name = param_files['motifs_model_name']
    cnn_model_path = os.path.join(pytorch_data_dir,
                                  f'{model_name}-version_{version}-epochs_{num_epochs}.pt')

    # check if model already saved (don't learn again)
    if not file_exists(cnn_model_path):
        model_data_params = {'role_embeddings_csv': param_files['node_role_embeddings_csv'],
                             'content_embeddings_csv': param_files['node_avg_content_embeddings_csv'],
                             'node_embeddings_csv': param_files['node_embeddings_csv'],
                             'node_map_file': param_files['node_map_file'],
                             'labeled_data_csv': param_files['motif_type_training_csv'],
                             'test_prop': test_prop,
                             'version': version}

        # get dataset for training
        calc_node_avg_var_content_embeddings(param_files)  # added to learn_parameters
        create_node_embeddings(param_files)  # concats roles & avg. content embeddings
        motif_dataset, node_data = get_motif_types_dataset(pytorch_data_dir, model_data_params, num_cv_folds)

        # learn model
        logging.info("Train motif type CNN")
        model, results = learn_cnn(model_name, motif_dataset, node_data.embeddings, num_epochs, pytorch_data_dir)

        # save model
        torch.save(model, cnn_model_path)

    return cnn_model_path


def get_motif_type_training_data(motifs: Set[FrozenSet[int]],
                                 motif_types: Dict[FrozenSet[int], int],
                                 nodes_df: pd.DataFrame,
                                 num_motifs: Dict[int, int],
                                 sample_size: int,  # 100k each
                                 params_data: Dict[str, Any]
                                 ) -> Union[str, os.PathLike]:
    logging.info('Get motif type training data')
    sample_types = [3, 2, 1, 0]
    filter_cnn = params_data['filter_cnn_label']
    training_csv = os.path.join(params_data['params_dir'],
                                f'motif_types-training_sample{filter_cnn}-balanced_{sample_size * len(sample_types)}.csv')
    # OPTIMIZE: get motif counts
    counts = {i: 0 for i in sample_types}
    for m in motif_types:
        counts[motif_types[m]] += 1

    len_samples = {i: min([sample_size, counts[i]]) for i in sample_types}
    len_samples[0] = sample_size
    total_sample = sum([len_samples[i] for i in sample_types])

    if not file_exists(training_csv):
        subsets = []
        # make 300k balanced dataset
        for i in sample_types:
            logging.info(f'Sampling type i={i}')
            sampled_file = os.path.join(params_data['tmp_files_dir'], f"sample_i_{i}.msg")
            if not file_exists(sampled_file):
                if i > 0:
                    sample_i = [sorted(list(m)) for m in motifs if motif_types[m] == i]
                    sample_i = subsample_motifs(sample_i, num_motifs[i], len_samples[i])
                else:
                    # add empty examples
                    sample_i = np.array(get_some_empty_examples(motifs, nodes_df, sample_size))

                msgpack_save(sampled_file, sample_i)
            else:
                sample_i = msgpack_load(sampled_file)

            # logging.info(f"sample_{i}.shape = {sample_i.shape}")
            if sample_i.shape[0] > 0 and sample_i.shape[1] == 3:
                # make df for motifs type i sampled
                df = pd.DataFrame({'u': sample_i[:, 0], 'v': sample_i[:, 1], 'w': sample_i[:, 2]})

                df['type'] = i
                df['weight'] = total_sample / len_samples[i]
                subsets.append(df)
            else:
                logging.error(f'Could not sample any motifs of type i={i} for training CSV!!!')

        # create dataframe with all motif type samples
        training_df = pd.concat(subsets, ignore_index=True)

        write_dataframe(training_df, training_csv)

    return training_csv


def create_node_embeddings(param_files: Dict[str, Any]):
    logging.info("Get node embeddings (roles + content)")
    if not file_exists(param_files['node_embeddings_csv']):
        # get role embeddings
        role_embeddings_df = read_dataframe(param_files['node_role_embeddings_csv'])
        # get content embeddings
        content_embeddings_df = get_content_embeddings(param_files)
        # join embeddings
        node_embeddings_df = build_node_embeddings(role_embeddings_df, content_embeddings_df, param_files)
        # save joined embeddings
        write_dataframe(node_embeddings_df, param_files['node_embeddings_csv'])


def build_node_embeddings(role_embeddings_df: pd.DataFrame,
                          content_embeddings_df: pd.DataFrame,
                          param_files: Dict[str, Any]) -> pd.DataFrame:
    node_map = read_node_map(param_files['node_map_file'])
    map_col = ['node']
    content_embeddings_df[map_col] = df_apply_node_mapping(content_embeddings_df, node_map, map_col)
    return pd.merge(role_embeddings_df, content_embeddings_df,
                    how='left', on='node').sort_values(by=['node']).reset_index(drop=True)


def build_node_role_embeddings_df(node_role_embeddings, nodes: List[int]) -> pd.DataFrame:
    # save embeddings
    embeddings_df = pd.DataFrame(node_role_embeddings).astype("float")
    embeddings_df['node'] = nodes
    embeddings_df.columns = roles + ['node']
    # re-order columns
    embeddings_df = embeddings_df[['node'] + roles]
    return embeddings_df


def get_content_embeddings(param_files: Dict[str, Any]) -> pd.DataFrame:
    # refactor: keep version paper experiments (average)
    # get content embeddings
    if param_files['node_content_embedding_option'] == 'average':  # DYANE paper version
        content_embeddings_df = read_dataframe(param_files['node_avg_content_embeddings_csv'])
    elif param_files['node_content_embedding_option'] == 'center':
        content_embeddings_df = read_dataframe(param_files['node_ctr_content_embeddings_csv'])
    else:
        content_embeddings_df = read_dataframe(param_files['node_content_embed_all_csv'])
    return content_embeddings_df


def update_node_embeddings(meta_model: Union[RidgeCV, KernelRidge],
                           role_counts_df: pd.DataFrame,
                           motifs_cnn: Type[torch.nn.Module],
                           param_files: Dict[str, Any]):
    # get predicted embeddings from meta-model
    y_pred = meta_model.predict(role_counts_df[roles])
    role_embeddings_df = build_node_role_embeddings_df(y_pred, list(role_counts_df.index.values))

    # get content embeddings
    content_embeddings_df = get_content_embeddings(param_files)

    # join embeddings
    node_embeddings_df = build_node_embeddings(role_embeddings_df, content_embeddings_df, param_files)

    # update embeddings for CNN model
    embeddings = torch.from_numpy(node_embeddings_df.drop(columns=['node']).to_numpy())

    new_embeddings = embeddings.clone().detach().cpu().numpy()  # DEBUG

    motifs_cnn.update_embeddings(embeddings.to(device))


def calc_node_avg_var_content_embeddings(param_files: Dict[str, Any]):
    logging.info("Get mean and variance for nodes' content embeddings")
    variances_csv = param_files['node_var_content_embeddings_csv']
    averages_csv = param_files['node_avg_content_embeddings_csv']
    centers_csv = param_files['node_ctr_content_embeddings_csv']

    if not file_exists(variances_csv) or not file_exists(averages_csv):
        # get nodes
        nodes = get_nodes_data(param_files['nodes_csv'])['node'].tolist()

        # get authored data
        content_authored_df = read_dataframe(param_files['content_authored_csv'])
        cols = list(content_authored_df.columns)
        if 'personId:STARTID' in cols and 'paperId:ENDID' in cols:
            content_authored_df.rename(columns={'personId:STARTID': 'node',
                                                'paperId:ENDID': 'contentId'},
                                       inplace=True)

        # get content data
        content_embeddings_df = read_dataframe(param_files['all_content_embeddings_csv'])
        cols = list(content_embeddings_df.columns)
        if 'paperId:ENDID' in cols:
            content_embeddings_df.rename(columns={'paperId:ID': 'contentId'}, inplace=True)
        elif 'post_id' in cols:
            content_embeddings_df.rename(columns={'post_id': 'contentId'}, inplace=True)

        variances_df = []
        averages_df = []
        centers_df = []

        for v in tqdm(nodes, desc="Nodes"):
            # get author's papers (already sorted by year)
            v_contentIds = content_authored_df[content_authored_df['node'] == v]['contentId'].tolist()

            # get those papers' embeddings
            v_embeddings_df = content_embeddings_df[content_embeddings_df['contentId'].isin(v_contentIds)]

            # convert embeddings back to tensors
            v_content_tensors = torch.from_numpy(v_embeddings_df[v_embeddings_df.columns.drop('contentId')].to_numpy())

            # get embedding mean and variance
            v_df = pd.DataFrame({'node': [v]})
            var, mean = torch.var_mean(v_content_tensors, dim=0, unbiased=True)
            v_var_df = pd.concat([v_df, pd.DataFrame([var.numpy()])], axis=1)
            v_avg_df = pd.concat([v_df, pd.DataFrame([mean.numpy()])], axis=1)
            variances_df.append(v_var_df)
            averages_df.append(v_avg_df)

            # get embedding closest to centroid
            avg_emb = mean.numpy()
            embs = v_content_tensors.numpy()
            distances = [cosine(avg_emb, e) for e in embs]
            ctr = embs[np.argmin(distances)]

            v_ctr_df = pd.concat([v_df, pd.DataFrame([ctr])], axis=1)
            centers_df.append(v_ctr_df)

        variances_df = pd.concat(variances_df, ignore_index=True)
        averages_df = pd.concat(averages_df, ignore_index=True)
        centers_df = pd.concat(centers_df, ignore_index=True)

        write_dataframe(variances_df, variances_csv)
        write_dataframe(averages_df, averages_csv)
        write_dataframe(centers_df, centers_csv)
