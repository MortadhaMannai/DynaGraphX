import logging
from typing import List
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, Pipeline
from tqdm import tqdm


class EmbeddingPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_args = ['max_length',
                           'stride']

        preprocess_kwargs = {arg: kwargs[arg] for arg in preprocess_args if arg in kwargs}
        preprocess_kwargs = {'truncation': 'longest_first',
                             'return_special_tokens_mask': True,
                             'return_overflowing_tokens': True,
                             'return_offsets_mapping': True,
                             'padding': 'max_length',
                             'return_tensors': 'pt',
                             **preprocess_kwargs}
        forward_kwargs = {}
        postprocess_kwargs = {}
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, text_list, **kwargs):
        # tokenize
        encoded_input = self.tokenizer(text_list, **kwargs)

        # Map from a feat. to its corresponding example.
        if 'overflow_to_sample_mapping' in encoded_input:
            sample_mapping = encoded_input.pop("overflow_to_sample_mapping")
        # Map from token to character position in the original context.
        if 'offset_mapping' in encoded_input:
            offset_mapping = encoded_input.pop("offset_mapping", None)
        if 'special_tokens_mask' in encoded_input:
            special_tokens_mask = encoded_input.pop("special_tokens_mask")

        return encoded_input

    def _forward(self, model_inputs, **kwargs):
        # # Map from a feat. to its corresponding example.
        # if 'overflow_to_sample_mapping' in model_inputs:
        #     sample_mapping = model_inputs.pop("overflow_to_sample_mapping")
        # # Map from token to character position in the original context.
        # if 'offset_mapping' in model_inputs:
        #     offset_mapping = model_inputs.pop("offset_mapping", None)
        # if 'special_tokens_mask' in model_inputs:
        #     special_tokens_mask = model_inputs.pop("special_tokens_mask")

        all_outputs = torch.Tensor().to(device=0)
        num_chunks = len(model_inputs["input_ids"])

        for i in range(num_chunks):
            model_input = {k: torch.unsqueeze(v[i], dim=0) for k, v in model_inputs.items()}
            with torch.no_grad():
                # First element of model_output contains all token embeddings
                outputs = self.model(**model_input)[0]
            # Perform pooling
            outputs = self._mean_pooling(outputs, model_input['attention_mask'])
            # Normalize embeddings
            outputs = F.normalize(outputs, p=2, dim=1)
            # all_outputs = torch.cat((all_outputs, outputs), dim=1)
            all_outputs = torch.cat((all_outputs, outputs), dim=0)

        return all_outputs

    def postprocess(self, model_outputs, **kwargs):
        return torch.mean(model_outputs, dim=0).detach()

    def _mean_pooling(self, token_embeddings, attention_mask):
        # Mean Pooling - Take attention mask into account for correct averaging
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed_documents(content_text: List[str], model: AutoModel, tokenizer: AutoTokenizer,
                    max_length: int = 512, stride: int = 0, batch_size: int = 1):
    logging.info('Embedding documents...')
    pipe = EmbeddingPipeline(model=model,
                             tokenizer=tokenizer,
                             max_length=max_length,
                             stride=stride,
                             device=0
                             )

    content_embeddings = [out for out in tqdm(pipe(content_text, batch_size=batch_size),
                                              total=len(content_text),
                                              desc="Embedding documents")]

    content_embeddings = torch.concat([t.unsqueeze(0) for t in content_embeddings], dim=0).numpy()

    return content_embeddings


def get_embeddings_df(content_df: pd.DataFrame, id_col: str, title_col: str, text_col: str, model_name: str, **kwargs):
    # Get model & Tokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 'sentence-transformers/all-distilroberta-v1'
    # 'allenai/specter2'

    # Embed all documents
    content_text = get_document_text(content_df, title_col, text_col, tokenizer.sep_token)
    corpus_embeddings = embed_documents(content_text, model, tokenizer, **kwargs)

    # Create embeddings dataframe
    embeddings_df = pd.DataFrame(corpus_embeddings).astype("float")
    embeddings_df[id_col] = list(content_df[id_col])

    return embeddings_df


def get_document_text(content_df: pd.DataFrame, title_col: str, text_col: str, sep_token=''):
    if title_col:
        content_text = [str(p[title_col]) + sep_token + str(p[text_col])
                        if str(p[title_col]) != ' ' else str(p[text_col])
                        for idx, p in tqdm(content_df.iterrows(), total=content_df.shape[0], desc="Convert text")]
    else:
        content_text = content_df[text_col].tolist()

    return content_text
