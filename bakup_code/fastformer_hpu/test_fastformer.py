from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import polars as pl
import numpy as np
import time, os, gc
from ebrec.utils._constants import *
from ebrec.utils._behaviors import *
from ebrec.utils._articles import convert_text2encoding_with_transformers
from ebrec.utils._polars import concat_str_columns, slice_join_dataframes
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._nlp import get_transformers_word_embeddings
import torch
from torch.utils.data import DataLoader
from ebrec.models.fastformer.fastformer import Fastformer
from ebrec.models.fastformer.config import FastFormerConfig
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from ebrec.utils._python import (
    rank_predictions_by_score,
    write_submission_file
)
from ebrec.models.fastformer.dataloader_hpu import *

model_weights_path = "/home/data/models/FastFormer/ebnerd_large/weights/weights_e0"
TRANSFORMER_MODEL_NAME = f"/home/data/models/bert-base-multilingual-cased"
article_path = "/home/data/dataset/origin/ebnerd_large/articles.parquet"

device = torch.device("hpu")
df_validation = pl.read_parquet("./data/validation.parquet")
print(df_validation.shape)

transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
word2vec_embedding = transformer_model.embeddings.word_embeddings

df_articles = pl.read_parquet(article_path)
df_articles, cat_cal = concat_str_columns(df_articles, columns=[DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL])
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=30
)
article_mapping = create_article_id_to_value_mapping(df=df_articles, value_col=token_col_title)

validation_dataloader = DataLoader(
    FastformerDataset(
        behaviors=df_validation,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        article_dict=article_mapping,
        batch_size=8,
        device=device,
        shuffle=False,
    )
)

model_config = FastFormerConfig(num_hidden_layers=4, num_attention_heads=8)
model = Fastformer(model_config,
        word_embedding=word2vec_embedding)
model.to(device)

criterion = torch.nn.BCELoss()
model.load_state_dict(torch.load(model_weights_path), strict=True)
model.train(False)
model.eval()
with torch.no_grad():
    for inputs, labels in tqdm(validation_dataloader,total=validation_dataloader.__len__()):
        inputs, labels = batch_input_label_concatenation(inputs, labels)
        t1 = time.time()
        outputs = model(*inputs)
        t2 = time.time()
        outputs.cpu()
        t3 = time.time()
        print(f"Tensor shape: {outputs.shape}")
        print(f"forward cost: {t2-t1} seconds")
        print(f"to CPU cost: {t3-t2} seconds")
        batch_loss = criterion(outputs, labels)
        htcore.mark_step()

   
