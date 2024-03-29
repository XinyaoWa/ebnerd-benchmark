from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import polars as pl
import numpy as np
from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL, 
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_TITLE_COL, 
    DEFAULT_USER_COL, 
)
from ebrec.utils._behaviors import (
    create_binary_labels_column, 
    sampling_strategy_wu2019,
    add_known_user_column,
    add_prediction_scores,
    truncate_history, 
)
from ebrec.utils._articles import convert_text2encoding_with_transformers
from ebrec.utils._polars import concat_str_columns, slice_join_dataframes
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._nlp import get_transformers_word_embeddings
import numpy as np
import torch
from ebrec.models.fastformer.dataloader import FastformerDataset
from ebrec.models.fastformer.dataloader import train as train_fastformer
from ebrec.models.fastformer.dataloader import evaluate as evaluate_fastformer
from ebrec.models.fastformer.dataloader import compute_auc_from_fixed_pos_neg_samples
from torch.utils.data import DataLoader
from ebrec.models.fastformer.fastformer import Fastformer
from ebrec.models.fastformer.config import FastFormerConfig
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore

sample_num = 10 # for debug, disable by set to -1

MODEL_NAME = "FastFormer"
path = Path("/home/data/dataset/origin/ebnerd_demo/")
LOG_DIR = f"/home/data/models/{MODEL_NAME}/log"
MODEL_WEIGHTS = f"/home/data/models/{MODEL_NAME}/weights/weights"
TRANSFORMER_MODEL_NAME = "/home/data/models/bert-base-multilingual-cased"
N_SAMPLES = "n"

BATCH_SIZE = 2
num_epochs=1
num_hidden_layers = 4
num_attention_heads = 4

MAX_TITLE_LENGTH = 30
HISTORY_SIZE = 30
COLUMNS = [DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL,N_SAMPLES]
TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]

def ebnerd_from_path(path:Path, history_size:int = 30) -> pl.DataFrame:
    """
    Load ebnerd - function 
    """
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=history_size,
            padding_value=0,
        )
    )
    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .with_columns(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len().alias(N_SAMPLES))
        .collect()
        .pipe(
            slice_join_dataframes, df2=df_history.collect(), on=DEFAULT_USER_COL, how="left"
        )
    )
    return df_behaviors

df_train = (
    ebnerd_from_path(path.joinpath("train"), history_size=HISTORY_SIZE)
    .select(COLUMNS)
    .pipe(sampling_strategy_wu2019,npratio=4,shuffle=True,with_replacement=True, seed=123)
    .pipe(create_binary_labels_column)
)

df_validation = (
    ebnerd_from_path(path.joinpath("validation"), history_size=HISTORY_SIZE)
    .select(COLUMNS)
    .pipe(create_binary_labels_column)
)

if sample_num > 0:
    df_train = df_train.sample(n=sample_num)
    df_validation = df_validation.sample(n=sample_num)

label_lengths = df_validation[DEFAULT_INVIEW_ARTICLES_COL].list.len().to_list()

# LOAD HUGGINGFACE:
transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

# We'll init the word embeddings using the 
word2vec_embedding = get_transformers_word_embeddings(transformer_model)
# 

df_articles = pl.read_parquet(path.joinpath("articles.parquet"))
df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)
article_mapping = create_article_id_to_value_mapping(df=df_articles, value_col=token_col_title)

train_dataloader = DataLoader(
    FastformerDataset(
        behaviors=df_train,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        article_dict=article_mapping,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
)

test_dataloader = DataLoader(
    FastformerDataset(
        behaviors=df_validation,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        article_dict=article_mapping,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
)

model_config = FastFormerConfig(num_hidden_layers=num_hidden_layers,
                            num_attention_heads=num_attention_heads)
model = Fastformer(model_config)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
loss = torch.nn.BCELoss()

##evaluate
all_outputs, all_labels, val_loss = evaluate_fastformer(
                                    model=model,
                                    dataloader=test_dataloader,
                                    criterion=loss)

df_validation = add_prediction_scores(df_validation, all_outputs.tolist()).pipe(
    add_known_user_column, known_users=df_train[DEFAULT_USER_COL]
)

metrics = MetricEvaluator(
    labels=df_validation["labels"].to_list(),
    predictions=df_validation["scores"].to_list(),
    metric_functions=[AucScore(), MrrScore(),  NdcgScore(k=5),  NdcgScore(k=10)],
)
result = metrics.evaluate()
print(result)

##train
model = train_fastformer(model=model, \
                 train_dataloader=train_dataloader, \
                 df_train=df_train, \
                 criterion=loss, 
                 optimizer=optimizer,
                 scheduler=scheduler,
                 num_epochs=num_epochs, \
                 val_dataloader=test_dataloader, \
                 df_validation=df_validation,\
                 state_dict_path=MODEL_WEIGHTS, \
                 monitor_metric="loss"
                )