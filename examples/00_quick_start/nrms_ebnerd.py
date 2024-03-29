from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import tensorflow as tf
import polars as pl
import time, os

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
from ebrec.models.newsrec.dataloader import NRMSDataLoader
from ebrec.models.newsrec.model_config import hparams_nrms
from ebrec.models.newsrec import NRMSModel

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
        .collect()
        .pipe(
            slice_join_dataframes, df2=df_history.collect(), on=DEFAULT_USER_COL, how="left"
        )
    )
    return df_behaviors

data_path = "/home/data/recsys2024"
path = Path(f"{data_path}/dataset/origin/ebnerd_large/")
TRANSFORMER_MODEL_NAME = f"{data_path}/models/bert-base-multilingual-cased"
MODEL_NAME = "NRMS"
LOG_DIR = f"{data_path}/models/test/runs/{MODEL_NAME}"
MODEL_WEIGHTS = f"{data_path}/models/test/data/state_dict/{MODEL_NAME}/weights"
start_time = time.time()
COLUMNS = [DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL]
HISTORY_SIZE = 30
N_SAMPLES = -1

if os.path.isfile(path.joinpath("train.parquet")):
    df_train = pl.read_parquet(path.joinpath("train.parquet"))
else:
    df_train = (
        ebnerd_from_path(path.joinpath("train"), history_size=HISTORY_SIZE)
        .select(COLUMNS)
        .pipe(sampling_strategy_wu2019,npratio=4,shuffle=True,with_replacement=True, seed=123)
        .pipe(create_binary_labels_column) 
    )
    df_train.write_parquet(path.joinpath("train.parquet"))
print(df_train.head(5))

if os.path.isfile(path.joinpath("validation.parquet")):
    df_validation = pl.read_parquet(path.joinpath("validation.parquet"))
else:
    df_validation = (
        ebnerd_from_path(path.joinpath("validation"), history_size=HISTORY_SIZE)
        .select(COLUMNS)
        .pipe(create_binary_labels_column) 
    )
    df_validation.write_parquet(path.joinpath("validation.parquet"))

if N_SAMPLES > 0:
    df_train = df_train.sample(n=N_SAMPLES)
    df_validation = df_validation.sample(n=N_SAMPLES)
print(df_train.head(5))

df_articles = pl.read_parquet(path.joinpath("articles.parquet"))


TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
MAX_TITLE_LENGTH = 30

transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

word2vec_embedding = get_transformers_word_embeddings(transformer_model)

df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)
article_mapping = create_article_id_to_value_mapping(df=df_articles, value_col=token_col_title)

train_dataloader = NRMSDataLoader(
    behaviors=df_train,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=512,
)
val_dataloader = NRMSDataLoader(
    behaviors=df_validation,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=512,
)
print(f"data loaded!! Cost {time.time()-start_time} seconds")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_WEIGHTS, save_best_only=True, save_weights_only=True, verbose=1
)

hparams_nrms.history_size = HISTORY_SIZE
model = NRMSModel(
    hparams=hparams_nrms,
    word2vec_embedding=word2vec_embedding,
    seed=42,
)

hist = model.model.fit(
    train_dataloader,
    validation_data=val_dataloader,
    epochs=1,
    callbacks=[tensorboard_callback, early_stopping, modelcheckpoint],
)

model.model.load_weights(filepath=MODEL_WEIGHTS)

pred_validation = model.scorer.predict(val_dataloader)

df_validation = add_prediction_scores(df_validation, pred_validation.tolist()).pipe(
    add_known_user_column, known_users=df_train[DEFAULT_USER_COL]
)

from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
metrics = MetricEvaluator(
    labels=df_validation["labels"].to_list(),
    predictions=df_validation["scores"].to_list(),
    metric_functions=[AucScore(), MrrScore(),  NdcgScore(k=5),  NdcgScore(k=10)],
)
print(metrics.evaluate())
print(f"Finished!! Totally cost {time.time()-start_time} seconds")