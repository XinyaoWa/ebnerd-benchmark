from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import polars as pl
import numpy as np
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

sample_num = 256 # for debug, disable by set to -1

MODEL_NAME = "FastFormer"
dataset_name = "ebnerd_demo"
path = Path(f"/home/recsys2024/data/dataset/origin/{dataset_name}/")
LOG_DIR = f"/home/recsys2024/data/models/{MODEL_NAME}/{dataset_name}/log"
MODEL_WEIGHTS = f"/home/recsys2024/data/models/{MODEL_NAME}/{dataset_name}/weights/weights"
TRANSFORMER_MODEL_NAME = "/home/recsys2024/data/models/bert-base-multilingual-cased"
N_SAMPLES = "n"

BATCH_SIZE = 128
num_epochs=1
num_hidden_layers = 4
num_attention_heads = 8

MAX_TITLE_LENGTH = 30
HISTORY_SIZE = 30
COLUMNS = [DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL,N_SAMPLES]
TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
device = torch.device("hpu")
#torch.set_default_device(device)

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

time0 = time.time()
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
if os.path.isfile(path.joinpath("validation.parquet")):
    df_validation = pl.read_parquet(path.joinpath("validation.parquet"))
else:
    df_validation = (
        ebnerd_from_path(path.joinpath("validation"), history_size=HISTORY_SIZE)
        .select(COLUMNS)
        .pipe(create_binary_labels_column)
    )
    df_validation.write_parquet(path.joinpath("validation.parquet"))

if sample_num > 0:
    df_train = df_train.sample(n=sample_num)
    df_validation = df_validation.sample(n=sample_num)

label_lengths = df_validation[DEFAULT_INVIEW_ARTICLES_COL].list.len().to_list()
time1 = time.time()
print(f"Dataset Loaded!! Cost {time1-time0} seconds")


# LOAD HUGGINGFACE:
transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

# We'll init the word embeddings using the 
# word2vec_embedding = get_transformers_word_embeddings(transformer_model)
word2vec_embedding = transformer_model.embeddings.word_embeddings
time2 = time.time()
print(f"Embedding Loaded!! Cost {time2-time1} seconds")

df_articles = pl.read_parquet(path.joinpath("articles.parquet"))
df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)
article_mapping = create_article_id_to_value_mapping(df=df_articles, value_col=token_col_title)
time3 = time.time()
print(f"article_mapping Loaded!! Cost {time3-time2} seconds")

train_ds = FastformerDataset(
        behaviors=df_train,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        article_dict=article_mapping,
        batch_size=BATCH_SIZE,
        device=device,
        shuffle=False,
    )

valid_ds = FastformerDataset(
        behaviors=df_validation,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        article_dict=article_mapping,
        batch_size=BATCH_SIZE,
        device=device,
        shuffle=False,
    )

############################################################
import sys
from transformers import TrainingArguments, HfArgumentParser
from transformers import Trainer, EvalPrediction
from dataclasses import dataclass, field
# import evaluate 
from optimum.habana import GaudiTrainer
from optimum.habana import GaudiTrainingArguments

def init_training_args(training_args):
    training_args_dict = training_args.to_dict()
    training_args_dict.update({
        "use_habana": True,
        "use_lazy_mode": True,
        "gaudi_config_name": "config/gaudi_config.json",
    })
    training_args = GaudiTrainingArguments(**training_args_dict)
    return training_args


# def compute_metrics(p: EvalPrediction):
#     return metric.compute(prediction_scores=p.predictions, references=p.label_ids)
    

if __name__ == "__main__":
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_args_into_dataclasses()
    training_args = init_training_args(training_args[0])

    model_config = FastFormerConfig(num_hidden_layers=num_hidden_layers,
                            num_attention_heads=num_attention_heads)
    model = Fastformer(model_config,
            word_embedding=word2vec_embedding)

    
    trainer = GaudiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds
    )
    
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(valid_ds)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    print(f"Finished!! Cost {time5-time0} seconds")