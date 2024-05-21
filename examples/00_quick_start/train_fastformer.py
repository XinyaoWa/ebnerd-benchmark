from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import polars as pl
import numpy as np
import time, os
import gc
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
from torch.utils.data import DataLoader
from ebrec.models.fastformer.fastformer import Fastformer
from ebrec.models.fastformer.config import FastFormerConfig
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from ebrec.utils._python import (
    rank_predictions_by_score,
    write_submission_file
)

sample_num = -1 # for debug, disable by set to -1
train_flag = False
eval_flag = False
prediction_flag = True
model_weights_path = "/home/data/models/FastFormer/ebnerd_large/weights/weights_e0"
use_device = "hpu"
dataset_name = "ebnerd_large"

MODEL_NAME = "FastFormer"
data_path = "/home/data/" ##vsr216, vsr134, gaudi-190
# data_path = "/home/recsys2024/data" # gaudi-414
path = Path(f"{data_path}/dataset/origin/{dataset_name}/")
test_path = Path(f"{data_path}/dataset/origin/ebnerd_testset/")
LOG_DIR = f"{data_path}/models/{MODEL_NAME}/{dataset_name}/log"
MODEL_WEIGHTS = f"{data_path}/models/{MODEL_NAME}/{dataset_name}/weights/weights"
out_dir = f"{data_path}/models/{MODEL_NAME}/{dataset_name}/out"
TRANSFORMER_MODEL_NAME = f"{data_path}/models/bert-base-multilingual-cased"
N_SAMPLES = "n"

BATCH_SIZE = 64
num_epochs=1
num_hidden_layers = 4
num_attention_heads = 8

MAX_TITLE_LENGTH = 30
HISTORY_SIZE = 30
COLUMNS = [DEFAULT_IMPRESSION_ID_COL,DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL,N_SAMPLES]
TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
#torch.set_default_device(device)
if use_device == "hpu":
    device = torch.device("hpu")
    from ebrec.models.fastformer.dataloader_hpu import FastformerDataset, compute_auc_from_fixed_pos_neg_samples
    from ebrec.models.fastformer.dataloader_hpu import train as train_fastformer
    from ebrec.models.fastformer.dataloader_hpu import evaluate as evaluate_fastformer
    from ebrec.models.fastformer.dataloader_hpu import predict as predict_fastformer
elif use_device == "gpu":
    device = torch.device("cuda:0")
    from ebrec.models.fastformer.dataloader import FastformerDataset, compute_auc_from_fixed_pos_neg_samples
    from ebrec.models.fastformer.dataloader import train as train_fastformer
    from ebrec.models.fastformer.dataloader import evaluate as evaluate_fastformer
    from ebrec.models.fastformer.dataloader import predict as predict_fastformer
elif  use_device == "cpu":
    device = torch.device("cpu")
    from ebrec.models.fastformer.dataloader import FastformerDataset, compute_auc_from_fixed_pos_neg_samples
    from ebrec.models.fastformer.dataloader import train as train_fastformer
    from ebrec.models.fastformer.dataloader import evaluate as evaluate_fastformer
    from ebrec.models.fastformer.dataloader import predict as predict_fastformer

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
if prediction_flag:
    if os.path.isfile(test_path.joinpath("test.parquet")):
        df_test = pl.read_parquet(test_path.joinpath("test.parquet"))
    else:
        df_test = (
            ebnerd_from_path(test_path.joinpath("test"), history_size=HISTORY_SIZE)
            .with_columns(pl.Series(DEFAULT_CLICKED_ARTICLES_COL, [[]]))
            .select(COLUMNS)
            .pipe(create_binary_labels_column)
        )
        df_test.write_parquet(test_path.joinpath("test.parquet"))

if sample_num > 0:
    df_train = df_train[:sample_num]
    df_validation = df_validation[:sample_num]
    if prediction_flag:
        df_test = df_test[:sample_num]
gc.collect()

print(f"train shape: {df_train.shape}")
print(f"validation shape: {df_validation.shape}")
if prediction_flag:
    print(f"test shape: {df_test.shape}")
    print(f"test:\n{df_test.head(5)}")
else:
    print(f"validation:\n{df_validation.head(5)}")

label_lengths = df_validation[DEFAULT_INVIEW_ARTICLES_COL].list.len().to_list()
if prediction_flag:
    label_lengths_test = df_test[DEFAULT_INVIEW_ARTICLES_COL].list.len().to_list()
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

train_dataloader = DataLoader(
    FastformerDataset(
        behaviors=df_train,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        article_dict=article_mapping,
        batch_size=BATCH_SIZE,
        device=device,
        shuffle=False,
    )
)

validation_dataloader = DataLoader(
    FastformerDataset(
        behaviors=df_validation,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        article_dict=article_mapping,
        batch_size=BATCH_SIZE,
        device=device,
        shuffle=False,
    )
)

test_dataloader = DataLoader(
    FastformerDataset(
        behaviors=df_test,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        article_dict=article_mapping,
        batch_size=BATCH_SIZE,
        device=device,
        shuffle=False,
    )
)

model_config = FastFormerConfig(num_hidden_layers=num_hidden_layers,
                            num_attention_heads=num_attention_heads)
model = Fastformer(model_config,
        word_embedding=word2vec_embedding)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
loss = torch.nn.BCELoss()
time4 = time.time()
print(f"Prepared!! Cost {time4-time3} seconds")

##evaluate
if eval_flag:
    model.load_state_dict(torch.load(model_weights_path), strict=True)
    model.train(False)
    all_outputs, all_labels, val_loss = evaluate_fastformer(
                                        model=model,
                                        dataloader=validation_dataloader,
                                        criterion=loss,
                                        device=device,
                                        out_dir=out_dir)
    print(f"all_outputs shape: {all_outputs.shape}")
    print(f"all_labels shape: {all_labels.shape}")
    print(f"sum of labels: {sum(label_lengths)}")
    save_d = {"labels": all_labels, "outputs": all_outputs}  
    torch.save(save_d, os.path.join(out_dir,"outputs"))
    print(f"save outputs to: {os.path.join(out_dir,'outputs')}")
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
    time5 = time.time()
    print(f"Finished!! Cost {time5-time0} seconds")
    exit(0)

if prediction_flag:
    model.load_state_dict(torch.load(model_weights_path), strict=True)
    model.train(False)
    all_outputs = predict_fastformer(
                                model=model,
                                dataloader=test_dataloader,
                                device=device,
                                out_dir=out_dir)
    print(f"all_outputs shape: {all_outputs.shape}")
    print(f"sum of labels: {sum(label_lengths_test)}")
    save_d = {"outputs": all_outputs}  
    torch.save(save_d, os.path.join(out_dir,"outputs"))
    print(f"save outputs to: {os.path.join(out_dir,'outputs')}")
    
    df_test = add_prediction_scores(df_test, all_outputs.tolist()).pipe(
        add_known_user_column, known_users=df_train[DEFAULT_USER_COL]
    )
    print(df_test.head(5))
    df_test = df_test.with_columns(
        pl.col("scores")
        .map_elements(lambda x: list(rank_predictions_by_score(x)))
        .alias("prediction_scores")
    )
    print(df_test.head(5))

    impression_ids = df_test[DEFAULT_IMPRESSION_ID_COL].to_list()
    prediction_scores = df_test["prediction_scores"].to_list()

    filename_zip=f"predictions_{MODEL_NAME}_{dataset_name}.zip"
    write_submission_file(
        impression_ids=impression_ids,
        prediction_scores=prediction_scores,
        path=os.path.join(out_dir,"predictions.txt"),
        filename_zip=filename_zip,
    )
    print(f"save predictions to: {os.path.join(out_dir,filename_zip)}")
    time5 = time.time()
    print(f"Finished!! Cost {time5-time0} seconds")
    exit(0)

##train
if train_flag:
    model = train_fastformer(model=model, \
                    train_dataloader=train_dataloader, \
                    df_train=df_train, \
                    criterion=loss, 
                    optimizer=optimizer,
                    scheduler=scheduler,
                    num_epochs=num_epochs, \
                    val_dataloader=validation_dataloader, \
                    df_validation=df_validation,\
                    state_dict_path=MODEL_WEIGHTS, \
                    monitor_metric="auc",
                    device=device
                    )
    time5 = time.time()
    print(f"Finished!! Cost {time5-time0} seconds")