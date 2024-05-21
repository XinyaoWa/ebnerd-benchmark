from dataclasses import dataclass
from tqdm import tqdm
import polars as pl
import numpy as np
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import torch
from ebrec.utils._constants import  DEFAULT_USER_COL

from ebrec.utils._constants import DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_LABELS_COL

from ebrec.utils._python import (
    repeat_by_list_values_from_matrix,
    convert_to_nested_list,
    make_lookup_objects,
)
from ebrec.utils._articles_behaviors import map_list_article_id_to_value
from ebrec.utils._polars import shuffle_rows

from ebrec.evaluation import AucScore
from ebrec.utils._torch import save_checkpoint
from ebrec.utils._behaviors import  add_known_user_column,  add_prediction_scores
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
import habana_frameworks.torch.core as htcore

@dataclass
class FastformerDataset(Dataset):
    """_summary_
    The batch-size is aggragating multiple impressions and processing them simultaneous, which
    has a major effect on the training time. Hence, you should put the batch_size=1 in the 'DataLoader'
    and just use FastformerDataset batch_size.

    Note, the outut is then (1, output_shape), where the 1 is the DataLoader batch_size.
    """

    behaviors: pl.DataFrame
    history_column: str
    article_dict: dict[int, pl.Series]
    batch_size: int = 64
    shuffle: bool = True
    device: str = "cpu"
    seed: int = None
    labels_col: str = DEFAULT_LABELS_COL
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL
    n_samples_col: str = "n_samples"

    def __post_init__(self):
        self.unknown_index = [0]
        if self.shuffle:
            self.behaviors = shuffle_rows(self.behaviors, seed=self.seed)
        self.behaviors = self.behaviors.with_columns(
            pl.col(self.labels_col).list.len().alias(self.n_samples_col)
        )
        self.lookup_indexes, self.lookup_matrix = make_lookup_objects(
            self.article_dict, unknown_representation="zeros"
        )

    def __len__(self):
        """
        Number of batch steps in the data
        """
        return int(np.ceil(self.behaviors.shape[0] / self.batch_size))

    def __getitem__(self, index: int):
        """
        Get the batch of samples for the given index.

        Note: The dataset class provides a single index for each iteration. The batching is done internally in this method
        to utilize and optimize for speed. This can be seen as a mini-batching approach.

        Args:
            index (int): An integer index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input features and labels as torch Tensors.
                Note, the output of the PyTorch DataLoader is (1, *shape), where 1 is the DataLoader's batch_size.
        """
        # Clever way to batch the data:
        batch_indices = range(index * self.batch_size, (index + 1) * self.batch_size)
        batch = self.behaviors[batch_indices]
        if self.shuffle:
            batch = shuffle_rows(batch, seed=self.seed)
        # =>
        x = (
            batch.drop(self.labels_col)
            .pipe(
                map_list_article_id_to_value,
                behaviors_column=self.history_column,
                mapping=self.lookup_indexes,
                fill_nulls=self.unknown_index,
            )
            .pipe(
                map_list_article_id_to_value,
                behaviors_column=self.inview_col,
                mapping=self.lookup_indexes,
                fill_nulls=self.unknown_index,
            )
        )
        # =>
        repeats = np.array(batch[self.n_samples_col])
        # =>
        history_input = repeat_by_list_values_from_matrix(
            input_array=x[self.history_column].to_list(),
            matrix=self.lookup_matrix,
            repeats=repeats,
        ).squeeze(2)
        # =>
        candidate_input = self.lookup_matrix[x[self.inview_col].explode().to_list()]
        # =>
        history_input = torch.Tensor(history_input).type(torch.int).to(self.device)
        candidate_input = torch.Tensor(candidate_input).type(torch.int).to(self.device)
        y = (
            torch.Tensor(batch[self.labels_col].explode())
            .view(-1, 1)
            .type(torch.float)
            .to(self.device)
        )
        # ========================
        return (history_input, candidate_input), y


def batch_input_label_concatenation(
    inputs: tuple[torch.Tensor], labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """ """
    return (inputs[0].squeeze(0), inputs[1].squeeze(0)), labels.squeeze(0)


def compute_auc_from_fixed_pos_neg_samples(
    y_true: list[float], y_pred: list[float]
) -> float:
    #
    n_samples = int(np.sum(y_true))
    y_true = convert_to_nested_list(y_true, n_samples)
    y_pred = convert_to_nested_list(y_pred, n_samples)
    val_auc = AucScore().calculate(y_true=y_true, y_pred=y_pred)
    return val_auc


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler,
    criterion=None,
    df_train=None,
    num_epochs: int = 5,
    val_dataloader: DataLoader = None,
    df_validation=None,
    state_dict_path: str = "model_state_dict.pt",
    patience: int = None,
    summary_writer: SummaryWriter = None,
    gradient_accumulation_steps: int = 1,
    tqdm_disable: bool = False,
    tqdm_ncol: int = 80,
    monitor_metric: str = "loss",
    device=torch.device("hpu:0")
) -> nn.Module:
    """ """
    min_val_loss = np.inf
    max_val_auc = -np.inf
    early_stop = 0
    global_steps = 0
    total_batches = len(train_dataloader)
    running_loss = 0.0
    running_samples = 0
    # ==> TRAIN LOOP:
    for epoch in range(num_epochs):
        # => Set the model to train mode
        model.train(True)
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch [{epoch + 1}/{num_epochs}]",
            disable=tqdm_disable,
            ncols=tqdm_ncol,
        )
        # => Zero the parameter gradients
        optimizer.zero_grad()
        print_step = 50
        print_step_count = 0
        for batch_idx, (inputs, labels) in enumerate(progress_bar, start=1):
            # => Move inputs and labels to device
            inputs, labels = batch_input_label_concatenation(inputs, labels)
            # => Forward pass
            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            # => Backward pass and optimization
            loss.backward()
            htcore.mark_step()
            # => Update training loss
            global_steps += 1
            # running_loss += loss.item() * len(outputs)
            running_samples += len(outputs)
            # if print_step_count % print_step == 0:
            #     print("loss: ", loss)
            print_step_count += 1
            # current_loss = running_loss / running_samples
            # progress_bar.set_postfix({"Loss": round(current_loss, 6)})
            # =>
            # if summary_writer is not None:
            #     summary_writer.add_scalar(
            #         tag="Train/Loss",
            #         scalar_value=current_loss,
            #         global_step=global_steps,
            #     )
            # => Accumulated gradient step:
            # if (
            #     batch_idx % gradient_accumulation_steps == 0
            #     or batch_idx == total_batches
            # ):
                # => Take step and zero gradients
            optimizer.step()
            optimizer.zero_grad()
            htcore.mark_step()
        # print("loss: ", loss)

        scheduler.step()
        save_checkpoint(model, path=state_dict_path+f"_e{epoch}")
        # ==> EVAL LOOP:
        if val_dataloader:
            model.train(False)
            all_outputs, all_labels, val_loss = evaluate(
                model=model,
                dataloader=val_dataloader,
                criterion=criterion,
                tqdm_disable=tqdm_disable,
            )

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

            # if summary_writer is not None:
            #     summary_writer.add_scalar(
            #         tag="Val/Loss", scalar_value=val_loss, global_step=global_steps
            #     )

            if monitor_metric == "auc":
                val_auc = result.evaluations["auc"]
                # val_auc = compute_auc_from_fixed_pos_neg_samples(
                #     y_true=np.ravel(all_labels.tolist()),
                #     y_pred=np.ravel(all_outputs.tolist()),
                # )
                print(f"Val/AUC : {round(val_auc, 6)}")
                if summary_writer is not None:
                    summary_writer.add_scalar(
                        tag="Val/AUC", scalar_value=val_auc, global_step=global_steps
                    )

            # => MODEL CHECKPOINT
            # if monitor_metric == "loss" and val_loss < min_val_loss:
            #     save_checkpoint(model, path=state_dict_path)
            #     min_val_loss = val_loss
            #     early_stop = 0
            # elif monitor_metric == "auc" and val_auc > max_val_auc:
            #     save_checkpoint(model, path=state_dict_path)
            #     max_val_auc = val_auc
            #     early_stop = 0
            # else:
            #     early_stop += 1
            # => EARLYSTOP
            # if patience is not None and early_stop == patience:
            #     break

    if summary_writer is not None:
        summary_writer.close()

    if val_dataloader:
        model.load_state_dict(torch.load(state_dict_path+f"_e{num_epochs-1}"), strict=True)

    return model

def predict(
    model: nn.Module,
    dataloader: DataLoader,
    tqdm_disable: bool = False,
    tqdm_ncol: int = 80,
    device: str = "cpu",
    out_dir=None
) -> tuple[list[float], list[float], float]:
    model.eval()
    all_outputs = []
    n_samples = 0
    shape_list = []
    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc="Predicting",
            total=dataloader.__len__(),
            disable=tqdm_disable,
            ncols=tqdm_ncol,
        )
        for inputs, labels in progress_bar:
            inputs, labels = batch_input_label_concatenation(inputs, labels)
            # Forward pass
            outputs = model(*inputs)
            shape_list.append([labels.shape[0], outputs.shape[0]])
            htcore.mark_step()
            n_samples += len(outputs)
            all_outputs.append(outputs)
        all_outputs = torch.cat(all_outputs, dim=0)

        shape_np = np.array(shape_list)
        out_dir = out_dir if out_dir is not None else "tmp"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_dir.joinpath("shape.txt"), shape_np)
        print(f"save shape to: {out_dir.joinpath('shape.txt')}")
    return all_outputs
