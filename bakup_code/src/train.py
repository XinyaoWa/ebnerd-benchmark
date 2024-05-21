import sys
from transformers import TrainingArguments, HfArgumentParser
from transformers import Trainer, EvalPrediction
from dataclasses import dataclass, field
from transformers4rec import torch as tr
import evaluate 
import pandas as pd

@dataclass
class OtherArguments:
    use_hpu: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use habana gaudi2 for training"
                "should only be set to `True` if you have gaudi2"
            )
        },
    )


def init_training_args(training_args, other_args):
    training_args_dict = training_args.to_dict()
    if not other_args.use_hpu:
        training_args = TrainingArguments(**training_args_dict)
    else:
        from optimum.habana import GaudiTrainingArguments
        training_args_dict.update({
            "use_habana": True,
            "use_lazy_mode": True,
            "gaudi_config_name": "config/gaudi_config.json",
        })
        training_args = GaudiTrainingArguments(**training_args_dict)
    return training_args


def compute_metrics(p: EvalPrediction):
    return metric.compute(prediction_scores=p.predictions, references=p.label_ids)

def get_model():
    inputs = tr.TabularSequenceFeatures.from_schema(
            schema,
            max_sequence_length=101,
            continuous_projection=64,
            masking="clm",
            d_output=100,
    )

    transformer_config = tr.XLNetConfig.build(
        d_model=64, n_head=4, n_layer=2, total_seq_length=101
    )
    body = tr.SequentialBlock(
        inputs, tr.MLPBlock([64]), tr.TransformerBlock(transformer_config, masking=inputs.masking)
    )
    metrics = [NDCGAt(top_ks=[20, 40], labels_onehot=True),  
            RecallAt(top_ks=[20, 40], labels_onehot=True)]
    head = tr.Head(
        body,
        tr.NextItemPredictionTask(target_dim=28144,
                                    weight_tying=True, 
                                    metrics=metrics),
        inputs=inputs,
    )

    model = tr.Model(head)

if __name__ == "__main__":
    data_path = "/home/data/dataset/origin/ebnerd_large"

    parser = HfArgumentParser((OtherArguments, TrainingArguments))
    other_args, training_args = parser.parse_args_into_dataclasses()
    training_args = init_training_args(training_args, other_args)

    train_ds = pd.read_parquet(os.path.join(data_path,"train.parquet"))
    valid_ds = pd.read_parquet(os.path.join(data_path,"validation.parquet"))

    

    if other_args.use_hpu:
        from optimum.habana import GaudiTrainer
        choose_trainer = GaudiTrainer
    else:
        choose_trainer = Trainer
    trainer = choose_trainer(
        model=ctr_model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=valid_ds if training_args.do_eval else None,
        compute_metrics=compute_metrics,
    )
    
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_ds)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(valid_ds)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)







