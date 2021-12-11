# This is an adaptation of run_glue_fusion.py in adapter-transformers repository
# 
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from transformers import (
    AdapterArguments,
    AdapterTrainer,
    AutoConfig,
    AutoModelWithHeads,
    AutoTokenizer,
    EvalPrediction,
    GlueDataset,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers import GlueDataTrainingArguments as DataTrainingArguments

from datasets import load_dataset
import transformers.adapters.composition as ac

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        #metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
        metadata={"help": "The name of the task to train on:.."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            f" Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    output_mode = "classification"
    #try:
    #    num_labels = glue_tasks_num_labels[data_args.task_name]
    #    output_mode = glue_output_modes[data_args.task_name]
    #except KeyError:
    #    raise ValueError("Task not found: %s" % (data_args.task_name))
    if data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset(
            "csv", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
        )
    else:
        # Loading a dataset from local json files
        datasets = load_dataset(
            "json", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
        )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    label_list = None
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:          
            #label_list = datasets["train"].features["label"].names                       
            label_list =datasets["train"].unique("label")                       
            num_labels = len(label_list)            
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    
    # Load pretrained model and tokenizer    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    # We use the AutoModelWithHeads class here for better adapter support.
    model = AutoModelWithHeads.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )       
    model.add_classification_head(
        data_args.task_name,
        num_labels=num_labels,
        id2label={i: v for i, v in enumerate(label_list)} if num_labels > 0 else None,
    )
    
    # experiment
    LANG_ADAPTER_PATH = "tmp/test-anomaly-hdfs-houlsby/mlm"
    TASK_ADAPTER_PATH = "tmp/test-anomaly-hdfs-houlsby/hanomaly"  
    LANG_ADAPTER_PATH_ = "tmp/test-anomaly-firewall-pfeiffer-last2/mlm"
    TASK_ADAPTER_PATH_ = "tmp/test-anomaly-firewall-pfeiffer-last2/fanomaly"
    firewall_adapter = model.load_adapter(LANG_ADAPTER_PATH, load_as="mlm-firewall", with_head=False)
    fanomaly_adapter =model.load_adapter(TASK_ADAPTER_PATH, with_head=False)
    hdfs_adapter = model.load_adapter(LANG_ADAPTER_PATH_, load_as="mlm-hdfs", with_head=False)
    hanomaly_adapter = model.load_adapter(TASK_ADAPTER_PATH_, with_head=False)    
    
    fusion_setup = ac.Fuse(fanomaly_adapter, hanomaly_adapter)
    model.add_adapter_fusion(fusion_setup)
    # fuse two stacks
    model.set_active_adapters(ac.Fuse(ac.Stack(firewall_adapter, fanomaly_adapter),\
                                      ac.Stack(hdfs_adapter, hanomaly_adapter)))   

    model.train_adapter_fusion(fusion_setup)
   

    # Get datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}    

    def preprocess_function(examples):
        # Tokenize the texts       
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        result = tokenizer(examples[non_label_column_names[0]], padding=padding, max_length=max_length, truncation=True)
        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]   

    def compute_metrics(p: EvalPrediction):
        labels = p.label_ids
        preds = p.predictions.argmax(-1)
        ## DEFINITELY Check here
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1)   
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
    }   

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics        
    )

    # Training
    if training_args.do_train:        
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None,            
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev"))

        for eval_dataset in eval_datasets:
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{data_args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(data_args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)
      
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
