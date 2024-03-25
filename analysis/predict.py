import os
import csv
import argparse

import torch
import datasets
from datasets import load_dataset
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd

import transformers
from accelerate import Accelerator
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

task_to_keys = {
    "biasbios": ["title_scrubbed", "title_scrubbed_gender_swapped", "gender_title_scrubbed"],
    "jigsaw": ["comment_text", "gender_swapped_comment"]
}

task_target_labels = {
    "biasbios": "title",
    "jigsaw": "label"
}

task_to_id_key = {
    "biasbios": "idx",
    "jigsaw": "id"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        help="name of the tokenizer",
        required=True,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument("--output_dir", type=str, default=None, required=True, help="Where to store the output.")
    parser.add_argument("--output_filename", type=str, default=None, required=True, help="output filename")
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.test_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."

    return args


def main():
    args = parse_args()
    
    extension = args.test_file.split(".")[-1]
    raw_dataset = load_dataset(extension, data_files=args.test_file)["train"]
    raw_dataset = raw_dataset.add_column("idx", list(range(len(raw_dataset))))
    
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)

    id_to_label = model.config.id2label
    label_to_id = model.config.label2id
    label_column = task_target_labels[args.task_name]
    id_column = task_to_id_key[args.task_name]

    def preprocess_function(examples, column_name):
        # Tokenize the texts
        texts = (examples[column_name],)
        result = tokenizer(*texts, padding=False, max_length=512, truncation=True)
        result["labels"] = [label_to_id[str(l)] for l in examples[label_column]]
        result["idxs"] = examples["idx"]
        return result

    df = None
    # target_label_id = label_to_id[raw_dataset[0][label_column]]

    for column_name in task_to_keys[args.task_name]:
        temp_df = pd.DataFrame({})
        with accelerator.main_process_first():
            processed_dataset = raw_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=raw_dataset.column_names,
                desc="Running tokenizer on dataset",
                fn_kwargs={"column_name": column_name}
            )

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
        eval_dataloader = DataLoader(processed_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

        model.eval()
        samples_seen = 0
        all_preds = []
        all_probs = []
        all_idxs = []
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Running evaluation")):
            with torch.no_grad():
                outputs = model(**{k:v for k, v in batch.items() if k != "idxs"})
            predictions = outputs.logits.argmax(dim=-1)
            probs = outputs.logits.softmax(dim=-1)
            predictions, probs, references, idxs = accelerator.gather((predictions, probs, batch["labels"], batch["idxs"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                # probs = probs[:, target_label_id].detach()
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    # probs = probs[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                    idxs = idxs[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]

            all_preds.extend(predictions.tolist())
            all_probs.extend(probs.tolist())
            all_idxs.extend(idxs.tolist())

        
        if accelerator.is_main_process:
            all_preds = [id_to_label[pred] for pred in all_preds]
            temp_df[f"{column_name}_pred"] = all_preds
            # temp_df[f"{column_name}_prob"] = all_probs
            temp_df["idx"] = all_idxs
            if df is None:
                df = temp_df
            else:
                df = df.merge(temp_df, on="idx")

    if accelerator.is_main_process:
        df["label"] = raw_dataset.select(all_idxs)[label_column]
        df["gender"] = raw_dataset.select(all_idxs)["gender"]
        if args.task_name == "jigsaw":
            df["id"] = raw_dataset.select(all_idxs)["id"]

        df.to_csv(os.path.join(args.output_dir, args.output_filename), index=False)

if __name__ == "__main__":
    main()