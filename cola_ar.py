import os
import numpy as np
from datasets import load_metric
from functools import partial
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
)
from utils import read_splits
from razdel import tokenize

# Load metrics
ACCURACY = load_metric("accuracy", keep_in_memory=True)
MCC = load_metric("matthews_correlation", keep_in_memory=True)

# Model paths
MODEL_TO_HUB_NAME = {
    "zh-roberta": "./checkpoints_best/chinese-roberta",
    "rubert-base": "sberbank-ai/ruBert-base",
    "rubert-large": "sberbank-ai/ruBert-large",
    "ruroberta-large": "sberbank-ai/ruRoberta-large",
    "xlmr-base": "xlm-roberta-base",
    "rembert": "google/rembert",
}

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    acc_result = ACCURACY.compute(predictions=preds, references=p.label_ids)
    mcc_result = MCC.compute(predictions=preds, references=p.label_ids)
    return {"accuracy": acc_result["accuracy"], "mcc": mcc_result["matthews_correlation"]}

def preprocess_examples(examples, tokenizer):
    result = tokenizer(examples["sentence"], padding=True, truncation=True, max_length=512, return_tensors="pt")
    if "label" in examples:
        result["label"] = examples["label"]
    result["length"] = [len(list(tokenize(sentence))) for sentence in examples["sentence"]]
    return result

def main(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    splits = read_splits(as_datasets=True)

    tokenized_splits = splits.map(
        partial(preprocess_examples, tokenizer=tokenizer),
        batched=True,
        remove_columns=["sentence"],
        keep_in_memory=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    training_args = TrainingArguments(
        output_dir="./checkpoints/temp",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=0,
        seed=42,
        dataloader_num_workers=4,
        group_by_length=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_splits["test"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    eval_result = trainer.evaluate()
    print(f"Evaluation results: {eval_result}")

if __name__ == "__main__":
    model_name = "roberta-base"
    model_path = "/home/a401/桌面/办公/YANG/cola/CoLA-baselines-master/multi/checkpoints"  # Replace with your model path
    main(model_name, model_path)
