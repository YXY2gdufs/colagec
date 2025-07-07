import os
from functools import partial
import numpy as np
from datasets import load_metric
from razdel import tokenize
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
    BertTokenizer,
)

MODEL_TO_HUB_NAME = {
    "zh-roberta": "./checkpoints_best/chinese-roberta",
    "rubert-base": "sberbank-ai/ruBert-base",
    "rubert-large": "sberbank-ai/ruBert-large",
    "ruroberta-large": "sberbank-ai/ruRoberta-large",
    "xlmr-base": "xlm-roberta-base",
    "rembert": "google/rembert",
}


from utils import read_splits

ACCURACY = load_metric("accuracy", keep_in_memory=True)
MCC = load_metric("matthews_correlation", keep_in_memory=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

N_SEEDS = 10
N_EPOCHS = 5
LR_VALUES = (1e-5, 3e-5, 5e-5)
DECAY_VALUES = (1e-4, 1e-2, 0.1)
BATCH_SIZES = (32, 64)

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

def main(model_name):
    if "rubert" in model_name:
        tokenizer = BertTokenizer.from_pretrained(MODEL_TO_HUB_NAME[model_name])
    else:
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_HUB_NAME[model_name])
        tokenizer = AutoTokenizer.from_pretrained('./model/roberta-base', model_path=True)
    splits = read_splits(as_datasets=True)
    tokenized_splits = splits.map(partial(preprocess_examples, tokenizer=tokenizer), batched=True, remove_columns=["sentence"], keep_in_memory=True)
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    dev_metrics_per_run = np.empty((N_SEEDS, len(LR_VALUES), len(DECAY_VALUES), len(BATCH_SIZES), 2))

    for i, learning_rate in enumerate(LR_VALUES):
        for j, weight_decay in enumerate(DECAY_VALUES):
            for k, batch_size in enumerate(BATCH_SIZES):
                for seed in range(N_SEEDS):
                    set_seed(seed)
                    model = AutoModelForSequenceClassification.from_pretrained('./model/xlmr-base')
                    training_args = TrainingArguments(
                        output_dir="./checkpoints",
                        overwrite_output_dir=True,
                        evaluation_strategy="epoch",
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        num_train_epochs=N_EPOCHS,
                        warmup_ratio=0.1,
                        optim="adamw_torch",
                        save_strategy="epoch",
                        save_total_limit=0,  # 不保存模型
                        seed=seed,
                        fp16=False,
                        fp16_full_eval=False,
                        tf32=False,
                        dataloader_num_workers=4,
                        group_by_length=True,
                        report_to="none",
                        load_best_model_at_end=False,  # 不加载最佳模型
                        metric_for_best_model="eval_mcc",
                    )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_splits["train"],
                        eval_dataset=tokenized_splits["dev"],
                        compute_metrics=compute_metrics,
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                    )

                    trainer.train()
                    dev_predictions = trainer.predict(test_dataset=tokenized_splits["dev"])
                    print(f"Dev accuracy: {dev_predictions.metrics['accuracy']}, Dev MCC: {dev_predictions.metrics['mcc']}")
                    dev_metrics_per_run[seed, i, j, k] = (dev_predictions.metrics["accuracy"], dev_predictions.metrics["mcc"])

    # 保存验证集上的得分
    np.save("dev_metrics_per_run.npy", dev_metrics_per_run)
if __name__ == "__main__":
    model_name = "xlmr-base"
    main(model_name)

