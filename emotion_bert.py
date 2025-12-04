#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, recall_score
import wandb


TRANSCRIPTION_CSV = "/home/hail/emotionproject/datasets/jvnv_v1/transcription.csv"
WRIME_URL = "https://raw.githubusercontent.com/ids-cv/wrime/master/wrime-ver1.tsv"

EMO_LABELS = [
    "Joy",
    "Sadness",
    "Anticipation",
    "Surprise",
    "Anger",
    "Fear",
    "Disgust",
    "Trust",
]
id2label = {i: lab for i, lab in enumerate(EMO_LABELS)}
label2id = {lab: i for i, lab in id2label.items()}

TRANSCRIPTION_MAP = {
    "happy": "Joy",
    "sad": "Sadness",
    "anger": "Anger",
    "fear": "Fear",
    "disgust": "Disgust",
    "surprise": "Surprise",
}

MODEL_NAME = "ku-nlp/deberta-v2-base-japanese"
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LR = 2e-5
NUM_EPOCHS = 3
MAX_LEN = 128
SEED = 42

WANDB_PROJECT = "emotion"
WANDB_RUN_NAME = "wrime_bert_7"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics_with_sklearn(
    all_labels: List[np.ndarray], all_preds: List[np.ndarray]
) -> Dict[str, float]:
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    overall_acc = accuracy_score(y_true, y_pred)
    num_labels = len(EMO_LABELS)
    per_class_recall = recall_score(
        y_true,
        y_pred,
        labels=list(range(num_labels)),
        average=None,
        zero_division=0.0,
    )

    result = {
        "overall_accuracy": overall_acc,
        "per_class_recall": {
            id2label[i]: float(per_class_recall[i]) for i in range(num_labels)
        },
    }
    return result


def load_wrime_ver1() -> Dict[str, Dataset]:
    raw = load_dataset(
        "csv",
        data_files=WRIME_URL,
        delimiter="\t",
    )["train"]

    writer_emotion_cols = [f"Writer_{emo}" for emo in EMO_LABELS]

    def add_labels(example):
        scores = [example[col] for col in writer_emotion_cols]
        idx = int(np.argmax(scores))
        example["emotion_name"] = EMO_LABELS[idx]
        example["labels"] = idx
        example["sentence"] = example["Sentence"]
        return example

    raw = raw.map(add_labels)

    train_ds = raw.filter(lambda x: x["Train/Dev/Test"] == "train")
    val_ds = raw.filter(lambda x: x["Train/Dev/Test"] == "dev")
    test_ds = raw.filter(lambda x: x["Train/Dev/Test"] == "test")

    keep_cols = ["sentence", "emotion_name", "labels"]

    def drop_others(ds):
        drop_cols = [c for c in ds.column_names if c not in keep_cols]
        if drop_cols:
            ds = ds.remove_columns(drop_cols)
        return ds

    train_ds = drop_others(train_ds)
    val_ds = drop_others(val_ds)
    test_ds = drop_others(test_ds)

    return {
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    }


def load_transcription_csv(path: str) -> Dataset:
    df_raw = pd.read_csv(path, header=None, names=["raw"])
    split = df_raw["raw"].str.split("|", n=2, expand=True)
    split.columns = ["utt_id", "short", "sentence"]
    split["raw_label"] = split["utt_id"].str.split("_").str[0]
    split["emotion_name"] = split["raw_label"].map(TRANSCRIPTION_MAP)
    split = split[split["emotion_name"].notna()].copy()
    split["labels"] = split["emotion_name"].map(label2id)

    ds = Dataset.from_pandas(
        split[["sentence", "emotion_name", "labels"]],
        preserve_index=False,
    )
    return ds


def tokenize_function(batch, tokenizer, max_length: int):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding=False,
        max_length=max_length,
    )


def prepare_dataloaders(tokenizer):
    wrime = load_wrime_ver1()
    trans_all = load_transcription_csv(TRANSCRIPTION_CSV)

    train_ds = wrime["train"]
    val_ds = wrime["validation"]
    test_ds = concatenate_datasets([wrime["test"], trans_all])

    def tok_fn(batch):
        return tokenize_function(batch, tokenizer, MAX_LEN)

    train_tok = train_ds.map(tok_fn, batched=True)
    val_tok = val_ds.map(tok_fn, batched=True)
    test_tok = test_ds.map(tok_fn, batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    train_tok.set_format(type="torch", columns=cols)
    val_tok.set_format(type="torch", columns=cols)
    test_tok.set_format(type="torch", columns=cols)

    data_collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        train_tok, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator
    )
    val_loader = DataLoader(
        val_tok, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=data_collator
    )
    test_loader = DataLoader(
        test_tok, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=data_collator
    )

    return train_loader, val_loader, test_loader


def evaluate(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str,
    global_step: int,
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            labels = batch["labels"]

            wandb.log(
                {
                    f"{split_name}/loss_batch": loss.item(),
                    f"{split_name}/batch_idx": batch_idx,
                },
                step=global_step,
            )

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            preds = torch.argmax(logits, dim=-1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    if total_examples == 0:
        avg_loss = 0.0
        overall_acc = 0.0
        per_class_acc = {name: 0.0 for name in EMO_LABELS}
    else:
        avg_loss = total_loss / total_examples
        metrics = compute_metrics_with_sklearn(all_labels, all_preds)
        overall_acc = metrics["overall_accuracy"]
        per_class_acc = metrics["per_class_recall"]

    log_dict = {
        f"{split_name}/loss_epoch": avg_loss,
        f"{split_name}/accuracy": overall_acc,
    }
    for emo_name, acc in per_class_acc.items():
        log_dict[f"{split_name}/acc_{emo_name}"] = acc

    wandb.log(log_dict, step=global_step)

    return avg_loss, overall_acc, per_class_acc


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        entity="hails",
        config={
            "model_name": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "lr": LR,
            "epochs": NUM_EPOCHS,
            "max_len": MAX_LEN,
            "labels": EMO_LABELS,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, val_loader, test_loader = prepare_dataloaders(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(EMO_LABELS),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    global_step = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_examples = 0

        all_train_labels: List[np.ndarray] = []
        all_train_preds: List[np.ndarray] = []

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            labels = batch["labels"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_examples += batch_size

            preds = torch.argmax(logits, dim=-1)

            all_train_labels.append(labels.cpu().numpy())
            all_train_preds.append(preds.cpu().numpy())

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/epoch": epoch,
                    "train/batch_idx": batch_idx,
                },
                step=global_step,
            )

            global_step += 1

        if epoch_examples == 0:
            train_loss_epoch = 0.0
            train_acc_epoch = 0.0
            train_per_class_acc = {name: 0.0 for name in EMO_LABELS}
        else:
            train_loss_epoch = epoch_loss / epoch_examples
            metrics = compute_metrics_with_sklearn(
                all_train_labels, all_train_preds
            )
            train_acc_epoch = metrics["overall_accuracy"]
            train_per_class_acc = metrics["per_class_recall"]

        log_dict = {
            "train/loss_epoch": train_loss_epoch,
            "train/accuracy": train_acc_epoch,
            "train/epoch_end": epoch,
        }
        for emo_name, acc in train_per_class_acc.items():
            log_dict[f"train/acc_{emo_name}"] = acc

        wandb.log(log_dict, step=global_step)

        print(
            f"[Epoch {epoch+1}/{NUM_EPOCHS}] "
            f"train_loss={train_loss_epoch:.4f}, train_acc={train_acc_epoch:.4f}"
        )

        val_loss, val_acc, _ = evaluate(
            model, val_loader, device, split_name="val", global_step=global_step
        )
        print(
            f"[Epoch {epoch+1}/{NUM_EPOCHS}] "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    test_loss, test_acc, _ = evaluate(
        model, test_loader, device, split_name="test", global_step=global_step
    )
    print(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
