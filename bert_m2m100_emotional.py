import os
import random
import re
import json
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from collections import Counter

# ---------------------------------------------------------
# [중요] 시스템 설정
# ---------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# [!!! 중요 !!!] W&B API Key 설정
os.environ["WANDB_API_KEY"] = "e758b93c3e805dafd9d187ec1c0b1b984fe6256f"

from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,          
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup, 
)
from sklearn.metrics import accuracy_score, recall_score, f1_score
import wandb

# ---------------------------------------------------------
# 1. 설정 및 하이퍼파라미터
# ---------------------------------------------------------
WRIME_URL = "https://raw.githubusercontent.com/ids-cv/wrime/master/wrime-ver1.tsv"
CSV_PATHS = [
    "/home/hail/emotionproject/datasets/jvnv_v1/transcription.csv",
    "/home/hail/emotionproject/datasets/japanese_emotions.csv",
]
FREEFORM_TXT = "/home/hail/emotionproject/datasets/ja_augmented_lexicon.txt"

TRANSLATED_DATASET_DIR = "./translated_dataset_m2m_xlmroberta"
JSONL_DIR = "./wrime_jsonl_xlmroberta"

M2M_MODEL_NAME = "facebook/m2m100_418M"       
MODEL_NAME = "xlm-roberta-base"   # XLM-RoBERTa 유지

EMO_LABELS = [
    "Joy", "Sadness", "Anticipation", "Surprise", 
    "Anger", "Fear", "Disgust", "Trust"
]
id2label = {i: lab for i, lab in enumerate(EMO_LABELS)}
label2id = {lab: i for i, lab in id2label.items()}

# 하이퍼파라미터
BATCH_SIZE = 16       
EVAL_BATCH_SIZE = 32  
MAP_BATCH_SIZE = 16   
LR = 2e-5             
NUM_EPOCHS = 3        # 🟢 [수정] 3 에포크로 고정
MAX_LEN = 128
SEED = 42
LOG_INTERVAL = 10        
VALIDATION_INTERVAL = 100 

# W&B 설정
WANDB_ENTITY = "hails"
WANDB_PROJECT = "Emotional_Traslate_Bert_m2m100"
WANDB_RUN_NAME = "XLM_RoBERTa_Fixed_Epoch_3" # 🟢 [수정] 실행 이름 변경 (비교용)

# ---------------------------------------------------------
# 2. 유틸리티 함수
# ---------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics_detailed(all_labels: list, all_preds: list) -> dict:
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    overall_acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    
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
        "f1_macro": f1_macro,
        "per_class_recall": {
            id2label[i]: float(per_class_recall[i]) for i in range(num_labels)
        },
    }
    return result

# ---------------------------------------------------------
# 3. 데이터 로드 및 전처리
# ---------------------------------------------------------
def process_csv_data(csv_paths: list) -> DatasetDict:
    all_datasets = []
    RAW_TO_EMO_MAP = {
        "happy": "Joy", "sad": "Sadness", "anger": "Anger", "fear": "Fear", 
        "disgust": "Disgust", "surprise": "Surprise", 
        "joy": "Joy", "sadness": "Sadness", "anticipation": "Anticipation", 
        "trust": "Trust",
    }

    for path in csv_paths:
        if not os.path.exists(path): continue
        try: df = pd.read_csv(path)
        except: continue
        
        if 'transcription.csv' in path: 
            df = df.rename(columns={'text': 'Sentence_std', 'emotion_label': 'raw_label'})
        elif 'japanese_emotions.csv' in path: 
            df = df.rename(columns={'tweet': 'Sentence_std', 'sentiment': 'raw_label'})
        
        if 'Sentence_std' in df.columns and 'raw_label' in df.columns:
            def map_and_filter(row):
                raw = str(row['raw_label']).strip().lower()
                mapped_label = RAW_TO_EMO_MAP.get(raw, None)
                if mapped_label in EMO_LABELS:
                    row['labels'] = label2id[mapped_label]
                    return row
                return None 
            
            df = df[['Sentence_std', 'raw_label']].copy()
            ds = Dataset.from_pandas(df)
            ds = ds.map(map_and_filter, remove_columns=['raw_label'])
            ds = ds.filter(lambda x: x.get('labels') is not None and x.get('Sentence_std') is not None)
            if len(ds) > 0: all_datasets.append(ds)

    if not all_datasets:
        empty_ds = Dataset.from_dict({"Sentence_std": [], "labels": []})
        return DatasetDict({"validation_csv": empty_ds, "test_csv": empty_ds})

    full_csv_ds = concatenate_datasets(all_datasets)
    split_ds = full_csv_ds.train_test_split(test_size=0.5, seed=SEED)
    return DatasetDict({"validation_csv": split_ds["train"], "test_csv": split_ds["test"]})


def prepare_raw_dataset(csv_paths: list) -> DatasetDict:
    print("\n1. WRIME 데이터셋 로드 및 분리 중...")
    raw_wrime = load_dataset("csv", data_files=WRIME_URL, delimiter="\t")["train"]
    WRIME_SCORE_COLS = [f"Writer_{e}" for e in EMO_LABELS]
    
    def wrime_process(ex):
        scores = [ex[c] for c in WRIME_SCORE_COLS]
        ex["labels"] = int(np.argmax(scores)) 
        ex["Sentence_std"] = ex.get("Sentence", "")
        return ex

    raw_wrime = raw_wrime.map(wrime_process)
    
    wrime_splits = {
        "train": raw_wrime.filter(lambda x: x["Train/Dev/Test"]=="train"),
        "validation": raw_wrime.filter(lambda x: x["Train/Dev/Test"]=="dev"),
        "test": raw_wrime.filter(lambda x: x["Train/Dev/Test"]=="test")
    }
    
    for split in wrime_splits:
        wrime_splits[split] = wrime_splits[split].select_columns(["Sentence_std", "labels"])
    
    print("\n2. 외부 CSV 데이터 로드 중...")
    csv_splits = process_csv_data(csv_paths)
    
    print("\n3. 데이터셋 병합 (Train=WRIME, Val/Test=WRIME+CSV)...")
    train_ds = wrime_splits["train"]
    val_ds = concatenate_datasets([wrime_splits["validation"], csv_splits["validation_csv"]])
    test_ds = concatenate_datasets([wrime_splits["test"], csv_splits["test_csv"]])
    
    return DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})


def prepare_translated_dataloaders(device):
    if os.path.exists(TRANSLATED_DATASET_DIR):
        print(f"\n[Cache] 번역된 데이터셋 로드: {TRANSLATED_DATASET_DIR}")
        tokenized_dataset = DatasetDict.load_from_disk(TRANSLATED_DATASET_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    else:
        raw_dataset = prepare_raw_dataset(CSV_PATHS)
        print("\n[M2M100] 번역기 로드 및 전체 데이터 번역 시작 (JA -> EN)...")
        
        m2m_tokenizer = AutoTokenizer.from_pretrained(M2M_MODEL_NAME, src_lang="ja")
        m2m_model = AutoModelForSeq2SeqLM.from_pretrained(M2M_MODEL_NAME).to(device)
        m2m_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        def translate_and_tokenize(examples):
            torch.cuda.empty_cache()
            m2m_inputs = m2m_tokenizer(
                examples["Sentence_std"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN
            ).to(device)

            with torch.no_grad():
                generated_tokens = m2m_model.generate(
                    **m2m_inputs,
                    forced_bos_token_id=m2m_tokenizer.get_lang_id("en")
                )
            translated_texts = m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            tokenized_inputs = tokenizer(
                translated_texts,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
            )
            return tokenized_inputs

        tokenized_dataset = raw_dataset.map(
            translate_and_tokenize,
            batched=True,
            batch_size=MAP_BATCH_SIZE, 
            remove_columns=["Sentence_std"]
        )
        print(f"\n[Cache] 번역된 데이터셋 저장: {TRANSLATED_DATASET_DIR}")
        tokenized_dataset.save_to_disk(TRANSLATED_DATASET_DIR)
        del m2m_model, m2m_tokenizer
        torch.cuda.empty_cache()

    cols = ["input_ids", "attention_mask", "labels"]
    tokenized_dataset.set_format(type="torch", columns=cols)
    data_collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(tokenized_dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator, num_workers=4, pin_memory=True)
    val_loader = DataLoader(tokenized_dataset["validation"], batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=data_collator, num_workers=4)
    test_loader = DataLoader(tokenized_dataset["test"], batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=data_collator, num_workers=4)

    return train_loader, val_loader, test_loader, tokenizer, tokenized_dataset["train"]

# ---------------------------------------------------------
# 4. 평가 함수
# ---------------------------------------------------------
def evaluate(model, dataloader, device, split_name, global_step):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            labels = batch["labels"]

            wandb.log({f"{split_name}/loss_batch": loss.item(), f"{split_name}/batch_idx": batch_idx}, step=global_step)

            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)
            preds = torch.argmax(logits, dim=-1)
            
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    if total_examples == 0: return 0.0, 0.0, {}
    
    avg_loss = total_loss / total_examples
    metrics = compute_metrics_detailed(all_labels, all_preds)
    
    log_dict = {
        f"{split_name}/loss_epoch": avg_loss,
        f"{split_name}/accuracy": metrics["overall_accuracy"],
        f"{split_name}/f1_macro": metrics["f1_macro"],
    }
    for emo_name, acc in metrics["per_class_recall"].items():
        log_dict[f"{split_name}/acc_{emo_name}"] = acc

    wandb.log(log_dict, step=global_step)
    
    model.train() 
    return avg_loss, metrics["overall_accuracy"], metrics["per_class_recall"]

# ---------------------------------------------------------
# 5. 메인 함수 (Early Stopping 제거 & Epoch 고정)
# ---------------------------------------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device: {device}")

    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={"architecture": "M2M(JA->EN) -> XLM-R", "batch_size": BATCH_SIZE, "lr": LR, "epochs": NUM_EPOCHS},
    )

    train_loader, val_loader, test_loader, tokenizer, train_ds_obj = prepare_translated_dataloaders(device)

    train_labels = np.array(train_ds_obj["labels"]) 
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    num_classes = len(EMO_LABELS)

    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        w = total_samples / (num_classes * count) if count > 0 else 1.0
        weights.append(w)
    
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    print(f" -> 적용된 Class Weights: {weights}")

    print(f"\n[Main] Model 로드: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(EMO_LABELS), id2label=id2label, label2id=label2id)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    
    global_step = 0
    final_model_path = "./XLM_RoBERTa_M2M_Fixed_3_Final" # 🟢 [수정] 저장 경로

    print(f"\n[Main] 학습 시작... (Total Steps: {total_steps}, Epochs: {NUM_EPOCHS})")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_examples = 0
        all_train_labels, all_train_preds = [], []

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]

            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * labels.size(0)
            epoch_examples += labels.size(0)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_train_labels.extend(labels.cpu().tolist())
            all_train_preds.extend(preds.cpu().tolist())

            wandb.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=global_step)
            
            if global_step % LOG_INTERVAL == 0:
                print(f"   [Epoch {epoch+1}] Step {global_step}/{total_steps} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.8f}")
            
            if global_step > 0 and global_step % VALIDATION_INTERVAL == 0:
                val_loss, val_acc, _ = evaluate(model, val_loader, device, "val", global_step)
                print(f"   >>> [Step {global_step} Validation] Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            global_step += 1

        train_loss_epoch = epoch_loss / epoch_examples
        metrics = compute_metrics_detailed(all_train_labels, all_train_preds)
        
        log_dict = {"train/loss_epoch": train_loss_epoch, "train/accuracy": metrics["overall_accuracy"], "train/epoch_end": epoch}
        for emo_name, acc in metrics["per_class_recall"].items():
            log_dict[f"train/acc_{emo_name}"] = acc
        wandb.log(log_dict, step=global_step)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss_epoch:.4f}, Acc: {metrics['overall_accuracy']:.4f}")
        
        # 🟢 [수정] Early Stopping 로직 제거 -> 무조건 평가하고 다음 에포크 진행
        val_loss, val_acc, _ = evaluate(model, val_loader, device, "val", global_step)
        print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    print("\n[Main] 학습 종료. 최종 모델 저장...")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print("\n[Main] 최종 테스트 평가 (Final Model 로드)...")
    final_model = AutoModelForSequenceClassification.from_pretrained(final_model_path)
    final_model.to(device)
    test_loss, test_acc, _ = evaluate(final_model, test_loader, device, "test", global_step)
    print(f"[TEST] Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    wandb.finish()

    # ---------------------------------------------------------
    # 6. 실시간 테스트
    # ---------------------------------------------------------
    print("\n--- 실시간 번역 및 감정 분석 테스트 (XLM-R) ---")
    m2m_tokenizer = AutoTokenizer.from_pretrained(M2M_MODEL_NAME, src_lang="ja")
    m2m_model = AutoModelForSeq2SeqLM.from_pretrained(M2M_MODEL_NAME).to(device)
    m2m_model.eval()
    model.eval()
    RE_KOREAN = re.compile(r'[가-힣]')
    RE_JAPANESE = re.compile(r'[ぁ-ゟ゠-ヿ一-龯]') 

    while True:
        try:
            print("\n한국어 또는 일본어 문장을 입력하세요. (종료: '0')")
            input_text = input(" [입력] > ").strip()
            if input_text == "0": break
            if not input_text: continue

            src_lang, tgt_lang_trans = None, None
            if RE_KOREAN.search(input_text): src_lang, tgt_lang_trans = "ko", "ja"
            elif RE_JAPANESE.search(input_text): src_lang, tgt_lang_trans = "ja", "ko"
            else: continue

            m2m_tokenizer.src_lang = src_lang
            m2m_inputs = m2m_tokenizer(input_text, return_tensors="pt").to(device)
            with torch.no_grad():
                trans_tokens = m2m_model.generate(**m2m_inputs, forced_bos_token_id=m2m_tokenizer.get_lang_id(tgt_lang_trans))
            print(f" [번역] {m2m_tokenizer.decode(trans_tokens[0], skip_special_tokens=True)}")

            m2m_tokenizer.src_lang = src_lang
            with torch.no_grad():
                en_tokens = m2m_model.generate(**m2m_inputs, forced_bos_token_id=m2m_tokenizer.get_lang_id("en"))
            en_text = m2m_tokenizer.decode(en_tokens[0], skip_special_tokens=True)
            
            inputs = tokenizer(en_text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            print(f" [영어 변환] {en_text}")
            print(f" [예측 감정] {id2label[torch.argmax(logits, dim=-1).item()]}")

        except Exception as e:
            print(f"오류: {e}")
            break

if __name__ == "__main__":
    main()