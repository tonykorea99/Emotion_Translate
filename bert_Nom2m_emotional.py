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
os.environ["WANDB_API_KEY"] = "e758b93c3e805dafd9d187ec1c0b1b984fe6256f"

from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup, 
    AutoModelForSeq2SeqLM 
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

# 🟢 [수정] Large 모델 적용 (성능 극대화)
MODEL_NAME = "cl-tohoku/bert-large-japanese-v2"   
M2M_MODEL_NAME = "facebook/m2m100_418M" 

EMO_LABELS = [
    "Joy", "Sadness", "Anticipation", "Surprise", 
    "Anger", "Fear", "Disgust", "Trust"
]
id2label = {i: lab for i, lab in enumerate(EMO_LABELS)}
label2id = {lab: i for i, lab in id2label.items()}

# 🟢 [수정] 배치 줄임 (Large 모델 메모리 때문) & 에폭 늘림
BATCH_SIZE = 16       
EVAL_BATCH_SIZE = 32  
LR = 1e-5             # Large 모델은 더 세심하게 학습 (2e-5 -> 1e-5)
NUM_EPOCHS = 5        # 5 에폭 도전
MAX_LEN = 128
SEED = 42
LOG_INTERVAL = 10        
VALIDATION_INTERVAL = 100 

# 🟢 [핵심] 확실한 감정만 학습 (강도 2 이상)
EMOTION_INTENSITY_THRESHOLD = 2 

WANDB_ENTITY = "hails"
WANDB_PROJECT = "Emotional_Traslate_Bert_m2m100"
WANDB_RUN_NAME = "JP_Large_Filtered_Data_Final" 

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
# 3. 데이터 로드 및 전처리 (필터링 적용)
# ---------------------------------------------------------
def process_csv_data(csv_paths: list) -> DatasetDict:
    all_datasets = []
    # (외부 데이터는 강도 정보가 없으므로 그대로 사용)
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
                # 간단 매핑 (필요시 수정)
                mapping = {"happy":"Joy", "sad":"Sadness", "anger":"Anger", "fear":"Fear", "disgust":"Disgust", "surprise":"Surprise", "joy":"Joy", "sadness":"Sadness", "trust":"Trust", "anticipation":"Anticipation"}
                mapped = mapping.get(raw)
                if mapped in EMO_LABELS:
                    row['labels'] = label2id[mapped]
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


def prepare_datasets(csv_paths: list):
    print("\n[Data] WRIME 및 외부 데이터 로드 중...")
    
    # 1. WRIME 로드
    raw_wrime = load_dataset("csv", data_files=WRIME_URL, delimiter="\t")["train"]
    WRIME_SCORE_COLS = [f"Writer_{e}" for e in EMO_LABELS]
    
    def wrime_process(ex):
        scores = [ex[c] for c in WRIME_SCORE_COLS]
        max_score = int(np.max(scores)) 
        ex["labels"] = int(np.argmax(scores)) 
        ex["max_intensity"] = max_score # 강도 저장
        ex["Sentence_std"] = ex.get("Sentence", "")
        return ex

    raw_wrime = raw_wrime.map(wrime_process)
    
    # 🟢 [필터링] 확실한 감정만 남기기
    print(f" -> 필터링 전: {len(raw_wrime)}개")
    raw_wrime = raw_wrime.filter(lambda x: x["max_intensity"] >= EMOTION_INTENSITY_THRESHOLD)
    print(f" -> 필터링 후 (Intensity >= {EMOTION_INTENSITY_THRESHOLD}): {len(raw_wrime)}개")
    
    wrime_splits = {
        "train": raw_wrime.filter(lambda x: x["Train/Dev/Test"]=="train"),
        "validation": raw_wrime.filter(lambda x: x["Train/Dev/Test"]=="dev"),
        "test": raw_wrime.filter(lambda x: x["Train/Dev/Test"]=="test")
    }
    
    for split in wrime_splits:
        wrime_splits[split] = wrime_splits[split].select_columns(["Sentence_std", "labels"])
    
    # 2. CSV 로드
    csv_splits = process_csv_data(csv_paths)
    
    # 3. 합치기
    dataset = DatasetDict({
        "train": wrime_splits["train"],
        "validation": concatenate_datasets([wrime_splits["validation"], csv_splits["validation_csv"]]),
        "test": concatenate_datasets([wrime_splits["test"], csv_splits["test_csv"]])
    })

    print(f" -> Final Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} | Test: {len(dataset['test'])}")

    # 4. 토큰화
    print(f"\n[Tokenizer] 모델: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["Sentence_std"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized_datasets, tokenizer

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
# 5. 메인 함수
# ---------------------------------------------------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device: {device}")

    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "architecture": "JP Large + Filter(Int>=2)", 
            "model": MODEL_NAME,
            "batch_size": BATCH_SIZE, 
            "lr": LR, 
            "epochs": NUM_EPOCHS
        },
    )

    # 데이터 준비
    tokenized_datasets, tokenizer = prepare_datasets(CSV_PATHS)
    
    data_collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator, num_workers=4, pin_memory=True)
    val_loader = DataLoader(tokenized_datasets["validation"], batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=data_collator, num_workers=4)
    test_loader = DataLoader(tokenized_datasets["test"], batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=data_collator, num_workers=4)

    # 클래스 가중치 계산
    print("\n[Main] 클래스 가중치 계산...")
    train_labels = np.array(tokenized_datasets["train"]["labels"])
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
    print(f" -> Class Weights: {weights}")

    # 모델 로드
    print(f"\n[Main] 일본어 Large BERT 로드: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(EMO_LABELS), 
        id2label=id2label, 
        label2id=label2id
    )
    model.to(device)

    # 학습 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    
    global_step = 0
    best_val_acc = 0.0
    final_model_path = "./JP_Large_Filtered_Best"

    print(f"\n[Main] 학습 시작... (Epochs: {NUM_EPOCHS})")

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
                print(f"   [Epoch {epoch+1}] Step {global_step}/{total_steps} | Loss: {loss.item():.4f}")
            
            if global_step > 0 and global_step % VALIDATION_INTERVAL == 0:
                val_loss, val_acc, _ = evaluate(model, val_loader, device, "val", global_step)
                print(f"   >>> [Val] Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                
                # Acc 기준 저장
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f"      ★ Best Acc Updated! ({best_val_acc:.4f})")
                    model.save_pretrained(final_model_path)
                    tokenizer.save_pretrained(final_model_path)

            global_step += 1

        train_loss_epoch = epoch_loss / epoch_examples
        metrics = compute_metrics_detailed(all_train_labels, all_train_preds)
        
        log_dict = {"train/loss_epoch": train_loss_epoch, "train/accuracy": metrics["overall_accuracy"], "train/epoch_end": epoch}
        wandb.log(log_dict, step=global_step)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss_epoch:.4f}, Acc: {metrics['overall_accuracy']:.4f}")
        
        val_loss, val_acc, _ = evaluate(model, val_loader, device, "val", global_step)
        print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(final_model_path)
            tokenizer.save_pretrained(final_model_path)

    print(f"\n[Main] 학습 종료. 최고 성능({best_val_acc:.4f}) 모델을 로드하여 테스트합니다.")
    best_model = AutoModelForSequenceClassification.from_pretrained(final_model_path)
    best_model.to(device)

    print("\n[Main] 최종 테스트 평가...")
    test_loss, test_acc, _ = evaluate(best_model, test_loader, device, "test", global_step)
    print(f"[TEST] Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    wandb.finish()

    # ---------------------------------------------------------
    # 6. 실시간 테스트
    # ---------------------------------------------------------
    print("\n--- 실시간 감정 분석 테스트 (Large Model) ---")
    m2m_tokenizer = AutoTokenizer.from_pretrained(M2M_MODEL_NAME, src_lang="ko")
    m2m_model = AutoModelForSeq2SeqLM.from_pretrained(M2M_MODEL_NAME).to(device)
    m2m_model.eval()
    
    best_model.eval()
    
    RE_KOREAN = re.compile(r'[가-힣]')

    while True:
        try:
            print("\n한국어 또는 일본어 문장을 입력하세요. (종료: '0')")
            input_text = input(" [입력] > ").strip()
            if input_text == "0": break
            if not input_text: continue

            final_input_text = input_text
            
            if RE_KOREAN.search(input_text):
                print(" [감지] 한국어 -> 일본어 번역 중...")
                m2m_tokenizer.src_lang = "ko"
                m2m_inputs = m2m_tokenizer(input_text, return_tensors="pt").to(device)
                with torch.no_grad():
                    gen_tokens = m2m_model.generate(**m2m_inputs, forced_bos_token_id=m2m_tokenizer.get_lang_id("ja"))
                final_input_text = m2m_tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                print(f" [번역된 일본어] {final_input_text}")
            else:
                print(" [감지] 일본어 (직접 입력)")

            inputs = tokenizer(final_input_text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
            with torch.no_grad():
                logits = best_model(**inputs).logits
            
            pred_id = torch.argmax(logits, dim=-1).item()
            print(f" [예측 감정] {id2label[pred_id]}")

        except Exception as e:
            print(f"오류: {e}")
            break

if __name__ == "__main__":
    main()