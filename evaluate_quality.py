import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from sacrebleu.metrics import BLEU
from bert_score import score

M2M_MODEL = "facebook/m2m100_418M"
EMO_MODEL = "./JP_Large_Filtered_Best"

EMO_LABELS = [
    "Joy", "Sadness", "Anticipation", "Surprise",
    "Anger", "Fear", "Disgust", "Trust"
]

TEST_DATA = [
    {"text": "아 진짜 짜증나게 하네, 다 망쳤어.", "label": "Anger"},
    {"text": "합격해서 너무 기뻐! 진짜 날아갈 것 같아.", "label": "Joy"},
    {"text": "집에 혼자 있으니까 너무 외롭고 우울해.", "label": "Sadness"},
    {"text": "이 음식에서는 정말 역겨운 냄새가 나.", "label": "Disgust"},
    {"text": "갑자기 튀어나와서 진짜 깜짝 놀랐잖아!", "label": "Surprise"},
    {"text": "밤길이 너무 어두워서 무서워 죽겠어.", "label": "Fear"},
    {"text": "앞으로의 미래가 너무 기대된다.", "label": "Anticipation"},
    {"text": "당신을 전적으로 믿습니다.", "label": "Trust"}
]


class EmotionPivotTranslator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tok = AutoTokenizer.from_pretrained(M2M_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(M2M_MODEL).to(self.device)
        self.model.eval()

        self.emo_tok = AutoTokenizer.from_pretrained(EMO_MODEL)
        self.emo_model = AutoModelForSequenceClassification.from_pretrained(EMO_MODEL).to(self.device)
        self.emo_model.eval()

        self.bleu = BLEU()

    def _translate(self, text, src, tgt, beams=5, ret=1):
        self.tok.src_lang = src
        inputs = self.tok(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tok.get_lang_id(tgt),
                num_beams=beams,
                num_return_sequences=ret,
                max_length=256,
                early_stopping=True
            )
        return [self.tok.decode(o, skip_special_tokens=True) for o in out]

    def _emo_score(self, ja, emotion):
        idx = EMO_LABELS.index(emotion)
        inputs = self.emo_tok(ja, return_tensors="pt", truncation=True, max_length=128).to(self.device)

        with torch.no_grad():
            logits = self.emo_model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        return probs[idx].item()

    def _emo_predict_label(self, ja):
        inputs = self.emo_tok(ja, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            logits = self.emo_model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()
        return EMO_LABELS[pred_id]

    def ko_to_ja(self, ko_list, emotion, k=5):
        result = []
        for ko in ko_list:
            en = self._translate(ko, "ko", "en")[0]
            ja_candidates = self._translate(en, "en", "ja", beams=k, ret=k)

            best = None
            best_score = -1.0
            for ja in ja_candidates:
                s = self._emo_score(ja, emotion)
                if s > best_score:
                    best_score = s
                    best = ja

            result.append({
                "ko": ko,
                "en": en,
                "best_ja": best,
                "candidates": ja_candidates
            })
        return result

    def run_evaluation(self, test_data, k=5):
        results = []

        print("\n" + "=" * 100)
        print(f"{'원문 (KR)':<40} | {'번역 (JA)':<40} | {'Target':<12} | {'Pred':<12}")
        print("=" * 100)

        for item in test_data:
            ko_text = item["text"]
            true_label = item["label"]

            trans = self.ko_to_ja([ko_text], emotion=true_label, k=k)[0]
            ja_text = trans["best_ja"]

            back_en = self._translate(ja_text, "ja", "en")[0]
            back_ko = self._translate(back_en, "en", "ko")[0]

            pred_label = self._emo_predict_label(ja_text)
            is_match = (pred_label == true_label)

            print(f"{ko_text:<40} | {ja_text:<40} | {true_label:<12} | {pred_label:<12} {'O' if is_match else 'X'}")

            results.append({
                "original": ko_text,
                "back_translated": back_ko,
                "match": is_match
            })

        refs = [r["original"] for r in results]
        hyps = [r["back_translated"] for r in results]

        emo_acc = sum(r["match"] for r in results) / len(results) * 100.0

        try:
            P, R, F1 = score(hyps, refs, lang="ko", verbose=False)
            bert_score_avg = F1.mean().item() * 100.0
        except:
            bert_score_avg = 0.0

        bleu_score = self.bleu.corpus_score(hyps, [refs]).score

        print("\n" + "=" * 60)
        print("  [Results]")
        print("=" * 60)
        print(f" 1. 감정 보존율 (Emotion Consistency):  {emo_acc:.2f}%")
        print(f" 2. 의미 보존율 (BERTScore):          {bert_score_avg:.2f}점")
        print(f" 3. 역번역 정확도 (BLEU):             {bleu_score:.2f}점")
        print("=" * 60)


if __name__ == "__main__":
    tr = EmotionPivotTranslator()
    tr.run_evaluation(TEST_DATA, k=5)
