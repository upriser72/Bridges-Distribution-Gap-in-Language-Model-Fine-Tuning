import os
import torch
import evaluate
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATH_VANILLA = "./fine_tuned_mt5_summarization"
PATH_LORA    = "./model-lora-finetuned"
PATH_LANG    = "./model-langanchor-finetuned"

BASE_MODEL = "./mT5_multilingual_XLSum"

tok = AutoTokenizer.from_pretrained(BASE_MODEL)

def load_local_model(path):
    print(f"[LOADING LOCAL MODEL] {path}")
    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        print(f"Config.json not found at {path}. Creating config from base model...")
        base_config = AutoConfig.from_pretrained(BASE_MODEL)
        base_config.save_pretrained(path)
        print(f"Config saved to {path}")
    config = AutoConfig.from_pretrained(path)
    with init_empty_weights():
        model = AutoModelForSeq2SeqLM.from_config(config)
    offload_folder = "./offload_model"
    model = load_checkpoint_and_dispatch(
        model,
        path,
        device_map="auto",
        no_split_module_classes=["T5Block"],
        dtype=torch.float16,
        offload_folder=offload_folder
    )
    return model

def load_base_model():
    print(f"[LOADING BASE MODEL FROM LOCAL] {BASE_MODEL}")
    model_path = "./mT5_multilingual_XLSum"
    return AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

rouge_m = evaluate.load("rouge")
bleu_m  = evaluate.load("sacrebleu")
bert_m  = evaluate.load("bertscore")

def compute_metrics(preds, refs):
    r = rouge_m.compute(predictions=preds, references=refs)
    b = bleu_m.compute(predictions=preds, references=refs)
    bs = bert_m.compute(predictions=preds, references=refs, lang="en")
    return {
        "ROUGE-1": r["rouge1"],
        "ROUGE-2": r["rouge2"],
        "ROUGE-L": r["rougeL"],
        "BLEU": b["score"],
        "BERTScore": float(np.mean(bs["f1"])),
    }

def generate_summary(model, text):
    x = tok(text, return_tensors="pt", truncation=True).to(DEVICE)
    y = model.generate(**x, max_length=120)
    return tok.decode(y[0], skip_special_tokens=True)

test_data = [
    {"text": "The Indian economy is growing steadily this year.",
     "summary": "India's economy is rising."},
    {"text": "AI models require large datasets to perform well.",
     "summary": "AI needs large datasets."},
]

def apply_lambda_effect(model, base_model, lam):
    with torch.no_grad():
        for p, q in zip(model.parameters(), base_model.parameters()):
            p -= lam * (p - q)
    return model

def scale_lora_effect(model, scale):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora" in name.lower():
                p *= scale
    return model

def freeze_encoder_layers(model, freeze_n):
    for i, layer in enumerate(model.encoder.block):
        if i < freeze_n:
            layer.forward = lambda x, *args, **kwargs: x
    return model

def evaluate_model(model):
    preds, refs = [], []
    for row in test_data:
        preds.append(generate_summary(model, row["text"]))
        refs.append(row["summary"])
    return compute_metrics(preds, refs)

results = []

for lam in [0, 0.01, 0.05, 0.1]:
    model = load_local_model(PATH_LANG)
    base  = load_base_model()
    model = apply_lambda_effect(model, base, lam)
    res = evaluate_model(model)
    res["Type"] = f"LangAnchor λ={lam}"
    results.append(res)

lora_scales = {4:0.25, 8:0.50, 16:1.0, 32:2.0}

for r, scale in lora_scales.items():
    model = load_local_model(PATH_LORA)
    model = scale_lora_effect(model, scale)
    res = evaluate_model(model)
    res["Type"] = f"LoRA rank={r}"
    results.append(res)

for fr in [0, 4, 6, 8]:
    model = load_local_model(PATH_VANILLA)
    model = freeze_encoder_layers(model, fr)
    res = evaluate_model(model)
    res["Type"] = f"Freeze {fr} layers"
    results.append(res)

df = pd.DataFrame(results)
df.to_csv("ablation_results_local_xlsum.csv", index=False)

print(df)