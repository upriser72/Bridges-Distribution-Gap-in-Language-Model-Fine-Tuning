# Preserving Multilingual Geometry During Fine-Tuning

> Comparing full fine-tuning, LoRA, and a proposed hidden-state anchoring method for preserving cross-lingual capabilities in mT5.

This research project studies **catastrophic forgetting and representation drift in multilingual sequence-to-sequence models**.

The project fine-tunes a pretrained **mT5 multilingual summarization model** using **English-only summarization data**, and investigates how fine-tuning affects the model's multilingual capabilities and internal representation geometry.

Three approaches are compared:

1. **Vanilla Full Fine-Tuning** — updates all model parameters.
2. **LoRA (Low-Rank Adaptation)** — adds low-rank trainable adapters to attention projections.
3. **LangAnchor** — the proposed method, which regularizes fine-tuned hidden states against the pretrained model's representations.

# Project Overview

Multilingual pretrained models such as mT5 learn shared representations across many languages.

However, when such a model is fine-tuned on a narrow dataset containing only one language, its pretrained multilingual representation space may change.

This project investigates the following question:

> **Can a multilingual model be fine-tuned for a downstream task while preserving the multilingual knowledge learned during pretraining?**

To investigate this, an mT5 multilingual summarization model is fine-tuned using **English-only XLSum data**.

The resulting models are evaluated on multiple languages to study:

* downstream summarization performance,
* multilingual generalization,
* perplexity,
* and representation stability.

---

# Research Problem

Fine-tuning a pretrained multilingual model can create a trade-off:

```text
                         Fine-Tuning
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          Task Adaptation          Knowledge Preservation
                 │                         │
                 ▼                         ▼
        Better downstream          Preserve multilingual
          task performance             representations
                 │                         │
                 └────────────┬────────────┘
                              ▼
                    Representation Drift
```

When the model is trained only on English data, its internal representations may move away from the multilingual geometry learned during pretraining.

This phenomenon is investigated as a form of **catastrophic forgetting / multilingual degradation**.

---

# Fine-Tuning Approach

The project uses **Supervised Fine-Tuning (SFT)**.

The training data contains paired examples:

```text
Article ─────────────────► Reference Summary
   X                              Y
   │                              │
   └──────── Supervised Training ┘
                  │
                  ▼
             Fine-tune mT5
```

For each article-summary pair:

```text
Article
   │
   ▼
mT5 Encoder
   │
   ▼
mT5 Decoder
   │
   ▼
Predicted Summary
   │
   ▼
Compare with Reference Summary
   │
   ▼
Token-Level Cross-Entropy Loss
   │
   ▼
Backpropagation
   │
   ▼
Update Model Parameters
```

The project therefore uses:

> **Supervised sequence-to-sequence fine-tuning for abstractive summarization.**

The task prefix:

```text
summarize:
```

is used to indicate the summarization task.

---

# Objectives

The main objectives are:

* Fine-tune a pretrained multilingual mT5 model for abstractive summarization.
* Study the effect of English-only fine-tuning on multilingual capabilities.
* Compare full-parameter fine-tuning with parameter-efficient LoRA.
* Introduce LangAnchor as a hidden-state regularization method.
* Reduce hidden-state representation drift.
* Evaluate summarization quality using ROUGE, BLEU, and BERTScore.
* Measure perplexity.
* Evaluate multilingual behavior across multiple languages.
* Study the trade-off between downstream adaptation and multilingual knowledge preservation.

---

# Dataset

The project uses the **XLSum multilingual summarization dataset**.

The main experimental setup uses **10,000 English article-summary pairs**.

```text
             10,000 English XLSum Samples
                         │
                         ▼
                  Data Cleaning
                         │
                         ▼
                 Dataset Split
                         │
                         ▼
              SentencePiece Tokenizer
                         │
                         ▼
                    mT5 Model
```

Each training example contains:

```text
Input  → News Article
Target → Reference Summary
```

The model learns to generate the reference summary from the article.

---

# System Architecture

The complete system consists of a pretrained mT5 model, supervised fine-tuning, three alternative adaptation strategies, and multilingual evaluation.

```text
                         ┌──────────────────────────────┐
                         │     Pretrained mT5 Model     │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │      English XLSum Data      │
                         │                              │
                         │     Article → Summary        │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │     SentencePiece Tokenizer  │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
             ┌──────────────────────────┼──────────────────────────┐
             │                          │                          │
             ▼                          ▼                          ▼
   ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
   │ Vanilla Fine-Tuning│      │       LoRA        │      │     LangAnchor     │
   │                   │      │                   │      │                   │
   │ Update ALL        │      │ Frozen mT5        │      │ Fine-tuning       │
   │ parameters        │      │ + LoRA adapters    │      │ + Anchor loss     │
   └─────────┬─────────┘      └─────────┬─────────┘      └─────────┬─────────┘
             │                          │                          │
             ▼                          ▼                          ▼
   ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
   │ Vanilla Model     │      │ LoRA Model        │      │ LangAnchor Model  │
   └─────────┬─────────┘      └─────────┬─────────┘      └─────────┬─────────┘
             │                          │                          │
             └──────────────────────────┼──────────────────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │        Model Evaluation      │
                         │                              │
                         │ ROUGE | BLEU | BERTScore     │
                         │ Perplexity | Drift           │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │    Multilingual Evaluation   │
                         │                              │
                         │ English | French | Hindi     │
                         │ Spanish                      │
                         └──────────────────────────────┘
```

The three fine-tuning approaches are **independent experimental branches** starting from the same pretrained mT5 model. **Vanilla Fine-Tuning is not followed by LoRA, and LoRA is not followed by LangAnchor.**

---

# 1. Vanilla Full-Parameter Fine-Tuning

Vanilla fine-tuning updates all model parameters.

```text
                 Pretrained mT5
                      │
                      ▼
                Input Article
                      │
                      ▼
                  Encoder
                      │
                      ▼
                  Decoder
                      │
                      ▼
             Predicted Summary
                      │
                      ▼
              Cross-Entropy Loss
                      │
                      ▼
                 Backpropagation
                      │
                      ▼
             Update All Parameters
```

The training objective is token-level cross-entropy:

$$
\mathcal{L}_{CE}
=
-\sum_t
\log P(y_t \mid y_{<t},x)
$$

where:

* \(x\) = input article
* \(y_t\) = target summary token
* \(y_{<t}\) = previous target tokens

Because the complete model is updated, Vanilla Fine-Tuning provides maximum adaptation capacity but can also produce larger changes in pretrained representations.

---

# 2. LoRA Fine-Tuning

**LoRA (Low-Rank Adaptation)** keeps the pretrained model parameters frozen and introduces trainable low-rank matrices.

```text
                    Pretrained mT5
                          │
                 ┌────────┴────────┐
                 │                 │
                 ▼                 ▼
          Frozen Parameters    LoRA Adapters
                                   │
                                   ▼
                              Trainable
                                   │
                 └────────┬────────┘
                          │
                          ▼
                    Model Output
                          │
                          ▼
                    Summary Loss
```

LoRA represents the adapted weight as:

$$
W' = W + BA
$$

where:

* \(W\) = pretrained weight
* \(A\) and \(B\) = trainable low-rank matrices

### LoRA Configuration

| Parameter                     |           Value |
| ----------------------------- | --------------: |
| Rank                          |              16 |
| Alpha                         |              16 |
| Dropout                       |            0.05 |
| Target modules                | Q/V projections |
| Encoder                       |             Yes |
| Decoder                       |             Yes |
| Trainable parameter reduction |            ~97% |

LoRA provides parameter-efficient adaptation while keeping the majority of the pretrained model frozen.

---

# 3. LangAnchor

LangAnchor is the proposed method for preserving the pretrained multilingual representation geometry.

The method compares hidden representations from:

* the pretrained model, and
* the fine-tuned model.

```text
                         Input Article
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          Pretrained mT5            Fine-Tuned mT5
                 │                         │
                 ▼                         ▼
              h_base                    h_tuned
                 │                         │
                 └────────────┬────────────┘
                              │
                              ▼
                   Representation Difference
                              │
                              ▼
                        Anchor Loss
                              │
                              ▼
                  Combined Training Loss
```

The combined objective is:

$$
\mathcal{L}
=
\mathcal{L}_{CE}
+
\lambda\mathcal{L}_{anchor}
$$

where the anchoring term is based on the distance between the hidden states:

$$
\mathcal{L}_{anchor}
=
\left\|
h_{tuned}-h_{base}
\right\|_2^2
$$

The anchor coefficient controls how strongly the fine-tuned model is encouraged to remain close to the pretrained representation space.

The reported configuration uses:

$$
\lambda = 0.05
$$

and applies the anchoring mechanism to **mid-to-upper encoder layers**.

---

# Representation Drift

Representation drift refers to the change in the model's internal hidden representations after fine-tuning.

```text
                 PRETRAINED MODEL
                        │
                        ▼
                     h_base
                        │
                        │
                        │ Representation
                        │ Drift
                        │
                        ▼
                     h_tuned
                        │
                        ▼
                  FINE-TUNED MODEL
```

A larger distance between \(h_{base}\) and \(h_{tuned}\) indicates greater representational change.

LangAnchor introduces an explicit regularization term to control this drift.

---

# Training Pipeline

The complete supervised fine-tuning pipeline is:

```text
Raw XLSum Dataset
       │
       ▼
Data Cleaning
       │
       ▼
Train / Evaluation Split
       │
       ▼
SentencePiece Tokenization
       │
       ▼
Input IDs + Attention Masks + Labels
       │
       ▼
      mT5
       │
       ▼
     Encoder
       │
       ▼
     Decoder
       │
       ▼
Generated Summary
       │
       ▼
Cross-Entropy Loss
       │
       ▼
Backpropagation
       │
       ▼
Parameter Update
```

For LangAnchor, the training objective additionally includes the representation anchoring loss.

---

# Evaluation

The project evaluates both **task performance** and **multilingual behavior**.

```text
                    Fine-Tuned Models
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
          Vanilla         LoRA       LangAnchor
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                   Summary Generation
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
          ROUGE          BLEU       BERTScore
                           │
                           ▼
                       Perplexity
                           │
                           ▼
                  Multilingual Testing
```

---

# Final Multilingual Evaluation

The final evaluation script compares four model variants:

1. **mT5 Base**
2. **Vanilla Fine-Tuned mT5**
3. **LoRA Fine-Tuned mT5**
4. **LangAnchor Fine-Tuned mT5**

The evaluation uses four manually specified examples:

```text
English
French
Hindi
Spanish
```

The final evaluation was executed on **CPU**.

The models generate summaries using beam search with:

```text
num_beams = 4
```

The evaluation calculates:

* ROUGE-1
* ROUGE-2
* ROUGE-L
* BLEU
* BERTScore
* Perplexity
* Inference time

---

# Quantitative Results

The following table contains the results obtained from the final multilingual evaluation script.

| Model        |    ROUGE-1 |    ROUGE-2 |    ROUGE-L |        BLEU |  BERTScore | Perplexity ↓ | Time (s) |
| ------------ | ---------: | ---------: | ---------: | ----------: | ---------: | -----------: | -------: |
| **mT5 Base** | **0.4126** | **0.3676** | **0.4126** | **17.1513** | **0.9317** |       3.3336 |     9.06 |
| Vanilla FT   |     0.1477 |     0.0808 |     0.1477 |      2.6565 |     0.8726 |      11.7661 | **6.98** |
| LoRA         |     0.3188 |     0.2604 |     0.3188 |     15.0482 |     0.9177 |       2.5572 |     7.96 |
| LangAnchor   |     0.2326 |     0.1295 |     0.2326 |      7.2771 |     0.9079 |       2.2652 |     8.98 |

### Metric Interpretation

* **ROUGE-1 / ROUGE-2 / ROUGE-L:** lexical overlap between generated and reference summaries.
* **BLEU:** n-gram overlap between generated and reference text.
* **BERTScore:** semantic similarity between generated and reference summaries.
* **Perplexity:** model prediction uncertainty; lower is better.
* **Time:** total evaluation/generation time measured by the evaluation script.

---

# Results Analysis

## mT5 Base

The pretrained mT5 model achieved the highest scores on this small diagnostic evaluation:

* ROUGE-1: **0.4126**
* ROUGE-2: **0.3676**
* ROUGE-L: **0.4126**
* BLEU: **17.1513**
* BERTScore: **0.9317**

This shows that, on these four examples, the pretrained model already generated strong multilingual outputs.

---

## Vanilla Fine-Tuning

Vanilla Fine-Tuning produced the weakest metrics among the three fine-tuned models:

* ROUGE-1: **0.1477**
* ROUGE-2: **0.0808**
* ROUGE-L: **0.1477**
* BLEU: **2.6565**
* BERTScore: **0.8726**
* Perplexity: **11.7661**

The generated outputs also showed repetition and multilingual degradation.

For example, the Hindi output contained mixed-language text, while some outputs showed repeated phrases.

---

## LoRA

LoRA achieved the strongest results among the **three fine-tuned models**:

* ROUGE-1: **0.3188**
* ROUGE-2: **0.2604**
* ROUGE-L: **0.3188**
* BLEU: **15.0482**
* BERTScore: **0.9177**
* Perplexity: **2.5572**

Therefore, in this evaluation:

> **LoRA provided the strongest summarization performance among the fine-tuned approaches.**

---

## LangAnchor

LangAnchor achieved:

* ROUGE-1: **0.2326**
* ROUGE-2: **0.1295**
* ROUGE-L: **0.2326**
* BLEU: **7.2771**
* BERTScore: **0.9079**
* Perplexity: **2.2652**

The most notable result is its **lowest perplexity** among the evaluated models.

LangAnchor's primary objective is not simply maximizing summarization metrics. Its main objective is to preserve the pretrained multilingual representation geometry while allowing downstream adaptation.

---

# Qualitative Results

The final evaluation generated the following outputs.

### English

**Reference:**

```text
India's economy is expanding.
```

**mT5 Base:**

```text
India's economy is at its highest rate in more than a decade.
```

**Vanilla FT:**

```text
Indian economy is growing steadily this year.
Indian economy is growing steadily this year.
Indian economy is growing steadily this year.
```

**LoRA:**

```text
India's economy is at its highest rate in more than a decade.
```

**LangAnchor:**

```text
The Indian economy is growing sharply in the past few years.
```

---

### French

The LoRA and LangAnchor models generated fluent French summaries, while Vanilla Fine-Tuning produced repetition.

---

### Hindi

LoRA and LangAnchor retained coherent Hindi generation.

Vanilla Fine-Tuning produced a mixed Hindi/English output:

```text
PM 'tई शिक्षा नीति' घोषणा in UAE.
```

This illustrates the multilingual degradation observed in the qualitative evaluation.

---

### Spanish

LoRA and LangAnchor produced coherent Spanish summaries.

Vanilla Fine-Tuning produced a mixed-language output:

```text
El clima is changing rápidamente en all el continent.
```

---

# Ablation Study

The project investigates three ablation dimensions.

## 1. LangAnchor Coefficient

$$
\lambda \in \{0, 0.01, 0.05, 0.10\}
$$

This studies the effect of different anchoring strengths.

---

## 2. LoRA Rank

$$
r \in \{4,8,16,32\}
$$

This studies the effect of adapter capacity.

The main LoRA configuration uses:

$$
r=16
$$

---

## 3. Encoder Layer Freezing

$$
F \in \{0,4,6,8\}
$$

This studies the effect of freezing different numbers of encoder layers.

---

## Ablation Implementation Note

The current ablation script evaluates different settings using existing fine-tuned checkpoints.

It does **not** retrain an independent model from scratch for every value of:

* \(\lambda\)
* LoRA rank
* frozen layers

Therefore, these experiments should be described as **exploratory ablation evaluations**, rather than controlled retraining experiments.

---

# Key Findings

The project demonstrates three important observations.

### 1. Full Fine-Tuning Can Cause Multilingual Degradation

Updating all parameters using English-only data can significantly alter multilingual behavior.

```text
English-only Fine-Tuning
          │
          ▼
   Update All Parameters
          │
          ▼
Representation Drift
          │
          ▼
Multilingual Degradation
```

---

### 2. LoRA Provides Strong Fine-Tuned Performance

Among the three fine-tuned models, LoRA achieved the strongest reported summarization metrics in the final evaluation.

It also reduces the number of trainable parameters by approximately **97%** compared with full fine-tuning.

---

### 3. LangAnchor Focuses on Representation Preservation

LangAnchor explicitly adds a hidden-state anchoring objective.

```text
             Fine-Tuning
                  │
                  ▼
          Task Adaptation
                  │
                  +
          Anchor Regularization
                  │
                  ▼
       Representation Preservation
```

The reported project results attribute approximately **40% reduction in representation drift** to LangAnchor.

# Important Experimental Note

The quantitative results should be interpreted according to the evaluation setup.

The final evaluation uses **four manually specified examples** across English, French, Hindi, and Spanish. Therefore, it is best considered a **small multilingual diagnostic evaluation**, not a statistically robust benchmark.

The reported project results and the final diagnostic evaluation should also be distinguished:

* The **project report** contains the main experimental comparison and reported results.
* The **current evaluation script** reproduces the reported Vanilla, LoRA, and LangAnchor metric values and additionally evaluates the pretrained mT5 base model.
* The **ablation script** explores parameter variations using existing checkpoints rather than retraining independent models for every ablation configuration.

---

# Technologies Used

* **Python**
* **PyTorch**
* **Hugging Face Transformers**
* **mT5**
* **SentencePiece**
* **PEFT**
* **LoRA**
* **ROUGE**
* **SacreBLEU**
* **BERTScore**
* **Pandas**
* **NumPy**
* **Evaluate**

---

# Project Structure

```text
.
├── README.md
│
├── fine_tuning.py
├── abiliation.py
│
├── Summarization_dataset.csv
│
├── mT5-multilingual-XLSum/
│
├── fine_tuned_mt5_summarization/
├── model-lora-finetuned_2/
├── model-langanchor-finetuned/
│
└── multilingual_eval_results.csv
```

---

# End-to-End Workflow

```text
                    XLSum Dataset
                         │
                         ▼
               English Article-Summary Pairs
                         │
                         ▼
                SentencePiece Tokenizer
                         │
                         ▼
                Pretrained mT5 Model
                         │
            ┌────────────┼────────────┐
            │            │            │
            ▼            ▼            ▼
         Vanilla        LoRA      LangAnchor
            │            │            │
            ▼            ▼            ▼
       All Params    LoRA Params   Anchor Loss
            │            │            │
            └────────────┼────────────┘
                         │
                         ▼
                  Fine-Tuned Models
                         │
                         ▼
            Multilingual Evaluation
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
      English          French           Hindi
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
                      Spanish
                         │
                         ▼
             ┌──────────────────────┐
             │      Evaluation      │
             │                      │
             │ ROUGE                │
             │ BLEU                 │
             │ BERTScore            │
             │ Perplexity           │
             │ Inference Time       │
             └──────────┬───────────┘
                        │
                        ▼
             Multilingual Analysis
```

---

# Conclusion

This project investigates the effect of supervised fine-tuning on the multilingual capabilities of pretrained mT5 models.

Three approaches are compared:

```text
Vanilla Fine-Tuning
        │
        ├── Maximum parameter adaptation
        └── Greater representation changes


LoRA
        │
        ├── Parameter-efficient adaptation
        └── Strong fine-tuned summarization performance


LangAnchor
        │
        ├── Task adaptation
        ├── Hidden-state anchoring
        └── Representation preservation
```

The final evaluation shows that **LoRA provides the strongest summarization performance among the fine-tuned models**, while **LangAnchor achieves the lowest perplexity and is specifically designed to control representation drift**.

The central research insight is:

> **Fine-tuning a multilingual model changes not only its output behavior but potentially its internal multilingual representation geometry. Explicitly constraining this representation change provides a mechanism for studying and mitigating multilingual forgetting.**

---

# Team

* **Surya Kant Mani**
* **Karan Shardul**
* **Vanshika Srivastava**
* **Kaivalya Vanmali**

---

# References

1. Xue et al., **mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer**
2. Hu et al., **LoRA: Low-Rank Adaptation of Large Language Models**
3. Hasan et al., **XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages**
4. Hugging Face Transformers Documentation
5. Hugging Face PEFT Documentation
