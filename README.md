# Preserving Multilingual Geometry During Fine-Tuning

> Comparing full fine-tuning, LoRA, and a proposed hidden-state anchoring
> method for preserving cross-lingual capabilities in mT5.

This student research project studies catastrophic forgetting in multilingual
sequence-to-sequence models. It fine-tunes an mT5 multilingual summarization
checkpoint on English-only data, then measures how well the model retains its
English, Hindi, and Marathi capabilities.

Three strategies are compared:

1. **Vanilla fine-tuning** updates all model parameters.
2. **LoRA** adds low-rank trainable adapters to attention projections.
3. **LangAnchor**, the project's proposed method, regularizes fine-tuned hidden
   states against the pretrained model's representations.

Summarization is a controlled probe for multilingual stability. The full
methodology and results are in
[`ML Project Final/Project Report Final.pdf`](ML%20Project%20Final/Project%20Report%20Final.pdf).

## Reported findings

According to the report:

- full-parameter English fine-tuning causes the greatest multilingual
  degradation;
- LoRA gives the strongest summarization metrics while greatly reducing
  trainable parameters;
- LangAnchor best preserves multilingual geometry and reduces hidden-state
  drift.

These are reported findings, not results automatically reproduced here.

### Quantitative results

Results averaged across English, Hindi, and Marathi in the project report:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore | PPL (lower is better) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mT5 base | 0.4126 | 0.3676 | 0.4126 | 17.15 | 0.9317 | 3.33 |
| Vanilla FT | 0.1477 | 0.0808 | 0.1477 | 2.66 | 0.8726 | 11.76 |
| LoRA | 0.3188 | 0.2604 | 0.3188 | 15.05 | 0.9177 | 2.56 |
| LangAnchor | 0.2326 | 0.1295 | 0.2326 | 7.28 | 0.9079 | **2.27** |

Language-specific ROUGE-L:

| Model | English | Hindi | Marathi |
| --- | ---: | ---: | ---: |
| Vanilla FT | 0.212 | 0.091 | 0.031 |
| LoRA | **0.398** | **0.301** | **0.257** |
| LangAnchor | 0.344 | 0.241 | 0.212 |

LoRA used rank 16, alpha 16, dropout 0.05, and Q/V projection adapters,
reducing trainable parameters by approximately 97% versus full fine-tuning.
LangAnchor used an anchoring coefficient of 0.05. The report attributes a 40%
reduction in representation drift to LangAnchor.

## How it works

```text
10,000 English XLSum samples
             |
             v
 mT5-Multilingual-XLSum & mT5Base
             |
     +-------+--------+
     |       |        |
  Vanilla   LoRA   LangAnchor
     |       |        |
     +-------+--------+
             |
             v
 English / Hindi / Marathi evaluation
             |
             v
 ROUGE, BLEU, BERTScore, perplexity,
 and layer-wise hidden-state drift
```


## Team

The report credits:

- Surya Kant Mani 
- Karan Shardul 
- Vanshika Srivastava 
- Kaivalya Vanmali 


