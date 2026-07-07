# LensWord — Model Card

## Model Description

**Model name:** LensWord Bidirectional LSTM  
**Model type:** Sequence classification — 3-class sentiment analysis  
**Language:** English  
**License:** MIT  
**Authors:** Betty George & Miheret Woldegabrial  
**Program:** AI/ML Engineering Program, Apeiron AI Training, 2026  
**GitHub:** github.com/BettyG-ship-it/lensword

---

## Intended Use

### Primary Use
Classifying e-commerce product reviews as Positive, Neutral, or Negative to help customer service teams prioritize and respond to feedback.

### Intended Users
- Customer service teams at e-commerce companies
- Business analysts monitoring product feedback
- Developers building customer feedback pipelines

### Out-of-Scope Use
- Non-English text
- Reviews outside the product/service domain
- Medical, legal, or financial sentiment analysis
- Social media posts or news articles

---

## Training Data

| Source | Size | Access |
|---|---|---|
| Amazon Alexa Reviews | 3,149 reviews | Kaggle |
| Yelp Reviews (filtered ≤50 words) | 8,000 sampled | HuggingFace |

**Label construction rule:**
- 1-2 stars → Negative (label 0)
- 3 stars → Neutral (label 1)
- 4-5 stars → Positive (label 2)

**Important:** The Neutral class is an artifact of this star-to-class mapping. Zero-shot models that do not know this rule will perform poorly on Neutral reviews.

---

## Model Architecture

| Component | Details |
|---|---|
| Model definition | src/model.py — single source of truth |
| Embedding | 4,340 vocab × 64 dims, padding_idx=0 |
| BiLSTM Layer 1 | Hidden: 64, Bidirectional, Dropout: 0.4 |
| BiLSTM Layer 2 | Hidden: 64, Bidirectional, Dropout: 0.4 |
| Dropout | p=0.4 |
| Fully Connected | 128 → 3 classes |
| Total Parameters | 444,035 |

---

## Performance

All figures are mean ± standard deviation across 3 random seeds (42, 7, 123) on a held-out test set of 1,030 reviews saved at split time in test_texts.csv.

| Metric | Value |
|---|---|
| Test Accuracy (mean ± std) | 71.88% ± 0.59% |
| Macro F1 (mean ± std) | 0.7201 ± 0.0052 |
| Negative F1 (seed 42) | 0.7467 |
| Neutral F1 (seed 42) | 0.6396 |
| Positive F1 (seed 42) | 0.7925 |

All reported figures are loaded from models/metrics.json — not hardcoded.

---

## Complete Limitations

### Model Limitations
1. **Overfitting** — ~21% gap between training accuracy (93.55%) and test accuracy (72.62%) due to limited dataset size (7,720 training samples for 444,035 parameters)
2. **Negation handling** — LSTM reads sequentially and did not learn "hate" as a strong negative signal due to low frequency in training data. Handled at application layer via word override lists
3. **Neutral class weakest** — F1: 0.6396. 3-star label is an artifact of our star-to-class rule and is genuinely ambiguous
4. **Confident on out-of-domain text** — non-review input classified with high confidence. Handled via vocabulary-based is_product_review() guard
5. **English only** — no multilingual support
6. **Small vocabulary** — 4,340 words. Rare words replaced with UNK token
7. **Fixed sequence length** — reviews truncated to 50 tokens. Long reviews lose information
8. **LSTM vs BERT** — LSTM reads sequentially; BERT uses attention and handles negation significantly better. DistilBERT fine-tuning is the highest-priority future work
9. **Miscalibrated confidence** — high confidence does not reliably indicate correct prediction. 75% threshold is a heuristic workaround

### Data Limitations
10. **Label construction is a heuristic** — not ground truth
11. **Domain mismatch** — Amazon Alexa (smart speaker) + Yelp (restaurants) combined
12. **Class imbalance** — Amazon dataset was 87% Positive. Compensated but not perfectly balanced
13. **Limited dataset size** — 7,720 training samples. Industry models train on millions

### RAG Limitations
14. **Knowledge base has 33 entries** — production system would need hundreds
15. **In-memory ChromaDB** — data lost on restart, recreated on every startup
16. **Static response templates** — not personalized to specific customer or issue

### Application & Deployment Limitations
17. **Word override lists are manual** — hardcoded, words not in list may be misclassified
18. **Conversation flow is scripted** — does not re-classify follow-up messages
19. **No authentication** — acceptable for localhost demo only
20. **CORS wildcard** — should be restricted in production
21. **No persistent storage** — customer feedback not saved
22. **Local deployment only** — HuggingFace Spaces deployment is future work
23. **No CI/CD pipeline** — future work
24. **Single Docker container** — production system would use microservices

### Comparison Limitations
25. **Zero-shot comparison not perfectly fair** — LSTM trained on exact distribution; HuggingFace models were not
26. **Only one trained architecture** — no comparison against CNN, Transformer, or fine-tuned BERT

---

## Ethical Considerations

- Model should not be used as the sole decision-maker for customer escalations
- Human review recommended for all HIGH priority cases
- Confidence scores below 75% should trigger human review
- Model trained on English Amazon and Yelp reviews — may exhibit bias toward English-speaking customer demographics
- Word override lists introduce manual bias — maintained and reviewed by the development team

---

## How to Use

```python
import torch
import pickle
import sys
sys.path.append('src')
from model import LensWordLSTM
from src.config import *

with open('data/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)

model = LensWordLSTM(
    vocab_size=MAX_VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
)
model.load_state_dict(torch.load('models/lensword_model.pt', weights_only=True))
model.eval()
```

Or via API:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product broke after two days"}'
```

---

## Citation

```
@misc{lensword2026,
  author = {George, Betty and Woldegabrial, Miheret},
  title = {LensWord: Deep Learning Sentiment Analysis for E-Commerce Reviews},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/BettyG-ship-it/lensword}
}
```
