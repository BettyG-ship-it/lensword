LensWord — Model Card
Model Description
Model name: LensWord Bidirectional LSTM
Model type: Sequence classification — 3-class sentiment analysis
Language: English
License: MIT
Authors: Betty George & Miheret Woldegabrial
Program: AI/ML Engineering Program, Apeiron AI Training, 2026

Intended Use
Primary Use
Classifying e-commerce product reviews as Positive, Neutral, or Negative to help customer service teams prioritize and respond to feedback.

Intended Users
Customer service teams at e-commerce companies
Business analysts monitoring product feedback
Developers building customer feedback pipelines
Out-of-Scope Use
Non-English text
Reviews outside the product/service domain
Medical, legal, or financial sentiment analysis
Social media posts or news articles
Training Data
Source	Size	Access
Amazon Alexa Reviews	3,149 reviews	Kaggle
Yelp Reviews (filtered ≤50 words)	8,000 sampled	HuggingFace
Label construction rule:

1-2 stars → Negative (label 0)
3 stars → Neutral (label 1)
4-5 stars → Positive (label 2)
Important: The Neutral class is an artifact of this star-to-class mapping. Zero-shot models that do not know this rule will perform poorly on Neutral reviews.

Model Architecture
Component	Details
Embedding	4,340 vocab × 64 dims
BiLSTM Layer 1	Hidden: 64, Bidirectional, Dropout: 0.4
BiLSTM Layer 2	Hidden: 64, Bidirectional, Dropout: 0.4
Fully Connected	128 → 3 classes
Total Parameters	444,035
Performance
All figures are mean ± standard deviation across 3 random seeds (42, 7, 123) on a held-out test set of 1,030 reviews.

Metric	Value
Test Accuracy	71.88% ± 0.59%
Macro F1	0.7201 ± 0.0052
Negative F1	0.7467 (seed 42)
Neutral F1	0.6396 (seed 42)
Positive F1	0.7925 (seed 42)
Limitations
Overfitting — 10%+ gap between training and validation accuracy due to limited dataset size (7,720 training samples)
Negation handling — short phrases with implicit negative sentiment (e.g. "I hate this") may be misclassified; handled at application layer via word override lists and confidence threshold
Neutral class — hardest to classify (F1: 0.6396); 3-star reviews are genuinely ambiguous
Domain specificity — trained on product/service reviews only; out-of-domain text handled via vocabulary-based detection
English only — no multilingual support
Calibration — confidence scores may be miscalibrated especially after epoch 5 where validation loss diverged
Ethical Considerations
Model should not be used as the sole decision-maker for customer escalations
Human review is recommended for all HIGH priority cases
Confidence scores below 75% should trigger human review
Model was trained on English Amazon and Yelp reviews — may exhibit bias toward English-speaking customer demographics
How to Use
import torch
import pickle
from src.config import *

# Load vocabulary
with open('data/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)

# Load model
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
Or use the FastAPI endpoint:

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product broke after two days"}'
Citation
@misc{lensword2026,
  author = {George, Betty and Woldegabrial, Miheret},
  title = {LensWord: Deep Learning Sentiment Analysis for E-Commerce Product Reviews},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/BettyG-ship-it/lensword}
}