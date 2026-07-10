LensWord Model Card
Authors: Betty George and Miheret Woldegabrial Program: AI/ML Engineering Program, Apeiron AI Training, July 2026 GitHub: github.com/BettyG-ship-it/lensword

What this model does
LensWord classifies e-commerce product reviews into three categories: Negative, Neutral, and Positive. It uses a Bidirectional LSTM that reads each review forward and backward simultaneously, giving it context from both directions before making a prediction.

We also fine-tuned DistilBERT on the same dataset (Notebook 06). It handles negation better and achieves higher Macro F1.

Models
BiLSTM (primary, in production) Architecture: Bidirectional LSTM, 2 layers, hidden dim 64, dropout 0.4 Parameters: 444,035 Vocabulary: 4,340 words fitted on training data only Sequence length: 50 tokens

DistilBERT (fine-tuned, Notebook 06) Base model: distilbert-base-uncased from HuggingFace Fine-tuned on our amazon_yelp_combined.csv Parameters: 66 million Max sequence length: 128 tokens

Training data
We combined two datasets. Amazon Alexa Reviews from Kaggle gave us 3,149 reviews. Yelp Reviews from HuggingFace gave us 8,000 sampled reviews. We sampled Yelp as 3,500 Negative, 3,500 Neutral, and 1,000 Positive to correct for Amazon's heavy Positive skew (87% Positive).

Label mapping:

1 to 2 stars means Negative (label 0)
3 stars means Neutral (label 1)
4 to 5 stars means Positive (label 2)
This is a heuristic, not ground truth. The Neutral class is an artifact of our star-to-class rule.

Total after deduplication: 10,299 reviews

Final split after deduplication:

Training: 7,720 reviews
Validation: 1,544 reviews
Test: 1,030 reviews
Performance
BiLSTM results (3 seeds):

Seed	Accuracy	Macro F1
42	72.62%	0.7263
7	71.84%	0.7205
123	71.17%	0.7136
Mean and std	71.88% plus or minus 0.59%	0.7201 plus or minus 0.0052
Per-class F1 (seed 42):

Negative: 0.7467
Neutral: 0.6396
Positive: 0.7925
DistilBERT fine-tuned results:

Test Accuracy: 80.29%
Test Macro F1: 0.8082
Negative F1: 0.8126
Neutral F1: 0.7490
Positive F1: 0.8630
All results computed on the same 1,030 held-out test rows from test_texts.csv.

Intended use
This model is for e-commerce customer service teams who need to classify product reviews and prioritize responses. It works best on English product and service reviews in the same domain as our training data (Amazon product reviews and Yelp restaurant reviews).

What it should not be used for
Do not use this model for medical, legal, or financial sentiment analysis. Do not use it on social media posts, news articles, or other text domains far from product reviews. Do not use it as the only signal for high-stakes decisions without human review.

What happened with the first pipeline
Our first pipeline reported 88.85% accuracy. We found four errors -- data leakage, vocabulary leakage, wrong checkpoint metric, and invalid SMOTE. We fixed all four and rebuilt. The honest result is what you see above.

Limitations
Negation handling: The BiLSTM sometimes gets negation wrong. "I hate this product" was classified as Positive in testing because "hate" appeared rarely in negative training examples. We handle this at the application layer with word overrides. DistilBERT fixes this at the model level.

Neutral class is weakest: F1 of 0.6396 for BiLSTM, 0.7490 for DistilBERT. Three-star reviews are genuinely ambiguous and the label is defined by our star-to-class rule which zero-shot models cannot know.

Overfitting: The BiLSTM shows about a 21% gap between training accuracy (93.55%) and test accuracy (72.62%). This is because 7,720 training samples is not enough for 444,035 parameters. DistilBERT pre-training reduces this problem significantly.

English only: Both models were trained on English reviews only.

Small vocabulary: The BiLSTM vocabulary has 4,340 words. Rare words become unknown tokens.

Fixed sequence length: Reviews longer than 50 tokens get truncated. Long reviews lose information.

Domain specific: Both models were trained on Amazon and Yelp reviews. Performance on other domains may be lower.

Label construction: Our 3-class label system is based on star ratings. This is a proxy for sentiment, not ground truth.

Dataset size: 7,720 training samples is small compared to production sentiment models. More data would improve both models.

System details
Knowledge base: 33 RAG entries across 7 complaint categories. Protection guards: 18 across api.py, customer.html, index.html, and Docker. LangGraph nodes: 13 across three conversation flows.

Ethical considerations
The model should not be used as the sole decision-maker for customer escalations. High-priority cases should always have human review. The model was trained on Amazon and Yelp reviews which reflect English-speaking customer demographics. Performance may vary for other customer populations. The word override lists we use introduce some manual bias. These lists are maintained by our team and documented in customer.html.

How to use
Via API:

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product broke after two days"}'
Via Python:

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
Citation
@misc{lensword2026,
  author = {George, Betty and Woldegabrial, Miheret},
  title = {LensWord: Deep Learning Sentiment Analysis for E-Commerce Reviews},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/BettyG-ship-it/lensword}
}