# LensWord - Project Configuration

# Data
DATA_PATH = "data/amazon_reviews.csv"
TEXT_COLUMN = "verified_reviews"
LABEL_COLUMN = "sentiment"

# Preprocessing
MAX_VOCAB_SIZE = 4340
MAX_SEQ_LENGTH = 50
TEST_SIZE = 0.1
VAL_SIZE = 0.1

# Model
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
NUM_CLASSES = 3

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Output
MODEL_SAVE_PATH = "../models/lensword_model.pt"