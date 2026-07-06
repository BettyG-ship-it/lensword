# LensWord — Model Definition
# Single source of truth for LensWordLSTM architecture
# Imported by: src/api.py, notebooks/03, notebooks/04

import torch
import torch.nn as nn


class LensWordLSTM(nn.Module):
    """
    Bidirectional LSTM for 3-class sentiment classification.

    Architecture:
        Embedding → BiLSTM Layer 1 → BiLSTM Layer 2
        → Dropout → Fully Connected → 3 classes

    Args:
        vocab_size:    Size of vocabulary (default 4340)
        embedding_dim: Embedding dimensions (default 64)
        hidden_dim:    LSTM hidden dimensions (default 64)
        num_layers:    Number of LSTM layers (default 2)
        num_classes:   Number of output classes (default 3)
        dropout:       Dropout probability (default 0.4)
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int, num_classes: int, dropout: float):
        super(LensWordLSTM, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Concatenate final forward and backward hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output