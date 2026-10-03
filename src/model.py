"""CRNN with two heads: shared encoder, intent
head + slot head.

  log-mel (1 x 64 x 480)
    -> 3 x [Conv2d 3x3 -> BatchNorm2d -> ReLU -> MaxPool 2x2]  (1->32->64->128)
    -> reshape to 1024 features per time step
    -> BiGRU (128 hidden units per direction -> 256)
    -> mean over time -> Dropout(0.3)
    -> intent head: Linear 256 -> n_intents  (19 intents)
    -> slot head:   Linear 256 -> n_slots    (18 slot values + NO_SLOT)

The intent head picks the command; the slot head picks the value
("6 AM", "red", ... or NO_SLOT when the command has no slot). Both heads
share the encoder, so both tasks regularize the same features
(multi-task learning).

BatchNorm note: at inference it is a fixed per-channel scale and shift,
so ONNX Runtime folds it into the preceding convolution — zero extra
latency on the Raspberry Pi. Softmax is applied outside the model.
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):

    def __init__(self, n_mels=64, n_intents=19, n_slots=19):
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1: (1, 64, 480) -> (32, 32, 240)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: (32, 32, 240) -> (64, 16, 120)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: (64, 16, 120) -> (128, 8, 60)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # after the CNN: 128 channels x 8 mel bins = 1024 features per time step
        self.gru = nn.GRU(input_size=128 * (n_mels // 8), hidden_size=128,
                          batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.intent_head = nn.Linear(256, n_intents)
        self.slot_head = nn.Linear(256, n_slots)

    def forward(self, x):
        x = self.cnn(x)                                  # (batch, 128, 8, 60)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)   # (batch, 60, 1024)
        x, _ = self.gru(x)                               # (batch, 60, 256)
        x = x.mean(dim=1)                                # (batch, 256)
        x = self.dropout(x)
        return self.intent_head(x), self.slot_head(x)    # two logits tensors