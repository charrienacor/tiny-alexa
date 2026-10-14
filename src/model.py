"""Two-head TCN for spoken-command classification (intent + slot).

Input: log-mel (B, 1, 64, 480) -> strided stem convs (480 -> 240 -> 120
frames) -> 5 dilated blocks -> global average pooling -> two linear heads
-> intent_logits (B, 19), slot_logits (B, 19).

Convolutions only (no RNN ops) and early downsampling keep it fast on the
Pi 4: 0.23 M params, ~0.9 MB fp32. The receptive field (1005 frames)
exceeds the 480-frame input, so the pooled vector sees the whole command.
The heads are decoded jointly over the 31 valid (intent, slot)
combinations (see train.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


IN_CHANNELS = 64        # N_MELS from features.py (input shape (B, 1, 64, 480))
KERNEL = 5              # 5 frames = 50 ms context
C1, C2 = 32, 64         # channels after downsampling stage 1 and 2
N_BLOCKS = 5            # dilated blocks; dilation doubles per block
LAYERS_PER_BLOCK = 2    # convs per block (standard TCN)
BLOCK_DROPOUT = 0.1     # off at eval / export
DOWN_FACTOR = 4         # two stride-2 convs: 1 downsampled frame = 4 orig frames


class DilatedBlock(nn.Module):
    """Two dilated convs with a residual connection.

    Padding is symmetric, not causal: after global average pooling the
    sequence order is irrelevant, so causal padding would only waste
    compute.
    """

    def __init__(self, channels, kernel, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel,
                               dilation=dilation, padding=(kernel - 1) * dilation // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel,
                               dilation=dilation, padding=(kernel - 1) * dilation // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(F.relu(self.bn1(self.conv1(x))))
        h = self.bn2(self.conv2(h))
        return F.relu(h + x)   # residual


class TCN(nn.Module):
    """Two-head temporal convolutional network.

    Args:
        n_intents: number of intent classes (19)
        n_slots:   number of slot classes incl. NO_SLOT (19)
        channels:  width of the dilated blocks (default 64)
        n_blocks:  number of dilated blocks (default 5; dilation doubles
                   per block)
        kernel:    conv kernel size in frames (default 5 = 50 ms)
    """

    def __init__(self, n_intents=19, n_slots=19,
                 channels=C2, n_blocks=N_BLOCKS, kernel=KERNEL):
        super().__init__()
        self.channels = channels
        self.n_blocks = n_blocks
        self.kernel = kernel
        # input arrives as (B, 1, 64, 480): channel dim is 64 (N_MELS)
        self.ds1 = nn.Conv1d(IN_CHANNELS, C1, kernel, stride=2, padding=kernel // 2)
        self.bn1 = nn.BatchNorm1d(C1)
        self.ds2 = nn.Conv1d(C1, channels, kernel, stride=2, padding=kernel // 2)
        self.bn2 = nn.BatchNorm1d(channels)

        self.blocks = nn.ModuleList()
        for b in range(n_blocks):
            base_dilation = 2 ** b          # 1, 2, 4, 8, 16
            self.blocks.append(DilatedBlock(channels, kernel, base_dilation,
                                             BLOCK_DROPOUT))

        self.pool = nn.AdaptiveAvgPool1d(1)   # global average over time
        self.intent_head = nn.Linear(channels, n_intents)
        self.slot_head = nn.Linear(channels, n_slots)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, logmel):
        """(B, 1, 64, 480) [or (B, 64, 480)] -> (intent_logits, slot_logits).

        Collapses the singleton channel dim to the (B, C, T) form the
        1-D convolutions expect.
        """
        x = logmel
        # dataset yields (1, 64, 480)/item -> batch is (B, 1, 64, 480);
        # reshape to the (B, C, T) form the 1-D convs expect (also accepts 3-D)
        x = x.reshape(x.size(0), IN_CHANNELS, x.size(-1))
        x = F.relu(self.bn1(self.ds1(x)))      # 480 -> 240 frames
        x = F.relu(self.bn2(self.ds2(x)))      # 240 -> 120 frames
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)          # (B, C)
        return self.intent_head(x), self.slot_head(x)

    @property
    def receptive_field(self):
        """Receptive field of the deepest layer, in original input frames.

        1005 for the default config, i.e. > the 480-frame input, so the
        pooled vector sees the entire command.
        """
        k = self.kernel
        r, s = 1, 1
        r += (k - 1) * s; s *= 2          # ds1
        r += (k - 1) * s; s *= 2          # ds2
        for b in range(self.n_blocks):
            for i in range(LAYERS_PER_BLOCK):
                d = 2 ** b
                r += (k - 1) * d * s
        return r

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = TCN(n_intents=19, n_slots=19)
    n = m.count_parameters()
    print(f"parameters      : {n:,}  ({n * 4 / 1e6:.2f} MB fp32, "
          f"~{n / 1e6:.2f} MB int8 weights)")
    print(f"receptive field : {m.receptive_field} frames "
          f"(input is 480 frames = 4.8 s)")
    print(f"blocks          : {N_BLOCKS} x {LAYERS_PER_BLOCK} dilated convs, "
          f"k={KERNEL}, C={C2}, dilations 1..{2 ** (N_BLOCKS - 1)})")
    x = torch.randn(2, 1, 64, 480)
    m.eval()
    with torch.no_grad():
        i, s = m(x)
        i2, s2 = m(x)
    assert i.shape == (2, 19) and s.shape == (2, 19)
    assert torch.equal(i, i2) and torch.equal(s, s2), "eval mode must be deterministic"
    print(f"forward ok      : intent {tuple(i.shape)}, slot {tuple(s.shape)}, "
          f"deterministic in eval")
