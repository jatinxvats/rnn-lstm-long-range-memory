# src/lstm.py

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm_cell = nn.LSTMCell(embed_dim, hidden_size)

        self.W_hy = nn.Linear(hidden_size, vocab_size, bias=True)

    def forward(self, x: torch.Tensor,
                h0: torch.Tensor = None, c0: torch.Tensor = None,
                track_gradients: bool = False):

        batch_size, seq_len = x.shape

        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.zeros(batch_size, self.hidden_size, device=x.device)

        embeds = self.embedding(x)

        h_t, c_t = h0, c0
        logits = []
        hidden_states = []

        for t in range(seq_len):
            x_t = embeds[:, t, :]

            h_t, c_t = self.lstm_cell(x_t, (h_t, c_t))

            if track_gradients:
                h_t.retain_grad()
                hidden_states.append(h_t)

            logits.append(self.W_hy(h_t))

        logits = torch.stack(logits, dim=1)

        return logits, h_t, c_t, hidden_states

    def init_hidden(self, batch_size: int, device: torch.device):

        h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(batch_size, self.hidden_size, device=device)
        return h0, c0