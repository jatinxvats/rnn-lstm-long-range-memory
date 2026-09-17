# src/rnn.py
import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.W_xh = nn.Linear(embed_dim, hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hy = nn.Linear(hidden_size, vocab_size, bias=True)

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None,
                track_gradients: bool = False):
       
        batch_size, seq_len = x.shape

        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)

        embeds = self.embedding(x)

        h_t = h0
        logits = []
        hidden_states = []

        for t in range(seq_len):
            x_t = embeds[:, t, :]

            h_t = torch.tanh(self.W_xh(x_t) + self.W_hh(h_t))

            if track_gradients:
                h_t.retain_grad()
                hidden_states.append(h_t)

            logits.append(self.W_hy(h_t))

        
        logits = torch.stack(logits, dim=1)

        return logits, h_t, hidden_states


    def init_hidden(self, batch_size: int, device: torch.device):
        
        return torch.zeros(batch_size, self.hidden_size, device=device)