# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(model, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device,
                    model_type: str = "rnn") -> float:

    model.train()
    total_loss = 0.0
    h = model.init_hidden(loader.batch_size, device)

    if model_type == "lstm":
        h, c = h

    for batch_idx, (input_seq, target_seq) in enumerate(loader):
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        if model_type == "rnn":
            h = h.detach()
            logits, h, _ = model(input_seq, h)
        else:
            h = h.detach()
            c = c.detach()
            logits, h, c, _ = model(input_seq, h, c)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            target_seq.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader: DataLoader, criterion: nn.Module,
             device: torch.device, model_type: str = "rnn") -> float:

    model.eval()
    total_loss = 0.0
    h = model.init_hidden(loader.batch_size, device)

    if model_type == "lstm":
        h, c = h

    with torch.no_grad():
        for input_seq, target_seq in loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            if model_type == "rnn":
                logits, h, _ = model(input_seq, h)
            else:
                logits, h, c, _ = model(input_seq, h, c)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_seq.view(-1)
            )
            total_loss += loss.item()

    return total_loss / len(loader)


def train(model, train_loader: DataLoader, val_loader: DataLoader,
          num_epochs: int, lr: float = 0.001, device: torch.device = torch.device("cpu"),
          model_type: str = "rnn", save_path: str = None):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device, model_type)
        val_loss = evaluate(model, val_loader, criterion, device, model_type)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Train Perplexity: {torch.exp(torch.tensor(train_loss)):.2f} | "
              f"Val Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")

        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, save_path)
            print(f"  → Saved best checkpoint (val loss: {val_loss:.4f})")

    return train_losses, val_losses