import torch


@torch.no_grad()
def get_loss(model: torch.nn.Module, dl: torch.utils.data.DataLoader, device: str) -> float:
    model.eval()
    losses = []
    for batch in dl:
        (X_board, X_text, y_sequence, pad_mask) = batch.to(device)
        _, loss = model(X_board, X_text, pad_mask, y_sequence)
        losses.append(loss.item())
    return sum(losses) / len(losses)