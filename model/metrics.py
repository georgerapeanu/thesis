import torch
from model.commentary_models import AlphazeroTransformerModel, AlphazeroMultipleHeadsModel


@torch.no_grad()
def get_loss(model: AlphazeroTransformerModel | AlphazeroMultipleHeadsModel, dl: torch.utils.data.DataLoader, device: str) -> float:
    model.eval()
    losses = []
    for batch in dl:
        (X_board, X_text, y_sequence, pad_mask, types) = batch
        (X_board, X_text, y_sequence, pad_mask, types) = (X_board.to(device), X_text.to(device), y_sequence.to(device), pad_mask.to(device), types)
        if isinstance(model, AlphazeroMultipleHeadsModel):
            _, loss = model(X_board, X_text, pad_mask, y_sequence, types)
        else:
            _, loss = model(X_board, X_text, pad_mask, y_sequence)
        losses.append(loss.item())
    return sum(losses) / len(losses)