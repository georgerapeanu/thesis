import torch
from typing import *
from nltk.translate.bleu_score import sentence_bleu


@torch.no_grad()
def get_loss(model: torch.nn.Module, dl: torch.utils.data.DataLoader, device: str) -> float:
    model.eval()
    losses = []
    for batch in dl:
        (X_board, X_text, y_sequence, pad_mask) = batch
        (X_board, X_text, y_sequence, pad_mask) = (X_board.to(device), X_text.to(device), y_sequence.to(device), pad_mask.to(device))
        _, loss = model(X_board, X_text, pad_mask, y_sequence)
        losses.append(loss.item())
    return sum(losses) / len(losses)


class BLEU:
    def __init__(self, references: List[str]):
        self.__references = map(lambda x: x.strip().split(),  references)

    def __call__(self, candidate: str):
        return sentence_bleu(self.__references, candidate.strip().split())
