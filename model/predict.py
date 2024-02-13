import sentencepiece
import torch

from model.model import Model
from utils.configs import SharedConfig


class Predictor:
    def __init__(self, shared_config: SharedConfig):
        self.__sp = sentencepiece.SentencePieceProcessor(model_file=shared_config['sentencepiece_path'])
        self.__context_length = shared_config['context_length']

    def tokens_to_string(self, tokens: torch.Tensor) -> str:
        return self.__sp.decode(tokens.view(-1).tolist()).replace("<n>", "\n")

    def predict(self, model: Model, X_board: torch.tensor, text: str, max_new_tokens: int, device: str) -> str:
        tokens = self.__sp.encode(text.strip().replace('\n', '<n>'))
        tokens = [self.__sp.bos_id()] + tokens
        tokens = torch.Tensor(tokens).unsqueeze(0).int().to(device)
        model.eval()
        tokens = model.generate(X_board.unsqueeze(0), tokens, max_new_tokens, device)
        return self.tokens_to_string(tokens)