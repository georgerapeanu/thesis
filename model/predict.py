import sentencepiece
import torch

from model.commentary_models import AlphazeroTransformerModel, AlphazeroModelResidualEncoder
from utils.configs import SharedConfig


class Predictor:
    def __init__(
            self,
            context_length: int,
            sp: sentencepiece.SentencePieceProcessor
    ):
        self.__sp = sp
        self.__context_length = context_length

    def tokens_to_string(self, tokens: torch.Tensor) -> str:
        return self.__sp.decode(tokens.view(-1).tolist()).replace("<n>", "\n")

    def predict(self, model, X_board: torch.tensor, text: str, max_new_tokens: int) -> str:
        tokens = self.__sp.encode(text.strip().replace('\n', '<n>'))
        tokens = [self.__sp.bos_id()] + tokens
        X_board = X_board.to(model.device)
        tokens = torch.Tensor(tokens).unsqueeze(0).int().to(model.device)
        tokens = model.generate(X_board.unsqueeze(0), tokens, max_new_tokens)
        return self.tokens_to_string(tokens)