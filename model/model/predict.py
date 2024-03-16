import sentencepiece
import torch


class AlphaZeroPredictor:
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


class ActualBoardPredictor:
    def __init__(
            self,
            context_length: int,
            sp: sentencepiece.SentencePieceProcessor
    ):
        self.__sp = sp
        self.__context_length = context_length

    def tokens_to_string(self, tokens: torch.Tensor) -> str:
        return self.__sp.decode(tokens.view(-1).tolist()).replace("<n>", "\n")

    def predict(self, model, X_board, X_strength, X_reps, X_state, text: str, max_new_tokens: int) -> str:
        tokens = self.__sp.encode(text.strip().replace('\n', '<n>'))
        tokens = [self.__sp.bos_id()] + tokens
        X_board = X_board.to(model.device)
        X_strength = X_strength.to(model.device)
        X_reps = X_reps.to(model.device)
        X_state = X_state.to(model.device)
        tokens = torch.Tensor(tokens).unsqueeze(0).int().to(model.device)
        tokens = model.generate(X_board.unsqueeze(0), X_strength.unsqueeze(0), X_reps.unsqueeze(0), X_state.unsqueeze(0), tokens, max_new_tokens)
        return self.tokens_to_string(tokens)