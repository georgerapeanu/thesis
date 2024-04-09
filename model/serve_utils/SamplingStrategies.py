from abc import abstractmethod

import torch
from overrides import overrides


class AbstractSamplingStrategy:
    @abstractmethod
    def execute(self, logits: torch.Tensor) -> torch.Tensor:
        pass


class MultinomialSamplingStrategy(AbstractSamplingStrategy):
    def __init__(self, temperature):
        self.__temperature = temperature
    @overrides
    def execute(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits / self.__temperature
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)


class TopKSamplingStrategy(AbstractSamplingStrategy):
    def __init__(self, temperature):
        self.__temperature = temperature
    @overrides
    def execute(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits / self.__temperature
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return torch.topk(probabilities, k=1, dim=-1)[1]
