"""
Facade(restrictionez acces la componente)
Singleton(una singura duh)
Chain of responsability(pt validari)
Strategy(pt modul de sampling)
Observer pe frontend pt raspuns
"""
import json
from io import BytesIO
from typing import Iterator, Tuple, List

import hydra
import requests
import sentencepiece
import logging
import struct

import torch

from serve_utils.SamplingStrategies import MultinomialSamplingStrategy, TopKSamplingStrategy, AbstractSamplingStrategy
from serve_utils.Validators import JsonSchemaValidator, BoardsValidator, MaxNewTokensValidator, TargetTypeValidator, \
    TemperatureValidator
from ring.func.lru_cache import LruCache

logger = logging.getLogger(__name__)


class ServeProxyUtilsFacadeSingleton(object):
    #singleton through new
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ServeProxyUtilsFacadeSingleton, cls).__new__(cls)
            cls.instance.to_initialize = True
        return cls.instance

    def __init__(self):
        if hasattr(self, 'to_initialize') and self.to_initialize:
            del self.to_initialize
            with hydra.initialize(version_base="1.2", config_path="../conf"):
                self.__cfg = hydra.compose(config_name="serve_proxy_config")

            self.__sp = sentencepiece.SentencePieceProcessor(self.__cfg["sentencepiece_path"])
            validator = MaxNewTokensValidator()
            self.TARGET_TYPES_TO_IDS = {
                'MoveDesc': 0,
                'MoveQuality': 1,
                'Comparative': 2,
                "Strategy": 3,
                "Context": 4
            }
            validator = TargetTypeValidator(self.TARGET_TYPES_TO_IDS, validator)
            validator = TemperatureValidator(validator)
            validator = BoardsValidator(self.__cfg, validator)
            validator = JsonSchemaValidator(validator)
            self.__validator = validator
            self.__cache = LruCache(maxsize=self.__cfg["cache_size"])
            self.__model_url = self.__cfg["model"]["url"]

    def validate_request(self, request_data):
        data = json.loads(request_data)
        self.__validator.validate(data)

    def unpack(self, stream, fmt):
        size = struct.calcsize(fmt)
        buf = stream.read(size)
        return struct.unpack(fmt, buf)

    def consume_bytesio_stream(self, stream: BytesIO) -> Iterator[Tuple[List[float], int]]:
        while True:
            try:
                unpacked = self.unpack(stream, f"!{self.__sp.vocab_size()}fI")
                yield list(unpacked[:-1]), unpacked[-1]
            except struct.error:
                break

    def get_next_token_tensor(self, sampler: AbstractSamplingStrategy, probabilities: torch.Tensor) -> str:
        token = sampler.execute(probabilities).view(1).item()
        return self.__sp.IdToPiece(token)

    def get_next_token(self, sampler, probabilities: List[float]) -> str:
        return self.get_next_token_tensor(sampler, torch.tensor(probabilities).view(1, -1))

    def get_commentary(self, request_data) -> Iterator[str]:
        logger.warning(f"Received request: {request_data}")
        data = json.loads(request_data)

        temperature = 1.0 if 'temperature' not in data else data.get('temperature')
        do_sample = False if 'do_sample' not in data else data.get('do_sample')
        max_new_tokens = 1000 if 'max_new_tokens' not in data else data.get('max_new_tokens')
        prefix = '' if 'prefix' not in data else data.get('prefix')
        data['prefix'] = prefix

        sampler = MultinomialSamplingStrategy(temperature) if do_sample else TopKSamplingStrategy(temperature)

        while max_new_tokens > 0:
            key = json.dumps(data)
            if self.__cache.has(key):
                probabilities = self.__cache.get(key)
                max_new_tokens -= 1
                token = self.get_next_token(sampler, probabilities)
                token = token.replace("▁", " ")
                if len(data['prefix']) == 0:
                    token = token.strip()
                data['prefix'] += token
                yield token
                if token == self.__sp.eos_id():
                    max_new_tokens = 0
                    break
            else:
                break
        if max_new_tokens > 0:
            data['max_new_tokens'] = max_new_tokens
            s = requests.Session()

            with s.post(self.__model_url + "/get_commentary_execution", json=data, stream=True) as resp:
                for (probabilities, token) in self.consume_bytesio_stream(resp.raw):
                    self.__cache.set(json.dumps(data), probabilities)
                    token = self.__sp.IdToPiece(token)
                    token = token.replace("▁", " ")
                    if len(data['prefix']) == 0:
                        token = token.strip()
                    data['prefix'] += token
                    yield token


if __name__ == '__main__':
    a = ServeProxyUtilsFacadeSingleton()
    b = ServeProxyUtilsFacadeSingleton()

    print(a is b)