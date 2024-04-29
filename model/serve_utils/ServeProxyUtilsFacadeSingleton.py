"""
Facade(restrictionez acces la componente)
Singleton(una singura duh)
Chain of responsability(pt validari)
Strategy(pt modul de sampling)
Observer pe frontend pt raspuns
"""
import json
from io import BytesIO
from typing import Iterator, Tuple, List, Optional

import hydra
import numpy as np
import requests
import sentencepiece
import logging
import struct

import torch

from serve_utils.SamplingStrategies import MultinomialSamplingStrategy, TopKSamplingStrategy, AbstractSamplingStrategy
from serve_utils.Validators import JsonSchemaValidator, BoardsValidator, MaxNewTokensValidator, TargetTypeValidator, \
    TemperatureValidator, TopKValidator
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
            self.TARGET_TYPES_TO_IDS = {
                'MoveDesc': 0,
                'MoveQuality': 1,
                'Comparative': 2,
                "Strategy": 3,
                "Context": 4
            }
            validator = TargetTypeValidator(self.TARGET_TYPES_TO_IDS)
            validator = TemperatureValidator(validator)
            validator = BoardsValidator(self.__cfg, validator)
            self.__commentary_validator = JsonSchemaValidator({
                "type": "object",
                "properties": {
                    "past_boards": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "current_board": {
                        "type": "string"
                    },
                    "temperature": {
                        "type": "number"
                    },
                    "do_sample": {
                        "type": "boolean"
                    },
                    "target_type": {
                        "type": "string"
                    },
                    "max_new_tokens": {
                        "type": "number"
                    },
                    "prefix": {
                        "type": "string"
                    }
                },
                "required": ["past_boards", "current_board"]
            }, MaxNewTokensValidator(self.__cfg['max_new_tokens'], self.__cfg['max_new_tokens'], validator))

            validator = TopKValidator(self.__cfg['topk_max'], validator)
            self.__topk_validator = JsonSchemaValidator({
                "type": "object",
                "properties": {
                    "past_boards": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "current_board": {
                        "type": "string"
                    },
                    "temperature": {
                        "type": "number"
                    },
                    "do_sample": {
                        "type": "boolean"
                    },
                    "target_type": {
                        "type": "string"
                    },
                    "prefix": {
                        "type": "string"
                    },
                    "topk": {
                        "type": "number"
                    }
                },
                "required": ["past_boards", "current_board"]
            }, validator)
            self.__cache = LruCache(maxsize=self.__cfg["cache_size"])
            self.__model_url = self.__cfg["model"]["url"]

    def validate_commentary_request(self, request_data):
        data = json.loads(request_data)
        self.__commentary_validator.validate(data)

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

    def get_next_token_tensor(self, sampler: AbstractSamplingStrategy, logits: torch.Tensor) -> int:
        token = sampler.execute(logits).view(1).item()
        return token

    def get_next_token(self, sampler, logits: List[float]) -> int:
        return self.get_next_token_tensor(sampler, torch.tensor(logits).view(1, -1))

    def request_to_key(self, past_boards: List[str], current_board: str, target_type: Optional[str], prefix: str) -> str:
        return json.dumps({
            "past_boards": past_boards,
            "current_board": current_board,
            "target_type": target_type,
            "prefix": prefix
        })


    def get_commentary(self, request_data) -> Iterator[str]:
        logger.warning(f"Received request: {request_data}")
        data = json.loads(request_data)

        past_boards = data['past_boards']
        current_board = data['current_board']
        target_type = None if 'target_type' not in data else data.get('target_type')
        temperature = 1.0 if 'temperature' not in data else data.get('temperature')
        do_sample = False if 'do_sample' not in data else data.get('do_sample')
        max_new_tokens = 1000 if 'max_new_tokens' not in data else data.get('max_new_tokens')
        prefix = '' if 'prefix' not in data else data.get('prefix')
        data['prefix'] = prefix

        sampler = MultinomialSamplingStrategy(temperature) if do_sample else TopKSamplingStrategy(temperature)

        while max_new_tokens > 0:
            key = self.request_to_key(
                past_boards=past_boards,
                current_board=current_board,
                target_type=target_type,
                prefix=data['prefix']
            )
            if self.__cache.has(key):
                logits = self.__cache.get(key)
                max_new_tokens -= 1
                token = self.get_next_token(sampler, logits)
                if token == self.__sp.eos_id():
                    max_new_tokens = 0
                    break
                token = self.__sp.IdToPiece(token)
                token = token.replace("▁", " ")
                if len(data['prefix']) == 0:
                    token = token.strip()
                data['prefix'] += token
                yield token
            else:
                break
        if max_new_tokens > 0:
            data['max_new_tokens'] = max_new_tokens
            s = requests.Session()

            with s.post(self.__model_url + "/get_commentary_execution", json=data, stream=True) as resp:
                for (logits, token) in self.consume_bytesio_stream(resp.raw):
                    key = self.request_to_key(
                        past_boards=past_boards,
                        current_board=current_board,
                        target_type=target_type,
                        prefix=data['prefix']
                    )
                    self.__cache.set(key, logits)
                    if token == self.__sp.eos_id():
                        break
                    token = self.__sp.IdToPiece(token)
                    token = token.replace("▁", " ")
                    if len(data['prefix']) == 0:
                        token = token.strip()
                    data['prefix'] += token
                    yield token

    def get_topk(self, request_data) -> List[Tuple[str, float]]:
        logger.info("received topk request_data: {}".format(request_data))
        data = json.loads(request_data)
        self.__topk_validator.validate(data)

        data['max_new_tokens'] = 1
        if 'topk' in data:
            topk = data.get('topk')
            del data['topk']
        else:
            topk = 10

        past_boards = data['past_boards']
        current_board = data['current_board']
        target_type = None if 'target_type' not in data else data.get('target_type')
        prefix = '' if 'prefix' not in data else data.get('prefix')
        data['prefix'] = prefix

        key = self.request_to_key(
            past_boards=past_boards,
            current_board=current_board,
            target_type=target_type,
            prefix=data['prefix']
        )
        if not self.__cache.has(key):
            s = requests.Session()

            with s.post(self.__model_url + "/get_commentary_execution", json=data, stream=True) as resp:
                logits, _ = next(self.consume_bytesio_stream(resp.raw))
            self.__cache.set(key, logits)
        logits = self.__cache.get(key)

        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        values, indices = torch.topk(probabilities, k=topk, dim=-1)
        values = values.tolist()
        indices = list(map(lambda i: self.__sp.IdToPiece(i), indices.tolist()))

        return list(zip(values, indices))


if __name__ == '__main__':
    a = ServeProxyUtilsFacadeSingleton()
    b = ServeProxyUtilsFacadeSingleton()

    print(a is b)