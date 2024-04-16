"""
Facade(restrictionez acces la componente)
Singleton(una singura duh)
Chain of responsability(pt validari)
Strategy(pt modul de sampling)
Observer pe frontend pt raspuns
"""
import json

import chess
import hydra
import jsonschema
import sentencepiece
import stockfish
import torch
from flask import Response

from data.ActualBoardCommentaryDataset import ActualBoardCommentaryDataset
from serve_utils.SamplingStrategies import MultinomialSamplingStrategy, TopKSamplingStrategy
from serve_utils.Validators import JsonSchemaValidator, BoardsValidator, MaxNewTokensValidator, TargetTypeValidator, \
    TemperatureValidator


class ServeUtilsFacadeSingleton(object):
    #singleton through new
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ServeUtilsFacadeSingleton, cls).__new__(cls)
            cls.instance.to_initialize = True
        return cls.instance

    def __init__(self):
        if hasattr(self, 'to_initialize') and self.to_initialize:
            del self.to_initialize
            with hydra.initialize(version_base="1.2", config_path="../conf"):
                self.__cfg = hydra.compose(config_name="serve_config")

            self.__model = torch.jit.load(self.__cfg["model_path"])
            self.__sp = sentencepiece.SentencePieceProcessor(self.__cfg["sentencepiece_path"])
            self.__engine = stockfish.Stockfish(self.__cfg['engine_config']['location'], parameters={
                "Threads": self.__cfg['engine_config']['threads'],
                "Hash": self.__cfg['engine_config']['hash'],
                "Minimum Thinking Time": self.__cfg['engine_config']['minimum_thinking_time']
            })
            self.__engine.set_depth(self.__cfg['engine_config']['engine_depth'])
            self.TARGET_TYPES_TO_IDS = {
                'MoveDesc': 0,
                'MoveQuality': 1,
                'Comparative': 2,
                "Strategy": 3,
                "Context": 4
            }

            validator = MaxNewTokensValidator()
            validator = TargetTypeValidator(self.TARGET_TYPES_TO_IDS, validator)
            validator = TemperatureValidator(validator)
            validator = BoardsValidator(self.__cfg, validator)
            validator = JsonSchemaValidator(validator)
            self.__validator = validator

    def __evaluation_to_value(self, evaluation):
        if evaluation['type'] == 'cp':
            return evaluation['value']
        else:
            return self.__cfg['engine_config']['mate_value'] if evaluation['value'] > 0 else -self.__cfg['engine_config'][
                'mate_value']

    def __position_to_value(self, position: str):
        self.__engine.set_fen_position(position)
        return self.__evaluation_to_value(self.__engine.get_evaluation())

    def validate_request(self, request_data):
        data = json.loads(request_data)
        self.__validator.validate(data)

    def get_commentary(self, request_data):
        data = json.loads(request_data)

        past_boards = data.get('past_boards')
        current_board = data.get('current_board')
        temperature = 1.0 if 'temperature' not in data else data.get('temperature')
        do_sample = False if 'do_sample' not in data else data.get('do_sample')
        target_type = None if 'target_type' not in data else data.get('target_type')
        if target_type is not None:
            target_type = torch.tensor(self.TARGET_TYPES_TO_IDS[target_type])
        max_new_tokens = 1000 if 'max_new_tokens' not in data else data.get('max_new_tokens')
        prefix = '' if 'prefix' not in data else data.get('prefix')

        sampler = MultinomialSamplingStrategy(temperature) if do_sample else TopKSamplingStrategy(temperature)

        (X_board, X_strength, X_reps, X_state, _, _) = ActualBoardCommentaryDataset.raw_data_to_data(
            (
                list(zip(past_boards, list(map(lambda x: self.__position_to_value(x), past_boards)))),
                (current_board, self.__position_to_value(current_board)),
                torch.zeros(0),
                torch.zeros(0)
            ),
            self.__cfg['count_past_boards'],
            self.__cfg['engine_config']['mate_value']
        )
        X_text = torch.tensor([self.__sp.bos_id()] + self.__sp.encode(prefix))
        X_text = X_text.unsqueeze(0)
        X_board = X_board.unsqueeze(0)
        X_strength = X_strength.unsqueeze(0)
        X_reps = X_reps.unsqueeze(0)
        X_state = X_state.unsqueeze(0)

        with torch.no_grad():
            for i in range(max_new_tokens):
                X_text = X_text if X_text.size(1) < self.__cfg.context_length else X_text[:, -self.__cfg.context_length:]
                logits, _ = self.__model(X_board, X_strength, X_reps, X_state, X_text,
                                  (torch.zeros(1, X_text.size(1)) == 1).to(X_board.device), target_type=target_type)

                text_next = sampler.execute(logits[:, -1, :])
                X_text = torch.cat([X_text, text_next], dim=1)
                if text_next == self.__model.eos_id:
                    break
                skip_length = 0
                if X_text.size(1) > 1:
                    skip_length = len(self.__sp.decode(X_text[0, -2].view(-1).tolist()).replace("<n>", "\n"))
                yield self.__sp.decode(X_text[0, -2:].view(-1).tolist()).replace("<n>", "\n")[skip_length:]


if __name__ == '__main__':
    a = ServeUtilsFacadeSingleton()
    b = ServeUtilsFacadeSingleton()

    print(a is b)