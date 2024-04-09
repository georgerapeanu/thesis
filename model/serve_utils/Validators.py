from abc import abstractmethod

import chess
import jsonschema


# Template
# chain of command


class AbstractValidator:
    def __init__(self, next_validator=None):
        self.__next_validator = next_validator

    def validate(self, data):
        self._validate_self(data)
        if self.__next_validator is not None:
            self.__next_validator.validate(data)

    @abstractmethod
    def _validate_self(self, data):
        pass

    def set_next(self, next_validator):
        self.__next_validator = next_validator


class JsonSchemaValidator(AbstractValidator):
    def __init__(self, next_validator=None):
        super().__init__(next_validator)
        self.__payload_schema = {
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
                }
            },
            "required": ["past_boards", "current_board"]
        }

    @abstractmethod
    def _validate_self(self, data):
        jsonschema.validate(instance=data, schema=self.__payload_schema)


class BoardsValidator(AbstractValidator):
    def __init__(self, cfg, next_validator=None):
        self.__count_past_boards = cfg['count_past_boards']
        super().__init__(next_validator)

    @abstractmethod
    def _validate_self(self, data):
        past_boards = data.get('past_boards')
        current_board = data.get('current_board')

        if len(past_boards) != self.__count_past_boards:
            raise ValueError("Length of past boards array is not the same as the one configured in config")

        for board in past_boards + [current_board]:
            try:
                chess.Board(board)
            except ValueError:
                raise ValueError("Board FEN is invalid")


class TemperatureValidator(AbstractValidator):
    def __init__(self, next_validator=None):
        super().__init__(next_validator)

    @abstractmethod
    def _validate_self(self, data):
        temperature = 1.0 if 'temperature' not in data else data.get('temperature')

        if temperature <= 0:
            raise ValueError("Temperature should be bigger than 0")


class TargetTypeValidator(AbstractValidator):
    def __init__(self, TARGET_TYPES_TO_IDS, next_validator=None):
        self.__TARGET_TYPES_TO_IDS = TARGET_TYPES_TO_IDS
        super().__init__(next_validator)

    @abstractmethod
    def _validate_self(self, data):
        target_type = None if 'target_type' not in data else data.get('target_type')

        if target_type is not None and target_type not in self.__TARGET_TYPES_TO_IDS:
            raise ValueError("Target type is not recognized")


class MaxNewTokensValidator(AbstractValidator):
    def __init__(self, next_validator=None):
        super().__init__(next_validator)

    @abstractmethod
    def _validate_self(self, data):
        max_new_tokens = 1000 if 'max_new_tokens' not in data else data.get('max_new_tokens')

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens should be bigger than 0")
