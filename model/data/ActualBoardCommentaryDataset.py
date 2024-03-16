import itertools
import os

import chess
import stockfish
import torch
import torchtext.vocab
from torch.utils.data import Dataset
from typing import *
import polars as pl
from torch.nn.utils.rnn import pad_sequence
import sentencepiece
from data.create_data_type_train_file_cli import TYPES
from omegaconf import ListConfig, DictConfig


class ActualBoardCommentaryDataset(Dataset):
    def __init__(
        self,
        config: DictConfig,
        engine_config: DictConfig,
        sp: sentencepiece.SentencePieceProcessor,
    ):
        self.mate_value = engine_config.mate_value
        self.count_past_boards = config.count_past_boards
        self.__sp = sp
        self.__raw_data = []
        self.__data = []

        for filename in os.listdir(os.path.join(config.processed_path, config.split)):
            local_data = pl.read_parquet(os.path.join(config.processed_path, config.split, filename)).rows(named=True)
            past_boards = []
            for row in local_data:
                past_boards.append((row['past_board'], row['past_strength']))
                current_board = (row['current_board'], row['current_strength'])

                if len(row['commentary'].strip()) == 0:
                    continue
                take = False

                types = torch.zeros(len(TYPES), dtype=torch.bool)
                for type in config.target_types:
                    if row[f"is_type_{type}"]:
                        take = True
                        types[type] = True

                if not take:
                    continue

                tokens = [self.__sp.bos_id()] + self.__sp.encode(row['commentary'].strip().replace('\n', '<n>')) + [self.__sp.eos_id()]
                if len(tokens) > config.context_length:
                    for i in range(0, len(tokens) - 1, config.stride_big_sequences):
                        to_add = ((
                            past_boards[max(0, len(past_boards) - config.count_past_boards):],
                            current_board,
                            tokens[i:min(len(tokens), i + config.context_length)],
                            types
                        ))

                        self.__raw_data.append(to_add)
                        self.__data.append(self.raw_data_to_data(to_add))
                else:
                    to_add = ((
                        past_boards[max(0, len(past_boards) - config.count_past_boards):],
                        current_board,
                        tokens,
                        types
                    ))

                    self.__raw_data.append(to_add)
                    self.__data.append(self.raw_data_to_data(to_add))

    def __get_positional_features(self, board: chess.Board) -> torch.tensor:
        board_stuff = torch.zeros(64, dtype=torch.int32)

        for i, (color, piece) in enumerate(itertools.product([chess.WHITE, chess.BLACK], [chess.PAWN, chess.ROOK, chess.KING, chess.BISHOP, chess.QUEEN, chess.KNIGHT])):
            board_data = board.pieces(piece, color).tolist()
            for j in range(64):
                if board_data[j]:
                    board_stuff[j] = i + 1

        # Repetitions -> 1
        repetitions = 0
        for count in range(3, 0, -1):
            if count == 0 or board.is_repetition(count):
                repetitions = count
                break

        return board_stuff, repetitions

    def __get_state_features(self, board: chess.Board) -> torch.tensor:
        answer = torch.zeros((7))
        # Color -> 1
        answer[0] = ((0 if board.turn == chess.WHITE else 1))
        # Total moves -> 1
        answer[1] = (board.fullmove_number)
        # P1 and P2 castling -> 4
        for i, color in enumerate([chess.WHITE, chess.BLACK]):
            answer[2 + 2 * i] = (int(board.has_kingside_castling_rights(color)))
            answer[3 + 2 * i] = (int(board.has_queenside_castling_rights(color)))

        # No progress count -> 1
        answer[6] = (board.halfmove_clock)

        return answer

    def __len__(self):
        return len(self.__raw_data)

    # repetitions, strength
    # Board x 64, Board x 1 (strength) , Board x 1 (rep), 7(state features), tokens, types
    def raw_data_to_data(self, raw_data):
        current_board, current_eval = chess.Board(raw_data[1][0]), raw_data[1][1]
        past_boards = [chess.Board(x[0]) for x in raw_data[0]]
        past_evals = [x[1] for x in raw_data[0]]

        boards = past_boards + [current_board]
        processed_boards = list(map(lambda x: self.__get_positional_features(x), boards))
        boards_tensor = torch.stack(list(map(lambda x: x[0], processed_boards)))
        reps_tensor = torch.tensor(list(map(lambda x: x[1], processed_boards))).unsqueeze(1)
        strengths_tensor = torch.tensor(past_evals + [current_eval]).unsqueeze(1)

        boards_tensor = torch.cat([torch.zeros(self.count_past_boards - len(past_boards), 64), boards_tensor], dim=0)
        strengths_tensor = torch.cat([torch.zeros(self.count_past_boards - len(past_boards), 1), strengths_tensor], dim=0)
        reps_tensor = torch.cat([torch.zeros(self.count_past_boards - len(past_boards), 1), reps_tensor], dim=0)

        state_tensor = self.__get_state_features(current_board)

        # return normalisations
        return (
            boards_tensor.int(),
            reps_tensor / 3.0,
            strengths_tensor / self.mate_value,
            state_tensor,
            torch.tensor(raw_data[2]),
            raw_data[3]
        )

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.tensor, torch.Tensor]:
        return self.__data[idx]

    def get_raw_data(self, idx: int) -> Tuple[Optional[str], str, Optional[int], int]:
        current_board, current_eval = self.__raw_data[idx][1]
        past_board, past_eval = None, None
        if len(self.__raw_data[idx][0]) != 0:
            past_board, past_eval = self.__raw_data[idx][0][-1]
        return current_board, past_board, current_eval, past_eval

    @staticmethod
    def get_board_token_size():
        return 13