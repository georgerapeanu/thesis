import os

import chess
import stockfish
import torch
import torchtext.vocab
from torch.utils.data import Dataset
from typing import *
import polars as pl
from torch.nn.utils.rnn import pad_sequence
from utils.configs import DataConfig
import sentencepiece
class CommentaryDataset(Dataset):
    def __init__(self, config: DataConfig):
        self.__config = config
        self.__deltas = CommentaryDataset.__all_move_deltas()
        self.__sp = sentencepiece.SentencePieceProcessor(model_file=self.__config['sentencepiece_path'])

        self.__raw_data = []
        for filename in os.listdir(os.path.join(self.__config['data_path'], self.__config['split'])):
            try:
                local_data = pl.read_parquet(os.path.join(self.__config['data_path'], self.__config['split'], filename)).rows()
                past_boards = []
                for row in local_data:
                    past_boards.append((row[0], row[3]))
                    current_board = (row[1], row[4])
                    if len(row[2].strip()) == 0:
                        pass
                    tokens = [self.__sp.bos_id()] + self.__sp.encode(row[2].strip().replace('\n', '<n>')) + [self.__sp.eos_id()]
                    if len(tokens) > config['context_length']:
                        for i in range(0, len(tokens) - 1 - config['context_length'], config['stride_big_sequences']):
                            self.__raw_data.append((past_boards[max(0, len(past_boards) - config['past_boards']):], current_board, tokens[i:i + config['context_length'] + 1]))
                    else:
                        self.__raw_data.append((past_boards[max(0, len(past_boards) - config['past_boards']):], current_board, tokens))
            except Exception as e:
                pass

    @staticmethod
    def __all_move_deltas() -> List[Tuple[int, int]]:
        deltas = []

        # queen moves -> 56
        for length in range(1, 8):
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    deltas.append((dx * length, dy * length))

        # knight moves -> 8
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dx_mult, dy_mult in [(2, 1), (1, 2)]:
                    deltas.append((dx_mult * dx, dy_mult * dy))

        cached_deltas = deltas
        # 64 moves
        return deltas

    def __get_positional_features(self, board: chess.Board, evaluation: int) -> torch.tensor:
        layers = torch.zeros([207, 8, 8], dtype=torch.int32)

        i = 0
        # P1 and P2 -> 12
        for color in [chess.WHITE, chess.BLACK]:
            for piece in [chess.PAWN, chess.ROOK, chess.KING, chess.BISHOP, chess.QUEEN, chess.KNIGHT]:
                layers[i, :, :] = torch.tensor(board.pieces(piece, color).tolist(), dtype=torch.int32).reshape((8, 8))
                i += 1

        # attackers mask -> 192
        for square in chess.SQUARES:
            layers[i, :, :] = (torch.tensor(board.attacks(square).tolist(), dtype=torch.int32).reshape((8, 8)))
            i += 1
            for color in [chess.WHITE, chess.BLACK]:
                layers[i, :, :] = (torch.tensor(board.pin(color, square).tolist(), dtype=torch.int32).reshape((8, 8)))
                i += 1

        # checks -> 1
        layers[i, :, :] = (torch.tensor(board.checkers().tolist(), dtype=torch.int32).reshape((8, 8)))
        i += 1
        # strength -> 1
        layers[i, :, :] = (torch.full([8, 8], evaluation))
        i += 1

        # Repetitions -> 1
        for count in range(3, -1, -1):
            if count == 0 or board.is_repetition(count):
                layers[i, :, :] = (torch.full((8, 8), count))
                i += 1
                break
        # Total 207, shape 207 x 8 x 8
        return layers

    def __get_state_features(self, board: chess.Board) -> torch.tensor:
        layers = torch.zeros((7, 8, 8))
        # Color -> 1
        layers[0].fill_((0 if board.turn is chess.WHITE else 1))
        # Total moves -> 1
        layers[1].fill_(board.fullmove_number)
        # P1 and P2 castling -> 4
        for i, color in enumerate([chess.WHITE, chess.BLACK]):
            layers[2 + 2 * i].fill_(int(board.has_kingside_castling_rights(color)))
            layers[3 + 2 * i].fill_(int(board.has_queenside_castling_rights(color)))

        # No progress count -> 1
        layers[6].fill_(board.halfmove_clock)

        # Shape 7 x 8 x 8
        return layers

    def __get_all_move_features(self, board: chess.Board) -> torch.tensor:
        FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ROWS = ['1', '2', '3', '4', '5', '6', '7', '8']

        move_features = torch.zeros([73, 8, 8])

        # moves - 64
        for i, (dx, dy) in enumerate(self.__deltas):
            for x in range(8):
                for y in range(8):
                    if x + dx < 0 or x + dx >= 8 or y + dy < 0 or y + dy >= 8:
                        continue
                    move = chess.Move.from_uci(FILES[x] + ROWS[y] + FILES[x + dx] + ROWS[y + dy])
                    if not board.is_legal(move):
                        move = chess.Move.from_uci(FILES[x] + ROWS[y] + FILES[x + dx] + ROWS[y + dy] + 'q')
                    if not board.is_legal(move):
                        continue
                    move_features[i, x, y] = 1

        i = len(self.__deltas)

        # underpromotions -> 9
        dy = (1 if board.turn == chess.WHITE else -1)
        target_y = (6 if board.turn == chess.WHITE else 1)
        for dx in [-1, 0, 1]:
            for piece in ['r', 'b', 'n']:
                for x in range(0, 8):
                    for y in range(0, 8):
                        if y != target_y:
                            continue
                        if x + dx < 0 or x + dx >= 8:
                            continue
                        move = chess.Move.from_uci(FILES[x] + ROWS[y] + FILES[x + dx] + ROWS[y + dy] + piece)
                        if not board.is_legal(move):
                            continue
                        move_features[i, x, y] = 1
                i += 1

        return move_features

    def __len__(self):
        return len(self.__raw_data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        POSITIONAL_SIZE = 207
        MOVE_SIZE = 73
        STATE_SIZE = 7
        answer_board = torch.zeros([STATE_SIZE + POSITIONAL_SIZE * self.__config['past_boards'] + (POSITIONAL_SIZE + MOVE_SIZE), 8, 8], dtype=torch.int32)

        current_board, current_eval = chess.Board(self.__raw_data[idx][1][0]), self.__raw_data[idx][1][1]
        past_boards = [chess.Board(x[0]) for x in self.__raw_data[idx][0]]
        past_evals = [x[1] for x in self.__raw_data[idx][0]]
        if current_board.turn == chess.BLACK:
            current_board = current_board.mirror()
            past_boards = [x.mirror() for x in past_boards]
            current_eval = -current_eval
            past_evals = [-x for x in past_evals]


        answer_board[-STATE_SIZE:, :, :] = self.__get_state_features(current_board)
        answer_board[-STATE_SIZE - MOVE_SIZE:-STATE_SIZE, :, :] = self.__get_all_move_features(current_board)
        answer_board[-STATE_SIZE - POSITIONAL_SIZE - MOVE_SIZE:-STATE_SIZE - MOVE_SIZE, :, :] = self.__get_positional_features(current_board, current_eval)
        for i in range(0, len(past_boards)):
            answer_board[
                -STATE_SIZE - (i + 2) * POSITIONAL_SIZE - MOVE_SIZE:-STATE_SIZE - (i + 1) * POSITIONAL_SIZE - MOVE_SIZE,
                :,
                :
            ] = (
                self.__get_positional_features(past_boards[-i], past_evals[-i])
            )

        return answer_board, torch.tensor(self.__raw_data[idx][2])

    def pad_id(self) -> int:
        return self.__sp.pad_id()

# outdated
# if __name__ == "__main__":
#     ds = CommentaryDataset({
#         'split': 'train',
#         'raw_data_path': '../raw_data',
#         'past_boards': 8,
#         'mate_value': 10000,
#         'context_length': 100,
#         'sentencepiece_path': '../artifacts/sp8000.model',
#         'stride_big_sequences': 1,
#         'engine_config': {
#             'threads': 4,
#             'hash': 64,
#             'minimum_thinking_time': 1,
#             'location': '../artifacts/stockfish-ubuntu-x86-64-avx2',
#             'engine_depth': 5,
#         }
#     })
#     print(ds[0])