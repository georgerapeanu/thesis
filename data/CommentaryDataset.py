import itertools
import multiprocessing
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
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
INV_FILES = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
ROWS = ['1', '2', '3', '4', '5', '6', '7', '8']
INV_ROWS = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7}
INV_PROMOTIONS = {'b': 0, 'r': 1, 'n': 2}
class CommentaryDataset(Dataset):
    def __init__(self, config: DataConfig):
        self.__config = config
        self.__deltas, self.__inv_deltas = CommentaryDataset.__all_move_deltas()
        self.__sp = sentencepiece.SentencePieceProcessor(model_file=self.__config['sentencepiece_path'])

        self.__data = []

        with multiprocessing.Pool(config['ds_num_workers']) as p:
            self.__data = list(itertools.chain(p.map(self.worker, os.listdir(os.path.join(self.__config['data_path'], self.__config['split'])))))


    def worker(self, filename):
        work_data = []
        local_data = pl.read_parquet(
            os.path.join(self.__config['data_path'], self.__config['split'], filename)).rows()
        past_boards = []
        for row in local_data:
            past_boards.append((row[0], row[3]))
            current_board = (row[1], row[4])
            if len(row[2].strip()) == 0:
                continue
            tokens = [self.__sp.bos_id()] + self.__sp.encode(row[2].strip().replace('\n', '<n>')) + [
                self.__sp.eos_id()]
            if len(tokens) > self.__config['context_length']:
                for i in range(0, len(tokens) - 1 - self.__config['context_length'], self.__config['stride_big_sequences']):
                    work_data.append(self.raw_data_to_data((past_boards[
                                                              max(0, len(past_boards) - self.__config['past_boards']):],
                                                              current_board,
                                                              tokens[i:i + self.__config['context_length'] + 1])))
            else:
                work_data.append(self.raw_data_to_data(
                    (past_boards[max(0, len(past_boards) - self.__config['past_boards']):], current_board, tokens)))

    @staticmethod
    def __all_move_deltas() -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
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
        inv_deltas = {}
        for i, d in enumerate(deltas):
            inv_deltas[d] = i
        # 64 moves
        return deltas, inv_deltas

    def __get_positional_features(self, board: chess.Board, evaluation: int) -> torch.tensor:
        layers = torch.zeros([15, 8, 8], dtype=torch.int32)

        # P1 and P2 -> 12
        for i, (color, piece) in enumerate(itertools.product([chess.WHITE, chess.BLACK], [chess.PAWN, chess.ROOK, chess.KING, chess.BISHOP, chess.QUEEN, chess.KNIGHT])):
                layers[i, :, :] = torch.tensor(board.pieces(piece, color).tolist(), dtype=torch.int32).reshape((8, 8))

        # checks -> 1
        layers[12, :, :] = (torch.tensor(board.checkers().tolist(), dtype=torch.int32).reshape((8, 8)))
        # strength -> 1
        layers[13, :, :] = (torch.full([8, 8], evaluation))

        # Repetitions -> 1
        for count in range(3, -1, -1):
            if count == 0 or board.is_repetition(count):
                layers[14, :, :] = (torch.full((8, 8), count))
                break
        # Total 15, shape 15 x 8 x 8
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
        move_features = torch.zeros([73, 8, 8])

        for move in board.legal_moves:
            str_move = str(move)
            from_square = (INV_FILES[str_move[0]], INV_ROWS[str_move[1]])
            to_square = (INV_FILES[str_move[2]], INV_ROWS[str_move[3]])
            delta = (to_square[0] - from_square[0], to_square[1] - from_square[1])
            promotion_type = None if len(str_move) == 4 else str_move[4]

            if promotion_type is None or promotion_type == 'q':
                index = self.__inv_deltas[delta]
            else:
                # doesn't distinguish between white and black promotions, maybe it's bad, not sure
                index = len(self.__deltas) + INV_PROMOTIONS[promotion_type] * 3 + 1 + delta[0]
            move_features[index, from_square[0], from_square[1]] = 1

        return move_features

    def __len__(self):
        return len(self.__data)

    def raw_data_to_data(self, raw_data):
        POSITIONAL_SIZE = 15
        MOVE_SIZE = 73
        STATE_SIZE = 7
        answer_board = torch.zeros(
            [STATE_SIZE + POSITIONAL_SIZE * self.__config['past_boards'] + (POSITIONAL_SIZE + MOVE_SIZE), 8, 8],
            dtype=torch.int32)

        current_board, current_eval = chess.Board(raw_data[1][0]), raw_data[1][1]
        past_boards = [chess.Board(x[0]) for x in raw_data[0]]
        past_evals = [x[1] for x in raw_data[0]]
        if current_board.turn == chess.BLACK:
            current_board = current_board.mirror()
            past_boards = [x.mirror() for x in past_boards]
            current_eval = -current_eval
            past_evals = [-x for x in past_evals]

        answer_board[-STATE_SIZE:, :, :] = self.__get_state_features(current_board)
        answer_board[-STATE_SIZE - MOVE_SIZE:-STATE_SIZE, :, :] = self.__get_all_move_features(current_board)
        answer_board[-STATE_SIZE - POSITIONAL_SIZE - MOVE_SIZE:-STATE_SIZE - MOVE_SIZE, :,
        :] = self.__get_positional_features(current_board, current_eval)
        for i in range(0, len(past_boards)):
            answer_board[
            -STATE_SIZE - (i + 2) * POSITIONAL_SIZE - MOVE_SIZE:-STATE_SIZE - (i + 1) * POSITIONAL_SIZE - MOVE_SIZE,
            :,
            :
            ] = (
                self.__get_positional_features(past_boards[-i], past_evals[-i])
            )

        return answer_board, torch.tensor(raw_data[2])

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.__data[idx]

    def pad_id(self) -> int:
        return self.__sp.pad_id()

    def bos_id(self) -> int:
        return self.__sp.bos_id()

    def eos_id(self) -> int:
        return self.__sp.eos_id()

    def vocab_size(self) -> int:
        return self.__sp.vocab_size()

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