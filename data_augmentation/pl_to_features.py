import multiprocessing
import os
import time

import chess
import stockfish
from chess import Board
import numpy as np
from typing import *
import polars as pl
import pickle

def extract_input_features_from_board(board: chess.Board) -> np.ndarray:
    layers = []

    #P1 and P2 -> 12
    for color in [chess.WHITE, chess.BLACK]:
        for piece in [chess.PAWN, chess.ROOK, chess.KING, chess.BISHOP, chess.QUEEN, chess.KNIGHT]:
            layers.append(np.array(board.pieces(piece, color).tolist()).astype(np.int64).reshape((8, 8)))

    # Repetitions -> 1
    for count in range(3, -1):
        if count == 0 or board.is_repetition(count):
            layers.append(np.full((8, 8), count))
            break
    # Total 13, shape 13 x 8 x 8
    return np.array(layers)


def extract_input_features_from_final_position(board: chess.Board) -> np.ndarray:
    layers = []
    # Color -> 1
    layers.append(np.full((8, 8), 0 if board.turn is chess.WHITE else 1))
    # Total moves -> 1
    layers.append(np.full((8, 8), board.fullmove_number))
    # P1 and P2 castling -> 4
    for color in [chess.WHITE, chess.BLACK]:
        if board.has_kingside_castling_rights(color):
            layers.append(np.full((8, 8), 1))
        else:
            layers.append(np.full((8, 8), 0))
        if board.has_queenside_castling_rights(color):
            layers.append(np.full((8, 8), 1))
        else:
            layers.append(np.full((8, 8), 0))

    # No progress count -> 1
    layers.append(np.full((8, 8), board.halfmove_clock))

    # Shape 7 x 8 x 8
    return np.array(layers)


cached_deltas = None
def get_all_strength_deltas() -> List[Tuple[int, int]]:
    global cached_deltas
    if cached_deltas is not None:
        return cached_deltas
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
    assert(len(deltas) == 64)
    return deltas


ENGINE_LOCATION = "../utils_bin/stockfish-ubuntu-x86-64-avx2"
ENGINE_DEPTH = 10
cached_engine = None
def get_engine() -> stockfish.Stockfish:
    global cached_engine
    if cached_engine is not None:
        return cached_engine

    cached_engine = stockfish.Stockfish(ENGINE_LOCATION, parameters={
        "Threads": 4,
        "Hash": 64,
    })
    cached_engine.set_depth(ENGINE_DEPTH)
    return cached_engine


FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
ROWS = ['1', '2', '3', '4', '5', '6', '7', '8']
MATE_VALUE = 1000000
def get_all__strength_features_from_board(board: chess.Board, depth: int) -> np.ndarray:
    invalid_entry = np.full(depth + 4, -MATE_VALUE if board.turn == chess.WHITE else MATE_VALUE)
    invalid_entry[depth + 1] = invalid_entry[depth + 2] = invalid_entry[depth + 3] = 0

    strength_features = np.ones([8, 8, 74, 1], dtype=np.int64)
    strength_features = strength_features @ invalid_entry.reshape((1, -1))

    deltas = get_all_strength_deltas()
    engine = get_engine()

    def evaluation_to_value(evaluation):
        if evaluation['type'] == 'cp':
            return evaluation['value']
        else:
            return MATE_VALUE if evaluation['value'] > 0 else -MATE_VALUE

    #moves - 64
    for i, (dx, dy) in enumerate(deltas):
        for x in range(8):
            for y in range(8):
                if x + dx < 0 or x + dx >= 8 or y + dy < 0 or y + dy >= 8:
                    continue
                move = chess.Move.from_uci(FILES[x] + ROWS[y] + FILES[x + dx] + ROWS[y + dy])
                if not board.is_legal(move):
                    move = chess.Move.from_uci(FILES[x] + ROWS[y] + FILES[x + dx] + ROWS[y + dy] + 'q')
                if not board.is_legal(move):
                    continue
                board.push(move)
                engine.set_fen_position(board.fen())
                board.pop()
                for d in range(0, depth):
                    engine.set_depth(d + 1)
                    strength_features[x, y, i, d] = evaluation_to_value(engine.get_evaluation())
                engine.set_depth(ENGINE_DEPTH)
                # True evaluation
                strength_features[x, y, i, depth] = evaluation_to_value(engine.get_evaluation())
                # is capture
                if board.is_capture(move):
                    strength_features[x, y, i, depth + 1] = 1
                else:
                    strength_features[x, y, i, depth + 1] = 0
                # is_check
                board.push(move)
                if board.is_check():
                    strength_features[x, y, i, depth + 2] = 1
                else:
                    strength_features[x, y, i, depth + 2] = 0
                board.pop()
                #is_checkmate
                if engine.get_evaluation()['type'] == 'mate':
                    strength_features[x, y, i, depth + 3] = 1
                else:
                    strength_features[x, y, i, depth + 3] = 0

    i = len(deltas)
    # current position
    for d in range(0, depth + 1):
        engine.set_fen_position(board.fen())
        if d == depth:
            engine.set_depth(ENGINE_DEPTH)
        else:
            engine.set_depth(d + 1)
        fill_value = evaluation_to_value(engine.get_evaluation())
        for x in range(0, 8):
            for y in range(0, 8):
                strength_features[x, y, i, d] = fill_value
    i += 1

    # underpromotions
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
                    board.push(move)
                    engine.set_fen_position(board.fen())
                    board.pop()
                    for d in range(0, depth):
                        engine.set_depth(d + 1)
                        strength_features[x, y, i, d] = evaluation_to_value(engine.get_evaluation())
                    engine.set_depth(ENGINE_DEPTH)
                    # True evaluation
                    strength_features[x, y, i, depth] = evaluation_to_value(engine.get_evaluation())
                    # is capture
                    if board.is_capture(move):
                        strength_features[x, y, i, depth + 1] = 1
                    else:
                        strength_features[x, y, i, depth + 1] = 0
                    # is_check
                    board.push(move)
                    if board.is_check():
                        strength_features[x, y, i, depth + 2] = 1
                    else:
                        strength_features[x, y, i, depth + 2] = 0
                    board.pop()
                    # is_checkmate
                    if engine.get_evaluation()['type'] == 'mate':
                        strength_features[x, y, i, depth + 3] = 1
                    else:
                        strength_features[x, y, i, depth + 3] = 0
            i += 1

    assert i == 74
    assert strength_features.shape == (8, 8, 74, depth + 4)
    return strength_features


PAST_BOARDS = 8
def single_pl_to_features(df: pl.DataFrame, depth:int):
    input_features_past_boards = np.zeros((PAST_BOARDS * 13, 8, 8), dtype=np.int64)
    mirrored_input_features_past_boards = np.zeros((PAST_BOARDS * 13, 8, 8), dtype=np.int64)

    inputs = []
    for row in df.rows(named=True):
        board = chess.Board(row['previous_board'])
        # Add previous board to first input part
        input_features_past_boards = np.concatenate([input_features_past_boards, extract_input_features_from_board(board)], axis=0)
        input_features_past_boards = input_features_past_boards[13:]
        board = board.mirror()
        mirrored_input_features_past_boards = np.concatenate([mirrored_input_features_past_boards, extract_input_features_from_board(board)], axis=0)
        mirrored_input_features_past_boards = mirrored_input_features_past_boards[13:]
        board = board.mirror()

        # Determine the move that happened
        real_move = None
        for move in board.legal_moves:
            board.push(move)
            if board.fen() == row['current_board']:
                real_move = move
            board.pop()
            if real_move is not None:
                break
        assert real_move is not None

        # build the whole first input part
        if board.turn == chess.WHITE:
            current_input_features = extract_input_features_from_final_position(board)
            first_input_part = np.concatenate([input_features_past_boards, current_input_features])
            pass
        else:
            current_input_features = extract_input_features_from_final_position(board.mirror())
            first_input_part = np.concatenate([mirrored_input_features_past_boards, current_input_features])
            pass

        # Second input part, oriented the same way
        # contains the played move board, and the best move
        if board.turn == chess.WHITE:
            actual_board = chess.Board(row['current_board'])
            best_board = chess.Board(board.fen())
            pass
        else:
            actual_board = chess.Board(row['current_board']).mirror()
            best_board = chess.Board(board.fen()).mirror()
            pass
        engine = get_engine()
        engine.set_depth(ENGINE_DEPTH)
        engine.set_fen_position(best_board.fen())
        best_move = engine.get_top_moves(1)[0]['Move']
        best_board.push_uci(best_move)

        second_input_part = np.concatenate(
            [np.expand_dims(get_all__strength_features_from_board(actual_board, depth), 0), np.expand_dims(get_all__strength_features_from_board(best_board, depth), 0)],
            0
        )

        third_input_part = f"{str(real_move)} {row['comment']}"

        input = {
            'past': first_input_part,
            'current': second_input_part,
            'comment': third_input_part
        }
        inputs.append(input)

    return inputs



SPLITS = [
    "test",
    "train",
    "valid"
]

FEATURE_DEPTH = 5

if __name__ == "__main__":
    dfs = []
    for split in SPLITS:
        for filename in os.listdir("../raw_data/"+split+"/"):
            dfs.append(f"../raw_data/{split}/" + filename)

    def worker(file: str):
        print(f"Porcessing {file}")
        try:
            df = pl.read_parquet(file)
            file = file.replace("raw", "processed").replace("parquet", "p")
            if os.path.exists(file):
                print("File already processed, skipping")
                return
            print(f"Set output to {file}")
            pickle.dump(single_pl_to_features(df, FEATURE_DEPTH), open(file, "wb"))
        except Exception as e:
            print(f"game {split}/{filename} didn't have any annotations, skipping")


    with multiprocessing.Pool(processes=8) as p:
        p.map(worker, dfs)
