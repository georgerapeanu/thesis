import time
import os
import requests
import re
import urllib
import polars as pl
import chess
import pickle
from typing import *


SAVE_PATH = ""


def parse_response_text(game_string: str):
    comments = re.findall("game_notes.*", game_string)[0].removesuffix("];").removeprefix("game_notes = [").strip()
    comments_list = list(map(lambda x: urllib.parse.unquote(x.strip().removeprefix("\"").removesuffix("\"").strip()),
                             comments.split(",")))

    moves = re.findall("game_movelist.*", game_string)[0].removesuffix("';").removeprefix("game_movelist = '").strip()
    moves_list = [moves[i:i + 5].removesuffix("-") for i in range(0, len(moves), 5)]

    data = []
    board = chess.Board()
    for move, comment in zip(moves_list, comments_list[1:]):
        prev_board = board.fen()
        board.push_uci(move)
        current_board = board.fen()
        data.append({
            'previous_board': prev_board,
            'current_board': current_board,
            'comment': comment
        })
    return pl.DataFrame(data)


def save_game(args: Tuple[str, int], skipIfExists=False):
    split, game_id = args
    if skipIfExists and os.path.exists(os.path.join(SAVE_PATH, f"{split}/{game_id}.parquet")):
        return
    print(f"Doing {split}/{game_id}")
    time.sleep(0.35)
    response = requests.get(f"https://gameknot.com/annotate.pl?id={game_id}")
    if response.status_code != 200:
        print(response.text)
        raise Exception("Something went bad")
    parse_response_text(response.text).write_parquet(os.path.join(SAVE_PATH, f"{split}/{game_id}.parquet"))
    print(f"Done {split}/{game_id}")

def crawl(pickle_path: str, raw_data_path: str):
    SPLITS = [
        ("train", os.path.join(pickle_path, "train_links.p")),
        ("test", os.path.join(pickle_path, "test_links.p")),
        ("valid", os.path.join(pickle_path, "valid_links.p")),
    ]
    global SAVE_PATH
    SAVE_PATH = raw_data_path

    WORK = [
        (x[0], int(game[1].split("gm=", 1)[1])) for x in SPLITS for game in pickle.load(open(x[1], "rb"))
    ]

    for i, work in enumerate(WORK):
        save_game(work, False)
        print(f"Done {i + 1}/{len(WORK)}")

