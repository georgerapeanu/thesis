import time
import os
import requests
import re
import urllib
import polars as pl
import chess
import pickle

def save_game(args, skipIfExists=False):
    split, game_id = args
    if skipIfExists and os.path.exists(f"../raw_data/{split}/{game_id}.parquet"):
        return
    print(f"Doing {split}/{game_id}")
    time.sleep(0.35)
    response = requests.get(f"https://gameknot.com/annotate.pl?id={game_id}")
    if response.status_code != 200:
        print(response.text)
        raise Exception("Something went bad")
    game_string = response.text

    comments = re.findall("game_notes.*", game_string)[0].removesuffix("];").removeprefix("game_notes = [").strip()
    comments_list = list(map(lambda x: urllib.parse.unquote(x.strip().removeprefix("\"").removesuffix("\"").strip()), comments.split(",")))

    moves = re.findall("game_movelist.*", game_string)[0].removesuffix("';").removeprefix("game_movelist = '").strip()
    moves_list = [moves[i:i + 5].removesuffix("-") for i in range(0, len(moves), 5)]

    data = []
    board = chess.Board()
    for move, comment in zip(moves_list, comments_list[1:]):
        prev_board = board.fen()
        board.push_uci(move)
        current_board = board.fen()
        if len(comment) == 0:
            continue
        data.append({
            'previous_board': prev_board,
            'current_board': current_board,
            'comment': comment
        })
    pl.DataFrame(data).write_parquet(f"../raw_data/{split}/{game_id}.parquet")
    print(f"Done {split}/{game_id}")


SPLITS = [
    ("train", "./train_links.p"),
    ("test", "./test_links.p"),
    ("valid", "./valid_links.p"),
]

WORK = [
    (x[0], int(game[1].split("gm=", 1)[1])) for x in SPLITS for game in pickle.load(open(x[1], "rb"))
]


if __name__ == '__main__':
    for i, work in enumerate(WORK):
        save_game(work, True)
        print(f"Done {i + 1}/{len(WORK)}")
