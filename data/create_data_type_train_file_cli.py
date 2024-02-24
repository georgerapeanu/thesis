import os.path
import traceback

import polars as pl
import random


TYPES = ['MoveDesc', 'MoveQuality', 'Comparative', 'Strategy', 'Context', 'General']

if __name__ == '__main__':
    commentaries = (list(iter(open("../artifacts/commentaries_raw.txt", "r"))))

    annotated_samples = []
    if os.path.exists("../artifacts/commentary_types.parquet"):
        annotated_samples = pl.read_parquet("../artifacts/commentary_types.parquet").rows(named=True)

    try:
        while True:
            sample = random.choice(commentaries)
            print(f"{len(annotated_samples)}: {sample}")
            print(f"Choose type {list(enumerate(TYPES))}")
            chosen_types = []
            while len(chosen_types) == 0 or len(list(filter(lambda x: x < 0 or x >= len(TYPES), chosen_types))) > 0:
                chosen_types = list(map(lambda x: int(x), [*input()]))

            annotated_samples.append({
                'commentary': sample,
                'type': ",".join(map(lambda x: str(x), chosen_types)),
                'type_str': ",".join([TYPES[x] for x in chosen_types])
            })
    except Exception:
        traceback.print_exc()
        pl.DataFrame(annotated_samples).write_parquet("../artifacts/commentary_types.parquet")
