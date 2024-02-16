import polars as pl
import random

random.seed(0)

TYPES = ['MoveDesc', 'MoveQuality', 'Comparative', 'Strategy', 'Context', 'General']
ANNOTATE_COUNT = 2000

if __name__ == '__main__':
    samples = random.sample(list(iter(open("../artifacts/commentaries.txt", "r"))), ANNOTATE_COUNT)

    annotated_samples = []
    for i, sample in enumerate(samples):
        print(f"{i}/{ANNOTATE_COUNT}: {sample}")
        print(f"Choose type {list(enumerate(TYPES))}")
        chosen_type = None
        while chosen_type is None:
            try:
                chosen_type = int(input())
            except Exception:
                pass
        annotated_samples.append({
            'commentary': sample,
            'type': chosen_type,
            'type_str': TYPES[chosen_type]
        })

    pl.DataFrame(annotated_samples).write_parquet("../artifacts/commentary_types.parquet")

#TODO review last 3 categories
# I think a lot of context got put into strategy, and vice versa
# also, a lot of general and context are swapped
# context should contain summaries about the game, thats what makes it different. maybe openings too(definetly)
# Move quality and comparative are also mixed
# Comparative and strategy might also be mixed