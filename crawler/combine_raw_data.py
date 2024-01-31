import polars as pl
import os

SPLITS = [
    "test",
    "train",
    "valid"
]

if __name__ == "__main__":
    for split in SPLITS:
        df = None
        for filename in os.listdir("../raw_data/"+split+"/"):
            try:
                local_df = pl.read_parquet(f"../raw_data/{split}/" + filename)
                if df is None:
                    df = local_df
                else:
                    df = pl.concat([df,local_df])
            except Exception as e:
                print(f"game {split}/{filename} didn't have any annotations, skipping")
        if df is not None:
            df.write_parquet("../combined_raw_data/"+split+".parquet")
