import os
import polars as pl

def extract(artifacts_path: str, raw_data_path: str):
    with open(os.path.join(artifacts_path, "commentaries_raw.txt"), "w") as f:
        for filename in os.listdir(os.path.join(raw_data_path, "train")):
            try:
                local_data = pl.read_parquet(os.path.join(raw_data_path, "train", filename)).rows()
                for _,_,comm in local_data:
                    if len(comm.strip()) > 0:
                        f.write(comm.strip().replace("\n", "<n>") + "\n")
            except Exception as e:
                pass

