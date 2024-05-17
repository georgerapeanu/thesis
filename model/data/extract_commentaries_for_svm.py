import os
import polars as pl
import polars.exceptions


def extract(artifacts_path: str, raw_data_path: str):
    with open(os.path.join(artifacts_path, "commentaries_raw.txt"), "w") as f:
        for filename in os.listdir(os.path.join(raw_data_path, "train")):
            try:
                local_data = pl.read_parquet(os.path.join(raw_data_path, "train", filename)).rows()
                for _,_,comm in local_data:
                    if len(comm.strip()) > 0:
                        f.write(comm.strip().replace("\n", "<n>") + "\n")
            except polars.exceptions.ComputeError as e:
                pass

if __name__ == "__main__":
    extract(raw_data_path="../raw_data", artifacts_path="../artifacts")