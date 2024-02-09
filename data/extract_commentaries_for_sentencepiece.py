import os
import polars as pl

if __name__ == '__main__':
    with open("../artifacts/commentaries.txt", "w") as f:
        for filename in os.listdir(os.path.join("../raw_data", "train")):
            try:
                local_data = pl.read_parquet(os.path.join("../raw_data", "train", filename)).rows()
                for _,_,comm in local_data:
                    if len(comm.strip()) > 0:
                        f.write(comm.strip().replace("\n", "<n>") + "\n")
            except Exception as e:
                pass
