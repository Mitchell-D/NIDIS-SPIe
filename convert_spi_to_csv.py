import numpy as np
import pandas as pd
import json
from pprint import pprint
from datetime import datetime
from pathlib import Path

if __name__=="__main__":
    spi_json_path = Path("data/crn_spi.json")
    data = json.load(spi_json_path.open("r"))
    csv_dir = Path("data/por-spi-csv")

    for sid in data.keys():
        date = [
            datetime.strptime(dstr, "%Y-%m-%d")
            for dstr in data[sid]["date"]
            ]
        dstr0 = date[0].strftime("%Y%m%d")
        dstrf = date[-1].strftime("%Y%m%d")
        out_csv_path = csv_dir.joinpath(f"spi-{sid}-{dstr0}-{dstrf}.csv")

        d = {
            "precip":data[sid]["precip"],
            "eff_precip":data[sid]["eff_precip"],
            "avg_precip":data[sid]["avg_precip"],
            }
        for w in data[sid]["spi"].keys():
            for fk in ["spi", "jng", "acc"]:
                d[f"{fk}_{w}"] = data[sid][fk][w]

        d["date"] = date
        df = pd.DataFrame(d)
        df.to_csv(out_csv_path, index=False, float_format="%.3f")
