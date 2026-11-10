import zarr
import numpy as np
import pickle as pkl
from pathlib import Path
from datetime import datetime

from plotting import plot_lines

if __name__=="__main__":
    zarr_store = Path("data/zarr/crn_hourly_por.zarr")
    fig_dir = Path("figures/subhourly")

    plot_variable = "swdown-avg"
    plot_stations = [
        "AL_Gadsden_19_N",
        "AL_Selma_13_WNW",
        "AL_Fairhope_3_NE",
        ]

    #start_time = datetime(2014, 1, 1, 0)
    #end_time = datetime(2014, 1, 31, 23)
    start_time = None
    end_time = None

    """ ------------( end normal configuration )------------ """

    zgrp = zarr.open(zarr_store, mode="r")

    lbl = zgrp.attrs["labels"]
    times = zgrp["time"][...].astype("M8[h]").astype(datetime).tolist()

    if start_time is not None:
        tix0 = times.index(start_time)
    else:
        tix0 = 0
    if end_time is not None:
        tixf = times.index(end_time)
    else:
        tixf = len(times) - 1
    fix = lbl["variables"].index(plot_variable)

    tdom = times[tix0:tixf]

    for sk in plot_stations:
        six = lbl["stations"].index(sk)
        x = zgrp["obs"][six, fix, tix0:tixf]

        plot_lines(
            domain=tdom,
            ylines=[x],
            fig_path=None,
            plot_spec={
                "title":f"{sk} {plot_variable}",
                },
            show=True,
            )
