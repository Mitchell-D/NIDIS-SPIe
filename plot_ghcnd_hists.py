import zarr
import numpy as np
import pickle as pkl
from pathlib import Path
from datetime import datetime

from plotting import plot_lines

if __name__=="__main__":
    zarr_store = Path("data/zarr/ghcnd.zarr")

    zgrp = zarr.open(zarr_store, mode="r")

    plot_var = "precip"
    hist_bounds = (0, 450)
    hist_res = 128

    for sid in zgrp["stations"].keys():
        x = zgrp[f"/stations/{sid}/{plot_var}"][...]
        x = x[np.isfinite(x)]
        x = x[x>0]
        print(x.size)
        h,bin_edges = np.histogram(x, bins=hist_res, range=hist_bounds)
        bins = (bin_edges[:-1] + bin_edges[1:]) / 2

        plot_lines(
            domain=bins,
            ylines=[h],
            plot_spec={
                "title":f"{sid} {plot_var}",
                "yscale":"log",
                },
            show=True,
            )


