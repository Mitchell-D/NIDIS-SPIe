import zarr
import numpy as np
import pickle as pkl
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
from pprint import pprint
import scipy.stats as spstats

from plotting import plot_bars_with_lines

if __name__=="__main__":
    zarr_store = Path("data/zarr/ghcnd.zarr")
    fig_dir = Path("figures/spi-err")

    sid = "USC00016161"
    window = 90
    doy_ix = 5

    zs = zarr.open(zarr_store, mode="r", path=f"/stations/{sid}")
    l_dist = zs.attrs["labels"]["dist"]
    l_window = zs.attrs["labels"]["window"]
    l_err = zs.attrs["labels"]["err"]

    wix = l_window.index(window)
    hist_x = zs["spi-hist-x"][...]
    hist_y = zs["spi-hist-y"][wix, doy_ix]
    dparams = dict(zip(l_dist, [
        [pv for pv in dparams if not np.isnan(pv)]
        for dparams in zs["spi-params"][:, wix, doy_ix].tolist()
        ]))
    pprint(dparams)

    pdfs = {}
    for dk,dp in dparams.items():
        dist = getattr(spstats, dk)
        pdfs[dk] = dist.pdf(hist_x, *dp)

    print(np.unique(hist_y))
    plot_bars_with_lines(
        domain=hist_x,
        bars=hist_y,
        lines=[pdfs[dk] for dk in l_dist],
        line_labels=l_dist,
        plot_spec={
            "title":f"Day {doy_ix+1} precip vs distributional forms " + \
                    f"({sid}, w{window})",
            "xlabel":f"{window}-day accumulated precip",
            "ylabel":"probability density",
            },
        show=True,
        )
