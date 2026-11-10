import zarr
import numpy as np
import pickle as pkl
from pathlib import Path
from datetime import datetime
import scipy.stats as spstats
#from fitter import Fitter

from plotting import plot_lines
from helpers import doy_window_masks,generate_daily,fit_and_check_dists

if __name__=="__main__":
    zarr_store = Path("data/zarr/crn_subhourly_2006-2026.zarr")

    get_stations = [
        "AL_Gadsden_19_N",
        "AL_Selma_13_WNW",
        "AL_Fairhope_3_NE",
        ]

    get_dists = [
        "gamma", "gumbel_l", "fisk", "lognorm",
        "logistic", "norm", "weibull_min",
        ]

    agg_windows = [30, 45, 60, 90]
    ## fraction of nan values allowed per day over which precip sum is ignored
    daily_nan_thresh = .3
    hist_bins = 64

    #start_time = datetime(2014, 1, 1, 0)
    #end_time = datetime(2014, 1, 31, 23)
    start_time = None
    end_time = None

    """ ------------( end normal configuration )------------ """

    err_labels = ["sse"]

    zgrp = zarr.open(zarr_store, mode="a")

    print(list(zgrp.keys()))
    if "spi_params" in zgrp.keys():
        del zgrp["spi_params"]
        del zgrp["spi_err"]
        del zgrp["err_label"]
        del zgrp["window_label"]
        del zgrp["dist_label"]
        #del zgrp["hist_precip"]
        #del zgrp["bins_precip"]
    zgrp.create_array(
        "spi_params",
        shape=(len(get_stations), len(get_dists), len(agg_windows), 366, 2),
        chunks=(1, 1, len(agg_windows), 366, 2),
        shards=(len(get_stations), len(get_dists), len(agg_windows), 366, 2),
        dimension_names=[
            "station_name", "dist_label", "window_label", "doy", "param2"
            ],
        dtype=np.float32,
        )
    zgrp.create_array(
        "spi_err",
        shape=(
            len(get_stations), len(get_dists), len(agg_windows), 366, 1
            ),
        chunks=(1, 1, len(agg_windows), 366, 2),
        shards=(len(get_stations), len(get_dists), len(agg_windows), 366, 2),
        dimension_names=[
            "station_name", "dist_label", "window_label", "doy", "err_label"
            ],
        dtype=np.float32,
        )

    zgrp.create_array(
        "err_label",
        data=np.array(err_labels, dtype="U16"),
        dimension_names=["err_label"],
        )

    zgrp.create_array(
        "window_label",
        dimension_names=["window_label"],
        data=np.array(agg_windows, dtype=np.uint16),
        )

    zgrp.create_array(
        "dist_label",
        dimension_names=["dist_label"],
        data=np.array(get_dists, dtype="U32"),
        )

    '''
    zgrp.create_array(
        "hist_precip",
        shape=(len(get_stations), len(agg_windows), 366, hist_bins),
        chunks=(1, 366, hist_bins),
        shards=(len(get_stations), len(agg_windows), 366, hist_bins),
        dimension_names=[
            "station_name", "window_label", "doy", "bins_precip"
            ],
        )

    zgrp.create_array(
        "bins_precip",
        shape=(hist_bins,),
        dimension_names=["bins_precip"],
        )
    '''

    zgrp.attrs.update({
        "labels":{
            **zgrp.attrs["labels"],
            "dist":get_dists,
            "window":agg_windows,
            "err":err_labels,
            }
        })

    zarr.consolidate_metadata(zarr_store)

    times = zgrp["time"][...].astype("M8[h]")
    dtimes = times.astype(datetime).tolist()

    if start_time is not None:
        tix0 = times.index(start_time)
    else:
        tix0 = 0
    if end_time is not None:
        tixf = times.index(end_time)
    else:
        tixf = len(times) - 1
    tdom = times[tix0:tixf]

    lbl = zgrp.attrs["labels"]
    fix = lbl["variables"].index("precip")
    mix_mean = lbl["stats"].index("mean")
    mix_count = lbl["stats"].index("count")

    for six,sk in enumerate(get_stations):
        six = lbl["stations"].index(sk)

        ## extract precip and convert average to sum
        precip_hourly = zgrp["obs"][six, fix, tix0:tixf]
        precip_hourly = precip_hourly[:,mix_mean] * precip_hourly[:,mix_count]

        ## convert to daily precip
        precip,dates = [],[]
        for d,vs in generate_daily(times, precip_hourly):
            dates.append(d)
            nan_frac = np.count_nonzero(np.isnan(vs)) / vs.size
            if nan_frac > daily_nan_thresh:
                precip.append(np.nan)
            else:
                precip.append(np.nansum(vs))
        dates = np.asarray(dates)
        precip = np.asarray(precip)

        tmp_err = np.empty((len(get_dists), len(agg_windows), 366, 1))
        tmp_params = np.empty((len(get_dists), len(agg_windows), 366, 2))
        #tmp_hist = np.empty((len(agg_windows), 366, hist_bins))

        for wix,wsize in enumerate(agg_windows):
            for dix,m_doy in enumerate(doy_window_masks(dates, wsize)):
                #params,sses,(hx,hy) = fit_and_check_dists(
                x = precip[m_doy]
                x = x[np.isfinite(x)]
                x = x[x>0]
                print(x.size)
                params,err = fit_and_check_dists(
                    x=x,
                    distributions=get_dists,
                    hist_bins=hist_bins,
                    #return_hist=True,
                    scipy_fit_kwargs={"floc":0},
                    )
                print(params)
                tmp_params[dix, wix, :, :] = np.asarray(params)
                tmp_err[dix, wix, :, :] = np.asarray(err)

        zgrp["spi_params"][six] = tmp_params
        zgrp["spi_err"][six] = tmp_err

        exit(0)

        '''
        plot_lines(
            domain=tdom,
            ylines=[x],
            fig_path=None,
            plot_spec={
                "title":f"{sk} {plot_variable}",
                },
            show=True,
            )
        '''
