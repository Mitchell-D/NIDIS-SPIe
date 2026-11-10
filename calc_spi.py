import zarr
import numpy as np
import pickle as pkl
from pathlib import Path
from datetime import datetime
import scipy.stats as spstats
#from fitter import Fitter

from plotting import plot_lines
from helpers import doy_window_masks,generate_daily,fit_and_check_dists

def get_station_spi(zarr_store, station_id, distributions, windows,
        start_day=None, end_day=None):
    zs = zarr.open(zarr_store, path=f"/stations/{station_id}", mode="a")
    dates = zs["date"][...].astype(datetime).tolist()
    if start_day is None:
        start_day = dates[0]
        dix0 = 0
        dix0 = dates.index(start_day)
    if end_day is None:
        end_day = dates[-1]
        dixf = len(dates) - 1
    else:
        dixf = dates.index(end_day)

    precip = zs["precip"][...]
    precip_flags = zs["precip-attr"][...]
    if "mdpr" in zs.keys():
        mdpr = zs["mdpr"][...]
        mdpr_flags = zs["mdpr"][...]
        dapr = zs["dapr"][...]
        dapr_flags = zs["dapr"][...]
        ix_md = np.where(np.isfinite(mdpr) & (dapr > 0))[0]
        assert np.all(np.isfinite(mdpr[ix_md]))
        for ix in ix_md:
            slc = slice(ix-int(dapr[ix])+1, ix+1)
            ## deal with some observations being present in the window
            if not np.all(np.isnan(precip[slc])):
                #print(
                #    "WARNING: obs overlapped by mdpr; ignoring",
                #    f"{dapr[ix]} days with {round(mdpr[ix])} mm precip,",
                #    f"falling back on {round(np.nansum(precip[slc]))} mm",
                #    station_id, dates[ix],
                #    )

                ## sometimes the multi day total is reported in the normal
                ## precip array on the final day
                if precip[ix] == mdpr[ix]:
                    precip[slc] = np.full((int(dapr[ix]),), mdpr[ix]/dapr[ix])
                ## otherwise distribute the total over NaN reports
                else:
                    inset_nans = np.isnan(precip[slc])
                    inset = np.where(
                        inset_nans,
                        mdpr[ix] / np.count_nonzero(inset_nans),
                        precip[slc],
                        )
                    precip[slc] = inset
            ## otherwise distribute evenly within the window
            else:
                precip[slc] = np.full((int(dapr[ix]),), mdpr[ix]/dapr[ix])
    return

    if "spi_params" in zs.keys():
        del zs["spi_params"]
        del zs["spi_err"]
    zs.create_array(
        "spi_params",
        shape=(len(distributions), len(windows), 366, 2),
        chunks=(1, len(windows), 366, 2),
        shards=(len(distributions), len(windows), 366, 2),
        dtype=np.float32,
        )
    zs.create_array(
        "spi_err",
        shape=(len(distributions), len(windows), 366, 1),
        chunks=(1, len(windows), 366, 2),
        shards=(len(distributions), len(windows), 366, 2),
        dtype=np.float32,
        )
    zs.attrs.update({
        "dist":distributions,
        "window":windows,
        "err":["sse"],
        })
    tmp_err = np.empty((len(distributions), len(windows), 366, 1))
    tmp_params = np.empty((len(distributions), len(windows), 366, 2))

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

if __name__=="__main__":
    #zarr_store = Path("data/zarr/crn_subhourly_2006-2026.zarr")
    zarr_store = Path("data/zarr/ghcnd.zarr")

    #get_stations = ["AL_Gadsden_19_N", "AL_Selma_13_WNW", "AL_Fairhope_3_NE"]
    get_stations = None

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

    zgrp = zarr.open(zarr_store, mode="r")
    all_stations = list(zgrp["stations"].keys())
    if get_stations is None:
        get_stations = all_stations

    for sid in get_stations:
        get_station_spi(
            zarr_store=zarr_store,
            station_id=sid,
            distributions=get_dists,
            windows=agg_windows,
            )

    #'''
    exit(0)
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
    #'''

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
