import zarr
import numpy as np
import pickle as pkl
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
import scipy.stats as spstats
#from fitter import Fitter

from plotting import plot_lines
from helpers import doy_window_slices,fit_and_check_dists

def mp_get_station_spi(args):
    return args,get_station_spi(**args)

def get_station_spi(
        zarr_store, station_id, distributions, windows, hist_bins, hist_bounds,
        start_day=None, end_day=None, nan_thresh=0.3,
        ):
    print()
    zs = zarr.open(zarr_store, path=f"/stations/{station_id}", mode="a")
    dates = zs["date"][...]
    dts = dates.astype(datetime).tolist()
    if start_day is None:
        start_day = dts[0]
        dix0 = 0
        dix0 = dts.index(start_day)
    if end_day is None:
        end_day = dts[-1]
        dixf = len(dts) - 1
    else:
        dixf = dts.index(end_day)

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
                #    station_id, dts[ix],
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

    err = np.empty((len(distributions), len(windows), 366, 3))
    params = np.empty((len(distributions), len(windows), 366, 3))
    ## final axis is (num samples, num zero precip samples)
    samples = np.empty((len(windows), 366, 2), dtype=np.uint16)
    hist_y = np.empty((len(windows), 366, hist_bins), dtype=np.uint16)
    hist_x = None
    for wix,wsize in enumerate(windows):
        for dix,slcs in enumerate(doy_window_slices(dates, wsize)):
            samples[wix, dix] = np.array([len(slcs), 0])
            pagg = []
            for s in slcs:
                x = precip[s]
                if np.count_nonzero(np.isnan(x))/x.size > nan_thresh:
                    continue
                xsum = np.nansum(x)
                if xsum == 0:
                    samples[wix, dix, 1] += 1
                else:
                    pagg.append(xsum)
            pagg = np.array(pagg)
            print(station_id, wix, dix, pagg.size, np.amin(pagg), np.amax(pagg))
            tmp_params,tmp_err,(hx,hy) = fit_and_check_dists(
                x=pagg,
                distributions=distributions,
                hist_bins=hist_bins,
                hist_bounds=hist_bounds,
                return_hist=True,
                scipy_fit_kwargs={"floc":0},
                nparams=2,
                )
            if hist_x is None:
                hist_x = hx
            hist_y[wix, dix] = hy
            tmp_params = np.array([
                p if len(p)==3 else [*p, np.nan]
                for p in tmp_params
                ])
            tmp_err = np.array(tmp_err).T
            #print(x.size, [[float(v) for v in p] for p in params])
            params[:, wix, dix, :] = np.asarray(tmp_params)
            err[:, wix, dix, :] = np.asarray(tmp_err)

    if "spi-params" in zs.keys():
        del zs["spi-samples"]
        del zs["spi-params"]
        del zs["spi-err"]
        del zs["spi-hist-x"]
        del zs["spi-hist-y"]
    zs.create_array("spi-samples", data=samples)
    zs.create_array("spi-params", data=params)
    zs.create_array("spi-err", data=err)
    zs.create_array("spi-hist-x", data=hist_x)
    zs.create_array("spi-hist-y", data=hist_y)
    zs.attrs.update({
        "labels":{
            "dist":distributions,
            "window":windows,
            "err":["sse", "ad", "aic"],
            "sample":["total", "zero"],
            }
        })
    print(f"Finished {station_id}")

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
    hist_bins = 128
    hist_bounds = (0, 1024)

    #start_time = datetime(2014, 1, 1, 0)
    #end_time = datetime(2014, 1, 31, 23)
    start_time = None
    end_time = None

    nworkers = 4

    """ ------------( end normal configuration )------------ """

    zgrp = zarr.open(zarr_store, mode="r")
    all_stations = list(zgrp["stations"].keys())
    if get_stations is None:
        get_stations = all_stations

    args = [{
        "zarr_store":zarr_store,
        "station_id":sid,
        "distributions":get_dists,
        "windows":agg_windows,
        "hist_bins":hist_bins,
        "hist_bounds":hist_bounds,
        "nan_thresh":daily_nan_thresh,
        } for sid in get_stations]
    with Pool(nworkers) as pool:
        for a,r in pool.imap_unordered(mp_get_station_spi, args):
            continue
    #'''
    print("finished")
