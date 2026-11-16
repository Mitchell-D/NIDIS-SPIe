import zarr
import json
import numpy as np
import pickle as pkl
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
import scipy.stats as spstats
#from fitter import Fitter

from plotting import plot_lines
from helpers import doy_window_slices,fit_and_check_dists,effect

def mp_get_station_spi(args):
    return args,get_station_spi(**args)

def get_station_spi(
        zarr_store, station_id, distributions, windows, hist_bins,
        hist_bounds_spi, hist_bounds_jng,
        start_day=None, end_day=None, nan_thresh=0.3,
        ):
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
    #precip_flags = zs["precip-attr"][...]
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

    eprecip = effect(precip / 25.4) * 25.4

    doy_avg = np.full((366,), np.nan)

    ## final axis is (num samples, num zero precip samples)
    samples_spi = np.empty((len(windows), 366, 2), dtype=np.uint16)
    err_spi = np.empty((len(distributions), len(windows), 366, 3))
    params_spi = np.empty((len(distributions), len(windows), 366, 3))
    hist_y_spi = np.empty((len(windows), 366, hist_bins), dtype=np.float32)
    hist_x_spi = None

    samples_jng = np.empty((len(windows), 366, 2), dtype=np.uint16)
    err_jng = np.empty((len(distributions), len(windows), 366, 3))
    params_jng = np.empty((len(distributions), len(windows), 366, 3))
    hist_y_jng = np.empty((len(windows), 366, hist_bins), dtype=np.float32)
    hist_x_jng = None
    for wix,wsize in enumerate(windows):
        for dix,slcs in enumerate(doy_window_slices(dates, wsize)):
            ## normal precip fitting
            samples_spi[wix, dix] = np.array([len(slcs), 0])
            pagg = []
            pavg = []
            for s in slcs:
                x = precip[s]
                if np.isfinite(x[-1]):
                    pavg.append(x[-1])
                if np.count_nonzero(np.isnan(x))/x.size > nan_thresh:
                    continue
                xsum = np.nansum(x)
                if xsum == 0:
                    samples_spi[wix, dix, 1] += 1
                else:
                    pagg.append(xsum)
            pagg = np.array(pagg)

            doy_avg[dix] = np.average(pavg)

            #print(station_id,wix,dix,pagg.size,np.amin(pagg),np.amax(pagg))
            tmp_params_spi,tmp_err_spi,(hx_spi,hy_spi) = fit_and_check_dists(
                x=pagg,
                distributions=distributions,
                hist_bins=hist_bins,
                hist_bounds=hist_bounds_spi,
                return_hist=True,
                scipy_fit_kwargs={"floc":0},
                nparams=2,
                )
            if hist_x_spi is None:
                hist_x_spi = hx_spi
            #print(np.unique(hy))
            hist_y_spi[wix, dix] = hy_spi
            tmp_params_spi = np.array([
                p if len(p)==3 else [*p, np.nan]
                for p in tmp_params_spi
                ])
            tmp_err_spi = np.array(tmp_err_spi).T
            params_spi[:, wix, dix, :] = np.asarray(tmp_params_spi)
            err_spi[:, wix, dix, :] = np.asarray(tmp_err_spi)

            ## effective precip fitting
            samples_jng[wix, dix] = np.array([len(slcs), 0])
            eagg = []
            for s in slcs:
                e = eprecip[s]
                if np.count_nonzero(np.isnan(e))/e.size > nan_thresh:
                    continue
                esum = np.nansum(e)
                if esum <= 0:
                    samples_jng[wix, dix, 1] += 1
                else:
                    eagg.append(esum)
            eagg = np.array(eagg)

            tmp_params_jng,tmp_err_jng,(hx_jng,hy_jng) = fit_and_check_dists(
                x=eagg,
                distributions=distributions,
                hist_bins=hist_bins,
                hist_bounds=hist_bounds_jng,
                return_hist=True,
                scipy_fit_kwargs={"floc":0},
                nparams=2,
                )
            if hist_x_jng is None:
                hist_x_jng = hx_jng
            #print(np.unique(hy))
            hist_y_jng[wix, dix] = hy_jng
            tmp_params_jng = np.array([
                p if len(p)==3 else [*p, np.nan]
                for p in tmp_params_jng
                ])
            tmp_err_jng = np.array(tmp_err_jng).T
            params_jng[:, wix, dix, :] = np.asarray(tmp_params_jng)
            err_jng[:, wix, dix, :] = np.asarray(tmp_err_jng)

    if "spi-params" in zs.keys():
        del zs["spi-samples"]
        del zs["spi-params"]
        del zs["spi-err"]
        del zs["spi-hist-x"]
        del zs["spi-hist-y"]

        del zs["jng-samples"]
        del zs["jng-params"]
        del zs["jng-err"]
        del zs["jng-hist-x"]
        del zs["jng-hist-y"]

        del zs["precip-avg"]

    zs.create_array("spi-samples", data=samples_spi)
    zs.create_array("spi-params", data=params_spi)
    zs.create_array("spi-err", data=err_spi)
    zs.create_array("spi-hist-x", data=hist_x_spi)
    zs.create_array("spi-hist-y", data=hist_y_spi)

    zs.create_array("jng-samples", data=samples_jng)
    zs.create_array("jng-params", data=params_jng)
    zs.create_array("jng-err", data=err_jng)
    zs.create_array("jng-hist-x", data=hist_x_jng)
    zs.create_array("jng-hist-y", data=hist_y_jng)

    zs.create_array("precip-avg", data=doy_avg)
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
    crn_ghcn_mapping_path = Path("data/crn_ghcn_station_mapping.json")

    #get_stations = ["AL_Gadsden_19_N", "AL_Selma_13_WNW", "AL_Fairhope_3_NE"]
    get_stations = None
    #get_stations = ["AL_Fairhope_3_NE", "GA_Newton_8_W", "GA_Newton_11_SW"]

    get_dists = [
        "gamma", "gumbel_r", "fisk", "lognorm",
        "logistic", "norm", "weibull_min",
        ]

    agg_windows = [15, 30, 45, 60, 90]
    ## fraction of nan values allowed per day over which precip sum is ignored
    daily_nan_thresh = .2
    hist_bins = 48
    hist_bounds_spi = (0, 768)
    hist_bounds_jng = (0, 448)

    #start_time = datetime(2014, 1, 1, 0)
    #end_time = datetime(2014, 1, 31, 23)
    start_time = None
    end_time = None

    nworkers = 6

    """ ------------( end normal configuration )------------ """

    zgrp = zarr.open(zarr_store, mode="r")
    all_stations = list(zgrp["stations"].keys())
    ghcn_stations = []
    if get_stations is None:
        ghcn_stations = all_stations
    else:
        cgsm = json.load(crn_ghcn_mapping_path.open("r"))
        for crn_key in get_stations:
            ghcn_stations += [v["ghcn_id"] for v in cgsm[crn_key]]
    ghcn_stations = list(set(ghcn_stations))

    args = [{
        "zarr_store":zarr_store,
        "station_id":sid,
        "distributions":get_dists,
        "windows":agg_windows,
        "hist_bins":hist_bins,
        "hist_bounds_spi":hist_bounds_spi,
        "hist_bounds_jng":hist_bounds_jng,
        "nan_thresh":daily_nan_thresh,
        } for sid in ghcn_stations]
    with Pool(nworkers) as pool:
        for a,r in pool.imap_unordered(mp_get_station_spi, args):
            continue
    #'''
    print("finished")
