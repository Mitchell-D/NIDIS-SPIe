import zarr
import numpy as np
import json
from pathlib import Path
from datetime import datetime,timedelta,date
from multiprocessing import Pool
from pprint import pprint
import scipy.stats as spstats
from numpy.lib.stride_tricks import sliding_window_view

from helpers import generate_daily,effect

def calc_crn_spi(
        zarr_store_crn, zarr_store_ghcn, crn_ghcn_mapping,
        window_dists_spi, window_dists_jng,
        start_day=None, end_day=None, nan_thresh=0.1, daily_nan_thresh=0.3,
        ):
    """
    :@param zarr_store_crn:
    :@param zarr_store_ghcn:
    :@param crn_ghcn_mapping: dict mapping crn station ids to the ghcn station
        ids used to calculate their distributions.
    :@param window_dists_spi: dict mapping int window sizes to dist forms for
        normal SPI
    :@param window_dists_jng: dict mapping int window sizes to dist forms for
        John Nielson-Gammon's effective precip estimate
    :@param start_day:
    :@param end_day:
    :@param nan_thresh: fraction of NaN crn values to allow to calculate sum
    :@param daily_nan_thresh: fraction of hourly NaN crn values per day allowed
        when aggregating to daily
    """
    assert set(window_dists_spi.keys()) == set(window_dists_jng.keys())
    max_window_days = max(window_dists_spi.keys())
    zcrn = zarr.open(zarr_store_crn, mode="r")
    zghcn = zarr.open(zarr_store_ghcn, mode="r")
    lc_station = zcrn.attrs["labels"]["stations"]
    lc_var = zcrn.attrs["labels"]["variables"]
    lc_stats = zcrn.attrs["labels"]["stats"]
    crn_times = zcrn["time"][...].astype("M8[h]")
    crn_dtimes = crn_times.astype(datetime).tolist()
    crn_dates = zcrn["time"][...].astype("M8[D]").astype(datetime).tolist()

    ## establish start and end times and indices wrt crn data available
    if start_day is not None:
        start_window = start_day - timedelta(days=max_window_days)
    else:
        start_window = crn_dates[0]
        start_day = start_window + timedelta(days=max_window_days)
    assert crn_dates[0] <= start_window <= crn_dates[-1]

    if end_day is None:
        end_day = crn_dates[-1]

    ## collect SPI data for each listed station combination
    spi = {}
    for sid_c,sid_g in crn_ghcn_mapping.items():
        spi[sid_c] = {
            "date":None,
            "precip":None,
            "eff_precip":None,
            "avg_precip":None,
            "spi":{w:[] for w in window_dists_spi.keys()},
            "jng":{w:[] for w in window_dists_spi.keys()},
            "acc":{w:[] for w in window_dists_spi.keys()},
            }
        six = lc_station.index(sid_c)
        vix = lc_var.index("precip")
        mix_mean = lc_stats.index("mean")
        mix_count = lc_stats.index("count")
        crn_precip_hourly = zcrn["obs"][six, vix]
        ## convert mean to total with count
        crn_precip_hourly = crn_precip_hourly[:, mix_mean] \
                * crn_precip_hourly[:, mix_count] \

        ## aggregate to daily precipitation, setting days with more nans than
        ## the threshold to nan
        precip,dates = [],[]
        for d,vs in generate_daily(crn_times, crn_precip_hourly):
            if d < start_window:
                continue
            if d > end_day:
                break
            dates.append(d)
            nan_frac = np.count_nonzero(np.isnan(vs) / vs.size)
            if nan_frac > daily_nan_thresh:
                precip.append(np.nan)
            else:
                precip.append(np.nansum(vs))

        ## for each window size, calculate the accumulated precip from crn
        acc_precip = {}
        acc_eprecip = {}
        dix0 = dates.index(start_day)
        dixf = dates.index(end_day)
        dates = np.asarray(dates[dix0:dixf+1])
        precip = np.asarray(precip)
        eprecip = effect(precip / 25.4) * 25.4
        eprecip[eprecip<0] = 0
        for wsize,_ in window_dists_spi.items():
            ## precip
            pwindow = sliding_window_view(precip[dix0-wsize+1:dixf+1], wsize)
            m_invalid = np.count_nonzero(np.isnan(pwindow), axis=1)
            m_invalid = (m_invalid / pwindow.shape[1]) > nan_thresh
            psum = np.where(m_invalid, np.nan, np.nansum(pwindow,axis=1))
            acc_precip[wsize] = psum

            ## jng effective precip
            epwindow = sliding_window_view(eprecip[dix0-wsize+1:dixf+1], wsize)
            m_invalid = np.count_nonzero(np.isnan(epwindow), axis=1)
            m_invalid = (m_invalid / epwindow.shape[1]) > nan_thresh
            epsum = np.where(m_invalid, np.nan, np.nansum(epwindow,axis=1))
            acc_eprecip[wsize] = epsum

        zgs = zghcn[f"/stations/{sid_g}"]
        lg_window = zgs.attrs["labels"]["window"]
        lg_dist = zgs.attrs["labels"]["dist"]
        dist_params_spi = zgs["spi-params"][...]
        dist_params_jng = zgs["jng-params"][...]
        precip_doy_avg = zgs["precip-avg"][...]

        ## for now, don't worry about zero-accumulation scaling (stagge, 2015)
        #spi_samples = zgs["spi-samples"][...]
        #print(np.unique(spi_samples[0,:,0], return_counts=True))
        #print(np.unique(spi_samples[0,:,1], return_counts=True))

        ## add precip and effective precip to the dict
        spi[sid_c]["date"] = [
                d.strftime("%Y-%m-%d")
                for d in dates.astype("M8[D]").astype(date)
                ]
        spi[sid_c]["precip"] = precip[dix0:dixf+1].tolist()
        spi[sid_c]["eff_precip"] = eprecip[dix0:dixf+1].tolist()

        ## for each window, calculate z score wrt ghcn distributions
        doy_ixs = (dates - dates.astype("M8[Y]")).astype(int)
        spi[sid_c]["avg_precip"] = np.take(precip_doy_avg, doy_ixs).tolist()
        for wsize in window_dists_spi.keys():
            wdist_spi = window_dists_spi[wsize]
            wdist_jng = window_dists_jng[wsize]
            dix_spi = lg_dist.index(wdist_spi)
            dix_jng = lg_dist.index(wdist_jng)
            wix = lg_window.index(wsize)
            doy_dists_spi = [
                getattr(spstats, wdist_spi)(*[
                    v for v in doy_params if not np.isnan(v)
                    ])
                for doy_params in dist_params_spi[dix_spi, wix].tolist()
                ]
            doy_dists_jng = [
                getattr(spstats, wdist_jng)(*[
                    v for v in doy_params if not np.isnan(v)
                    ])
                for doy_params in dist_params_jng[dix_jng, wix].tolist()
                ]
            psum = acc_precip[wsize]
            epsum = acc_eprecip[wsize]

            ## stand-in until probability mass shifting is implemented
            psum[psum <= 0] = 0.001
            epsum[epsum <= 0] = 0.001

            for i,doy in enumerate(doy_ixs):
                dist_spi = doy_dists_spi[doy]
                z_spi = spstats.norm.ppf(dist_spi.cdf(psum[i]))
                dist_jng = doy_dists_jng[doy]
                z_jng = spstats.norm.ppf(dist_jng.cdf(epsum[i]))
                spi[sid_c]["spi"][wsize].append(float(z_spi))
                spi[sid_c]["jng"][wsize].append(float(z_jng))
                spi[sid_c]["acc"][wsize].append(float(psum[i]))
    return spi

if __name__=="__main__":
    zarr_store_crn = Path("data/zarr/crn_subhourly_2006-2026.zarr")
    zarr_store_ghcn = Path("data/zarr/ghcnd.zarr")
    spi_json_path = Path("data/crn_spi.json")

    ## map crn stations to the ghcnd stations to get precip distributions from
    crn_ghcn_stations = {
        "AL_Fairhope_3_NE":"USC00012813",
        "GA_Newton_8_W":"USC00090979",
        "GA_Newton_11_SW":"USC00090979",
        }

    ## map window sizes to distributional forms to use
    window_dists_spi = {
        15:"weibull_min",
        30:"weibull_min",
        45:"gamma",
        60:"gamma",
        90:"gamma",
        }
    window_dists_jng = {
        15:"weibull_min",
        30:"weibull_min",
        45:"weibull_min",
        60:"weibull_min",
        90:"fisk",
        }

    ## fraction of NaN crn values to allow to calculate precip sum
    nan_thresh = 0.1
    ## fraction of hourly NaN crn values per day allowed when agg to daily
    daily_nan_thresh = 0.3

    #start_day = date(2024, 1, 1)
    #end_day = date(2024, 12, 31)
    start_day = None
    end_day = None

    """ ------------( end normal configuration )------------ """

    spi = calc_crn_spi(
        zarr_store_crn=zarr_store_crn,
        zarr_store_ghcn=zarr_store_ghcn,
        crn_ghcn_mapping=crn_ghcn_stations,
        window_dists_spi=window_dists_spi,
        window_dists_jng=window_dists_jng,
        start_day=start_day,
        end_day=end_day,
        nan_thresh=nan_thresh,
        daily_nan_thresh=daily_nan_thresh,
        )
    json.dump(spi, spi_json_path.open("w"))
