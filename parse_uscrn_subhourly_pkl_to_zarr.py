import zarr
import pickle as pkl
import numpy as np
from datetime import datetime,timedelta
from pathlib import Path
from multiprocessing import Pool
from pprint import pprint

## vectorized functions to map character fields to integers
char_mapping = {
    "stemp-type":np.vectorize({"R":1, "C":2, "U":3}.get),
    }

def mp_agg_to_hour(args):
    return args,agg_to_hour(**args)

'''
def agg_to_hour(
        start_time, ntimes, measurement_times, data_array, dtype, fill, **_
        ):
    """
    Given an array of sub-hourly observations and observation times,
    """
    out_arr = np.full((ntimes,), fill, dtype=dtype)
    nan_fill = np.isnan(fill)
    for i,mt in enumerate(measurement_times):
        tix = int(round((mt - start_time).total_seconds() / 60 / 60))
        if (nan_fill and not np.isnan(out_arr[tix])) \
                or (not nan_fill and out_arr[tix] != fill):
            print("err", mt, tix, out_arr[tix], fill, out_arr[tix] == fill)
            raise ValueError(f"Data collision at {mt}! overwriting")
        out_arr[tix] = data_array[i]
    return out_arr
'''

def agg_to_hour(
    data,
    observation_times,
    output_times,
    nan_fraction_threshold=0.5,
    quality_flags=None,
    valid_quality_flag=None,
    **_,
    ):
    """
    Aggregate sub-hourly observations into hourly statistics.

    Parameters
    ----------
    data : array_like, shape (N,)
        Float observational data. NaN values are permitted.

    observation_times : array_like, shape (N,)
        M8 datetime64 observation times, one per observation.

    output_times : array_like, shape (T,)
        Regular hourly datetime64 output times. Each output time
        represents the start of an hourly aggregation interval.

    nan_fraction_threshold : float, default=0.5
        Maximum permitted fraction of missing or invalid observations.
        If the fraction exceeds this threshold, all statistics except
        the valid observation count are set to NaN.

    quality_flags : array_like, shape (N,), optional
        Quality flags corresponding to the observations.

    valid_quality_flag : int or float, optional
        Quality flag value indicating valid observations. Required
        when quality_flags is provided.

    Returns
    -------
    result : ndarray, shape (T, 5)
        Columns:
            0: Minimum
            1: Maximum
            2: Mean
            3: Standard deviation (population, ddof=0)
            4: Number of valid observations

        The first four columns are NaN when the missing fraction
        exceeds nan_fraction_threshold.

    Notes
    -----
    Hourly intervals are [output_time, output_time + 1 hour).
    Observations outside the requested output intervals are ignored.
    """

    data = np.asarray(data, dtype=float)
    observation_times = np.asarray(observation_times, dtype="datetime64[ns]")
    output_times = np.asarray(output_times, dtype="datetime64[ns]")

    if data.ndim != 1 or observation_times.ndim != 1:
        raise ValueError("data and observation_times must be 1D.")

    if len(data) != len(observation_times):
        raise ValueError(
            "data and observation_times must have equal lengths."
            )

    if output_times.ndim != 1:
        raise ValueError("output_times must be 1D.")

    if not 0 <= nan_fraction_threshold <= 1:
        raise ValueError(
            "nan_fraction_threshold must be between 0 and 1."
            )

    if len(output_times) > 1:
        differences = np.diff(output_times)
        if not np.all(differences == np.timedelta64(1, "h")):
            raise ValueError("output_times must have hourly spacing.")

    n_hours = len(output_times)

    # Initialize output: min, max, mean, std, count
    result = np.full((n_hours, 5), np.nan, dtype=float)

    if n_hours == 0 or len(data) == 0:
        return result

    # Construct the validity mask.
    valid = np.isfinite(data)

    if quality_flags is not None:
        if valid_quality_flag is None:
            raise ValueError(
                "valid_quality_flag is required when quality_flags "
                "are provided."
                )

        quality_flags = np.asarray(quality_flags)

        if quality_flags.shape != data.shape:
            raise ValueError(
                "quality_flags must have the same shape as data."
                )

        valid &= quality_flags == valid_quality_flag

    # Calculate hourly bin indices.
    # Each bin covers [output_time, output_time + 1 hour).
    hour_ns = np.timedelta64(1, "h").astype("timedelta64[ns]").astype(np.int64)

    output_ns = output_times.astype(np.int64)
    observation_ns = observation_times.astype(np.int64)

    # Identify observations belonging to the requested time range.
    start_ns = output_ns[0]
    end_ns = output_ns[-1] + hour_ns

    in_range = ((observation_ns >= start_ns) & (observation_ns < end_ns))

    # Calculate bin index using integer arithmetic.
    bin_indices = (
        (observation_ns[in_range] - start_ns) // hour_ns
        ).astype(np.int64)

    selected_data = data[in_range]
    selected_valid = valid[in_range]

    # Number of observations in each hour, including invalid values.
    total_counts = np.bincount(
        bin_indices, minlength=n_hours
        )

    # Select only valid observations.
    valid_bins = bin_indices[selected_valid]
    valid_data = selected_data[selected_valid]

    # Count valid observations per hour.
    counts = np.bincount(
        valid_bins, minlength=n_hours
        )

    result[:, 4] = counts

    # Calculate fraction of missing/invalid observations.
    missing_fraction = np.ones(n_hours, dtype=float)

    np.divide(
        total_counts - counts,
        total_counts,
        out=missing_fraction,
        where=total_counts > 0,
        )

    # Identify hours that satisfy the missing-data threshold.
    acceptable = (
        (total_counts > 0) &
        (missing_fraction <= nan_fraction_threshold) &
        (counts > 0)
        )

    if not np.any(acceptable):
        return result

    # Calculate minimum and maximum.
    hourly_min = np.full(n_hours, np.inf)
    hourly_max = np.full(n_hours, -np.inf)

    np.minimum.at(hourly_min, valid_bins, valid_data)
    np.maximum.at(hourly_max, valid_bins, valid_data)

    # Calculate mean and population standard deviation.
    sums = np.bincount(
        valid_bins,
        weights=valid_data,
        minlength=n_hours,
        )

    means = np.full(n_hours, np.nan)

    np.divide(
        sums,
        counts,
        out=means,
        where=counts > 0,
        )

    squared_deviations = (valid_data - means[valid_bins]) ** 2

    sum_squared_deviations = np.bincount(
        valid_bins,
        weights=squared_deviations,
        minlength=n_hours,
        )

    std = np.full(n_hours, np.nan)

    np.sqrt(
        np.divide(
            sum_squared_deviations,
            counts,
            out=np.full(n_hours, np.nan),
            where=counts > 0,
            ),
        out=std,
        )

    # Populate statistics only for acceptable hours.
    result[acceptable, 0] = hourly_min[acceptable]
    result[acceptable, 1] = hourly_max[acceptable]
    result[acceptable, 2] = means[acceptable]
    result[acceptable, 3] = std[acceptable]

    return result

if __name__=="__main__":
    data_dir = Path("/rstor/mdodson/nidis_spie")
    pkl_dir = data_dir.joinpath("pkls/subhourly")
    zarr_store = data_dir.joinpath("zarr/crn_subhourly_2006-2026.zarr")

    start_time = datetime(2005, 4, 1, 0, 0)
    end_time = datetime(2026, 10, 1, 23, 0)
    nworkers = 20
    time_var = "time-utc" ## can also be "time-local"

    time_chunk_size = 24 * 64
    time_shard_size = time_chunk_size * 12
    station_chunk_size = 4
    station_shard_size = station_chunk_size * 6
    var_chunk_size = 4
    var_shard_size = var_chunk_size * 8
    flag_chunk_size = 4
    flag_shard_size = flag_chunk_size * 8

    keep_float_fields = [
        "temp", "precip", "swdown", "stemp", "rh", "sm-2in", "st-2in",
        "wetn", "wspd",
        ]

    flag_fields = {
        "swdown":"flag-swdown",
        "stemp":"flag-stemp",
        "rh":"flag-rh",
        "wetn":"flag-wetn",
        "wspd":"flag-wspd",
        "stemp":"flag-stemp",
        }

    nan_frac_thresh = 0.3

    get_stations = None

    """ ------------( end normal configuration )------------ """

    assert not zarr_store.exists() and zarr_store.parent.exists()

    ## truncate hours and get the full timeline
    start_time = datetime(
        start_time.year,
        start_time.month,
        day=start_time.day,
        hour=start_time.hour,
        )
    end_time = datetime(
        end_time.year,
        end_time.month,
        end_time.day,
        end_time.hour,
        )
    all_hours = []
    ct = start_time
    while ct <= end_time:
        all_hours.append(ct)
        ct += timedelta(hours=1)

    stat_labels = ["min", "max", "mean", "stddev", "count"]

    args = []
    station_coords = {}
    stations = []
    for p in pkl_dir.iterdir():
        sk,_,_ = p.stem.split("-")
        if get_stations is not None and sk not in get_stations:
            continue

        stations.append(sk)
        six = len(stations) - 1
        pdata = pkl.load(p.open("rb"))

        ## most stations should be consistent, but some vary by a bit likely
        ## due to rounding errors. arbitrarily choose to keep the min coord
        station_coords[sk] = (
            float(np.amin(pdata["lat"])),
            float(np.amin(pdata["lon"])),
            )
        for fix,fk in enumerate(keep_float_fields):
            #six = stations.index(sid)
            #fix = keep_float_fields.index(fk)
            args.append({
                "data":pdata[fk],
                "observation_times":pdata[time_var],
                "output_times":all_hours,
                "nan_fraction_threshold":nan_frac_thresh,
                "quality_flags":pdata.get(flag_fields.get(fk)),
                "valid_quality_flag":0,

                "fix":fix,
                "six":six,
                })

    sarr = np.full(
        (len(stations), len(keep_float_fields), len(all_hours), 5),
        np.nan,
        dtype=np.float32,
        )

    with Pool(nworkers) as pool:
        for a,r in pool.imap_unordered(mp_agg_to_hour, args):
            sarr[a["six"], a["fix"]] = r

    zgrp = zarr.open(zarr_store, mode="w")
    comp = zarr.codecs.BloscCodec(
        typesize=4,
        cname="zstd",
        clevel=4,
        shuffle="bitshuffle",
        blocksize=0,
        )
    zgrp.create_array(
        "time",
        shape=(len(all_hours),),
        chunks=(time_chunk_size,),
        shards=(time_shard_size,),
        dtype="M8[m]",
        dimension_names=["time"],
        )
    zgrp["time"][...] = all_hours
    zgrp.create_array(
        "station_name",
        shape=(len(stations),),
        fill_value="",
        dtype="U32",
        dimension_names=["station_name"]
        )
    zgrp["station_name"][...] = stations
    zgrp.create_array(
        "variable_label",
        shape=(len(keep_float_fields),),
        fill_value="",
        dtype="U32",
        dimension_names=["variable_label"]
        )
    zgrp["variable_label"][...] = keep_float_fields
    zgrp.create_array(
        "stat_label",
        shape=(len(stat_labels),),
        fill_value="",
        dtype="U16",
        dimension_names=["stat_label"]
        )
    zgrp["stat_label"][...] = stat_labels
    zgrp.create_array(
        "obs",
        shape=(len(stations), len(keep_float_fields), len(all_hours), 5),
        chunks=(station_chunk_size, var_chunk_size, time_chunk_size, 5),
        shards=(station_shard_size, var_shard_size, time_shard_size, 5),
        dtype=np.float32,
        fill_value=np.nan,
        dimension_names=[
            "station_name", "variable_label", "time", "stat_label"
            ],
        compressors=[comp],
        )
    zgrp["obs"][...] = sarr
    zgrp.attrs.update({
        "labels":{
            "stations":stations,
            "variables":keep_float_fields,
            "stats":stat_labels,
            },
        "time_var":time_var,
        "coords":station_coords,
        })
    zarr.consolidate_metadata(zarr_store)
    print("finished")
