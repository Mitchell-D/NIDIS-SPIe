import zarr
import pickle as pkl
import numpy as np
from datetime import datetime,timedelta
from pathlib import Path
from multiprocessing import Pool

if __name__=="__main__":
    data_dir = Path("/rstor/mdodson/nidis_spie")
    zarr_store = data_dir.joinpath("zarr/crn_hourly_por.zarr")

    nworkers = 20

    day_chunk_size = 24 * 64
    day_shard_size = time_chunk_size * 12
    station_chunk_size = 4
    station_shard_size = station_chunk_size * 6
    var_chunk_size = 4
    var_shard_size = var_chunk_size * 8
    flag_chunk_size = 4
    flag_shard_size = flag_chunk_size * 8

    """ ------------( end normal configuration )------------ """

    assert zarr_store.exists()

    zgrp = zarr.open(zarr_store, mode="a")

    stations = zgrp.attrs["labels"]["stations"]
    feats = zgrp.attrs["labels"]["variables"]
    flags = zgrp.attrs["labels"]["flags"]

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

    pdata = pkl.load(pkl_path.open("rb"))
    stations = list(pdata.keys())

    args = []
    station_coords = {}
    for sid in pdata.keys():
        ## most stations should be consistent, but some vary by a bit likely
        ## due to rounding errors. arbitrarily choose to keep the min coord
        station_coords[sid] = (
            float(np.amin(pdata[sid]["lat"])),
            float(np.amin(pdata[sid]["lon"])),
            )
        for fix,fk in enumerate(keep_float_fields):
            six = stations.index(sid)
            #fix = keep_float_fields.index(fk)
            args.append({
                "start_time":start_time,
                "ntimes":len(all_hours),
                "measurement_times":pdata[sid][time_var],
                "data_array":pdata[sid][fk],
                "dtype":np.float32,
                "fill":np.nan,
                "fix":fix,
                "six":six,
                })
    print("extracted float data")

    sarr = np.full(
        (len(stations), len(keep_float_fields), len(all_hours)),
        np.nan,
        dtype=np.float32,
        )

    with Pool(nworkers) as pool:
        for a,r in pool.imap_unordered(mp_snap_to_hour, args):
            sarr[a["six"], a["fix"]] = r

    args = []
    for sid in pdata.keys():
        for fix,fk in enumerate(keep_flag_fields):
            six = stations.index(sid)
            #fix = keep_float_fields.index(fk)
            args.append({
                "start_time":start_time,
                "ntimes":len(all_hours),
                "measurement_times":pdata[sid][time_var],
                "data_array":pdata[sid][fk],
                "dtype":np.uint8,
                "fill":255,
                "fix":fix,
                "six":six,
                })
    print("extracted flag data")

    farr = np.full(
        (len(stations), len(keep_flag_fields), len(all_hours)),
        255,
        dtype=np.uint8,
        )

    with Pool(nworkers) as pool:
        for a,r in pool.imap_unordered(mp_snap_to_hour, args):
            farr[a["six"], a["fix"]] = r

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
        chunks=(station_chunk_size,),
        shards=(station_shard_size,),
        fill_value="",
        dtype="U32",
        dimension_names=["station_name"]
        )
    zgrp["station_name"][...] = stations
    zgrp.create_array(
        "variable_label",
        shape=(len(keep_float_fields),),
        chunks=(var_chunk_size,),
        shards=(var_shard_size,),
        fill_value="",
        dtype="U32",
        dimension_names=["variable_label"]
        )
    zgrp["variable_label"][...] = keep_float_fields
    zgrp.create_array(
        "flag_label",
        shape=(len(keep_flag_fields),),
        chunks=(flag_chunk_size,),
        shards=(flag_shard_size,),
        fill_value="",
        dtype="U32",
        dimension_names=["flag_label"]
        )
    zgrp["flag_label"][...] = keep_flag_fields
    zgrp.create_array(
        "obs",
        shape=(len(stations), len(keep_float_fields), len(all_hours)),
        chunks=(station_chunk_size, var_chunk_size, time_chunk_size),
        shards=(station_shard_size, var_shard_size, time_shard_size),
        dtype=np.float32,
        fill_value=np.nan,
        dimension_names=["station_name", "variable_label", "time"],
        compressors=[comp],
        )
    zgrp["obs"][...] = sarr
    zgrp.create_array(
        "flags",
        shape=(len(stations), len(keep_flag_fields), len(all_hours)),
        chunks=(station_chunk_size, flag_chunk_size, time_chunk_size),
        shards=(station_shard_size, flag_shard_size, time_shard_size),
        dtype=np.uint8,
        fill_value=255,
        dimension_names=["station_name", "flag_label", "time"],
        compressors=[comp],
        )
    zgrp["flags"][...] = farr
    zgrp.attrs.update({
        "labels":{
            "stations":stations,
            "variables":keep_float_fields,
            "flags":keep_flag_fields,
            },
        "time_var":time_var,
        "coords":station_coords,
        })
    zarr.consolidate_metadata(zarr_store)
    print("finished")
