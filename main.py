import numpy as np
import pickle as pkl
import json
import zarr
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime

from acquire_uscrn_subhourly import download_uscrn_stations
from extract_uscrn_pkl_subhourly import mp_parse_uscrn_subhourly
from parse_uscrn_subhourly_pkl_to_zarr import parse_uscrn_subhourly_pkl_to_zarr
from check_ghcnd_inventory import download_closest_ghcnd
from parse_ghcn_daily_to_zarr import parse_ghcnd_csv
from calc_spi_dists import mp_get_station_spi

if __name__=="__main__":
    ## uscrn source data download location
    src_csv_dir = Path("data/source/subhourly")
    src_concat_dir = Path("data/source/subhourly_concat")
    ## uscrn source pkl extract location
    por_pkl_dir = Path("data/pkls")
    ## uscrn zarr store path
    crn_zarr_path = Path("data/zarr/crn_subhourly_2006-2026.zarr")
    ## ghcn daily station inventory location
    ghcn_inventory_path = Path("data/ghcnd-inventory.txt")
    ## mapping from crn stations to nearby ghcn daily stations
    crn_ghcn_mapping_json_path = Path("data/crn_ghcn_station_mapping.json")
    ghcnd_src_dir = Path("data/ghcnd")
    ghcnd_zarr_path = Path("data/zarr/ghcnd.zarr")

    concat_csv_template = "{sid}-{y0}-{yf}.txt"
    por_pkl_template = "{sid}-{y0}-{yf}.pkl"
    ghcnd_csv_template = "{ghcn_id}.csv"

    ## download options
    download_start_year = 2006
    download_end_year = 2026
    get_crn_stations = [
        "AL_Russellville_4_SSE",
        "AL_Fairhope_3_NE",
        "AL_Cullman_3_ENE",
        "AL_Troy_2_W",
        "AL_Talladega_10_NNE",
        "AL_Selma_6_SSE",
        "AL_Scottsboro_2_NE",
        "AL_Thomasville_2_S",
        "AL_Highland_Home_2_S",
        "AL_Gainesville_2_NE",
        "AL_Selma_13_WNW",
        "AL_Valley_Head_1_SSW",
        "AL_Greensboro_2_WNW",
        "AL_Northport_2_S",
        "AL_Courtland_2_WSW",
        "AL_Gadsden_19_N",
        "AL_Brewton_3_NNE",
        "AL_Muscle_Shoals_2_N",
        "AL_Clanton_2_NE",
        "GA_Newton_8_W",
        "GA_Newton_11_SW",
        ]
    uscrn_base_url = "https://www.ncei.noaa.gov/" + \
            "pub/data/uscrn/products/subhourly01/"

    ## pkl parsing options
    get_pkl_fields = [
        "UTC_DATE", "UTC_TIME", "LST_DATE", "LST_TIME", "LONGITUDE",
        "LATITUDE", "AIR_TEMPERATURE", "PRECIPITATION", "SOLAR_RADIATION",
        "SR_FLAG", "SURFACE_TEMPERATURE", "ST_TYPE", "ST_FLAG",
        "RELATIVE_HUMIDITY", "RH_FLAG", "SOIL_MOISTURE_5",
        "SOIL_TEMPERATURE_5", "WETNESS", "WET_FLAG", "WIND_1_5", "WIND_FLAG",
        ]

    ## zarr storage options
    zarr_start_time = datetime(2005, 4, 1, 0, 0)
    zarr_end_time = datetime(2026, 11, 16, 23, 0)
    pkl_time_var = "time-utc"
    keep_float_fields = [
        "temp", "precip", "swdown", "stemp", "rh", "sm-2in", "st-2in",
        "wetn", "wspd",
        ]
    flag_field_mapping = {
        "swdown":"flag-swdown",
        "stemp":"flag-stemp",
        "rh":"flag-rh",
        "wetn":"flag-wetn",
        "wspd":"flag-wspd",
        "stemp":"flag-stemp",
        }
    time_chunk_size = 24 * 64
    time_shard_size = time_chunk_size * 12
    station_chunk_size = 4
    station_shard_size = station_chunk_size * 6
    var_chunk_size = 4
    var_shard_size = var_chunk_size * 8
    flag_chunk_size = 4
    flag_shard_size = flag_chunk_size * 8
    zarr_get_stations = None ## all of them
    subhourly_nan_frac_thresh = 0.3

    ## ghcn download options
    degree_thresh = 0.3
    por_years_thresh = 35
    ghcn_base_url = "https://www.ncei.noaa.gov/data/" + \
        "global-historical-climatology-network-daily/access/"

    ## ghcn parsing options
    extract_ghcn_vars = ["PRCP", "TMAX", "TMIN", "DAPR", "MDPR"]
    extract_ghcn_attrs = [
        "PRCP_ATTRIBUTES", "TMAX_ATTRIBUTES", "TMIN_ATTRIBUTES",
        "DAPR_ATTRIBUTES", "MDPR_ATTRIBUTES"
        ]

    ## distribution calculation options
    fit_distributions = [
        "gamma", "gumbel_r", "fisk", "lognorm",
        "logistic", "norm", "weibull_min",
        ]
    aggregation_windows = [15, 30, 45, 60, 90]
    window_nan_threshold = .2
    hist_bins = 48
    hist_bounds_spi = (0, 768)
    hist_bounds_jng = (0, 768)
    hist_bounds_wspi = (0, 384)
    wspi_weight_scale = 5
    wspi_weight_method = "sigmoid"

    nworkers = 10

    download_source_csvs = True
    extract_uscrn_pkls = True
    parse_to_zarr = True
    download_ghcnd = True
    extract_ghcnd = True
    get_distributions = True

    """ ------------( end normal configuration )------------ """

    if download_source_csvs:
        print("\nDOWNLOADING USCRN SOURCE DATA")
        download_uscrn_stations(
            start_year=download_start_year,
            end_year=download_end_year,
            station_names=get_crn_stations,
            out_dir=src_csv_dir,
            base_url=uscrn_base_url,
            )

        sfiles = {}
        for p in src_csv_dir.iterdir():
            _,_,yyyy,sid = p.stem.split("-")
            yyyy = int(yyyy)
            if not sid in get_crn_stations:
                continue
            if not yyyy in range(download_start_year, download_end_year+1):
                continue
            if not sid in sfiles.keys():
                sfiles[sid] = []
            sfiles[sid].append((yyyy, p))

        for sid in sfiles.keys():
            sfiles[sid] = list(sorted(sfiles[sid], key=lambda v:v[0]))
            y0 = sfiles[sid][0][0]
            yf = sfiles[sid][-1][0]
            out_path = src_concat_dir.joinpath(
                concat_csv_template.format(sid=sid, y0=y0, yf=yf)
                )
            if out_path.exists():
                print(f"skipping existing concat csv: {out_path.name}")
            with out_path.open("w") as ofp:
                for _,p in sfiles[sid]:
                    with p.open("r") as ifp:
                        ofp.write(ifp.read())

    if extract_uscrn_pkls:
        print("\nEXTRACTING USCRN PKLS")
        args = []
        for p in src_concat_dir.iterdir():
            sid,y0,yf = p.stem.split("-")
            out_path = por_pkl_dir.joinpath(
                por_pkl_template.format(sid=sid, y0=y0, yf=yf)
                )
            if out_path.exists():
                continue
            args.append({
                "csv_file":p,
                "get_fields":get_pkl_fields,
                ## passthrough
                "out_path":out_path,
                })
        with Pool(nworkers) as pool:
            for a,r in pool.imap_unordered(mp_parse_uscrn_subhourly, args):
                pkl.dump(r, a["out_path"].open("wb"))
                print(f"stored {a['out_path'].as_posix()}")

    if parse_to_zarr:
        print("\nPARSING USCRN PKLS TO ZARR")
        parse_uscrn_subhourly_pkl_to_zarr(
            zarr_store=crn_zarr_path,
            pkl_dir=por_pkl_dir,
            start_time=zarr_start_time,
            end_time=zarr_end_time,
            time_var=pkl_time_var,
            keep_float_fields=keep_float_fields,
            flag_field_mapping=flag_field_mapping,
            time_chunk_size=time_chunk_size,
            time_shard_size=time_shard_size,
            station_chunk_size=station_chunk_size,
            station_shard_size=station_shard_size,
            var_chunk_size=var_chunk_size,
            var_shard_size=var_shard_size,
            flag_chunk_size=flag_chunk_size,
            flag_shard_size=flag_shard_size,
            get_stations=zarr_get_stations,
            nan_frac_thresh=subhourly_nan_frac_thresh,
            nworkers=nworkers,
            )

    if download_ghcnd:
        print("\nDOWNLOADING GHCNd STATIONS")
        download_closest_ghcnd(
            uscrn_zarr_store=crn_zarr_path,
            ghcn_inventory_path=ghcn_inventory_path,
            mapping_json_path=crn_ghcn_mapping_json_path,
            ghcn_out_dir=ghcnd_src_dir,
            ghcn_base_url=ghcn_base_url,
            ghcn_csv_template=ghcnd_csv_template,
            degree_threshold=degree_thresh,
            por_years_threshold=por_years_thresh,
            check_close_to_stations=get_crn_stations,
            )

    if extract_ghcnd:
        print("\nEXTRACTING GHCNd STATIONS")
        if ghcnd_zarr_path.exists():
            raise ValueError(
                f"GHCNd zarr store exists: {ghcnd_zarr_path.as_posix()}"
                )
        zgrp = zarr.open(ghcnd_zarr_path, mode="w")
        zgrp.create_group("stations")
        zgrp.attrs.update({
            "flags":{
                "precip":[
                    "measurement", "quality", "source", "time"
                    ],
                "other":[
                    "measurement", "quality", "source",
                    ],
                }
            })
        for p in ghcnd_src_dir.iterdir():
            meta,dates,vals,attrs = parse_ghcnd_csv(
                csv_path=p,
                extract_vars=extract_ghcn_vars,
                extract_attrs=extract_ghcn_attrs,
                )
            zs = zgrp["stations"].create_group(meta["station"])
            zs.create_array("date", data=dates)
            zs.attrs.update(meta)
            for vk,vv in vals.items():
                zs.create_array(vk, data=vv)
            for ak,av in attrs.items():
                zs.create_array(ak, data=av)

    if get_distributions:
        print("\nFITTING DISTRIBUTIONS")
        zgrp = zarr.open(ghcnd_zarr_path, mode="r")
        all_stations = list(zgrp["stations"].keys())
        ghcn_stations = []
        if get_crn_stations is None:
            ghcn_stations = all_stations
        else:
            cgsm = json.load(crn_ghcn_mapping_json_path.open("r"))
            for crn_key in get_crn_stations:
                ghcn_stations += [v["ghcn_id"] for v in cgsm[crn_key]]
        ghcn_stations = list(set(ghcn_stations))

        args = [{
            "zarr_store":ghcnd_zarr_path,
            "station_id":sid,
            "distributions":fit_distributions,
            "windows":aggregation_windows,
            "hist_bins":hist_bins,
            "hist_bounds_spi":hist_bounds_spi,
            "hist_bounds_wspi":hist_bounds_wspi,
            "hist_bounds_jng":hist_bounds_jng,
            "weight_method":wspi_weight_method,
            "weight_scale":wspi_weight_scale,
            "nan_thresh":window_nan_threshold,
            } for sid in ghcn_stations]
        with Pool(nworkers) as pool:
            for a,r in pool.imap_unordered(mp_get_station_spi, args):
                continue
