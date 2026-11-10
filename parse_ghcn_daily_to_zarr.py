import numpy as np
import zarr
import datetime
import json
import calendar
import sys
import time
import csv

from pprint import pprint
from multiprocessing import Pool
from pathlib import Path

field_convert = {
    "id":lambda v:str(v.strip()),
    "date":lambda v:datetime.date.strptime(v, "%Y%m%d"),
    "element":lambda v:v.strip(),
    "data_value":lambda v:int(v),
    "m_flag":lambda v:str(v.strip()),
    "q_flag":lambda v:str(v.strip()),
    "s_flag":lambda v:str(v.strip()),
    "obs_time":lambda v:str(v.strip()),
    }

var_name_mapping = {
    "PRCP":"precip", ## tenths of mm
    "SNOW":"snof", ## mm
    "SNWD":"snod", ## mm
    "TMAX":"tmax", ## tenths of degrees C
    "TMIN":"tmin", ## tenths of degrees C
    "MDSF":"mdsf", ## multi-day snowfall total
    "AWND":"wspd", ## average wind speed (m/s)
    "MDPR":"mdpr", ## multi-day precip total
    "DAPR":"dapr", ## days of recorded precip in MDPR
    "DWPR":"dwpr", ## days with nonzero precip in MDPR
    }

attr_name_mapping = {
    "PRCP_ATTRIBUTES":"precip-attr",
    "TMAX_ATTRIBUTES":"tmax-attr",
    "TMIN_ATTRIBUTES":"tmin-attr",
    "DAPR_ATTRIBUTES":"dapr-attr",
    "MDPR_ATTRIBUTES":"mdpr-attr",
    }

element_dtypes = {
    "PRCP":np.float32,
    "SNOW":np.int16,
    "SNWD":np.int16,
    "TMAX":np.float32,
    "TMIN":np.float32,
    "MDSF":np.int16,
    "AWND":np.float32,
    "MDPR":np.float32,
    "DAPR":np.int16,
    "DWPR":np.int16,
    }

element_fill = {
    "PRCP":np.nan,
    "SNOW":-32768,
    "SNWD":-32768,
    "TMAX":np.nan,
    "TMIN":np.nan,
    "MDSF":-32768,
    "AWND":np.nan,
    "MDPR":np.nan,
    "DAPR":-32768,
    "DWPR":-32768,
    }

element_convert = {
    "PRCP":lambda v:v/10,
    "SNOW":lambda v:v,
    "SNWD":lambda v:v,
    "TMAX":lambda v:v/10,
    "TMIN":lambda v:v/10,
    "MDSF":lambda v:v,
    "AWND":lambda v:v/10,
    "MDPR":lambda v:v/10,
    "DAPR":lambda v:v,
    "DWPR":lambda v:v,
    }

def mp_parse_ghcnd_csv(args):
    return args,parse_ghcnd_csv(**args)

def parse_ghcnd_csv(csv_path:Path, extract_vars:list, extract_attrs:list):
    """
    Parse a daily single-station PoR GHCN file into a dictionary of station
    metadata, and a dicts mapping variable names to (N,) arrays of daily
    values, or (N,4) arrays of flag character values.
    """
    print("Processing ", csv_path.name)
    clines = csv_path.open("r").readlines()
    header = clines.pop(0)
    fields = [
        h.strip().replace('"',"")
          for h in header.replace("\n","").split(",")
          ]
    hixs = {
        "DATE":fields.index("DATE"),
        "LATITUDE":fields.index("LATITUDE"),
        "LONGITUDE":fields.index("LONGITUDE"),
        "ELEVATION":fields.index("ELEVATION"),
        "NAME":fields.index("NAME"),
        "STATION":fields.index("STATION"),
        }

    dates = []
    vixs = {}
    aixs = {}
    for vk in extract_vars:
        if vk not in fields:
            continue
        vixs[vk] = fields.index(vk)
    for ak in extract_attrs:
        if ak not in fields:
            continue
        aixs[ak] = fields.index(ak)

    out_vars = {vk:[] for vk in vixs.keys()}
    out_attrs = {ak:[] for ak in aixs.keys()}
    reader = csv.reader(clines)
    meta = None
    for cl in reader:
        if meta is None:
            meta = {
                "lat":float(cl[hixs["LATITUDE"]]),
                "lon":float(cl[hixs["LONGITUDE"]]),
                "elev":float(cl[hixs["ELEVATION"]]),
                "name":cl[hixs["NAME"]],
                "station":cl[hixs["STATION"]],
                }

        dates.append(datetime.datetime.strptime(cl[hixs["DATE"]], "%Y-%m-%d"))
        for vk,vix in vixs.items():
            out_vars[vk].append(cl[vix].strip())
        for ak,aix in aixs.items():
            out_attrs[ak].append(cl[aix].strip().split(","))

    for vk in out_vars.keys():
        out_vars[vk] = np.array([
            np.nan if v == "" else v for v in out_vars[vk]
            ], dtype=np.float32)
    for ak in out_attrs.keys():
        amax = max(np.unique([len(v) for v in out_attrs[ak]]))
        out_attrs[ak] = np.array([
            ("",)*amax if len(v)==1 else v
            for v in out_attrs[ak]
            ], dtype="U8")

    dates = np.array(dates, dtype="datetime64[D]")
    print(dates[0], dates[-1])
    daily = np.arange(
        dates[0],
        dates[-1] + np.timedelta64(1, "D"),
        np.timedelta64(1, "D"),
        )
    dixs = np.searchsorted(daily, dates)

    final_vars = {}
    final_attrs = {}
    for vk,vv in out_vars.items():
        vkf = var_name_mapping[vk]
        final_vars[vkf] = np.full(
            (daily.size,), np.nan, dtype=np.float32
            )
        final_vars[vkf][dixs] = element_convert[vk](vv)
    for ak,av in out_attrs.items():
        akf = attr_name_mapping[ak]
        final_attrs[akf] = np.full(
            (daily.size, av.shape[1]), "", dtype="U8"
            )
        final_attrs[akf][dixs] = av
    return meta,daily,final_vars,final_attrs

if __name__=="__main__":
    zarr_store = Path("data/zarr/ghcnd.zarr")
    csv_dir = Path("data/ghcnd")

    get_vars = [
        "PRCP", "TMAX", "TMIN", "DAPR", "MDPR"
        ]
    get_attrs = [
        "PRCP_ATTRIBUTES", "TMAX_ATTRIBUTES", "TMIN_ATTRIBUTES",
        "DAPR_ATTRIBUTES", "MDPR_ATTRIBUTES"
        ]

    zgrp = zarr.open(zarr_store, mode="w")
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
    for p in csv_dir.iterdir():
        meta,dates,vals,attrs = parse_ghcnd_csv(
            csv_path=p,
            extract_vars=get_vars,
            extract_attrs=get_attrs,
            )
        zs = zgrp["stations"].create_group(meta["station"])
        zs.create_array("date", data=dates)
        zs.attrs.update(meta)
        for vk,vv in vals.items():
            zs.create_array(vk, data=vv)
        for ak,av in attrs.items():
            zs.create_array(ak, data=av)
