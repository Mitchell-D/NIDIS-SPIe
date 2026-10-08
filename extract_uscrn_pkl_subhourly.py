import numpy as np
import pickle as pkl
import zarr
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime,timedelta

def float_or_nan(v, mask_val=-9999):
    if v == "":
        return np.nan
    v = float(v)
    if v == mask_val:
        return np.nan
    return v

field_parsers = {
    "WBANNO":("wban", np.float32, lambda v: float_or_nan(v)),
    "UTC_DATE":(
        "date-utc",
        "M8[m]",
        lambda v: datetime.strptime(v, "%Y%m%d"),
        ),
    "UTC_TIME":(
        "time-utc",
        "M8[m]",
        lambda v: datetime.strptime(v, "%H%M"),
        ),
    "LST_DATE":(
        "date-local",
        "M8[m]",
        lambda v: datetime.strptime(v, "%Y%m%d"),
        ),
    "LST_TIME":(
        "time-local",
        "M8[m]",
        lambda v: datetime.strptime(v, "%H%M"),
        ),
    "LONGITUDE":("lon", np.float32, lambda v: float_or_nan(v)),
    "LATITUDE":("lat", np.float32, lambda v: float_or_nan(v)),
    "AIR_TEMPERATURE":("temp", np.float32, lambda v:float_or_nan(v)),
    "PRECIPITATION":("precip", np.float32, lambda v:float_or_nan(v)),
    "SOLAR_RADIATION":("swdown", np.float32, lambda v:float_or_nan(v)),
    "SR_FLAG":("flag-swdown", np.uint8, lambda v:np.uint8(v)),
    "SURFACE_TEMPERATURE":("stemp", np.float32, lambda v:float_or_nan(v)),
    "ST_TYPE":("stemp-type", "U1", lambda v:str(v)),
    "ST_FLAG":("flag-stemp", np.uint8, lambda v:np.uint8(v)),
    "RELATIVE_HUMIDITY":("rh", np.float32, lambda v:float_or_nan(v)),
    "RH_FLAG":("flag-rh", np.uint8, lambda v:np.uint8(v)),
    "SOIL_MOISTURE_5":("sm-2in", np.float32, lambda v:float_or_nan(v, -99.)),
    "SOIL_TEMPERATURE_5":("st-2in", np.float32, lambda v:float_or_nan(v)),
    "WETNESS":("wetn", np.float32, lambda v:float_or_nan(v)),
    "WET_FLAG":("flag-wetn", np.uint8, lambda v:np.uint8(v)),
    "WIND_1_5":("wspd", np.float32, lambda v:float_or_nan(v, -99.)),
    "WIND_FLAG":("flag-wspd", np.uint8, lambda v:np.uint8(v)),
    }

header = [
    "WBANNO",
    "UTC_DATE",
    "UTC_TIME",
    "LST_DATE",
    "LST_TIME",
    "CRX_VN",
    "LONGITUDE",
    "LATITUDE",
    "AIR_TEMPERATURE",
    "PRECIPITATION",
    "SOLAR_RADIATION",
    "SR_FLAG",
    "SURFACE_TEMPERATURE",
    "ST_TYPE",
    "ST_FLAG",
    "RELATIVE_HUMIDITY",
    "RH_FLAG",
    "SOIL_MOISTURE_5",
    "SOIL_TEMPERATURE_5",
    "WETNESS",
    "WET_FLAG",
    "WIND_1_5",
    "WIND_FLAG",
    ]

def mp_parse_uscrn_subhourly(args):
    return args,parse_uscrn_subhourly(**args)

def parse_uscrn_subhourly(csv_file:Path, get_fields, **_):
    """ """
    lines = csv_file.open("r").readlines()

    new_flabels = []
    flabel_mapping = {}
    col_ixs = []
    parsers = {}
    dtypes = {}
    for ofk in get_fields:
        nfk,fdt,fconv = field_parsers[ofk]
        new_flabels.append(nfk)
        flabel_mapping[ofk] = nfk
        dtypes[nfk] = fdt
        parsers[nfk] = fconv
        col_ixs.append(header.index(ofk))

    cols = list(zip(*[
        lstr.replace("\n", "").split()
        for lstr in lines
        ]))
    cols = {
        flabel_mapping[k]:c
        for i,(k,c) in enumerate(zip(header,cols))
        if i in col_ixs
        }

    for k in cols.keys():
        cols[k] = [parsers[k](v) for v in cols[k]]
        cols[k] = np.asarray(cols[k], dtype=dtypes[k])
    new_times_utc = []
    new_times_local = []
    cols["time-utc"] = cols["time-utc"].astype(object)
    cols["date-utc"] = cols["date-utc"].astype(object)
    cols["time-local"] = cols["time-local"].astype(object)
    cols["date-local"] = cols["date-local"].astype(object)
    for i in range(len(cols["date-utc"])):
        dtu = timedelta(
            hours=cols["time-utc"][i].hour,
            minutes=cols["time-utc"][i].minute,
            )
        new_times_utc.append(cols["date-utc"][i] + dtu)
        dtl = timedelta(
            hours=cols["time-local"][i].hour,
            minutes=cols["time-local"][i].minute,
            )
        new_times_local.append(cols["date-local"][i] + dtl)
    cols["time-utc"] = np.asarray(new_times_utc,dtype=dtypes["time-utc"])
    cols["time-local"] = np.asarray(new_times_local,dtype=dtypes["time-local"])
    del cols["date-utc"]
    del cols["date-local"]
    return cols

if __name__=="__main__":
    data_dir = Path("/rstor/mdodson/nidis_spie/")
    csv_dir = data_dir.joinpath("source/subhourly_concat")
    out_pkl_dir = data_dir.joinpath("pkls/subhourly")
    nworkers = 12
    get_fields = [
        "UTC_DATE",
        "UTC_TIME",
        "LST_DATE",
        "LST_TIME",
        "LONGITUDE",
        "LATITUDE",
        "AIR_TEMPERATURE",
        "PRECIPITATION",
        "SOLAR_RADIATION",
        "SR_FLAG",
        "SURFACE_TEMPERATURE",
        "ST_TYPE",
        "ST_FLAG",
        "RELATIVE_HUMIDITY",
        "RH_FLAG",
        "SOIL_MOISTURE_5",
        "SOIL_TEMPERATURE_5",
        "WETNESS",
        "WET_FLAG",
        "WIND_1_5",
        "WIND_FLAG",
        ]
    args = []
    for p in csv_dir.iterdir():
        sid,y0,yf = p.stem.split("-")
        args.append({
            "csv_file":p,
            "get_fields":get_fields,
            "sid":sid,
            "y0":y0,
            "yf":yf,
            })
    with Pool(nworkers) as pool:
        for a,r in pool.imap_unordered(mp_parse_uscrn_subhourly, args):
            out_path = out_pkl_dir.joinpath(f"{sid}-{y0}-{yf}.pkl")
            pkl.dump(r, out_path.open("wb"))
            print(f"stored {out_path.as_posix()}")

