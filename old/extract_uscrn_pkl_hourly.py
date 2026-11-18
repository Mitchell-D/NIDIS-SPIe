import numpy as np
import pickle as pkl
import zarr
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
    "Date UTC":(
        "time-utc",
        "M8[m]",
        lambda v: datetime.strptime(v[:-6], "%Y-%m-%d %H:%M:%S"),
        ),
    "Date":(
        "time-local",
        "M8[m]",
        lambda v: datetime.strptime(v[:-6], "%Y-%m-%d %H:%M:%S"),
        ),
    "LONGITUDE":("lon", np.float32, lambda v: float_or_nan(v)),
    "LATITUDE":("lat", np.float32, lambda v: float_or_nan(v)),
    "T_CALC":("temp-end", np.float32, lambda v: float_or_nan(v)),
    "T_MAX":("temp-max", np.float32, lambda v: float_or_nan(v)),
    "T_MIN":("temp-min", np.float32, lambda v: float_or_nan(v)),
    "T_HR_AVG":("temp-avg", np.float32, lambda v: float_or_nan(v)),
    "P_CALC":("precip", np.float32, lambda v: float_or_nan(v)),
    "SOLARAD":("swdown-avg", np.float32, lambda v: float_or_nan(v, -99999)),
    "SOLARAD_FLAG":("flag-swdown-avg", np.uint8, lambda v: np.uint8(v)),
    "SOLARAD_MAX":("swdown-max", np.float32, lambda v: float_or_nan(v, -99999)),
    "SOLARAD_MAX_FLAG":("flag-swdown-max", np.uint8, lambda v: np.uint8(v)),
    "SOLARAD_MIN":("swdown-min", np.float32, lambda v: float_or_nan(v, -99999)),
    "SOLARAD_MIN_FLAG":("flag-swdown-min", np.uint8, lambda v: np.uint8(v)),
    "SUR_TEMP_TYPE":("stemp-type", "U1", lambda v: str(v)),
    "SUR_TEMP":("stemp-avg", np.float32, lambda v: float_or_nan(v)),
    "SUR_TEMP_FLAG":("flag-stemp-avg", np.uint8, lambda v: np.uint8(v)),
    "SUR_TEMP_MAX":("stemp-max", np.float32, lambda v: float_or_nan(v)),
    "SUR_TEMP_MAX_FLAG":("flag-stemp-max", np.uint8, lambda v: np.uint8(v)),
    "SUR_TEMP_MIN":("stemp-min", np.float32, lambda v: float_or_nan(v)),
    "SUR_TEMP_MIN_FLAG":("flag-stemp-min", np.uint8, lambda v: np.uint8(v)),
    "RH_HR_AVG":("rh-avg", np.float32, lambda v: float_or_nan(v)),
    "RH_HR_AVG_FLAG":("flag-rh-avg", np.uint8, lambda v: np.uint8(v)),
    "SMSm2":("sm-2in", np.float32, lambda v: float_or_nan(v, -99)),
    "SMSm4":("sm-4in", np.float32, lambda v: float_or_nan(v, -99)),
    "SMSm8":("sm-8in", np.float32, lambda v: float_or_nan(v, -99)),
    "SMSm20":("sm-20in", np.float32, lambda v: float_or_nan(v, -99)),
    "SMSm40":("sm-40in", np.float32, lambda v: float_or_nan(v, -99)),
    "STOm2":("st-2in", np.float32, lambda v: float_or_nan(v)),
    "STOm4":("st-4in", np.float32, lambda v: float_or_nan(v)),
    "STOm8":("st-8in", np.float32, lambda v: float_or_nan(v)),
    "STOm20":("st-20in", np.float32, lambda v: float_or_nan(v)),
    "STOm40":("st-40in", np.float32, lambda v: float_or_nan(v)),
    "site":("site", "U32", lambda v: str(v))
    }

def parse_uscrn(csv_file:Path, get_fields):
    """ """
    lines = csv_file.open("r").readlines()
    h = lines.pop(0).replace("\n","").split(",")

    new_flabels = []
    flabel_mapping = {}
    col_ixs = []
    parsers = {}
    dtypes = {}
    #field_info = []
    for ofk in get_fields:
        nfk,fdt,fconv = field_parsers[ofk]
        #field_info.append((h.index(ofk), *field_parsers[ofk]))
        new_flabels.append(nfk)
        flabel_mapping[ofk] = nfk
        dtypes[nfk] = fdt
        parsers[nfk] = fconv
        col_ixs.append(h.index(ofk))
    cols = list(zip(*[
        lstr.replace("\n", "").split(",")
        for lstr in lines
        ]))
    cols = {
        flabel_mapping[k]:c
        for i,(k,c) in enumerate(zip(h,cols))
        if i in col_ixs
        }
    sids = list(set(cols["site"]))
    data = {}
    for s in sids:
        data[s] = {nfk:[] for nfk in new_flabels}

    for i,s in enumerate(cols["site"]):
        for k in new_flabels:
            if k == "site":
                continue
            data[s][k].append(cols[k][i])

    for s in data.keys():
        del data[s]["site"]
        for k in data[s].keys():
            data[s][k] = [parsers[k](v) for v in data[s][k]]
            data[s][k] = np.asarray(data[s][k], dtype=dtypes[k])

    return data

    '''
    data = {nfk:[] for _,nfk,_,_ in field_info}
    for lstr in lines:
        l = lstr.replace("\n", "").split(",")
        for fix,nfk,_,fconv in field_info:
            data[nfk].append(fconv(l[fix]))

    for _,nfk,fdt,_ in field_info:
        data[nfk] = np.asarray(data[nfk], dtype=fdt)
    return data
    '''


if __name__=="__main__":
    data_dir = Path("/rstor/mdodson/nidis_spie/")
    csv_path = data_dir.joinpath("source/crn_hourly_por.csv")
    out_pkl_path = data_dir.joinpath("pkls/crn_hourly_por.pkl")
    get_fields = [
        "site", "WBANNO", "Date UTC", "Date", "LONGITUDE", "LATITUDE",
        "T_CALC", "T_MAX", "T_MIN", "T_HR_AVG",
        "P_CALC",
        "SOLARAD", "SOLARAD_FLAG", "SOLARAD_MAX", "SOLARAD_MAX_FLAG",
        "SOLARAD_MIN", "SOLARAD_MIN_FLAG",
        "SUR_TEMP_TYPE", "SUR_TEMP", "SUR_TEMP_FLAG", "SUR_TEMP_MAX",
        "SUR_TEMP_MAX_FLAG", "SUR_TEMP_MIN", "SUR_TEMP_MIN_FLAG",
        "RH_HR_AVG", "RH_HR_AVG_FLAG",
        "SMSm2", "SMSm4", "SMSm8", "SMSm20", "SMSm40",
        "STOm2", "STOm4", "STOm8", "STOm20", "STOm40",
        ]
    r  = parse_uscrn(csv_path, get_fields)
    pkl.dump(r, out_pkl_path.open("wb"))
