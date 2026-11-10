import zarr
import numpy as np
import json
import requests
from pathlib import Path

def parse_ghcnd_inventory(inv_path:Path):
    """
    """
    inv = inv_path.open("r").readlines()

    slocs = {}
    var_pors = {}
    for l in inv:
        sid,lat,lon,var,y0,yf = [
            v for v in l.replace("\n","").split(" ")
            if not v == ""
            ]
        if sid not in slocs.keys():
            slocs[sid] = [float(lat), float(lon)]
            var_pors[sid] = {}
        assert var not in var_pors[sid].keys(), (sid, var)
        var_pors[sid][var] = (int(y0), int(yf))
    return slocs,var_pors

if __name__=="__main__":
    inv_path = Path("data/ghcnd-inventory.txt")
    zarr_store = Path("data/zarr/crn_subhourly_2006-2026.zarr")
    mapping_json = Path("data/crn_ghcn_station_mapping.json")
    ghcn_base_url = "https://www.ncei.noaa.gov/data/" + \
        "global-historical-climatology-network-daily/access/"
    out_dir = Path("data/ghcnd/")
    ghcn_csv_template = "{ghcn_id}.csv"
    degree_thresh = 0.3
    por_years_thresh = 25
    check_close_to_stations = [
        "AL_Gadsden_19_N",
        "AL_Selma_13_WNW",
        "AL_Fairhope_3_NE",
        ]
    check_closest = True
    download = True

    """ ------------( end normal configuration )------------ """

    if check_closest:
        slocs,var_pors = parse_ghcnd_inventory(inv_path)
        sids,lats,lons = zip(*[(k,*v) for k,v in slocs.items()])
        lats = np.array(lats)
        lons = np.array(lons)
        zgrp = zarr.open(zarr_store, mode="r")
        target_locs = {
            k:zgrp.attrs["coords"][k] for k in check_close_to_stations
            }
        closest = {}
        for crn_sid,(crn_lat,crn_lon) in target_locs.items():
            closest[crn_sid] = []
            dist = ((lats-crn_lat)**2 + (lons-crn_lon)**2)**(1/2)
            for ix in np.argwhere(dist < degree_thresh).squeeze():
                s = sids[ix]
                if "PRCP" not in var_pors[s].keys():
                    continue
                y0,yf = var_pors[s]["PRCP"]
                if yf-y0 < por_years_thresh:
                    continue
                closest[crn_sid].append({
                    "distance_deg":dist[ix],
                    "por":(y0, yf),
                    "ghcn_id":s,
                    "ghcn_loc":(lats[ix], lons[ix]),
                    })
        json.dump(closest, mapping_json.open("w"), indent=2)

    if download:
        closest = json.load(mapping_json.open("r"))
        for crn_sid,clist in closest.items():
            for gd in clist:
                fname = ghcn_csv_template.format(
                    ghcn_id=gd["ghcn_id"],
                    )
                url = ghcn_base_url + fname
                resp = requests.get(url, stream=True)
                with out_dir.joinpath(fname).open("wb") as outf:
                    for chunk in resp.iter_content(chunk_size=8192):
                        outf.write(chunk)

