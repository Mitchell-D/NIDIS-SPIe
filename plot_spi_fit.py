import zarr
import numpy as np
import json
import pickle as pkl
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
import scipy.stats as spstats

from plotting import plot_nested_bars,plot_lines

long_labels_err = {
    "sse":"Sum of Squared Errors",
    "ad":"Anderson-Darling Fit",
    "aic":"Akaike Information Criterion",
    }

if __name__=="__main__":
    zarr_store = Path("data/zarr/ghcnd.zarr")
    fig_dir = Path("figures/spi-err")
    crn_ghcn_mapping_path = Path("data/crn_ghcn_station_mapping.json")

    #get_stations = None
    get_stations = ["AL_Fairhope_3_NE", "GA_Newton_8_W", "GA_Newton_11_SW"]

    """ ------------( end normal configuration )------------ """

    zgrp = zarr.open(zarr_store, mode="r")

    ghcn_stations = []
    if get_stations is None:
        ghcn_stations = list(zgrp["stations"].keys())
    else:
        cgsm = json.load(crn_ghcn_mapping_path.open("r"))
        for crn_key in get_stations:
            ghcn_stations += [v["ghcn_id"] for v in cgsm[crn_key]]
    ghcn_stations = list(set(ghcn_stations))

    for sid in zgrp["stations"].keys():
        if sid not in ghcn_stations:
            continue
        zs = zgrp[f"/stations/{sid}"]

        l_dist = zs.attrs["labels"]["dist"]
        l_window = zs.attrs["labels"]["window"]
        l_err = zs.attrs["labels"]["err"]

        ## plot station-wise
        derr = zs["spi-err"][...]
        for eix,ek in enumerate(l_err):
            ev = derr[...,eix]
            eavg = np.average(ev, axis=2)
            edata = {}
            for wix,wk in enumerate(l_window):
                edata[wk] = {}
                for dix,dk in enumerate(l_dist):
                    eavg = np.average(ev[dix,wix])
                    p10,p90 = np.percentile(ev[dix,wix], [10, 90])
                    edata[wk][dk] = (eavg, eavg-p10, p90-eavg)
            plot_nested_bars(
                data_dict=edata,
                labels={},
                plot_error_bars=True,
                bar_colors=None,
                plot_spec={
                    "title":f"SPI {sid} {long_labels_err[ek]}",
                    "cmap":"rainbow",
                    "ylabel":long_labels_err[ek],
                    "xlabel":"Window size",
                    "figsize":(16,12),
                    "yscale":"log" if ek=="ad" else "linear",
                    "legend_ncols":2,
                    },
                #show=True,
                fig_path=fig_dir.joinpath(f"spi_err-bars_{ek}_{sid}.png"),
                group_order=list(sorted(l_window)),
                bar_order=list(sorted(l_dist)),
                )

            ## plot annual timeline of distribution error
            dom = np.arange(1, 367)
            for wix,wk in enumerate(l_window):
                dlabels,etl = zip(*[
                    (dk,ev[dix,wix]) for dix,dk in enumerate(l_dist)
                    ])
                plot_lines(
                    domain=dom,
                    ylines=etl,
                    fig_path=fig_dir.joinpath(
                        f"spi-err-timeline_{ek}_w{wk}_{sid}.png"
                        ),
                    labels=dlabels,
                    plot_spec={
                        "title":f"SPI {sid} {long_labels_err[ek]} w:{wk}",
                        "xlabel":"day of year",
                        "ylabel":long_labels_err[ek],
                        "yscale":"log" if ek=="ad" else "linear",
                        "cmap":"rainbow",
                        },
                    )

        ## plot station-wise
        derr = zs["jng-err"][...]
        for eix,ek in enumerate(l_err):
            ev = derr[...,eix]
            eavg = np.average(ev, axis=2)
            edata = {}
            for wix,wk in enumerate(l_window):
                edata[wk] = {}
                for dix,dk in enumerate(l_dist):
                    eavg = np.average(ev[dix,wix])
                    p10,p90 = np.percentile(ev[dix,wix], [10, 90])
                    edata[wk][dk] = (eavg, eavg-p10, p90-eavg)
            plot_nested_bars(
                data_dict=edata,
                labels={},
                plot_error_bars=True,
                bar_colors=None,
                plot_spec={
                    "title":f"JNG SPIe {sid} {long_labels_err[ek]}",
                    "cmap":"rainbow",
                    "ylabel":long_labels_err[ek],
                    "xlabel":"Window size",
                    "figsize":(16,12),
                    "yscale":"log" if ek=="ad" else "linear",
                    "legend_ncols":2,
                    },
                #show=True,
                fig_path=fig_dir.joinpath(f"jng_err-bars_{ek}_{sid}.png"),
                group_order=list(sorted(l_window)),
                bar_order=list(sorted(l_dist)),
                )

            ## plot annual timeline of distribution error
            dom = np.arange(1, 367)
            for wix,wk in enumerate(l_window):
                dlabels,etl = zip(*[
                    (dk,ev[dix,wix]) for dix,dk in enumerate(l_dist)
                    ])
                plot_lines(
                    domain=dom,
                    ylines=etl,
                    fig_path=fig_dir.joinpath(
                        f"jng-err-timeline_{ek}_w{wk}_{sid}.png"
                        ),
                    labels=dlabels,
                    plot_spec={
                        "title":f"JNG SPIe {sid} {long_labels_err[ek]} w:{wk}",
                        "xlabel":"day of year",
                        "ylabel":long_labels_err[ek],
                        "yscale":"log" if ek=="ad" else "linear",
                        "cmap":"rainbow",
                        },
                    )

    ## plot SPI error generalized over stations
    derr = np.concatenate([
        zgrp[f"/stations/{sid}/spi-err"][...]
        ], axis=2)

    for eix,ek in enumerate(l_err):
        ev = derr[...,eix]
        eavg = np.average(ev, axis=2)
        edata = {}
        for wix,wk in enumerate(l_window):
            edata[wk] = {}
            for dix,dk in enumerate(l_dist):
                eavg = np.average(ev[dix,wix])
                p10,p90 = np.percentile(ev[dix,wix], [10, 90])
                edata[wk][dk] = (eavg, eavg-p10, p90-eavg)
        plot_nested_bars(
            data_dict=edata,
            labels={},
            plot_error_bars=True,
            bar_colors=None,
            plot_spec={
                "title":f"SPI GHCNd {long_labels_err[ek]}",
                "cmap":"rainbow",
                "ylabel":long_labels_err[ek],
                "xlabel":"Window size",
                "figsize":(16,12),
                "yscale":"log" if ek=="ad" else "linear",
                "legend_ncols":2,
                },
            #show=True,
            fig_path=fig_dir.joinpath(f"spi_err-bars_{ek}_all-stations.png"),
            group_order=list(sorted(l_window)),
            bar_order=list(sorted(l_dist)),
            )

    ## plot JNG SPIe generalized over stations
    derr = np.concatenate([
        zgrp[f"/stations/{sid}/jng-err"][...]
        ], axis=2)

    for eix,ek in enumerate(l_err):
        ev = derr[...,eix]
        eavg = np.average(ev, axis=2)
        edata = {}
        for wix,wk in enumerate(l_window):
            edata[wk] = {}
            for dix,dk in enumerate(l_dist):
                eavg = np.average(ev[dix,wix])
                p10,p90 = np.percentile(ev[dix,wix], [10, 90])
                edata[wk][dk] = (eavg, eavg-p10, p90-eavg)
        plot_nested_bars(
            data_dict=edata,
            labels={},
            plot_error_bars=True,
            bar_colors=None,
            plot_spec={
                "title":f"JNG SPIe GHCNd {long_labels_err[ek]}",
                "cmap":"rainbow",
                "ylabel":long_labels_err[ek],
                "xlabel":"Window size",
                "figsize":(16,12),
                "yscale":"log" if ek=="ad" else "linear",
                "legend_ncols":2,
                },
            #show=True,
            fig_path=fig_dir.joinpath(f"jng_err-bars_{ek}_all-stations.png"),
            group_order=list(sorted(l_window)),
            bar_order=list(sorted(l_dist)),
            )

