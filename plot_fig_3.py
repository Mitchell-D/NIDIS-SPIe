import json
import numpy as np
import pickle as pkl
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
from pprint import pprint

from helpers import parse_usdm_csv
from plotting import plot_lines_multiy

if __name__=="__main__":
    spi_json_path = Path("data/crn_spi.json")
    fig_dir = Path("figures/spi_vs_acc")

    plot_station = "AL_Fairhope_3_NE"
    usdm_csv_path = Path("data/usdm/USDM-baldwin-county-al.csv")

    #plot_station = "GA_Newton_8_W"
    #plot_station = "GA_Newton_11_SW"
    #usdm_csv_path = Path("data/usdm/USDM-baker-county-ga.csv")

    plot_windows = [
        #15,
        30,
        #45,
        60,
        #90,
        ]

    inches = True

    plot_cumulative = [
        "precip",
        "eff_precip",
        "avg_precip",
        ]

    standard_precip_forms = [
        "spi",
        "wspi",
        #"jng",
        ]

    line_labels = {
        "avg_precip":"Mean Precip",
        "precip":"Cumulative Precip",
        "eff_precip":"Cumulative Effective Precip",
        "spi_15":"15-day SPI",
        "spi_30":"30-day SPI",
        "spi_45":"45-day SPI",
        "spi_60":"60-day SPI",
        "spi_90":"90-day SPI",
        "jng_15":"15-day SPIe",
        "jng_30":"30-day SPIe",
        "jng_45":"45-day SPIe",
        "jng_60":"60-day SPIe",
        "jng_90":"90-day SPIe",
        "wspi_15":"15-day wSPI",
        "wspi_30":"30-day wSPI",
        "wspi_45":"45-day wSPI",
        "wspi_60":"60-day wSPI",
        "wspi_90":"90-day wSPI",
        }

    line_colors = {
        "avg_precip":"#65d5db",
        "precip":"#6575db",
        "eff_precip":"#65db98",
        "spi_15":"#c3db65",
        "spi_30":"#dbcb65",
        "spi_45":"#dbaa65",
        "spi_60":"#db8865",
        "spi_90":"#db6565",
        "jng_15":"#c3db65",
        "jng_30":"#dbcb65",
        "jng_45":"#dbaa65",
        "jng_60":"#db8865",
        "jng_90":"#db6565",
        "wspi_15":"#c3db65",
        "wspi_30":"#dbcb65",
        "wspi_45":"#dbaa65",
        "wspi_60":"#db8865",
        "wspi_90":"#db6565",
        }
    line_style = {
        "avg_precip":"solid",
        "precip":"solid",
        "eff_precip":"solid",
        "spi_15":"solid",
        "spi_30":"solid",
        "spi_45":"solid",
        "spi_60":"solid",
        "spi_90":"solid",
        "jng_15":"dashed",
        "jng_30":"dashed",
        "jng_45":"dashed",
        "jng_60":"dashed",
        "jng_90":"dashed",
        "wspi_15":"dashdot",
        "wspi_30":"dashdot",
        "wspi_45":"dashdot",
        "wspi_60":"dashdot",
        "wspi_90":"dashdot",
        }

    usdm_field_colors = {
        "D0":"#ffff00",
        "D1":"#fcd37f",
        "D2":"#ffaa00",
        "D3":"#e60000",
        "D4":"#730000",
        }

    start_day = datetime(2024, 3, 14)
    end_day = datetime(2024, 11, 10)

    """ ------------( end normal configuration )------------ """

    data = json.load(spi_json_path.open("r"))[plot_station]
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in data["date"]]
    dix0 = dates.index(start_day)
    dixf = dates.index(end_day) + 1
    dates = dates[dix0:dixf]

    _,usdm_dates,usdm_fields,usdm_cover = parse_usdm_csv(usdm_csv_path)

    dix0_usdm = usdm_dates.index(start_day)
    dixf_usdm = usdm_dates.index(end_day) + 1
    usdm_cover = usdm_cover[dix0_usdm:dixf_usdm]

    '''
    ## determine the category of maximum drought designation within the county
    usdm_cat = []
    for i in range(usdm_cover.shape[0]):
        print(usdm_cover[i])
        v = np.argwhere(usdm_cover[i] > 0)
        usdm_cat.append(-1 if v.size == 0 else int(max(v)))
    '''

    ## determine the category of median drought designation within the county
    usdm_cat = []
    for i in range(usdm_cover.shape[0]):
        cp = 0
        for j in range(usdm_cover.shape[1]):
            jinv = usdm_cover.shape[1] - 1 - j
            cp += usdm_cover[i, jinv]
            if cp >= 50:
                break
        usdm_cat.append(-1 if cp == 0 else jinv)

    intervals = {}
    cur_start_val = None
    cur_start_index = 0
    for i,v in enumerate(usdm_cat):
        if cur_start_val is None:
            cur_start_val = v
            continue
        if cur_start_val != v:
            intervals[(cur_start_index, i)] = cur_start_val
            cur_start_index = i
            cur_start_val = v
    intervals[cur_start_index, i] = cur_start_val

    usdm_date_intervals = []
    for ix_range,ix_desig in intervals.items():
        if ix_desig == -1:
            continue
        usdm_desig = usdm_fields[ix_desig]
        usdm_date_intervals.append({
            "start":dates[ix_range[0]],
            "end":dates[ix_range[1]],
            "label":usdm_desig,
            "color":usdm_field_colors[usdm_desig],
            "alpha":0.3,
            })
    pprint(usdm_date_intervals)

    pc = {}
    for pck in plot_cumulative:
        tmpp = np.cumsum(data[pck][dix0:dixf])
        if inches:
            tmpp /= 25.4
        pc[pck] = tmpp.tolist()

    sp = {}
    for wk in map(str, plot_windows):
        for sk in standard_precip_forms:
            sp[f"{sk}_{wk}"] = data[sk][wk][dix0:dixf]

    sp_labels = list(sorted(sp.keys()))
    pc_labels = list(sorted(pc.keys()))
    all_labels = sp_labels + pc_labels

    plot_lines_multiy(
        domain=dates,
        ylines=[
            [sp[k] for k in sp_labels],
            [pc[k] for k in pc_labels],
            ],
        plot_spec={
            "title":f"Standard and Accumulated Precipitation {plot_station}",
            "line_labels":[line_labels[k] for k in all_labels],
            "line_colors":[line_colors[k] for k in all_labels],
            "line_style":[line_style[k] for k in all_labels],
            "y_labels":[
                "Standard Precipitation Z Score",
                f"Cumulative Precipitation ({['mm', 'in'][inches]})",
                ],
            "y_ranges":[(-3, 3), None],
            "legend_ncols":2,
            "legend_loc":"upper left",
            "legend_fontsize":8,
            "time_locator":"day",
            "time_locator_interval":14,
            "xtick_rotation":45,
            "xtick_align":"right",
            "grid":True,
            "hline_kwargs":{
                "y":0,
                #"color":"#db8865",
                "color":"black",
                "linestyle":"dotted",
                "linewidth":1,
                "alpha":1,
                "zorder":-1,
                },
            "grid_kwargs":{
                "axis":"x",
                },
            "shaded_intervals":usdm_date_intervals,
            },
        show=True,
        )
