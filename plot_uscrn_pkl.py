import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool

def mp_plot_lines(args):
    try:
        return args,plot_lines(**args)
    except Exception as e:
        print(f"Error plotting {args.get('out_path')}")
        return args,False

def plot_lines(xlines, ylines, plot_spec=None, out_path:Path=None, show=False):
    """ """
    ps = {
        "legend":True,
        }
    ps.update(plot_spec)
    if "fig_size" in ps.keys():
        fig,ax = plt.subplots(figsize=ps["fig_size"])
    else:
        fig,ax = plt.subplots()

    assert len(xlines) == len(ylines)
    for i,(x,y) in enumerate(zip(xlines,ylines)):
        options = {}
        if "label" in ps.keys():
            options["label"] = ps["label"][i]
        if "color" in ps.keys():
            options["color"] = ps["color"][i]
        if "linewidth" in ps.keys():
            options["lw"] = ps["linewidth"]
        ax.plot(x, y, **options)

    if "title" in ps:
        ax.set_title(ps["title"])
    if "xlabel" in ps:
        ax.set_xlabel(ps["xlabel"])
    if "ylabel" in ps:
        ax.set_ylabel(ps["ylabel"])
    if ps.get("legend", False):
        ax.legend(**ps.get("legend_kwargs", {}))
    if out_path is not None:
        fig.savefig(out_path.as_posix(), dpi=ps.get("dpi", 100))
    if show:
        plt.show()
    return True

if __name__=="__main__":
    data_dir = Path("/rstor/mdodson/nidis_spie/")
    pkl_path = data_dir.joinpath("pkls/crn_hourly_por.pkl")
    fig_dir = Path("/rhome/mdodson/NIDIS-SPIe/figures")

    plot_feats = [
        "temp-end", "temp-max", "temp-min", "temp-avg",
        "precip",
        "swdown-avg", "swdown-max", "swdown-min",
        "stemp-avg", "stemp-max", "stemp-min",
        "rh-avg",
        "sm-2in", "sm-4in", "sm-8in", "sm-20in", "sm-40in",
        "st-2in", "st-4in", "st-8in", "st-20in", "st-40in",
        ]
    print_flags = [
        "flag-swdown-avg", "flag-swdown-max", "flag-swdown-min",
        "flag-stemp-avg", "flag-stemp-max", "flag-stemp-min",
        "flag-rh-avg",
        ]
    #plot_stations = None ## plot all stations
    plot_stations = [
        "AL_Gadsden_19_N",
        "AL_Fairhope_3_NE",
        ]
    nworkers = 20

    r = pkl.load(pkl_path.open("rb"))
    plot_arrays = {
        "hist":{},
        "timeseries":{},
        }
    for sid in r.keys():
        if plot_stations is not None and sid not in plot_stations:
            continue
        for k,a in r[sid].items():
            if k in print_flags:
                val,count = np.unique(a, return_counts=True)
                print(f"flags {sid} {k}:", " ".join([
                    f"{v}:{int(c)}" for v,c in zip(val,count)
                    ]))
            if k in plot_feats:
                if k not in plot_arrays["hist"].keys():
                    plot_arrays["hist"][k] = {
                        "xlines":[],
                        "ylines":[],
                        "plot_spec":{
                            "title":k,
                            "xlabel":k,
                            "ylabel":"count",
                            "label":[],
                            "dpi":200,
                            "legend_kwargs":{
                                "ncol":3,
                                "fontsize":4,
                                },
                            "linewidth":1.5,
                            },
                        }
                    plot_arrays["timeseries"][k] = {
                        "xlines":[],
                        "ylines":[],
                        "plot_spec":{
                            "title":k,
                            "xlabel":"observation time",
                            "ylabel":k,
                            "label":[],
                            "dpi":200,
                            "legend_kwargs":{
                                "ncol":3,
                                "fontsize":4,
                                },
                            "fig_size":(16,8),
                            "linewidth":1.5,
                            },
                        }
                val,count = np.unique(a, return_counts=True)
                if k == "precip":
                    count[val==0.] = 0
                if k in ["swdown-avg", "swdown-min", "swdown-max"]:
                    count[val<=1.] = 0
                    val[val<=1.] = np.nan
                plot_arrays["hist"][k]["xlines"].append(val)
                plot_arrays["hist"][k]["ylines"].append(count)
                plot_arrays["hist"][k]["plot_spec"]["label"].append(sid)
                plot_arrays["timeseries"][k]["xlines"].append(
                        r[sid]["time-utc"].astype(object)
                        )
                plot_arrays["timeseries"][k]["ylines"].append(a)
                plot_arrays["timeseries"][k]["plot_spec"]["label"].append(sid)

    args = [{
        **pargs,
        "out_path":fig_dir.joinpath(f"{ptype}_{pfeat}.png"),
        }
        for ptype,pdict in plot_arrays.items()
        for pfeat,pargs in pdict.items()
        ]

    with Pool(nworkers) as pool:
        for a,r in pool.imap_unordered(mp_plot_lines, args):
            pass
    '''
    for ptype,pdict in plot_arrays.items():
        for pfeat,pargs in pdict.items():
            try:
                plot_lines(
                    **pargs,
                    out_path=fig_dir.joinpath(f"{ptype}_{pfeat}.png")
                    )
            except Exception as e:
                print(f"Error plotting {ptype} {pfeat}")
                print(e)
    '''
