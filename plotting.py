import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def plot_lines(domain:list, ylines:list, fig_path:Path=None,
               labels:list=[], multi_domain=False, plot_spec={},
               show:bool=False):
    """
    Plot a list of 1-d lines that share a domain and codomain.

    :@param domain: 1-d numpy array describing the common domain
    :@param ylines: list of at least 1 1-d array of data values to plot, which
            must be the same size as the domain array.
    :@param fig_path: Path to the location to store the figure. If None,
            doesn't store an image.
    :@param labels: list of string labels to include in a legend describing
            each line. If fewer labels than lines are provided, the labels
            will apply to the first of the lines.
    :@param multi_domain: If True, domain is assumed to be a list of domains
            corresponding to each of the ylines
    :@param plot_spec: Dictionary of plot options see the geo_plot module
            for plot_spec options, but the defaults are safe.
    :@param show: if True, shows the image in the matplotlib Agg client.
    """
    plt.clf()
    # Merge provided plot_spec with un-provided default values
    ps = {"xscale":"linear", "legend_font_size":8, "legend_ncols":1,
            "yscale":"linear", "date_format":"%Y-%m-%d",
            "colors":None, "linestyle":None}
    ps.update(plot_spec)

    if multi_domain:
        assert len(domain) == len(ylines)

    # Plot each
    fig, ax = plt.subplots(figsize=ps.get("fig_size"))

    if ps.get("zero_axis"):
        ax.axhline(0, color="black")
    if ps.get("zero_yaxis"):
        ax.axvline(0, color="black")

    colors = ps.get("colors")
    if colors:
        assert len(ylines)<=len(colors)
        if ps.get("cmap"):
            print(f"Warning: colors overriding cmap configuration: {cmap}")

    if cmap:=ps.get("cmap"):
        colors = plt.get_cmap(ps.get("cmap"), len(ylines)+2)
        colors = [colors(c) for c in range(1, len(ylines)+1)]

    linestyle = ps.get("linestyle")
    if linestyle:
        assert len(ylines)<=len(linestyle)
    for i in range(len(ylines)):
        if multi_domain:
            cur_domain = domain[i]
        else:
            cur_domain = domain
        ax.plot(cur_domain, ylines[i],
                label=labels[i] if len(labels) else "",
                linewidth=ps.get("line_width"),
                linestyle=None if not linestyle else linestyle[i],
                color=None if not colors else colors[i])

    ax.set_xlabel(ps.get("xlabel"),
            fontsize=ps.get("label_size"))
    ax.set_ylabel(ps.get("ylabel"),
            fontsize=ps.get("label_size"))
    ax.set_title(ps.get("title"),
            fontsize=ps.get("title_size"))
    if ps.get("xrange"):
        ax.set_xlim(*ps.get("xrange"))
    if ps.get("yrange"):
        ax.set_ylim(*ps.get("yrange"))
    ax.set_xscale(ps.get("xscale"))
    ax.set_yscale(ps.get("yscale"))

    if type(cur_domain[0])==datetime:
        ax.xaxis.set_major_formatter(
                mdates.DateFormatter(ps.get("date_format")))
        if ps.get("time_locator"):
            ax.xaxis.set_major_locator({
                "minute":mdates.MinuteLocator,
                "day":mdates.DayLocator,
                "weekday":mdates.WeekdayLocator,
                "month":mdates.MonthLocator,
                }[ps.get("time_locator")](
                    interval=ps.get("time_locator_interval")
                    ))

    if ps.get("xtick_rotation"):
        plt.tick_params(axis="x", **{"labelrotation":ps.get(
            "xtick_rotation")})
    if ps.get("ytick_rotation"):
        plt.tick_params(axis="y", **{"labelrotation":ps.get(
            "ytick_rotation")})
    if len(labels):
        plt.legend(fontsize=ps.get("legend_font_size"),
                   ncol=ps.get("legend_ncols"))
    if ps.get("xtick_align"):
        plt.setp(ax.get_xticklabels(),
                horizontalalignment=ps.get("xtick_align"))

    if ps.get("grid"):
        plt.grid()
    if show:
        plt.show()
    if not fig_path is None:
        fig.savefig(fig_path, bbox_inches="tight", dpi=ps.get("dpi"))
    plt.close()
    return


def plot_lines_multiy(domain, ylines, plot_spec={},
        show=False, fig_path=None):
    """
    Lines on the same plot sharing an x domain, but able to have one of
    multiple independent vertical axes.

    :@param ylines: list of lists such that the first level contains a sub-list
        for each group of lines that share a domain
    """
    ps = {"fig_size":None, "dpi":200, "spine_increment":.1,
          "date_format":"%Y-%m-%d", "xtick_rotation":30, "ytick_rotation":45,
          "ylabel_position":"top", "ylabel_pad":.2,}
    ps.update(plot_spec)
    if len(domain) != len(ylines[0][0]):
        raise ValueError(
                "Length of 'domain' must match length of each time series.")

    fig,host = plt.subplots(figsize=ps.get("fig_size"))
    fig.subplots_adjust(left=0.2 + ps.get("spine_increment") \
            * (len(ylines) - 1))

    nlines = sum(len(yl) for yl in ylines)

    axes = [host]
    colors = ps.get("line_colors", ["C" + str(i) for i in range(nlines)])
    line_labels = ps.get("line_labels", [""] * nlines)
    line_style = ps.get("line_style", ["-"] * nlines)
    y_labels = ps.get("y_labels", [""] * len(ylines))
    y_ranges = ps.get("y_ranges", [None] * len(ylines))
    y_scales = ps.get("y_scales", ["linear"] * len(ylines))

    assert len(colors) == nlines
    assert len(line_labels) == nlines
    assert len(line_style) == nlines
    assert len(y_labels) == len(ylines)
    assert len(y_ranges) == len(ylines)
    assert len(y_scales) == len(ylines)

    ## Create additional y-axes on the left, offset horizontally
    for i in range(1, len(ylines)):
        ax = host.twinx()
        #ax.spines["left"] = ax.spines["right"]
        ax.yaxis.set_label_position("left")
        ax.yaxis.set_ticks_position("left")
        ax.spines["left"].set_position(
                ("axes", -1*ps.get("spine_increment") * i))
        ax.set_yscale(y_scales[i])
        #ax.spines["right"].set_visible(False)
        axes.append(ax)

    ## Plot each series
    lc = 0
    lines = []
    for i, (ax, series) in enumerate(zip(axes, ylines)):
        for s in series:
            lines.append(ax.plot(
                domain, s, color=colors[lc], label=line_labels[lc],
                linestyle=line_style[lc]))
            lc += 1
        ax.set_ylabel(
                y_labels[i], color=colors[lc-1],
                fontsize=ps.get("label_fontsize"),
                loc=ps.get("ylabel_position"))
        ax.tick_params(axis="y", colors=colors[lc-1],)
        ax.tick_params(axis="y", rotation=ps.get("ytick_rotation"))
        if y_ranges[i] is not None:
            ax.set_ylim(y_ranges[i])


    host.grid(ps.get("grid", False), **ps.get("grid_kwargs", {}))
    host.set_xlabel(ps.get("xlabel", ""),
            fontsize=ps.get("label_fontsize"))
    if isinstance(domain[0], datetime):
        host.xaxis.set_major_formatter(
                mdates.DateFormatter(ps.get("date_format")))
    if ps.get("xticks"):
        #print(ps.get("xticks"), len(ps.get("xticks")))
        plt.xticks(range(len(domain)), ps.get("xticks"))
    host.tick_params(axis="x", rotation=ps.get("xtick_rotation"))
    host.legend(
            handles=[l[0] for l in lines],
            labels=line_labels,
            ncol=ps.get("legend_ncols", 1),
            fontsize=ps.get("legend_fontsize"),
            )
    if plot_spec.get("time_locator"):
        host.xaxis.set_major_locator({
            "minute":mdates.MinuteLocator,
            "day":mdates.DayLocator,
            "weekday":mdates.WeekdayLocator,
            "month":mdates.MonthLocator,
            }[plot_spec.get("time_locator")](
                interval=plot_spec.get("time_locator_interval")
                ))
    if plot_spec.get("xtick_align"):
        plt.setp(host.get_xticklabels(),
                horizontalalignment=plot_spec.get("xtick_align"))

    if ps.get("zero_axis"):
        host.axhline(0, color="black")

    plt.title(ps.get("title", ""),
            fontdict={"fontsize":ps.get("title_fontsize")})
    fig.tight_layout()
    if show:
        plt.show()
    if not fig_path is None:
        fig.savefig(fig_path, bbox_inches="tight", dpi=plot_spec.get("dpi"))
    plt.clf()
    plt.close()
