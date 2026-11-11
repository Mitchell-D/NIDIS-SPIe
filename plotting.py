import numpy as np
import matplotlib
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

def plot_nested_bars(data_dict:dict, labels:dict={}, plot_error_bars=False,
        bar_colors:list=None, plot_spec:dict={}, show=False, fig_path=None,
        group_order:list=None, bar_order:list=None):
    """
    Plot a bar graph of metrics nested 2 levels deep, with optional error bars.

    :@param data_dict: Dict nested 2 layers deep, where the first layer
        identifies the bar grouping, and the second layer identifies the
        subcategory of a data point within each bar grouping. The second layer
        should map to a number if plot_error_bars is False, or a 2-tuple of
        [data, error_bar_magnitude] if plot_error_bars is True.
    :@param labels: dict of optional labels to replace data_dict keys in the
        legend or x-axis, if the data_dict key matches a labels key
    :@param plot_error_bars: Determines whether to expect a 2-tuple including
        error bar data per bar, as specified above
    :@param plot_spec: Dict of configuration options for the plot
    """
    ps = {
        "xlabel":"", "ylabel":"", "text_size":12, "title":"", "dpi":80,
        "figsize":(12,12), "legend_ncols":1, "line_opacity":1,
        "cmap":"hsv", "label_fontsize":14, "title_fontsize":20,
        "legend_fontsize":14, "bar_spacing":1, "hline":None, "hline_kwargs":{},
        "yscale":"linear",
        }
    ps.update(plot_spec)
    fig,ax = plt.subplots()

    ## group keys
    gkeys = list(data_dict.keys()) if group_order is None else group_order
    ngroups = len(gkeys)
    group_starts = np.arange(ngroups)
    assert all(set(data_dict[k])==set(data_dict[gkeys[0]]) for k in gkeys[1:])
    ## bar keys
    bkeys = list(data_dict[gkeys[0]]) if bar_order is None else bar_order
    cm = matplotlib.cm.get_cmap(ps.get("cmap"), len(bkeys))

    bwidth = ps.get("bar_width", 1/(len(bkeys)+ps.get("bar_spacing")))

    bar_plots = []
    err_plots = []
    offset = 0
    for bix,bk in enumerate(bkeys):
        if plot_error_bars:
            tmp_data = [data_dict[gk][bk][0] for gk in gkeys]
            tmp_err = [data_dict[gk][bk][1] for gk in gkeys]
            ## if bounds are specified separately for (lower, upper)
            if isinstance(tmp_err[0], (list, tuple)):
                tmp_err = list(zip(*tmp_err))
        else:
            tmp_data = [data_dict[gk][bk] for gk in gkeys]
            tmp_err = None
        bar_plots.append(ax.bar(
                group_starts + offset,
                tmp_data,
                color=cm(bix) if bar_colors is None else bar_colors[bix],
                width=bwidth,
                label=labels.get(bk,bk),
                ))
        if plot_error_bars:
            err_plots.append(ax.errorbar(
                    group_starts + offset,
                    tmp_data,
                    yerr=tmp_err,
                    fmt=ps.get("err_fmt","o"),
                    color=ps.get("err_color","black"),
                    ))
        offset += bwidth

    if ps.get("hline"):
        plt.axhline(ps.get("hline"), **ps.get("hline_kwargs"))

    ax.set_xticks(
            group_starts+bwidth/2, [labels.get(gk,gk) for gk in gkeys],
            rotation=ps.get("xtick_rotation", 0),
            fontsize=ps.get("xtick_fontsize", ps.get("label_fontsize")),
            )
    ax.set_yscale(ps.get("yscale"))
    plt.yticks(fontsize=ps.get("label_fontsize"))

    ax.set_xlabel(ps.get("xlabel"), fontsize=ps.get("label_fontsize"))
    ax.set_ylabel(ps.get("ylabel"), fontsize=ps.get("label_fontsize"))
    if not ps.get("ylim") is None:
        ax.set_ylim(*ps.get("ylim"))
    ax.set_title(ps.get("title"), fontsize=ps.get("title_fontsize"))
    ax.legend(ncol=ps.get("legend_ncols"), fontsize=ps.get("legend_fontsize"))

    if show:
        plt.show()
    if fig_path:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(),bbox_inches="tight",dpi=ps.get("dpi"))
    plt.close()
    return

def plot_bars_with_lines(
    domain, bars, lines, fig_path=None, show=True,
    line_labels=None, plot_spec=None
    ):
    """
    Plot a bar graph with multiple overlaid line plots sharing an x-axis.

    Parameters
    ----------
    domain : 1d array-like
        Shared x-axis values.
    bars : 1d array-like
        Bar magnitudes.
    lines : list of 1d array-like
        Line series to overlay on the bar graph.
    fig_path : str or Path, optional
        Path to save the figure. Defaults to None.
    show : bool, default=True
        Whether to display the plot interactively.
    line_labels : list of str, optional
        Legend labels for each line series.
    plot_spec : dict, optional
        Dictionary of cosmetic plotting options.


    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    ax : matplotlib.axes.Axes
        Primary axes containing the bars and lines (or bars only
        when secondary_y=True).
    """

    # Convert inputs to numpy arrays
    domain = np.asarray(domain)
    bars = np.asarray(bars)
    lines = [np.asarray(line) for line in lines]

    # Default plotting specifications
    ps = {
        "title": None,
        "xlabel": None,
        "ylabel": None,
        "yscale": "linear",
        "figsize": (12, 6),
        "bar_color": "steelblue",
        "bar_alpha": 0.6,
        "bar_width": None,
        "line_cmap": "tab10",
        "line_width": 2.0,
        "line_styles": None,
        "line_markers": None,
        "grid": True,
        "legend": True,
        "legend_loc": "best",
        "dpi": 300,
        "secondary_y": False,
        "secondary_yscale": "linear",
    }

    if plot_spec is not None:
        ps.update(plot_spec)

    # Validate inputs
    if domain.ndim != 1 or bars.ndim != 1:
        raise ValueError("domain and bars must be 1D arrays.")

    if len(domain) != len(bars):
        raise ValueError("domain and bars must have the same length.")

    if not lines:
        lines = []

    for i, line in enumerate(lines):
        if line.ndim != 1:
            raise ValueError(f"Line {i} must be a 1D array.")
        if len(line) != len(domain):
            raise ValueError(
                f"Line {i} has length {len(line)}, "
                f"but domain has length {len(domain)}."
            )

    if line_labels is not None and len(line_labels) != len(lines):
        raise ValueError("line_labels must have one label per line.")

    # Set default labels
    if line_labels is None:
        line_labels = [None] * len(lines)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=ps["figsize"])

    # Plot bars
    if ps["bar_width"] is not None:
        bar_width = ps["bar_width"]
    elif len(domain) > 1:
        bar_width = 0.8 * np.min(np.diff(np.sort(domain)))
    else:
        bar_width = 0.8

    ax.bar(domain, bars, width=bar_width, color=ps["bar_color"],
           alpha=ps["bar_alpha"], label="Bars", zorder=1,)

    # Configure line colors
    cmap = plt.get_cmap(ps["line_cmap"])
    n_lines = len(lines)

    if n_lines > 0:
        colors = cmap(np.linspace(0, 1, n_lines))
    else:
        colors = []

    # Choose primary or secondary axes for lines
    if ps["secondary_y"]:
        line_ax = ax.twinx()
        line_ax.set_yscale(ps["secondary_yscale"])
    else:
        line_ax = ax

    # Plot lines
    for i, (line, label, color) in enumerate(
        zip(lines, line_labels, colors)
    ):
        style = (
            ps["line_styles"][i]
            if ps["line_styles"] is not None
            else "-"
        )
        marker = (
            ps["line_markers"][i]
            if ps["line_markers"] is not None
            else None
        )

        line_ax.plot(
            domain,
            line,
            color=color,
            linewidth=ps["line_width"],
            linestyle=style,
            marker=marker,
            label=label or f"Line {i + 1}",
            zorder=3,
        )

    # Configure axes
    ax.set_xlabel(ps["xlabel"])
    ax.set_ylabel(ps["ylabel"])
    ax.set_yscale(ps["yscale"])

    if line_ax is not ax:
        line_ax.set_ylabel(ps.get("secondary_ylabel", "Line values"))

    ax.set_title(ps["title"])

    if ps["grid"]:
        ax.grid(True, alpha=0.3, zorder=0)

    # Combine legends from both axes if necessary
    if ps["legend"]:
        handles, labels = ax.get_legend_handles_labels()

        if line_ax is not ax:
            line_handles, line_names = (
                line_ax.get_legend_handles_labels()
            )
            handles += line_handles
            labels += line_names

        if handles:
            ax.legend(handles, labels, loc=ps["legend_loc"])

    fig.tight_layout()

    # Save figure if requested
    if fig_path is not None:
        fig.savefig(
            Path(fig_path),
            dpi=ps["dpi"],
            bbox_inches="tight",
        )

    # Display figure if requested
    if show:
        plt.show()

    return fig, ax
