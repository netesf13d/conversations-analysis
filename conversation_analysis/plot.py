# -*- coding: utf-8 -*-
"""
This module implements functions to plot various quantities obtained in the
form of pandas Dataframes from conversation analysis (using the
<conversations.analysis> module).

It exposes the following functionality:
    - <pie_plot>
      Make pie charts for each column of a dataframe.
    - <bar_plot>
      Stacked bar plot of the selected column of a dataframe.
    - <stack_plot>
      Stacked area plot or streamgraph of the selected column of a dataframe.
"""

from itertools import pairwise

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch


# =============================================================================
# 
# =============================================================================

# First 24 xkcd colors from https://xkcd.com/color/rgb/
XKCD_COLORS = [
    '#7e1e9c', '#15b01a', '#0343df', '#ff81c0', '#653700', '#e50000',
    '#95d0fc', '#029386', '#f97306', '#96f97b', '#c20078', '#ffff14',
    '#75bbfd', '#929591', '#0cff0c', '#bf77f6', '#9a0eea', '#033500',
    '#06c2ac', '#c79fef', '#00035b', '#d1b26f', '#00ffff', '#06470c',
    ]


DEFAULT_LEGEND_KW = {
    'loc': 'upper center',
    'fontsize': 9
    }


# =============================================================================
# Plot utils
# =============================================================================

timedeltas = {'year': 31536000_000000000, 'month': 2592000_000000000,
              'day': 86400_000000000, 'hour': 3600_000000000,
              'minute': 60_000000000, 'second': 1_000000000,
              'microsecond': 1000}

divisors = {
    'year': [1, 2, 5, 10, 20, 50, 100],
    'month': [1, 2, 3, 6, 12],
    'day': [1, 2, 5, 10, 15, 30],
    'hour': [1, 2, 4, 6, 12, 24],
    'minute': [1, 2, 5, 10, 30, 60],
    'second': [1, 2, 5, 10, 30, 60],
    'microsecond': sum([[10**i, 2*10**i, 5*10**i] for i in range(6)], start=[])
    }

freq_names =  {'year': 'Y', 'month': 'M', 'day': 'D',
               'hour': 'h', 'minute': 'min', 'second': 's',
               'microsecond': 'us'}


def _get_ticks(t0: int, t1: int,
               nticks: int,
               mode: str,
               strftime: str = "%Y-%m-%d"
               )-> tuple[list[int], list[str], str]:
    """
    Return an appropriate set of x-axis ticks, tick labels and x-axis label
    corresponding to intuitive time intervals.

    Parameters
    ----------
    t0, t1 : int, int
        Start and stop times of the x-axis.
    nticks : int
        Number of ticks to position. The actual number of ticks returned is
        <= nticks, and corresponds to intuitive intervals (eg months,
        semesters, etc).
    mode : str {'abs', 'rel'}
        Select whether the x-axis corresponds to an absolute (timestamp) or
        relative (timedelta) time.
    strftime : str, optional
        Time-formatting string for the x-labels. Only applies if mode is 'abs'.
        The default is "%Y-%m-%d".

    Returns
    -------
    ticks : list[int]
        The tick positions.
    tick_labels : list[str]
        The tick labels.
    xlabel : str
        The x-axis label.

    """
    dt = t1 - t0
    unit = next(it:=iter(timedeltas))
    while (n := dt / timedeltas[unit]) < nticks - 1:
        unit = next(it)
    for div in divisors[unit]:
        if n / div < nticks:
            break
    n = round(dt / timedeltas[unit] / div) + 1

    if mode == 'abs':
        t0_ = pd.Timestamp(t0)
        tick0 = t0_.to_period(freq_names[unit]).to_timestamp()
        if tick0 < t0_:
            tick0_ = tick0 + pd.offsets.DateOffset(**{unit + 's': 1})
            if tick0_ - t0_ < t0_ - tick0:
                tick0 = tick0_
    
        ticks = [tick0 + pd.offsets.DateOffset(**{unit+'s': i*div})
                 for i in range(n)]
        tick_labels = [tick.strftime(strftime) for tick in ticks]
        ticks = pd.DatetimeIndex(ticks).astype('int64')
        
        return list(ticks), tick_labels, "date"
    
    if mode == 'rel':
        ticks = [pd.Timedelta(t0 + i * div * timedeltas[unit])
                 for i in range(n)]
        for xlabel, val in ticks[-2].components._asdict().items():
            if val > 0:
                break
        tick_labels = [str(getattr(tick.components, xlabel)) for tick in ticks]
        ticks = pd.TimedeltaIndex(ticks).astype('int64')
        
        return list(ticks), tick_labels, xlabel


def legend_params(dpi: float, legend_kw: dict)-> tuple[tuple[int, int], dict]:
    """
    Compute the size of the figure's legend object from the parameters passed
    to <Figure.legend>.

    Parameters
    ----------
    dpi : float
        Figure dots per inch.
    legend_kw : dict
        A dict of matplotlib Figure.legend kwargs to update the defaults. It
        must also contain the legend labels.

    Returns
    -------
    (tuple[int, int]
        The legend bbox size in inches.
    dict
        The legend kwargs.

    """
    # Set legend kwargs
    lkw = DEFAULT_LEGEND_KW | legend_kw
    # get legend bbox size in pixels
    fig = plt.figure(dpi=dpi)
    legend = fig.legend(handles=[Patch()]*len(lkw['labels']), **lkw)
    fig.canvas.draw()
    dx, dy = legend.legendPatch._width, legend.legendPatch._height
    plt.close(fig)
    
    return (dx/dpi, dy/dpi), lkw


# =============================================================================
# Plot functions
# =============================================================================

def pie_plot(dataframe: pd.DataFrame,
             *,
             color_palette: list = XKCD_COLORS,
             legend_kw: dict | None = None,
             dpi: float = 200,
             )-> tuple[plt.Figure, np.ndarray]:
    """
    Pie charts of all the columns the dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe containing the various quantities to display as pie charts.
        - Rows : Index, represent the group.
        - Columns : Index, represent the various quantities.
    color_palette : list, optional
        Color palette used for the different groups.
        The default is XKCD_COLORS.
    legend_kw : dict | None, optional
        Legend keyword arguments passed to the figure `legend` method.
        Can be used for instance to set a FontProperties object to use another
        font to display emojis (the default font DejaVu Sans does not).
        The default is None.
    dpi : float, optional
        Figure resolution in dots-per-inch. The default is 200.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure.
    axs : np.ndarray[matplotlib.axes._axes.Axes]
        Array containing the Axes.

    """
    # Legend setup
    lkw = {'labels': dataframe.index.to_list(),
           'ncols': min(4, len(dataframe))}
    if legend_kw is not None:
        lkw.update(legend_kw)
    (dx, dy), lkw = legend_params(dpi, lkw)
    
    # Figure setup
    nplots = len(dataframe.columns)
    nrows, ncols = (nplots + 2) // 3, min(nplots, 3)
    figsize = (max(dx+0.3, 3*ncols), 3.1*nrows + dy + 0.6)
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(
        nrows, ncols, hspace=0.05, wspace=0.07,
        left=0.015, bottom=0.08*nrows/figsize[1],
        right=0.985, top=3.1*nrows/figsize[1],
        figure=fig
        )
    axs = gs.subplots(sharex=True, sharey=True, squeeze=False,
                      subplot_kw={'frame_on': False})
    # Pie plot
    for quantity, ax in zip(dataframe.columns, axs.ravel()):
        data = dataframe[quantity]
        ax.set_title(f"{quantity} ({data.sum()})", y=1., pad=4,
                     fontweight='bold')
        
        if data.sum() == 0:
            ax.set_frame_on(True)
            continue
        # Plot
        wedges, _ = ax.pie(data, colors=color_palette)
        # Percent and values
        pct = data / data.sum()
        labels = [f"{p*100:.1f}%\n({d})" for p, d in zip(pct, data)]
        thetas = [np.deg2rad((w.theta2 + w.theta1) / 2) for w in wedges]
        xs = np.array([(0.72*x if pct.iloc[i] > 0.07 else 1.18*x)
                       for i, x in enumerate(np.cos(thetas))])
        ys = np.array([(0.72*y if pct.iloc[i] > 0.07 else 1.15*y)
                       for i, y in enumerate(np.sin(thetas))])
        for i, label in enumerate(labels):
            ax.annotate(label, xy=(xs[i], ys[i]),
                        ha='center', va='center', fontsize=9)
    
    fig.legend(handles=wedges, **lkw)
    
    return fig, axs


def bar_plot(dataframe: pd.DataFrame,
             print_percents: bool = False,
             *,
             color_palette: list = XKCD_COLORS,
             legend_kw: dict | None = None,
             dpi: float = 200,
             )-> tuple[plt.Figure, plt.Axes]:
    """
    Bar plot of the selected column (`quantity`) of the dataframe.
    The values for each group are stacked and represented vs the row index.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe containing the to display as a bar plot.
        - Rows : Index or MultiIndex, represent the binning entries.
        - Columns : 2-levels MultiIndex, represent (quantity, group)
    print_percents : bool, optional
        If true, displays percents and absolute values in bars which height is
        large enough. The text might nevertheless overflow horizontally if the
        bar width is small.
        The default is False.
    color_palette : list, optional
        Color palette used for the different groups.
        The default is XKCD_COLORS.
    legend_kw : dict | None, optional
        Legend keyword arguments passed to the figure `legend` method.
        Can be used for instance to set a FontProperties object to use another
        font to display emojis (the default font DejaVu Sans does not).
        The default is None.
    dpi : float, optional
        Figure resolution in dots-per-inch. The default is 200.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure.
    ax : matplotlib.axes._axes.Axes
        The Axes.

    """
    # Preprocessing
    tot_counts = dataframe.T.sum()
    max_counts = max(tot_counts)
    percents = dataframe.div(tot_counts, axis=0) * 100
    
    # Legend setup
    lkw = {'labels': dataframe.columns.to_list(),
           'ncols': min(3, len(dataframe.columns))}
    if legend_kw is not None:
        lkw.update(legend_kw)
    (dx, dy), lkw = legend_params(dpi, lkw)
    # Figure setup
    figsize = (7, 4 + dy + 0.38)
    y_ax = 4/figsize[1]
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes(rect=(l:=0.10, b:=0.12*y_ax, 0.98 - l, y_ax - b))
    # Bar plot
    handles = []
    bottom = np.zeros(len(dataframe))
    for i, (group, counts) in enumerate(dataframe.items()):
        bars = ax.bar(np.arange(len(dataframe)), counts, bottom=bottom,
                      color=color_palette[i])
        handles.append(bars)
        bottom += counts
        # bar labels
        if print_percents:
            for pct, bar, count in zip(percents.iloc[:, i], bars, counts):
                if count/max_counts > 0.075: # print only if enough space
                    ax.annotate(f"{pct:.1f}%\n({count})", xy=bar.get_center(),
                                ha='center', va='center', fontsize=7.5)
    
    # Set x ticks
    ax.set_xlim(-0.6, len(dataframe) - 0.4)
    if dataframe.index.nlevels == 1:
        idx = dataframe.index.to_list()
    else:
        idx = dataframe.index.get_level_values(0).to_list()
    tick_labels = np.unique(idx)
    ticks = [idx.index(tl) for tl in tick_labels]
    ax.set_xticks(ticks, tick_labels, minor=False)
    ax.set_xticks([i for i in range(len(dataframe)) if i not in set(ticks)],
                  minor=True)
    
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
    ax.set_ylabel("counts", fontsize=11)
    ax.set_xlabel(dataframe.index.names[0], fontsize=11)
    #ax.set_title(f"{quantity} (tot. {df.sum().sum()})", fontweight='bold')
    
    
    fig.legend(handles=handles, **lkw)
    
    return fig, ax


def stack_plot(dataframe: pd.DataFrame,
               baseline: str = 'wiggle',
               timescale: float | str = 'day',
               xlabel_strftime: str = '%Y-%m-%d',
               *,
               color_palette: list = XKCD_COLORS,
               legend_kw: dict | None = None,
               dpi: float = 200,
               )-> tuple[plt.Figure, plt.Axes]:
    """
    Stacked area plot or streamgraph of the selected column (`quantity`) of
    the dataframe. The values for each group are stacked.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe containing the to display as a bar plot.
        - Rows : MultiIndex, represent a Timestamp or Timedelta.
        - Columns : 2-levels MultiIndex, represent (quantity, group)

    baseline : str {'zero', 'sym', 'wiggle', 'weighted_wiggle'}, optional
        Method used to calculate the baseline.
        See matplotlib.axes.Axes.stackplot documentation for details.
        The default is 'wiggle'.
    timescale : float | str, optional
        Integration timescale of the counts.
        - str: {'day', 'hour', 'minute', 'second', 'microsecond'}
            For instance, if timescale is set as 'day', the ordinate is
            expressed in counts per day.
        - float: timedelta in seconds
            
        The default is 'day'.
    xlabel_strftime : str, optional
        String format for the x-label. Only used for absolute date plot.
        The default is '%Y-%m-%d'.
    color_palette : list, optional
        Color palette used for the different groups.
        The default is XKCD_COLORS.
    legend_kw : dict | None, optional
        Legend keyword arguments passed to the figure `legend` method.
        Can be used for instance to set a FontProperties object to use another
        font to display emojis (the default font DejaVu Sans does not).
        The default is None.
    dpi : float, optional
        Figure resolution in dots-per-inch. The default is 200.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure.
    ax : matplotlib.axes._axes.Axes
        The Axes.

    """
    ## Preprocessing: compute ticks
    if 'timestamp' in dataframe.index.names:
        times = dataframe.index.get_level_values('timestamp')
        major_ticks, tick_labels, xlabel = _get_ticks(
            times[0], times[-1], 9, mode='abs', strftime=xlabel_strftime)
    elif 'timedelta' in dataframe.index.names:
        times = dataframe.index.get_level_values('timedelta')
        major_ticks, tick_labels, xlabel = _get_ticks(
            times[0], times[-1], 9, mode='rel')
    # Set minor ticks
    minor_ticks = []
    for t0, t1 in pairwise(major_ticks):
        minor_ticks += _get_ticks(t0, t1, 7, mode='rel')[0]
    # Convert to seconds
    t = (times - times[0]) * 1e-9
    major_ticks = (np.array(major_ticks) - times[0]) * 1e-9
    minor_ticks = (np.array(minor_ticks) - times[0]) * 1e-9
    
    ## Preprocessing: rescale data
    dt = pd.Timedelta(times[1] - times[0]).value
    if isinstance(timescale, str):
        scale = timedeltas[timescale] / dt
        scale_str = timescale
    else:
        scale = timescale * 1e9 / dt
        scale_str = f"{timescale:.4g} s"
    df = dataframe * scale
    
    # Legend setup
    lkw = {'labels': df.columns.to_list(),
           'ncols': min(3, len(df.columns))}
    if legend_kw is not None:
        lkw.update(legend_kw)
    (dx, dy), lkw = legend_params(dpi, lkw)
    # Figure setup
    figsize = (7, 4 + dy + 0.38)
    y_ax = 4/figsize[1]
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes(rect=(l:=0.10, b:=0.12*y_ax, 0.98 - l, y_ax - b))

    polys = ax.stackplot(t, df.T, baseline=baseline,
                         colors=color_palette)
    # Set x-axis ticks and limits
    ax.set_xlim(0, t[-1])
    ax.set_xticks(major_ticks, tick_labels, minor=False)
    ax.set_xticks(minor_ticks, minor=True)

    ax.grid(visible=True)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
    ax.set_ylabel(f"counts / {scale_str}", fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    fig.legend(handles=polys, **lkw)
    
    return fig, ax