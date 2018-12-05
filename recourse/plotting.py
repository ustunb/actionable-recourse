from math import atan2, degrees
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.dates import date2num

import seaborn as sns
sns.set(style="white", palette="muted", color_codes = True)

#### GENERAL SETTINGS
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rc('legend',fontsize = 20)



#### PLOTS

def create_data_summary_plot(data_df, subplot_side_length = 3.0, subplot_font_scale = 0.5, max_title_length = 30):

    df = pd.DataFrame(data_df)

    # determine number of plots
    n_plots = len(data_df.columns)
    n_cols = int(np.round(np.sqrt(n_plots)))
    n_rows = int(np.round(n_plots / n_cols))
    assert n_cols * n_rows >= n_plots

    colnames = df.columns.tolist()
    plot_titles = [c[:max_title_length] for c in colnames]

    f, axarr = plt.subplots(n_rows, n_cols, figsize=(subplot_side_length * n_rows, subplot_side_length * n_cols), sharex = False, sharey = True)
    sns.despine(left = True)

    n = 0
    with sns.plotting_context(font_scale = subplot_font_scale):
        for i in range(n_rows):
            for j in range(n_cols):

                ax = axarr[i][j]

                if n < n_plots:

                    bar_color = 'g' if n == 0 else 'b'
                    vals = df[colnames[n]]
                    unique_vals = np.unique(vals)

                    sns.distplot(a = vals, kde = False, color = bar_color, ax = axarr[i][j], hist_kws = {'edgecolor': "k", 'linewidth': 0.5})

                    ax.xaxis.label.set_visible(False)
                    ax.text(.5, .9, plot_titles[n], horizontalalignment = 'center', transform = ax.transAxes, fontsize = 10)

                    if len(unique_vals) == 2:
                        ax.set_xticks(unique_vals)

                    n += 1
                else:
                    ax.set_visible(False)

    plt.tight_layout()
    plt.show()

    return f, axarr


def create_coefficient_path_plot(coefs_df, fig_size = (10, 8), label_coefs = True, label_halign = 'left', **kwargs):

    f = plt.figure(figsize = fig_size)
    ax = f.add_axes([0.15, 0.15, 0.8, 0.8])

    # plot coefficients
    coefs_df.T.plot(legend = False, ax = ax, alpha = 1.0)

    # labels etc
    ax.semilogx()
    ax.set_ylabel("Coefficient Values")
    ax.set_xlabel('$\ell_1$-penalty (Log Scale)')

    # values should be symmetric
    ymin, ymax = ax.get_ylim()
    ymax = np.maximum(np.abs(ymin), np.abs(ymax))
    ax.set_ylim(-ymax, ymax)

    # label each coefficient
    if label_coefs:
        coef_lines = ax.get_lines()
        n_lines = len(coef_lines)
        xmin = ax.get_xlim()[0]
        xvals = [2.5 * xmin] * n_lines
        label_lines(coef_lines, xvals = xvals, align = label_halign, clip_on = False, **kwargs)

    ax.hlines(0, *ax.get_xlim(), linestyle = 'dashed', alpha = 0.5)
    return f, ax


#### HELPER FUNCTIONS


def create_figure(fig_size = (6, 6)):

    f = plt.figure(figsize = fig_size)

    ax = f.add_axes([0.1, 0.15, 0.85, 0.8])
    ax.set_facecolor('white')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis = 'both', which = 'major', direction = 'in', length = 5, width = 2, bottom = True, left = True)

    return f, ax


def fix_font_sizes(ax):

    SMALL_SIZE = 22
    MEDIUM_SIZE = 26

    ax.xaxis.label.set_fontsize(MEDIUM_SIZE)
    ax.yaxis.label.set_fontsize(MEDIUM_SIZE)

    for tick_text in ax.get_xticklabels():
        tick_text.set_fontsize(SMALL_SIZE)

    for tick_text in ax.get_yticklabels():
        tick_text.set_fontsize(SMALL_SIZE)

    return ax


def label_line(line, x, label=None, align=True, max_length = 30, **kwargs):
    # adapted from https://github.com/cphyc/matplotlib-label-lines

    '''Label a single matplotlib line at position x'''
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    # Convert datetime objects to floats
    if isinstance(x, datetime):
        x = date2num(x)

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        # return

    # Find corresponding y co-ordinate and angle of the
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1]) * \
        (x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    label = label[:max_length]

    if align:
        # Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy, dx))

        # Transform to screen co-ordinates
        pt = np.array([x, y]).reshape((1, 2))
        trans_angle = ax.transData.transform_angles(np.array((ang, )), pt)[0]

    else:
        trans_angle = 0

    # Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = False

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x, y, label, rotation=trans_angle, **kwargs)


def label_lines(lines, align=True, xvals=None, **kwargs):
    # adapted from https://github.com/cphyc/matplotlib-label-lines
    '''Label all lines with their respective legends.
    xvals: (xfirst, xlast) or array of position. If a tuple is provided, the
    labels will be located between xfirst and xlast (in the axis units)
    '''
    ax = lines[0].axes
    labLines = []
    labels = []

    # Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if type(xvals) == tuple:
        xvals = np.linspace(xvals[0], xvals[1], len(labLines)+2)[1:-1]
    elif xvals is None:
        xmin, xmax = ax.get_xlim()
        xvals = np.linspace(xmin, xmax, len(labLines)+2)[1:-1]

    for line, x, label in zip(labLines, xvals, labels):
        label_line(line, x, label, align, **kwargs)