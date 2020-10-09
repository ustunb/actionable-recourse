# General Packages
from math import atan2, degrees
from datetime import datetime
from pathlib import Path
import time
import pprint
import numpy as np
import pandas as pd
import pickle

# Plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.dates import date2num
import seaborn as sns

# Scaling
from sklearn.preprocessing import StandardScaler

settings = {
    #
    # audit settings
    'data_name': 'credit',
    'method_name': 'logreg',
    'normalize_data': True,
    'force_rational_actions': False,
    #
    # script flags
    'audit_recourse': True,
    'plot_audits': True,
    'print_flag': True,
    'save_flag': True,
    'randomseed': 2338,
    #
    # placeholders
    'method_suffixes': [''],
    'audit_suffixes': [''],
    }

# Paths
repo_dir = Path(__file__).absolute().parent.parent
paper_dir = repo_dir / 'paper/'         # directory containing paper related info
data_dir = paper_dir / 'data/'          # directory containing data files
results_dir = paper_dir / 'results/'    # directory containing results

# create directories that don't exist
for d in [data_dir, results_dir]:
    d.mkdir(exist_ok = True)

# Formatting Options
np.set_printoptions(precision = 4, suppress = False)
pd.set_option('display.max_columns', 30)
pd.options.mode.chained_assignment = None
pp = pprint.PrettyPrinter(indent = 4)

# Plotting Settings
sns.set(style="white", palette="muted", color_codes = True)

plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rc('legend', fontsize = 20)


# file names
output_dir = results_dir / settings['data_name']
output_dir.mkdir(exist_ok = True)

if settings['normalize_data']:
    settings['method_suffixes'].append('normalized')

if settings['force_rational_actions']:
    settings['audit_suffixes'].append('rational')

# set file header
settings['dataset_file'] = '%s/%s_processed.csv' % (data_dir, settings['data_name'])
settings['file_header'] = '%s/%s_%s%s' % (output_dir, settings['data_name'], settings['method_name'], '_'.join(settings['method_suffixes']))
settings['audit_file_header'] = '%s%s' % (settings['file_header'], '_'.join(settings['audit_suffixes']))
settings['model_file'] = '%s_models.pkl' % settings['file_header']
settings['audit_file'] = '%s_audit_results.pkl' % settings['audit_file_header']

# Recourse Objects
from recourse.action_set import ActionSet
from recourse.builder import RecourseBuilder
from recourse.auditor import RecourseAuditor
from recourse.flipset import Flipset


### Helper Functions for Experimental Script

def load_data():
    """Helper function to load in data, and output that and optionally a scaler object:

    Output:
        data: dict with  the following fields
            outcome_name:               Name of the outcome variable (inferred as the first column.)
            variable_names:             A list of names indicating input columns.
            X:                          The input features for our model.
            y:                          The column of the dataframe indicating our outcome variable.
            scaler:                     The sklearn StandardScaler used to normalize the dataset, if we wish to scale.
            X_scaled:                   Scaled version of X, if we wish to scale
            X_train:                    The training set: set to the whole dataset if not scaled. Set to X_scaled if we do scale.

        scaler:
            Object used to scale data. If "scale" is set to None, then this is returned as None.
    """
    # data set
    data_df = pd.read_csv(settings['dataset_file'])
    data = {
        'outcome_name': data_df.columns[0],
        'variable_names': data_df.columns[1:].tolist(),
        'X': data_df.iloc[:, 1:],
        'y': data_df.iloc[:, 0]
    }

    scaler = None
    data['X_train'] = data['X']
    data['scaler'] = None
    if settings['normalize_data']:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        data['X_scaled'] = pd.DataFrame(scaler.fit_transform(data['X'].to_numpy(dtype=float), data['y'].values),
                                        columns=data['X'].columns)
        data['X_train'] = data['X_scaled']
        data['scaler'] = scaler

    return data, scaler


def undo_coefficient_scaling(clf = None, coefficients = None, intercept = 0.0, scaler = None):
    """
    given coefficients and data for scaled data, returns coefficients and intercept for unnormalized data

    w = w_scaled / sigma
    b = b_scaled - (w_scaled / sigma).dot(mu) = b_scaled - w.dot(mu)

    :param sklearn linear classifier
    :param coefficients: vector of coefficients
    :param intercept: scalar for the intercept function
    :param scaler: sklearn.Scaler or

    :return: coefficients and intercept for unnormalized data

    """
    if coefficients is None:

        assert clf is not None
        assert intercept == 0.0
        assert hasattr(clf, 'coef_')
        coefficients = clf.coef_
        intercept = clf.intercept_ if hasattr(clf, 'intercept_') else 0.0

    if scaler is None:

        w = np.array(coefficients)
        b = float(intercept)

    else:

        isinstance(scaler, StandardScaler)
        x_shift = np.array(scaler.mean_)
        x_scale = np.sqrt(scaler.var_)
        w = coefficients / x_scale
        b = intercept - np.dot(w, x_shift)

    w = np.array(w).flatten()
    b = float(b)
    return w, b


def get_coefficient_df(model_dict, variable_names = None, scaler = None):

    """
    extract coefficients of all models and store them into a data.frame

    :param model_dict: dictionary of models
    :param variable_names:
    :return:

    """
    # get the coefficient values
    assert isinstance(model_dict, dict)

    coef_df = []

    for k in sorted(model_dict.keys()):

        coef_vals = model_dict[k].coef_.flatten()
        intercept_val = model_dict[k].intercept_[0]
        coef_vals, intercept_val = undo_coefficient_scaling(coefficients = coef_vals, intercept = intercept_val, scaler = scaler)

        if variable_names is None:
            coef_vals = (pd.Series(coef_vals, index = ['x%d' % j for j in range(coef_vals)]).to_frame(k))
        else:
            coef_vals = (pd.Series(coef_vals, index = variable_names).to_frame(k))

        coef_df.append(coef_vals)

    return pd.concat(coef_df, axis = 1)


def format_gridsearch_df(grid_search_df, settings, n_coefficients, invert_C = True):
    """
    Take a fitted GridSearchCV and return:

     model_stats_df: data frame containing 1 row for fold x free parameter instance.
     columns include:
      - 'data_name',
      - 'method_name',
      - 'free_parameter_name',
      - 'free_parameter_value' (for each item in free parameter),
      - training error,
      - testing error,
      - n_coefficients

    :param grid_search_df:
    :param n_coefficients: size of input dataset
    :param invert_C: if C is a parameter, invert it (C = 1/lambda in l1 regression)
    :return:
    """
    train_score_df = (grid_search_df
                          .loc[:, filter(lambda x: 'train_score' in x and 'split' in x, grid_search_df.columns)]
                          .unstack()
                          .reset_index()
                          .rename(columns={'level_0': 'split_num', 0: 'train_score'})
                          .set_index('level_1')
                          .assign(split_num=lambda df: df.apply(lambda x: x['split_num'].replace('_train_score', ''), axis=1))
                          )

    test_score_df = (grid_search_df
                         .loc[:, filter(lambda x: 'test_score' in x and 'split' in x, grid_search_df.columns)]
                         .unstack()
                         .reset_index()
                         .rename(columns={'level_0': 'split_num', 0: 'test_score'})
                         .set_index('level_1')
                         .assign(split_num=lambda df: df.apply(lambda x: x['split_num'].replace('_test_score', ''), axis=1)))

    model_stats_df= pd.concat([train_score_df, test_score_df.drop('split_num', axis=1)], axis=1)
    model_stats_df['dataname'] = settings['data_name']
    param_df = (grid_search_df['params']
        .apply(pd.Series))
    if invert_C:
        param_df['C'] = 1 / param_df['C']
    param_df = (param_df.rename(
            columns={col: 'param %d: %s' % (idx, col) for idx, col in enumerate(param_df.columns)})
    ).assign(key=grid_search_df['key'])

    model_stats_df = (model_stats_df
        .merge(param_df, left_index=True, right_index=True)
        )
    return model_stats_df.assign(n_coefficients=n_coefficients)


def get_flipset_solutions(model, data, action_set, mip_cost_type = 'max', scaler = None, print_flag = True):

    """
    Run a basic audit of a model on the training dataset.

    :param model:
    :param data:
    :param action_set:
    :param mip_cost_type:
    :param scaler:
    :return:
    """

    if scaler is not None:
        yhat = model.predict(data['X_scaled'])
        coefficients, intercept = undo_coefficient_scaling(coefficients=np.array(model.coef_).flatten(), intercept = model.intercept_[0], scaler = scaler)
    else:
        yhat = model.predict(data['X'])
        coefficients, intercept = np.array(model.coef_).flatten(), model.intercept_[0]

    action_set.align(coefficients)

    # get defaults
    audit_results = []
    predicted_neg = np.flatnonzero(yhat < 1)

    if any(predicted_neg):

        U = data['X'].iloc[predicted_neg].values
        fb = RecourseBuilder(coefficients = coefficients, intercept = intercept, action_set = action_set, x = U[0], mip_cost_type = mip_cost_type)

        # basic audit
        start_time = time.time()
        if print_flag:
            for i, u in enumerate(U):
                fb.x = u
                info = fb.fit()
                audit_results.append(info)
                print_log('cost[%06d] = %1.2f' % (i, info['total_cost']))
        else:
            for i, u in enumerate(U):
                fb.x = u
                audit_results.append(fb.fit())

        print_log('runtime: solved %i IPs in %1.1f seconds' % (i, time.time() - start_time))

    return audit_results


def print_score_function(names, coefficients, intercept):
    s = ['score function =']
    s += ['%1.6f' % intercept]
    for n, w in zip(names, coefficients):
        if w >= 0.0:
            s += ['+\t%1.6f * %s' % (np.abs(w), n)]
        else:
            s += ['-\t%1.6f * %s' % (np.abs(w), n)]
    return '\n'.join(s)


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


def refomat_gridsearch_df(grid_search_df, settings, n_coefficients, invert_C=True):
    """
    Take a fitted GridSearchCV and return:
     model_stats_df: data frame containing 1 row for fold x free parameter instance.
     columns include:
      - 'data_name',
      - 'method_name',
      - 'free_parameter_name',
      - 'free_parameter_value' (for each item in free parameter),
      - training error,
      - testing error,
      - n_coefficients
    :param grid_search_df:
    :param n_coefficients: size of input dataset
    :param invert_C: if C is a parameter, invert it (C = 1/lambda in l1 regression)
    :return:
    """
    train_score_df = (grid_search_df
                        .loc[:, filter(lambda x: 'train_score' in x and 'split' in x, grid_search_df.columns)]
                        .unstack()
                        .reset_index()
                        .rename(columns={'level_0': 'split_num', 0: 'train_score'})
                        .set_index('level_1')
                        .assign(split_num=lambda df: df.apply(lambda x: x['split_num'].replace('_train_score', ''), axis=1))
                      )

    test_score_df = (grid_search_df
                        .loc[:, filter(lambda x: 'test_score' in x and 'split' in x, grid_search_df.columns)]
                        .unstack()
                        .reset_index()
                        .rename(columns={'level_0': 'split_num', 0: 'test_score'})
                        .set_index('level_1')
                        .assign(split_num=lambda df: df.apply(lambda x: x['split_num'].replace('_test_score', ''), axis=1)))

    model_stats_df= pd.concat([train_score_df, test_score_df.drop('split_num', axis=1)], axis=1)
    model_stats_df['dataname'] = settings['data_name']
    param_df = (grid_search_df['params']
                .apply(pd.Series))
    if invert_C:
        param_df['C'] = 1 / param_df['C']
    param_df = (param_df.rename(
        columns={col: 'param %d: %s' % (idx, col) for idx, col in enumerate(param_df.columns)})
    ).assign(key=grid_search_df['key'])

    model_stats_df = (model_stats_df
        .merge(param_df, left_index=True, right_index=True)
    )
    return model_stats_df.assign(n_coefficients=n_coefficients)