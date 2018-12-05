import numpy as np
import pandas as pd
from recourse.path import *
from recourse.helper_functions import print_log
from recourse.flipset import FlipsetBuilder

from sklearn.preprocessing import StandardScaler
import pickle
import time
import pprint

# Formatting Options
np.set_printoptions(precision = 4, suppress = False)
pd.set_option('display.max_columns', 30)
pd.options.mode.chained_assignment = None
pp = pprint.PrettyPrinter(indent = 4)

### Helper Functions for Experimental Script

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
        fb = FlipsetBuilder(coefficients = coefficients, intercept = intercept, action_set = action_set, x = U[0], mip_cost_type = mip_cost_type)

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

