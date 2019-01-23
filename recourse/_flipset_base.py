import numpy as np
from collections import defaultdict
from cplex import Cplex, SparsePair
import warnings
from itertools import chain
from recourse.action_set import ActionSet
import pandas as pd

# todo enumeration strategy
# todo method for enumeration
# todo improve cost function type implementation ('maxpct' / 'logpct' / 'euclidean')
# todo base_mip / rebuild mip when required

# table
# item | change_in_score | change_in_min_pctile | cost
# optional to score


class _FlipsetBase(object):

    _default_enumeration_type = 'size'
    _default_cost_function = 'size'
    _default_print_flag = True
    _default_check_flag = True
    _default_mip_cost_type = 'max'
    _valid_mip_cost_types = {'total', 'local', 'max'}

    def __init__(self, action_set, coefficients, intercept = 0.0, x = None, **kwargs):
        """
        :param x: vector of input variables for person x
        :param intercept: intercept value of score function
        :param coefs: coefficients of score function
        :param action_set: action set
        :param params: parameters for flipset form/flipset generation
                       (e.g. type of cost function to use / max items etc.)
        """

        # attach coefficients
        coefficients = np.array(coefficients).flatten()
        assert np.all(np.isfinite(coefficients))
        self._coefficients = coefficients

        # attach intercept
        intercept = float(intercept)
        assert np.isfinite(intercept)
        self._intercept = intercept

        # attach action set
        assert isinstance(action_set, ActionSet)
        assert len(action_set) == len(coefficients)
        if not action_set.aligned:
            action_set.align(coefficients)
        self._action_set = action_set

        # add indices
        self._variable_names = action_set.name
        self._variable_index = {n: j for j, n in enumerate(self._variable_names)}
        self._actionable_indices = [j for j, v in enumerate(action_set.actionable) if v]

        # flags
        self.print_flag = kwargs.get('print_flag', self._default_print_flag)
        self.check_flag = kwargs.get('check_flag', self._default_check_flag)

        # initialize Cplex MIP
        self._mip = None
        self._mip_cost_type = kwargs.get('mip_cost_type', self._default_mip_cost_type)
        self._mip_indices = dict()
        self._min_items = 0
        self._max_items = self.n_actionable

        # attach features
        self._x = None
        self.x = x

        assert self._check_rep()


    #### built-ins ####
    def __len__(self):
        raise len(self.action_set)


    #### internal representation ####
    def _check_rep(self):
        """
        :return: True if representation invariants are true
        """
        if self.check_flag:
            assert self.n_variables == len(self._coefficients)
            assert self.n_variables == len(self._action_set)
            assert self.n_variables == len(self._x)
            assert isinstance(self._intercept, float)
            assert self.action_set.aligned
            assert 0 <= self._min_items <= self._max_items <= self.n_variables
        return True


    @property
    def check_flag(self):
        return bool(self._check_flag)


    @check_flag.setter
    def check_flag(self, flag):
        if flag is None:
            self._check_flag = bool(self._default_check_flag)
        elif isinstance(flag, bool):
            self._check_flag = bool(flag)
        else:
            raise AttributeError('check_flag must be boolean or None')


    #### printing ####
    @property
    def print_flag(self):
        return bool(self._print_flag)


    @print_flag.setter
    def print_flag(self, flag):
        if flag is None:
            self._print_flag = bool(self._default_print_flag)
        elif isinstance(flag, bool):
            self._print_flag = bool(flag)
        else:
            raise AttributeError('print_flag must be boolean or None')


    def __repr__(self):
        return self.tabulate()


    def tabulate(self):
        return str(self.action_set)


    #### immutable properties ####

    @property
    def variable_names(self):
        return self._variable_names


    @property
    def n_variables(self):
        return len(self._coefficients)


    @property
    def variable_index(self):
        return self._variable_index


    @property
    def action_set(self):
        return self._action_set


    @property
    def actionable_indices(self):
        return self._actionable_indices


    @property
    def actionable_names(self):
        return self._action_set[self._actionable_indices].name


    @property
    def n_actionable(self):
        return len(self._actionable_indices)


    #### feature values ####

    @property
    def x(self):
        return np.array(self._x)

    @x.setter
    def x(self, x):
        assert isinstance(x, (np.ndarray, list))
        x = np.array(x, dtype = np.float_).flatten()
        assert len(x) == self.n_variables
        self._x = x
        ## TODO why is this here?? mixed functionality
        self.build_mip()

    #### model ####

    @property
    def coefficients(self):
        return self._coefficients


    @property
    def intercept(self):
        return float(self._intercept)


    def score(self, x = None):
        if x is None:
            x = self._x
        else:
            assert isinstance(x, (np.ndarray, list))
            x = np.array(x, dtype = np.float).flatten()
        return self._coefficients.dot(x) + self._intercept


    def prediction(self, x = None):
        return np.sign(self.score(x))

    #### flipset mip ####

    @property
    def mip(self):
        return self._mip


    @property
    def mip_cost_type(self):
        return self._mip_cost_type


    @mip_cost_type.setter
    def mip_cost_type(self, t = None):

        if t is None:
            t = str(self._default_mip_cost_type)
        else:
            assert t in self._valid_mip_cost_types, 'mip_cost_type must None, %r' % self._valid_mip_cost_types

        if t != self._mip_cost_type:
            self._mip_cost_type = t
            self.build_mip()


    @property
    def min_items(self):
        return int(self._min_items)


    @min_items.setter
    def min_items(self, k):
        if k is None:
            self._min_items = 0
        else:
            k = int(k)
            assert k >= 0
            self._min_items = k


    @property
    def max_items(self):
        return int(self._max_items)


    @max_items.setter
    def max_items(self, k):
        if k is None:
            self._max_items = self.n_actionable
        else:
            k = int(k)
            assert k >= 0
            self._max_items = min(k, self.n_actionable)


    def _get_mip_build_info(self, cost_function_type = 'percentile', validate = True):

        #
        build_info = {}
        indices = defaultdict(list)
        if self.mip_cost_type == 'local':
            cost_up = lambda c: np.log((1.0 - c[0])/(1.0 - c))
            cost_dn = lambda c: np.log((1.0 - c) / (1.0 - c[0]))
        else:
            cost_up = lambda c: c - c[0]
            cost_dn = lambda c: c[0] - c

        if cost_function_type == 'percentile':

            actions, percentiles = self._action_set.feasible_grid(x = self._x, return_actions = True, return_percentiles = True, return_immutable = False)

            for n, a in actions.items():

                if len(a) >= 2:

                    c = percentiles[n]
                    if np.isclose(a[-1], 0.0):
                        a = np.flip(a, axis = 0)
                        c = np.flip(c, axis = 0)
                        c = cost_dn(c)
                    else:
                        c = cost_up(c)

                    # override numerical issues
                    bug_idx = np.logical_or(np.less_equal(c, 0.0), np.isclose(a, 0.0, atol = 1e-8))
                    bug_idx = np.flatnonzero(bug_idx).tolist()
                    bug_idx.pop(0)
                    if len(bug_idx) > 0:
                        c = np.delete(c, bug_idx)
                        a = np.delete(a, bug_idx)

                    idx = self._variable_index[n]
                    w = float(self._coefficients[idx])
                    #da = np.diff(a)
                    dc = np.diff(c)

                    info = {
                        'idx': idx,
                        'coef': w,
                        'actions': a.tolist(),
                        'costs': c.tolist(),
                        'action_var_name': ['a[%d]' % idx],
                        'action_ind_names': ['u[%d][%d]' % (idx, k) for k in range(len(a))],
                        'cost_var_name': ['c[%d]' % idx]
                        }

                    build_info[n] = info

                    indices['var_idx'].append(idx)
                    indices['coefficients'].append(w)
                    indices['action_off_names'].append(info['action_ind_names'][0])
                    indices['action_ind_names'].extend(info['action_ind_names'])
                    indices['action_var_names'].extend(info['action_var_name'])
                    indices['cost_var_names'].extend(info['cost_var_name'])
                    indices['action_lb'].append(float(np.min(a)))
                    indices['action_ub'].append(float(np.max(a)))
                    # indices['action_df'].append(float(np.min(da)))
                    indices['cost_ub'].append(float(np.max(c)))
                    indices['cost_df'].append(float(np.min(dc)))


        if validate:
            assert self._check_mip_build_info(build_info)

        return build_info, indices


    @property
    def infeasible_info(self):

        info = {
            'feasible': False,
            'status': 'no solution exists',
            'status_code': float('nan'),
            #
            'total_cost': float('inf'),
            'actions': np.repeat(np.nan, self.n_variables),
            'costs': np.repeat(np.nan, self.n_variables),
            'upperbound': float('inf'),
            'lowerbound': float('inf'),
            'gap': float('inf'),
            #
            'iterations': 0,
            'nodes_processed': 0,
            'nodes_remaining': 0,
            'runtime': 0,
            }
        return info


    def _check_solution(self, info):
        """
        :return: return True if making the change from the Flipset will actually 'flip' the prediction for the classifier
        """

        if info['feasible']:
            a = info['actions']
            all_idx = np.arange(len(a))
            static_idx = np.flatnonzero(np.isclose(a, 0.0, rtol = 1e-4))
            action_idx = np.setdiff1d(all_idx, static_idx)
            n_items = len(action_idx)
            assert n_items >= 1
            assert self.min_items <= n_items <= self.max_items

            try:
                assert np.all(np.isin(action_idx, self.actionable_indices))
            except AssertionError:
                warnings.warn('action set no in self.actionable_indices')

            x = self.x
            try:
                assert np.not_equal(self.prediction(x), self.prediction(x + a))
            except AssertionError:
                s = self.score(x + a)
                assert not np.isclose(self.score(x + a), 0.0, atol = 1e-4)
                warnings.warn('numerical issue: near-zero score(x + a) = %1.8f' % s)

            try:
                # check costs change -> action
                assert np.all(np.greater(info['costs'][action_idx], 0.0))
                assert np.all(np.isclose(info['costs'][static_idx], 0.0))

                # check total cost
                if self.mip_cost_type == 'max':
                    if not np.isclose(info['total_cost'], np.max(info['costs']), rtol = 1e-4):
                        warnings.warn('numerical issue: max_cost is %1.2f but maximum of cost[j] is %1.2f' % (
                            info['total_cost'], np.max(info['costs'])))
                elif self.mip_cost_type == 'total':
                    assert np.isclose(info['total_cost'], np.sum(info['costs']))

            except AssertionError:
                warnings.warn('issue detected with %s' % str(info))

        return True




class Flipset(object):

    df_column_names = ['names',
                       'idx',
                       'size',
                       'total_cost',
                       'start_values',
                       'final_values',
                       'final_score',
                       'final_prediction',
                       'feasible',
                       'flipped']

    def __init__(self, x, variable_names, coefficients, intercept = 0.0):

        assert isinstance(x, (list, np.ndarray))

        x = np.array(x, dtype = np.float_).flatten()
        n_variables = len(x)

        assert isinstance(coefficients, (list, np.ndarray))
        intercept = float(intercept)
        assert np.isfinite([intercept])

        assert isinstance(variable_names, list)
        assert len(variable_names) == n_variables
        assert all(map(lambda s: isinstance(s, str), variable_names))

        self._x = x
        self._n_variables = n_variables
        self._variable_names = variable_names
        self._coefs = np.array(coefficients, dtype = np.float_).flatten()
        self._intercept = intercept
        self._items = []

        self._df = pd.DataFrame(columns = Flipset.df_column_names, dtype = object)
        self._sort_args = {'by': ['size', 'total_cost', 'final_score'], 'inplace': True, 'axis': 0}


    @property
    def x(self):
        return self._x


    @property
    def df(self):
        return self._df


    @property
    def sort_args(self):
        return self._sort_args


    def sort(self, **sort_args):

        if len(sort_args) == 0:
            self._df.sort_values(**self._sort_args)
            return

        if 'by' in sort_args:
            sort_names = sort_args['by']
        else:
            sort_names = list(sort_args.keys())
            sort_args = {}

        assert isinstance(sort_names, list)
        assert len(sort_names) > 0
        for s in sort_names:
            assert isinstance(s, str)
            assert s in self._df.columns

        sort_args['by'] = sort_names
        sort_args['inplace'] = True
        sort_args['axis'] = 0

        self._df.sort_values(**sort_args)
        self._sort_args = sort_args


    @property
    def n_variables(self):
        return self._n_variables


    @property
    def variable_names(self):
        return list(self._variable_names)


    @property
    def coefficients(self):
        return self._coefs


    @property
    def intercept(self):
        return self._intercept


    @property
    def items(self):
        return self._items


    @property
    def yhat(self):
        return self._intercept + np.dot(self._coefs, self._x)

    #### built ins ####


    def __len__(self):
        return len(self._items)


    def __repr__(self):
        return str(self._items)


    #### methods ####

    def predict(self, actions = None):
        return np.sign(self.score(actions))


    def score(self, actions = None):
        if actions is not None:
            return self._intercept + np.dot(self._coefs, self._x + actions)
        else:
            return self._intercept + np.dot(self._coefs, self._x)


    def add(self, items):
        if isinstance(items, dict):
            items = [items]
        assert isinstance(items, list)
        items = list(map(lambda i: self.validate_item(i), items))
        self._items.extend(items)
        self._update_df(items)


    def validate_action(self, a):
        a = np.array(a, dtype = np.float_).flatten()
        assert len(a) == self.n_variables, 'action vector must have %d elements' % self.n_variables
        assert np.all(np.isfinite(a)), 'actions must be finite'
        assert np.count_nonzero(a) >= 1, 'at least one action element must be non zero'
        assert np.not_equal(self.yhat, self.predict(a)), 'actions do not flip the prediction from %d' % self.yhat
        return a


    def validate_item(self, item):
        assert isinstance(item, dict)
        required_fields = ['feasible', 'actions', 'total_cost']
        for k in required_fields:
            assert k in item, 'item missing field %s' % k
        item['actions'] = self.validate_action(item['actions'])
        assert item['total_cost'] > 0.0, 'total cost must be positive'
        assert item['feasible'], 'item must be feasible'
        return item


    def item_to_df_row(self, item):
        x = self.x
        a = item['actions']
        h = self.predict(a)
        nnz_idx = np.flatnonzero(a)
        row = {
            'names': [self.variable_names[j] for j in nnz_idx],
            'idx': nnz_idx,
            'size': len(nnz_idx),
            'start_values': x[nnz_idx],
            'final_values': x[nnz_idx] + a[nnz_idx],
            'total_cost': float(item['total_cost']),
            'final_score': self.score(a),
            'final_prediction': h,
            'feasible': item['feasible'],
            'flipped': np.not_equal(h, self.yhat),
            }

        return row


    def _update_df(self, items):
        if len(items) > 0:
            row_data = list(map(lambda item: self.item_to_df_row(item), items))
            self._df = self._df.append(row_data, ignore_index = True)[self._df.columns.tolist()]
            self.sort()


    def view(self):
        return self._df


    def to_latex(self, name_formatter = '\\textit'):

        self.sort()
        tex_columns = ['names', 'start_values', 'final_values']
        tex_df = self._df[tex_columns]

        # split components for each item
        tex_df = tex_df.reset_index().rename(columns = {'index': 'item_id'})
        df_list = []
        for n in tex_columns:
            tmp = tex_df.set_index(['item_id'])[n].apply(pd.Series).stack()
            tmp = tmp.reset_index().rename(columns = {'level_1': 'var_id'})
            tmp_name = tmp.columns[-1]
            tmp = tmp.rename(columns = {tmp_name: n})
            df_list.append(tmp)

        # combine into a flattened list
        flat_df = df_list[0]
        for k in range(1, len(df_list)):
            flat_df = flat_df.merge(df_list[k])

        # drop the merge index
        flat_df = flat_df.drop(columns = ['var_id'])

        # index items by item_id
        flat_df = flat_df.sort_values(by = 'item_id')
        flat_df = flat_df.rename(columns = {'item_id': 'item'})
        flat_df = flat_df.set_index('item')


        # add another column for the latex arrow symbol
        idx = flat_df.columns.tolist().index('final_values')
        flat_df.insert(loc = idx, column = 'to', value = ['longrightarrow'] * len(flat_df))

        # name headers
        flat_df = flat_df.rename(columns = {
            'names': '\textsc{Feature Subset}',
            'start_values': '\textsc{Current Values}',
            'final_values': '\textsc{Required Values}'})

        # get raw tex table
        table = flat_df.to_latex(multirow = True, index = True, escape = False, na_rep = '-', column_format = 'rlccc')

        # manually wrap names with a formatter function
        if name_formatter is not None:
            for v in self.variable_names:
                table = table.replace(v, '%s{%s}' % (name_formatter, v))

        # add the backslash for the arrow
        table = table.replace('longrightarrow', '$\\longrightarrow$')

        # minor embellishments
        table = table.split('\n')
        table[2] = table[2].replace('to', '')
        table[2] = table[2].replace('{}', '')
        table.pop(3)
        table.pop(3)

        return '\n'.join(table)