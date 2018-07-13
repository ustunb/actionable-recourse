import numpy as np
from collections import defaultdict
from cplex import Cplex, SparsePair
import warnings
from itertools import chain
from recourse.action_set import ActionSet
from recourse.cplex_helper import set_mip_parameters, set_cpx_display_options, set_mip_time_limit, set_mip_node_limit, toggle_mip_preprocessing, DEFAULT_CPLEX_PARAMETERS
from recourse.debug import ipsh
import pandas as pd
# todo edit credit to increments of 50s
# todo enumeration strategy
# todo method for enumeration
# todo tabulate

# todo improve cost function type implementation ('maxpct' / 'logpct' / 'euclidean')
# todo base_mip / rebuild mip when required

# table
# item | change_in_score | change_in_min_pctile | cost
# optional to score


class FlipsetBuilder(object):

    _default_enumeration_type = 'size'
    _default_cost_function = 'size'
    _default_print_flag = True
    _default_check_flag = True
    _default_cplex_parameters = dict(DEFAULT_CPLEX_PARAMETERS)
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
        self._cpx_parameters = kwargs.get('cplex_parameters', self._default_cplex_parameters)
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
            self._check_flag = bool(FlipsetBuilder._default_check_flag)
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
            self._print_flag = bool(FlipsetBuilder._default_print_flag)
        elif isinstance(flag, bool):
            self._print_flag = bool(flag)
        else:
            raise AttributeError('print_flag must be boolean or None')


    def __repr__(self):
        return self.tabulate()


    def tabulate(self):
        return str(self.action_set)
        # t = PrettyTable()
        # t.add_column("name", self.name, align = "r")
        # t.add_column("vtype", self.actionable, align = "r")
        # t.add_column("step direction", self.step_direction, align = "r")
        # t.add_column("flip direction", self.flip_direction, align = "r")
        # t.add_column("actionable", self.actionable, align = "r")
        # t.add_column("grid size", self.size, align = "r")
        # t.add_column("lb", self.lb, align = "r")
        # t.add_column("ub", self.ub, align = "r")


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


    def set_item_limits(self, min_items = None, max_items = None):
        """
        changes limits for the number of items
        :param min_items:
        :param max_items:
        :return:
        """

        if min_items is None or max_items is None:
            return

        min_items = self.min_items if min_items is None else int(min_items)
        max_items = self.max_items if min_items is None else int(max_items)
        assert min_items <= max_items, 'incompatible sizes'

        min_items = int(min_items)
        if min_items != self.min_items:
            min_nnz_actions = float(self.n_actionable - min_items)
            self._mip.linear_constraints.set_rhs("min_items", min_nnz_actions)
            self.min_items = min_items

        if max_items != self.max_items:
            max_nnz_actions = float(self.n_actionable - max_items)
            self._mip.linear_constraints.set_rhs("max_items", max_nnz_actions)
            self.max_items = max_items



        return


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


    def _check_mip_build_info(self, build_info):

        for v in build_info.values():

            assert not np.isclose(v['coef'], 0.0)
            a = np.array(v['actions'])
            c = np.array(v['costs'])
            assert c[0] == 0.0
            assert a[0] == 0.0

            if np.sign(v['coef']) > 0:
                assert np.all(np.greater(a[1:], 0.0))
            else:
                assert np.all(np.less(a[1:], 0.0))

            assert len(a) >= 2
            assert len(a) == len(c)
            assert len(a) == len(np.unique(a))
            assert len(c) == len(np.unique(c))

        return True


    def build_mip(self):

        """
        returns an optimization problem that can be solved to determine an item in a flipset for x
        :return:
        """

        # setup MIP related parameters
        cost_type = self.mip_cost_type
        min_items = self.min_items
        max_items = self.max_items
        #assert min_items <= max_items

        # cost/action information
        build_info, indices = self._get_mip_build_info()

        # if build_info is empty, then reset mip and return
        if len(build_info) == 0:
            self._mip = None
            self._mip_indices = dict()
            return

        # initialize mip
        mip = Cplex()
        mip.set_problem_type(mip.problem_type.MILP)
        vars = mip.variables
        cons = mip.linear_constraints
        n_actionable = len(build_info)
        n_indicators = len(indices['action_ind_names'])

        # define a[j]
        vars.add(names = indices['action_var_names'],
                 types = ['C'] * n_actionable,
                 lb = indices['action_lb'],
                 ub = indices['action_ub'])

        # sum_j w[j] a[j] > -score
        cons.add(names = ['score'],
                 lin_expr = [SparsePair(ind = indices['action_var_names'], val = indices['coefficients'])],
                 senses = ['G'],
                 rhs = [-self.score()])

        # define indicators u[j][k] = 1 if a[j] = actions[j][k]
        vars.add(names = indices['action_ind_names'], types = ['B'] * n_indicators)

        # restrict a[j] to feasible values using a 1 of K constraint setup
        for info in build_info.values():

            # restrict a[j] to actions in feasible set and make sure exactly 1 indicator u[j][k] is on
            # 1. a[j]  =   sum_k u[j][k] * actions[j][k] - > 0.0   =   sum u[j][k] * actions[j][k] - a[j]
            # 2.sum_k u[j][k] = 1.0
            cons.add(names = ['set_a[%d]' % info['idx'], 'pick_a[%d]' % info['idx']],
                     lin_expr = [SparsePair(ind = info['action_var_name'] + info['action_ind_names'], val = [-1.0] + info['actions']),
                                 SparsePair(ind = info['action_ind_names'], val = [1.0] * len(info['actions']))],
                     senses = ["E", "E"],
                     rhs = [0.0, 1.0])

            # declare indicator variables as SOS set
            mip.SOS.add(type = "1", name = "sos_u[%d]" % info['idx'], SOS = SparsePair(ind = info['action_ind_names'], val = info['actions']))

        # limit number of features per action
        #
        # size := n_actionable - n_null where n_null := sum_j u[j][0] = sum_j 1[a[j] = 0]
        #
        # size <= max_size
        # n_actionable - sum_j u[j][0]  <=  max_size
        # n_actionable - max_size       <=  sum_j u[j][0]
        #
        # min_size <= size:
        # min_size          <=  n_actionable - sum_j u[j][0]
        # sum_j u[j][0]     <=  n_actionable - min_size
        min_items = max(min_items, 1)
        max_items = min(max_items, n_actionable)
        size_expr = SparsePair(ind = indices['action_off_names'], val = [1.0] * n_actionable)
        cons.add(names = ['max_items', 'min_items'],
                 lin_expr = [size_expr, size_expr],
                 senses = ['G', 'L'],
                 rhs = [float(n_actionable - max_items), float(n_actionable - min_items)])

        # add constraints for cost function
        if cost_type == 'max':

            indices['max_cost_var_name'] = ['max_cost']
            indices['epsilon'] = np.min(indices['cost_df']) / np.sum(indices['cost_ub'])
            vars.add(names = indices['max_cost_var_name'] + indices['cost_var_names'],
                     types = ['C'] * (n_actionable + 1),
                     obj = [1.0] + [indices['epsilon']] * n_actionable)
            #lb = [0.0] * (n_actionable + 1)) # default values are 0.0

            cost_constraints = {
                'names': [],
                'lin_expr': [],
                'senses': ["E", "G"] * n_actionable,
                'rhs': [0.0, 0.0] * n_actionable,
                }

            for info in build_info.values():

                cost_constraints['names'].extend([
                    'def_cost[%d]' % info['idx'],
                    'set_max_cost[%d]' % info['idx']
                    ])

                cost_constraints['lin_expr'].extend([
                    SparsePair(ind = info['cost_var_name'] + info['action_ind_names'], val = [-1.0] + info['costs']),
                    SparsePair(ind = indices['max_cost_var_name'] + info['cost_var_name'], val = [1.0, -1.0])
                    ])

            cons.add(**cost_constraints)

            # old code (commented out for speed)
            #
            # vars.add(names = indices['cost_var_names'],
            #          types = ['C'] * n_actionable,
            #          obj = [indices['epsilon']] * n_actionable,
            #          #ub = [CPX_INFINITY] * n_actionable, #indices['cost_ub'], #indices['cost_ub'],
            #          lb = [0.0] * n_actionable)
            #
            # vars.add(names = indices['max_cost_var_name'],
            #          types = ['C'],
            #          obj = [1.0],
            #          #ub = [np.max(indices['cost_ub'])],
            #          lb = [0.0])
            #
            # for info in build_info.values():
            #     cost[j] = sum c[j][k] u[j][k]
            #     cons.add(names = ['def_cost[%d]' % info['idx']],
            #              lin_expr = [SparsePair(ind = info['cost_var_name'] + info['action_ind_names'], val = [-1.0] + info['costs'])]
            #              senses = ["E"],
            #              rhs = [0.0])
            #
            #     max_cost > cost[j]
            #     cons.add(names = ['set_max_cost[%d]' % info['idx']],
            #              lin_expr = [SparsePair(ind = indices['max_cost_var_name'] + info['cost_var_name'], val = [1.0, -1.0])],
            #              senses = ["G"],
            #              rhs = [0.0])

        elif cost_type in ('total', 'local'):

            indices.pop('cost_var_names')
            objval_pairs = list(chain(*[list(zip(v['action_ind_names'], v['costs'])) for v in build_info.values()]))
            mip.objective.set_linear(objval_pairs)

        mip = self.set_mip_parameters(mip)
        self._mip = mip
        self.mip_indices = indices


    @property
    def solution_info(self):

        assert hasattr(self._mip, 'solution')
        mip = self._mip
        sol = mip.solution

        if sol.is_primal_feasible():

            indices = self._mip_indices
            variable_idx = indices['var_idx']

            # parse actions
            action_values = sol.get_values(indices['action_var_names'])

            if 'cost_var_names' in indices and self.mip_cost_type != 'total':
                cost_values = sol.get_values(indices['cost_var_names'])
            else:
                ind_idx = np.flatnonzero(np.array(sol.get_values(indices['action_ind_names'])))
                ind_names = [indices['action_ind_names'][int(k)] for k in ind_idx]
                cost_values = mip.objective.get_linear(ind_names)


            actions = np.zeros(self.n_variables)
            np.put(actions, variable_idx, action_values)

            costs = np.zeros(self.n_variables)
            np.put(costs, variable_idx, cost_values)

            info = {
                'feasible': True,
                'status': sol.get_status_string(),
                'status_code': sol.get_status(),
                #
                'actions': actions,
                'costs': costs,
                #
                'upperbound': sol.get_objective_value(),
                'lowerbound': sol.MIP.get_best_objective(),
                'gap': sol.MIP.get_mip_relative_gap(),
                #
                'iterations': sol.progress.get_num_iterations(),
                'nodes_processed': sol.progress.get_num_nodes_processed(),
                'nodes_remaining': sol.progress.get_num_nodes_remaining()
                }

            if self.mip_cost_type == 'max':
                info['total_cost'] = sol.get_values(indices['max_cost_var_name'])[0]
            else:
                info['total_cost'] = info['upperbound']

        else:

            info = self.infeasible_info
            info.update({
                'iterations': sol.progress.get_num_iterations(),
                'nodes_processed': sol.progress.get_num_nodes_processed(),
                'nodes_remaining': sol.progress.get_num_nodes_remaining()
                })

        return info

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
                # names = np.array(list(self.variable_names))
                # build_info, indices = self._get_mip_build_info()
                # sol = self.mip.solution
                # costs = sol.get_values(indices['cost_var_names'])
                # actions = sol.get_values(indices['action_var_names'])
                # nnz_ind = np.flatnonzero(np.array(actions))
                # nnz_cost_ind = np.flatnonzero(np.array(costs))
                # bug_ind = np.setdiff1d(nnz_ind, nnz_cost_ind)
                # bug_ind = list(np.array(indices['var_idx'])[bug_ind])
                #
                # bug_names = np.array(names[bug_ind]).tolist()
                # n = bug_names[0]
                # v = build_info[n]
                # v['costs']
                # v['actions']
                # sol.get_values(indices['action_var_names'])
                # ipsh()

        return True


    #### general mip stuff ####

    @property
    def mip_indices(self):
        return self._mip_indices


    @mip_indices.setter
    def mip_indices(self, indices):
        assert isinstance(indices, dict)
        self._mip_indices = indices


    def set_mip_parameters(self, cpx = None, param = None):

        if cpx is None:
            cpx = self._mip

        if param is None:
            param = self._cpx_parameters

        cpx = set_mip_parameters(cpx, param)
        return cpx


    def toggle_preprocessing(self, toggle = True):
        self._mip = toggle_mip_preprocessing(self._mip, toggle)

    #### solving, enumeration, validation ####


    def fit(self, time_limit = None, node_limit = None, display_flag = False):

        if self._mip is None:
            return self.infeasible_info

        mip = self.mip
        mip = set_cpx_display_options(mip, display_mip = False, display_lp = display_flag, display_parameters = display_flag)

        # update time limit
        if time_limit is not None:
            mip = set_mip_time_limit(mip, time_limit)

        if node_limit is not None:
            mip = set_mip_node_limit(mip, node_limit)

        # solve
        start_time = mip.get_time()
        mip.solve()
        end_time = mip.get_time() - start_time

        info = self.solution_info
        info['runtime'] = end_time
        # assert self._check_solution(info)
        return info


    def remove_all_features(self):
        mip = self.mip
        names = self.mip_indices['action_off_names']
        values = np.array(mip.solution.get_values(names))
        on_idx = np.flatnonzero(np.isclose(values, 0.0))
        mip.variables.set_lower_bounds([(names[j], 1.0) for j in on_idx])
        return


    def remove_feature_combination(self):

        mip = self.mip

        u_names = self.mip_indices['action_off_names']
        u = np.array(mip.solution.get_values(u_names))
        on_idx = np.isclose(u, 0.0)

        n = len(u_names)
        con_vals = np.ones(n, dtype = np.float_)
        con_vals[on_idx] = -1.0
        con_rhs = n - 1 - np.sum(on_idx)

        mip.linear_constraints.add(lin_expr = [SparsePair(ind = u_names, val = con_vals.tolist())],
                                   senses = ["L"],
                                   rhs = [float(con_rhs)])
        return


    def populate(self, total_items = 10, time_limit = None, node_limit = None, display_flag = False, enumeration_type = 'distinct_subsets'):

        mip = self.mip
        mip = set_cpx_display_options(mip, display_mip = False, display_lp = display_flag, display_parameters = display_flag)

        # update time limit
        if time_limit is not None:
            mip = set_mip_time_limit(mip, time_limit)

        if node_limit is not None:
            mip = set_mip_node_limit(mip, node_limit)

        if enumeration_type == 'mutually_exclusive':
            remove_solution = self.remove_all_features
        else:
            remove_solution = self.remove_feature_combination

        # enumerate soluitions
        k = 0
        all_info = []
        populate_start_time = mip.get_time()

        while k < total_items:

            # solve mip
            start_time = mip.get_time()
            mip.solve()
            run_time = mip.get_time() - start_time

            info = self.solution_info
            info['runtime'] = run_time

            if not info['feasible']:
                if self.print_flag:
                    print('recovered all minimum-cost items')
                break

            all_info.append(info)
            remove_solution()
            k += 1

        if self.print_flag:
            print('mined %d items in %1.1f seconds' % (k, mip.get_time() - populate_start_time))

        return all_info

    #### auditing ####
    # todo: basemip implementation
    #
    # def audit(self, target_sample):
    #     raise NotImplementedError()
    #     #if isinstance(target_sample, pd.DataFrame):
    #         #target_sample =
    #     #for u in
    #     # fb.x = u
    #     # audit_results.append(fb.fit())
    #
    # @property
    # def initial_mip(self):
    #     return self._base_mip
    #
    # def _setup_initial_mip(self):
    #     mip = Cplex()
    #     mip.set_problem_type(mip.problem_type.MILP)
    #     mip = self.set_mip_parameters(mip)
    #     # todo: add
    #
    #    vars.add(names = indices['action_var_names'], types = ['C'] * n_actionable, lb = indices['action_lb'], ub = indices['action_ub'])
    #
    #    cons.add(names = ['score'],
    #             lin_expr = [SparsePair(ind = indices['action_var_names'], val = indices['coefficients'])],
    #             senses = ['G'],
    #             rhs = [-self.score()])
    #
    #
    #     self._base_mip = mip






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








