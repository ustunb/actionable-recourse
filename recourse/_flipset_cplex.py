import numpy as np
from collections import defaultdict
from cplex import Cplex, SparsePair
import warnings
from itertools import chain
from recourse.action_set import ActionSet
from recourse._flipset_base import _FlipsetBase
from recourse.cplex_helper import set_mip_parameters, set_cpx_display_options, set_mip_time_limit, set_mip_node_limit, toggle_mip_preprocessing, DEFAULT_CPLEX_PARAMETERS
from recourse.debug import ipsh
import pandas as pd

# todo enumeration strategy
# todo method for enumeration
# todo improve cost function type implementation ('maxpct' / 'logpct' / 'euclidean')
# todo base_mip / rebuild mip when required

# table
# item | change_in_score | change_in_min_pctile | cost
# optional to score


class _FlipsetBuilderCPLEX(_FlipsetBase):

    _default_cplex_parameters = dict(DEFAULT_CPLEX_PARAMETERS)

    def __init__(self, action_set, coefficients, intercept = 0.0, x = None, **kwargs):
        """
        :param x: vector of input variables for person x
        :param intercept: intercept value of score function
        :param coefs: coefficients of score function
        :param action_set: action set
        :param params: parameters for flipset form/flipset generation
                       (e.g. type of cost function to use / max items etc.)
        """
        # initialize Cplex MIP
        self._cpx_parameters = kwargs.get('cplex_parameters', self._default_cplex_parameters)

        ## initialize base class
        super().__init__(action_set=action_set, coefficients=coefficients, intercept=intercept, x=x, **kwargs)

        return self



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











