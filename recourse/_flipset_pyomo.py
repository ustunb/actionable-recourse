import numpy as np
from collections import defaultdict
import warnings
from itertools import chain
from recourse.action_set import ActionSet
from recourse._flipset_base import _FlipsetBase
from pyomo.core import *
from pyomo import *
import pyomo
import pyomo.core
import pyomo.opt
import pyomo.environ

# table
# item | change_in_score | change_in_min_pctile | cost
# optional to score


class _FlipsetBuilderPyomo(_FlipsetBase):
    def __init__(self, action_set, coefficients, intercept = 0.0, x = None, **kwargs):
        self.built=False
        super().__init__(
            action_set=action_set,
            coefficients=coefficients,
            intercept=intercept,
            x=x,
            **kwargs
        )
        return self

    def _check_mip_build_info(self, build_info):
        ## TODO
        return True

    def _get_mip_build_info(self, cost_function_type = 'percentile', validate = True):
        build_info, indices = super()._get_mip_build_info(
            cost_function_type=cost_function_type, validate=validate)

        ## pyomo-specific processing
        c = []
        a = []
        for name in self.action_set._names:
            c.append(build_info.get(name, {'costs': []})['costs'])
            a.append(build_info.get(name, {'actions': []})['actions'])

        output_build_info = {}
        output_build_info['a'] = a
        output_build_info['c'] = c
        output_build_info['epsilon'] = min(indices['cost_df']) / sum(indices['cost_ub'])

        return output_build_info, indices

    def build_mip(self):
        """Build the model <b>object</b>."""
        self.model = AbstractModel()

        self.model.J = Set()
        self.model.K = Set(self.model.J)

        def jk_init(m):
            return [(j, k) for j in m.J for k in m.K[j]]

        self.model.JK = Set(initialize=jk_init, dimen=None)

        self.model.y_pred = Param()
        self.model.epsilon = Param()
        self.model.max_cost = Var()
        self.model.w = Param(self.model.J)
        self.model.a = Param(self.model.JK)
        self.model.c = Param(self.model.JK)
        self.model.u = Var(self.model.JK, within=Binary)

        # Make sure only one action is on at a time.
        def c1Rule(m, j):
            return (
                       sum([m.u[j, k] for k in m.K[j]])
                   ) == 1

        # 2.b: Action sets must flip the prediction of a linear classifier.
        def c2Rule(m):
            return (
                    sum((m.u[j, k] * m.a[j, k] * m.w[j]) for j, k in m.JK) >= -m.y_pred
            )

        # instantiate max cost
        def maxcost_rule(m, j, k):
            return m.max_cost >= (m.u[j, k] * m.c[j, k])

        # Set up objective for total sum.
        def obj_rule_percentile(m):
            return sum(m.u[j, k] * m.c[j, k] for j, k in m.JK)

        # Set up objective for max cost.
        def obj_rule_max(m):
            return (
                    sum(m.epsilon * m.u[j, k] * m.c[j, k] for j, k in m.JK) + (1 - m.epsilon) * m.max_cost
            )

        ##
        self.model.g = Objective(rule=obj_rule_max, sense=minimize)
        self.model.c1 = Constraint(self.model.J, rule=c1Rule)
        self.model.c2 = Constraint(rule=c2Rule)
        self.model.c3 = Constraint(self.model.JK, rule=maxcost_rule)
        self.built = True


    def instantiate_mip(self):
        build_info, indices = self._get_mip_build_info()
        if not self.built:
            self.build_mip()

        a = build_info['a']
        a_tuples = {}
        for i in range(len(a)):
            a_tuples[(i, 0)] = 0
            for j in range(len(a[i])):
                a_tuples[(i, j)] = a[i][j]

        c = build_info['c']
        c_tuples = {}
        for i in range(len(c)):
            c_tuples[(i, 0)] = 0
            for j in range(len(c[i])):
                c_tuples[(i, j)] = c[i][j]

        u_tuples = {}
        for i in range(len(c)):
            u_tuples[(i, 0)] = True
            for j in range(len(c[i])):
                u_tuples[(i, j)] = False

        epsilon = min(indices['cost_df']) / sum(indices['cost_ub'])
        data = {None: {
            'J':  {None: list(range(len(a)))},
            'K': {i: list(range(len(a[i]) or 1)) for i in range(len(a)) },
            'a': a_tuples,
            'c': c_tuples,
            'u': u_tuples,
            'w': {i: coef for i, coef in enumerate(self.coefficients)},
            'y_pred': {None: self.score()},
            'epsilon': {None: epsilon},
            'max_cost': {None: -1000}
        }}

        instance = self.model.create_instance(data)
        return instance

    def fit(self):
        instance = self.instantiate_mip()
        opt = pyomo.opt.SolverFactory('cbc')
        results = opt.solve(instance)

        ## add check
        output = {}
        for i in instance.JK:
            output[i] = {
                'a': instance.a[i],
                'u': instance.u[i](),
                'c': instance.c[i]
            }
        output['max_cost'] = instance.max_cost()
        return output