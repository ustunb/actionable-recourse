import warnings
import numpy as np
import pandas as pd
import itertools
from collections import namedtuple
from prettytable import PrettyTable
from recourse.helper_functions import parse_classifier_args
from scipy.stats import gaussian_kde as kde
from scipy.interpolate import interp1d

# todo: add doc string for feasible values
# todo: replace percentiles with scikit-learn API
# todo: get_feasible_values/get_flip_actions should include an option to also include all observed values
# todo: set default bounds / step types for each variable type

__all__ = ['ActionSet']


class ActionSet(object):
    """
    Class to represent and manipulate feasible actions for the features in a dataset
    """

    _default_print_flag = True
    _default_check_flag = True
    _default_bounds = (1, 99, 'percentile')
    _default_step_type = 'relative'

    def __init__(self, X, names = None, y_desired = 1, **kwargs):

        """
        Initializes ActionSet from data

        Required Inputs:

        :param X: pandas.DataFrame or numpy matrix representing a feature matrix (features are columns, samples are rows)
                  X must contain at least 1 column and at least 1 row

        :param names: list of strings containing variable names.
                      names is only required if X is a numpy matrix

        :param y_desired: value of the desired outcome that will be used to evaluate recourse
                          y_desired can either be +1 (default value) or -1
                          If y_desired = +1, then we consider x such that h(x) = -1 and look for actions a such that h(x+a) = +1
                          If y_desired = -1, then we consider x such that h(x) = +1, and look for actions a such that h(x+a) = -1

        Optional Keyword Arguments

        :param custom_bounds: dictionary of custom bounds
        :param default_bounds: tuple containing information for default bounds
                                - (lb, ub, type) where type = 'percentile' or 'absolute';
                                - (lb, ub) if type is omitted, it is assumed to be 'absolute'

        :param default_step_type:
        :param print_flag: set to True to print a table with the ActionSet as _repr_
        :param check_flag: set to True to check for internal errors
        """
        assert isinstance(X, (pd.DataFrame, np.ndarray)), '`X` must be pandas.DataFrame or numpy.ndarray'
        if isinstance(X, pd.DataFrame):
            names = X.columns.tolist()
            X = X.values

        # validate names
        assert _check_variable_names(names)

        # validate X
        xdim = X.shape
        assert len(xdim) == 2, '`values` must be a matrix'
        assert xdim[0] >= 1, '`values` must have at least 1 row'
        assert xdim[1] == len(names), '`values` must contain len(`names`) = %d columns' % len(names)
        assert np.array_equal(X, X + 0.0), 'values must be numeric'

        # parse key word arguments
        custom_bounds = kwargs.get('custom_bounds', {})
        default_bounds = kwargs.get('default_bounds', self._default_bounds)
        default_step_type = kwargs.get('default_step_type', self._default_step_type)
        self.print_flag = kwargs.get('print_flag', self._default_print_flag)
        self.check_flag = kwargs.get('check_flag', self._default_check_flag)

        # build action elements
        indices = {}
        elements = {}
        for j, n in enumerate(names):
            indices[n] = j
            variable_type = _determine_variable_type(values = X[:, j])
            if variable_type == bool:
                elements[n] = _ActionElement(name = n, values = X[:, j], step_type = 'absolute', step_size = 1, variable_type = bool, bounds = custom_bounds.get(n, (0, 1, 'absolute')))
            elif variable_type == int:
                elements[n] = _ActionElement(name = n, values = X[:, j], step_type = default_step_type, variable_type = int, bounds = custom_bounds.get(n, default_bounds))
            elif variable_type == float:
                elements[n] = _ActionElement(name = n, values = X[:, j], step_type = default_step_type, variable_type = float, bounds = custom_bounds.get(n, default_bounds))

        self._names = [str(n) for n in names]
        self._indices = indices
        self._elements = elements
        self._constraints = _ActionConstraints(names = self._names)

        # set y_desired
        assert y_desired in (-1, 0, 1), "y_desired must be +1 or -1"
        self._y_desired = 1 if y_desired > 0 else -1
        assert self._check_rep()


    def __len__(self):
        return len(self._names)


    def __iter__(self):
        return (self._elements[n] for n in self._names)


    def __getitem__(self, index):

        if isinstance(index, str):
            return self._elements[index]
        elif isinstance(index, (int, np.int_)):
            return self._elements[self._names[index]]
        elif isinstance(index, list):
            if isinstance(index[0], int):
                names = [self._names[j] for j in index]
            elif isinstance(index[0], bool):
                names = [self._names[j] for j, v in enumerate(index) if v]
            elif isinstance(index[0], str):
                names = index
            return _ActionSlice([self._elements[n] for n in names])
        elif isinstance(index, np.ndarray):
            names = np.array(self._names)[index].tolist()
            return _ActionSlice([self._elements[n] for n in names])
        elif isinstance(index, slice):
            return _ActionSlice([self._elements[n] for n in self._names[index]])
        else:
            raise IndexError('index must be str, int, a list of strings/int or a slice')


    def __setitem__(self, name, e):
        assert isinstance(e, _ActionElement), 'ActionSet can only contain ActionElements'
        assert name in self._names, 'no variable with name %s in ActionSet'
        self._elements.update({name: e})


    def __getattribute__(self, name):
        if name[0] == '_' or name in ('alignment_known') or not hasattr(_ActionElement, name):
            return object.__getattribute__(self, name)
        else:
            return [getattr(self._elements[n], name) for n, j in self._indices.items()]


    def __setattr__(self, name, value):
        if hasattr(self, '_elements') and hasattr(_ActionElement, name):
            attr_values = _expand_values(value, len(self))
            for n, j in self._indices.items():
                self._elements[n].__setattr__(name, attr_values[j])
        else:
            object.__setattr__(self, name, value)


    def __str__(self):
        return tabulate_actions(self)


    def __repr__(self):
        if self._print_flag:
            return str(self)

    ### validation ###
    @property
    def check_flag(self):
        return bool(self._check_flag)


    @check_flag.setter
    def check_flag(self, flag):
        assert isinstance(flag, bool)
        self._check_flag = bool(flag)


    def _check_rep(self):
        """:return: True if representation invariants are true."""
        if self._check_flag:
            elements = self._elements.values()
            aligned = [e.aligned for e in elements]
            assert all([isinstance(e, _ActionElement) for e in elements])
            assert all(aligned) or (not any(aligned))
        return True

    ### printing ###
    @property
    def print_flag(self):
        return bool(self._print_flag)


    @print_flag.setter
    def print_flag(self, flag):
        if flag is None:
            self._print_flag = bool(ActionSet._default_print_flag)
        elif isinstance(flag, bool):
            self._print_flag = bool(flag)
        else:
            raise AttributeError('print_flag must be boolean or None')


    @property
    def df(self):
        """
        :return: data frame containing key action set parameters
        """
        df = pd.DataFrame({'name': self.name,
                           'variable_type': self.variable_type,
                           'lb': self.lb,
                           'ub': self.ub,
                           'grid_size': self.size,
                           'step_size': self.step_size,
                           'actionable': self.actionable,
                           'compatible': self.compatible,
                           'step_direction': self.step_direction,
                           'flip_direction': self.flip_direction})
        return df


    def to_latex(self):
        """
        :return: formatted latex table summarizing the action set for publications
        """
        return texify_action_set(self)

    #### alignment ####
    @property
    def y_desired(self):
        """
        :return: value of the desired outcome
        If y_desired = +1 (default value), then we consider points such that h(x) = -1, and search for actions a such that h(x+a) = +1
        If y_desired = -1 (or y_desired = 0), then we consider points such that h(x) = +1, and look for actions a such that h(x+a) = -1
        """
        return int(self._y_desired)


    @y_desired.setter
    def y_desired(self, y):
        """
        value of the desired outcome
        If y_desired = +1 (default value), then we consider points such that h(x) = -1, and search for actions a such that h(x+a) = +1
        If y_desired = -1 (or y_desired = 0), then we consider points such that h(x) = +1, and look for actions a such that h(x+a) = -1
        :param y: integer value: either +1 or 0/-1. If user inputs y_desired = 0, we will convert this value to y_desired = -1.
        :return:
        """
        assert y in (-1, 0, 1), 'y_desired must be +1 or -1'
        y = int(y)
        if y != self._y_desired:
            self._y_desired = 1 if y > 0 else -1
            # if classifier was aligned, then re-set_alignment the classifier
            if self.alignment_known:
                for n, j in self._indices.items():
                    self._elements[n].flip_direction = -self._elements[n].flip_direction


    def set_alignment(self, *args, **kwargs):
        """
        uses a classifier to determine the "alignment" of actions on each variable
        if the coefficient for variable j is positive, then actions that increase variable j will flip prediction
        if the coefficient for variable j is negative, then actions that decrease variable j will flip prediction
        :param scikit-learn classifier object, or vector of coefficients
        :return:None
        """
        coefs, _ = parse_classifier_args(*args, **kwargs)
        assert len(coefs) == len(self)
        flips = np.sign(coefs) if self._y_desired > 0 else -np.sign(coefs)
        for n, j in self._indices.items():
            self._elements[n].flip_direction = flips[j]


    @property
    def alignment_known(self):
        """
        :return: True if action set has been aligned to a model
        """
        return all([e.aligned for e in self._elements.values()])


    #### grid generation  ####
    def feasible_grid(self, x, return_actions = True, return_percentiles = True, return_compatible = True):
        """
        returns feasible features when features are x
        :param x: list or np.array containing vector of feature values (must have same length as ActionSet)
        :param action_grid: set to True for returned grid to reflect changes to x
        :param return_percentiles: set to True to include percentiles in return values
        :param return_compatible: set to True to build a grid using only compatible variables
        :return: dictionary of the form {name: feasible_values}
        """
        assert isinstance(x, (list, np.ndarray)), 'feature values should be list or np.ndarray'
        assert len(x) == len(self), 'dimension mismatch x should have len %d' % len(self)
        assert np.isfinite(x).all(), 'x must contain finite values'

        if return_compatible:
            output = {n: self._elements[n].feasible_values(x[j], return_actions, return_percentiles) for n, j in self._indices.items() if self._elements[n].compatible}
        else:
            output = {n: self._elements[n].feasible_values(x[j], return_actions, return_percentiles) for n, j in self._indices.items()}


        if len(self.constraints) > 0:
            # if x[j] is included in a subset limit constraint, and x[j] = 1, then we must include actions to decrease a[j]
            subset_limit_names = self.constraints.constrained_names(constraint_type = SubsetLimitConstraint)
            for n in subset_limit_names:
                j = self._names.index(n)
                output[n] = self._elements[n].feasible_values(x[j], return_actions, return_percentiles, drop_suboptimal = False)

        if return_percentiles:
            return {n: v[0] for n, v in output.items()}, {n: v[1] for n, v in output.items()}

        return output


    #### constraints ####
    @property
    def constraints(self):
        return self._constraints


    def add_constraint(self, constraint_type, **constraint_args):
        """
        add a constraint on multiple behaviors

        :param constraint_type: string indicating the constraint type.
                                supported types include: 'subset_limit_constraint'

        :param constraints_args: named arguments for specific type of constraint

        :return: id: string that can be used to access the constraint as ActionSet.constraints[id]

        -----
        'subset_limit_constraint'
        -----

        arguments:

        :param names: list of strings containing variable names that will be included in the constraint
        :param lb: minimum number of variables that can be changed by a feasible action
        :param ub: maximum number of variables that can be changed by a feasible action
        :param id: string representing the name of the constraint:

        usage:

        Say a model uses a one-hot encoding of a categorical variable V with values {v1,v2,...vk},
        so that X would include k indicator variables:

        V_is_v1 = 1[V == v1]
        V_is_v2 = 1[V == v2]
        ...
        V_is_vk = 1[V == vk]

        In such cases, we can add a subset limit to ensure that at most one indicator can be on at a time.
        We can ensure this by adding:

        action_set.constraints.add(names = ['V_is_v1', 'V_is_v2', ... 'V_is_vk'], #names of indicator variables of V
                                   lb = 0,
                                   ub = 1)

        :return:
        """

        if constraint_type == 'subset_limit':
            id = self._constraints.add_subset_limit_constraint(**constraint_args)
        else:
            raise ValueError('unsupported constraint type')

        return id


    def remove_constraint(self, id):
        """
        :param id: constraint id
        :return: True if the constraint was removed
        """
        return self._constraints.remove(id)


#### Constraints ####
SubsetLimitConstraint = namedtuple('SubsetLimitConstraint', ['id', 'names', 'indices', 'lb', 'ub'])
_VALID_CONSTRAINT_TYPES = [SubsetLimitConstraint]

class _ActionConstraints(object):
    """
    Class to represent and manipulate constraints between variables
    """

    def __init__(self, names):
        self._names = names
        self._n_variables = len(names)
        self._id_counter = 0
        self._constraints = {}

    def __len__(self):
        return len(self._constraints)

    def __repr__(self):
        s = ['%r' % str(v) for v in self._constraints.values()]
        s = '{' + '\n'.join(s) + '}'
        return s

    def __iter__(self):
        return self._constraints.values().__iter__()

    def constrained_names(self, constraint_type = None):
        """
        :param constraint_type: must be a valid constraint type
        :return:
        """
        if constraint_type is None:
            names = [c.names for c in self._constraints.values()]
        else:
            assert constraint_type in _VALID_CONSTRAINT_TYPES
            names = [c.names for c in self._constraints.values() if isinstance(c, constraint_type)]

        # choose unique names
        names = set(itertools.chain.from_iterable(names))
        return names


    def remove(self, id):
        """
        :param id: constraint id
        :return: True if the constraint was removed
        """
        assert isinstance(id, str) and len(id) > 0
        if id in self._constraints:
            self._constraints.pop(id)
        else:
            raise ValueError('no constraint named id')
        return True

    def add_subset_limit_constraint(self, names, lb = 0, ub = 1, id = None):
        """
        add a constraint to limit the number variables in a subset of variables that can be changed at the same time
        :param names: list of strings containing variable names that will be included in the constraint
        :param lb: minimum number of variables that can be changed by a feasible action
        :param ub: maximum number of variables that can be changed by a feasible action
        :param id: string representing the name of the constraint:
        :return: string representing the name of the warning
        -----

        Say a model uses a one-hot encoding of a categorical variable V with values {v1,v2,...vk},
        so that X would include k indicator variables:

        V_is_v1 = 1[V == v1]
        V_is_v2 = 1[V == v2]
        ...
        V_is_vk = 1[V == vk]

        In such cases, we can add a subset limit to ensure that at most one indicator can be on at a time.
        We can ensure this by adding:

        action_set.constraints.add(names = ['V_is_v1', 'V_is_v2', ... 'V_is_vk'], #names of indicator variables of V
                                   lb = 0,
                                   ub = 1)

        :return:
        """
        # parse constraint id
        id = '%d' % self._id_counter if id is None else id
        assert isinstance(id, str)
        assert len(id) > 0

        # parse variable names
        assert _check_variable_names(names)
        assert 2 <= len(names) <= self._n_variables
        indices = [self._names.index(n) for n in names]

        # check bounds
        lb = int(lb)
        ub = int(ub)
        assert 0 <= lb <= self._n_variables, 'lb must be between 0 to %d' % self._n_variables
        assert 0 <= ub <= self._n_variables, 'ub must be between 0 to %d' % self._n_variables
        assert lb <= ub

        # sort names to match order of names in X
        sort_idx = np.argsort(indices)
        names = [names[i] for i in sort_idx]
        indices = [indices[i] for i in sort_idx]

        # issue a warning if we are overwriting the constraint
        if id in self._constraints:
            warnings.warn('Overwriting constraint %s' % id)

        # add constraint
        self._constraints[id] = SubsetLimitConstraint(id = id, names = names, indices = indices, lb = lb, ub = ub)

        # update constraint id
        self._id_counter += 1

        return id


#### Elements ####
class _ActionElement(object):
    """
    Internal class to represent and manipulate actions for one feature
    ActionSet = Collection of ActionElements for each feature
    """

    _default_check_flag = False
    _valid_step_types = {'relative', 'absolute'}
    _valid_variable_types = {bool, int, float}


    def __init__(self, name, values, bounds = None, variable_type = None, actionable = True, step_type = 'relative', step_direction = 0, step_size = 0.01):
        """
        Represent and manipulate feasible actions for one feature

        :param name: name of the variable (at least 1 character)
        :param values: values of the variable (must be non-empty, non-nan, finite)
        :param bounds: bounds (must be a tuple of the form (lb, ub) or (lb, ub, bound_type) where bound_type is a valid bound type
        :param variable_type: 'int', 'float', 'bool' / set to None (default) to determine automatically from values
        :param step_direction: set to +1 if variable can only increase; -1 if variable can only decrease
        :param actionable: set to True if the variable can be changed
        """

        # set name (immutable)
        assert isinstance(name, str), 'name must be string (or unicode)'
        assert len(name) >= 1, 'name must have at least 1 character'
        self._name = str(name)  # store defensive copy

        # set values (immutable)
        values = np.sort(np.copy(values).flatten())
        assert len(values) >= 1, 'must have at least 1 value'
        assert np.all(np.isfinite(values)), 'values must be finite'
        self._values = values

        # set variable type
        self.variable_type = variable_type

        # flip direction
        self._flip_direction = float('nan')
        self.actionable = actionable

        # set bounds
        self.bounds = bounds

        # step related properties
        self.step_type = step_type
        self.step_size = step_size
        self.step_direction = step_direction
        self._grid = np.array([])
        self.update_grid()

        # interpolation
        self._interpolator = None
        assert self._check_rep()


    def _check_rep(self, check_flag = True):
        """
        :return: True if all representation invariants are true
        """
        if check_flag:
            assert self.lb <= self.ub, 'lb must be <= ub'
            g = self._grid
            assert len(g) == len(np.unique(g)), 'grid is not unique'
            assert np.all(np.isfinite(g)), 'grid contains elements that are nan or inf'
            assert np.all(g[:-1] <= g[1:]), 'grid is not sorted'
        return True


    def __len__(self):
        return len(self._grid)


    def __repr__(self):
        return '%r: (%r, %r)' % (self._name, self._bounds.lb, self._bounds.ub)

    #### core properties ####

    @property
    def name(self):
        """:return: name of the variable"""
        return self._name


    @property
    def values(self):
        """:return: array containing observed values for this variable."""
        return np.copy(self._values)


    @property
    def actionable(self):
        """:return: True iff variable can be changed."""
        return self._mutable


    @actionable.setter
    def actionable(self, flag):
        assert np.isin(flag, (False, True)), 'compatible must be boolean'
        self._mutable = bool(flag)


    @property
    def compatible(self):
        """
        :return: True if variable is actionable, and there exists a feasible actions that will flip the prediction of a classifier
        """
        if not self.aligned:
            return float('nan')

        if not self.actionable:
            return False

        # if actionable, then check that directions OK
        sd = np.sign(self._step_direction)
        fd = np.sign(self._flip_direction)
        conflict = (fd == 0) or (fd * sd == -1)
        compatible = not conflict

        return compatible


    @property
    def variable_type(self):
        """:return: variable type."""
        return self._variable_type


    @variable_type.setter
    def variable_type(self, variable_type):
        if variable_type is None:
            self._variable_type = _determine_variable_type(self._values, self._name)
        else:
            assert variable_type in self._valid_variable_types
            self._variable_type = variable_type


    @property
    def size(self):
        """:return: # of points in action grid """
        # defined in addition to __len__ so that we can access len using ActionSet.__getattr__
        return len(self._grid)


    #### bounds ####

    @property
    def bound_type(self):
        return self._bounds.bound_type


    @bound_type.setter
    def bound_type(self):
        b = self._bounds
        if b.bound_type == 'percentile':
            b_new = _BoundElement(bound_type = 'percentile', lb = b.qlb, ub = b.qub, values = self._values)
        else:
            b_new = _BoundElement(bound_type = b.bound_type, lb = b.lb, ub = b.ub, values = self._values)
        self._bounds = b_new


    @property
    def bounds(self):
        return self._bounds


    @bounds.setter
    def bounds(self, b):
        if isinstance(b, (list, tuple)):
            if len(b) == 2:
                b = _BoundElement(values = self._values, lb = b[0], ub = b[1])
            elif len(b) == 3:
                b = _BoundElement(values = self._values, lb = b[0], ub = b[1], bound_type = b[2])
        elif b is None:
            b = _BoundElement(values = self._values)
        assert isinstance(b, _BoundElement), 'bounds must be a list/tuple of the form (lb, ub) or (lb, ub, bound_type)'
        self._bounds = b


    @property
    def lb(self):
        return self._bounds.lb


    @lb.setter
    def lb(self, value):
        b = self._bounds
        if b.bound_type == 'percentile':
            b_new = _BoundElement(bound_type = 'percentile', lb = value, ub = b.qub, values = self._values)
        else:
            b_new = _BoundElement(bound_type = b.bound_type, lb = value, ub = b.ub, values = self._values)
        self._bounds = b_new


    @property
    def ub(self):
        return self._bounds.ub


    @ub.setter
    def ub(self, value):
        b = self._bounds
        if b.bound_type == 'percentile':
            b_new = _BoundElement(bound_type = 'percentile', lb = b.qlb, ub = value, values = self._values)
        else:
            b_new = _BoundElement(bound_type = b.bound_type, lb = b.lb, ub = value, values = self._values)
        self._bounds = b_new


    #### alignment ####
    @property
    def aligned(self):
        return not np.isnan(self._flip_direction)


    @property
    def flip_direction(self):
        if self.aligned:
            return int(self._flip_direction)
        else:
            return float('nan')


    @flip_direction.setter
    def flip_direction(self, flip_direction):
        assert np.isfinite(flip_direction), "flip_direction must be finite"
        self._flip_direction = int(np.sign(flip_direction))

    #### grid directions ####

    @property
    def step_type(self):
        return self._step_type


    @step_type.setter
    def step_type(self, step_type):
        assert isinstance(step_type, str), '`step_type` must be str'
        assert step_type in self._valid_step_types, '`step_type` is %r (must be %r)' % (step_type, self._valid_step_types)
        self._step_type = str(step_type)


    @property
    def step_direction(self):
        return self._step_direction


    @step_direction.setter
    def step_direction(self, step_direction):
        assert np.isfinite(step_direction), "step_direction must be finite"
        self._step_direction = np.sign(step_direction)


    @property
    def step_size(self):
        return self._step_size


    @step_size.setter
    def step_size(self, s):
        assert isinstance(s, (float, int, bool, np.ndarray))
        assert np.greater(s, 0.0)
        if self._step_type == 'relative':
            assert np.less_equal(s, 1.0)
        self._step_size = float(s)


    @property
    def grid(self):
        return np.array(self._grid)


    def update_grid(self):
        """Generate grid of feasible values"""

        # end points
        start = self.lb
        stop = self.ub
        step = self.step_size

        if self._variable_type == int:
            start = np.floor(self.lb)
            stop = np.ceil(self.ub)

        if self.step_type == 'relative':
            step = np.multiply(step, stop - start)

        if self._variable_type == int:
            step = np.ceil(step)

        # generate grid
        grid = np.arange(start, stop + step, step)

        # cast grid
        if self._variable_type == int:
            grid = grid.astype('int')

        self._grid = grid


    #### kde and percentile computation ###
    @property
    def interpolator(self):
        if self._interpolator is None:
            self.update_interpolator()
        return self._interpolator


    def update_interpolator(self, left_buffer = 1e-6, right_buffer = 1e-6):
        # check buffers
        left_buffer = float(left_buffer)
        right_buffer = float(right_buffer)
        assert 0.0 <= left_buffer < 1.0
        assert 0.0 <= right_buffer < 1.0
        assert left_buffer + right_buffer < 1.0

        # build kde estimator using observed values
        kde_estimator = kde(self._values)

        # build the CDF over the grid
        pdf = kde_estimator(self._grid)
        cdf_raw = np.cumsum(pdf)
        total = cdf_raw[-1] + left_buffer + right_buffer
        cdf = (left_buffer + cdf_raw) / total
        self._interpolator = interp1d(x = self._grid, y = cdf, copy = False, fill_value = (left_buffer, 1.0 - right_buffer), bounds_error = False, assume_sorted = True)


    def percentile(self, x):
        return self.interpolator(x)


    #### methods ####
    def feasible_values(self, x, return_actions = True, return_percentiles = False, drop_suboptimal = True):

        """
        returns an array of feasible values or actions for this feature from a specific point x
        array of feasible values will always include x (or an action = 0.0)

        :param x: feature vector

        :param return_actions: if True, returns an array of actions a where x_new = x + a
                               if False, returns an array of new values x_new

        :param return_percentiles: if True, then percentiles of all new points will also be included

        :return:

        """
        assert np.isfinite(x), 'x must be finite.'
        assert return_actions is False or self.aligned, 'cannot determine feasible_actions before ActionSet is aligned'

        if self.actionable:

            # obtain a grid of all x-values between lb to ub
            x_new = self.grid

            # drop values that are infeasible due to constraints on the direction of change
            if self._step_direction > 0:
                x_new = np.extract(np.greater_equal(x_new, x), x_new)
            elif self._step_direction < 0:
                x_new = np.extract(np.less_equal(x_new, x), x_new)

            # include current point in the grid if it does not exist
            if not x in x_new:
                x_new = np.insert(x_new, np.searchsorted(x_new, x), x)

        else:
            x_new = np.array([x]) # if variable is not actionable, then x_new \in [x]


        # if drop suboptimal is true, then we drop actions are improve the chance of attaining y_desired
        if drop_suboptimal:
            if self._flip_direction > 0:
                x_new = np.extract(np.greater_equal(x_new, x), x_new)
            elif self._flip_direction < 0:
                x_new = np.extract(np.less_equal(x_new, x), x_new)

        if return_actions:
            vals = x_new - x
        else:
            vals = x_new

        if return_percentiles:
            return vals, self.percentile(x_new)

        return vals


class _BoundElement(object):
    """
    Immutable class to store the lower and upper bounds for a feature.
    """

    _valid_variable_types = {bool, int, float}
    _valid_bound_types = {'absolute', 'percentile'}
    _valid_bound_codes = {'a': 'absolute', 'p': 'percentile'}


    def __init__(self, bound_type = 'absolute', lb = None, ub = None, values = None, variable_type = None):
        """
        :param bound_type: `absolute` / `a` (default) or `percentile` / `p`

        :param lb:      value of lower bound (numeric);
                        set as min(values) by default;
                        must be within [0.0, 100.0] if bound_type is `percentile`

        :param ub:      value of upper bound (numeric);
                        set as max(values) by default;
                        must be within [0.0, 100.0] if bound_type is `percentile`

        :param values:  observed values for variable;
                        required if `bound_type` is `percentile`;
                        used to validate bounds if `bound_type` = `absolute`

        :param variable_type: the data type of the dimension this bound is being used for. Must be in
                        {int, float}
        """

        # set variable type
        if variable_type is None:
            assert values is not None
            variable_type = _determine_variable_type(values)
        else:
            assert variable_type in self._valid_variable_types
        self._variable_type = variable_type

        # set bound type
        assert isinstance(bound_type, str)
        if bound_type in self._valid_bound_codes:
            bound_type = self._valid_bound_codes[bound_type]
        else:
            assert bound_type in self._valid_bound_types

        self._bound_type = str(bound_type)

        if bound_type == 'percentile':
            assert values is not None
            values = np.array(values).flatten()

            assert isinstance(lb, (float, int, bool, np.ndarray))
            assert isinstance(ub, (float, int, bool, np.ndarray))
            assert 0.0 <= lb <= 100.0
            assert 0.0 <= ub <= 100.0

            self._qlb = lb
            self._qub = ub

            lb = np.percentile(values, lb)
            ub = np.percentile(values, ub)


        if bound_type == 'absolute':
            if lb is None:
                assert values is not None, "If using 'absolute' bounds without a specified lower bound, please pass in data."
                lb = np.min(values)
            else:
                assert isinstance(lb, (float, int, bool)) or (isinstance(lb, np.ndarray) and len(lb) == 1)

            if ub is None:
                assert values is not None, "If using 'absolute' bounds without a specified upper bound, please pass in data."
                ub = np.max(values)
            else:
                assert isinstance(ub, (float, int, bool)) or (isinstance(ub, np.ndarray) and len(ub) == 1)

            self._qlb = 0.0
            self._qub = 100.0

            # if values is not None:
            #     assert np.less_equal(lb, np.min(values))
            #     assert np.greater_equal(ub, np.max(values))

        assert np.less_equal(lb, ub), "Lower bound must be less than or equal to the upper bound."

        if variable_type == int:
            lb = np.floor(lb)
            ub = np.ceil(ub)

        # set lower bound and upper bound
        self._lb = float(lb)
        self._ub = float(ub)


    @property
    def bound_type(self):
        return self._bound_type


    @property
    def lb(self):
        """ value of the lower bound """
        return self._lb


    @property
    def ub(self):
        """ value of the lower bound """
        return self._ub


    @property
    def qlb(self):
        """ value of the lower bound (as a percentile) """
        return self._qlb


    @property
    def qub(self):
        """ value of the upper bound bound (as a percentile) """
        return self._qub


    def __repr__(self):
        return "(%r, %r, %r)" % (self._lb, self._ub, self._bound_type)


class _ActionSlice(object):
    """
    Class to set ActionElement properties by slicing.
    This class allows us to support commands like:
        a = ActionSet(...)
        a[1:2].ub = 2
    """

    def __init__(self, action_elements):
        self._indices = {e.name: j for j, e in enumerate(action_elements)}
        self._elements = {e.name: e for e in action_elements}

    def __getattr__(self, name):
        if name in ('_indices', '_elements'):
            object.__getattr__(self, name)
        else:
            return [getattr(self._elements[n], name) for n, j in self._indices.items()]

    def __setattr__(self, name, value):
        if name in ('_indices', '_elements'):
            object.__setattr__(self, name, value)
        else:
            assert hasattr(_ActionElement, name)
            attr_values = _expand_values(value, len(self._indices))
            for n, j in self._indices.items():
                setattr(self._elements[n], name, attr_values[j])

    def __len__(self):
        return len(self._indices)

    def __str__(self):
        return tabulate_actions(self)

    def __repr__(self):
        return str(self)


#### Helper Functions ####
def _check_variable_names(names):
    """
    checks variable names
    :param names: list of names for each feature in a dataset.
    :return:
    """
    assert isinstance(names, list), '`names` must be a list'
    assert all([isinstance(n, str) for n in names]), '`names` must be a list of strings'
    assert len(names) >= 1, '`names` must contain at least 1 element'
    assert all([len(n) > 0 for n in names]), 'elements of `names` must have at least 1 character'
    assert len(names) == len(set(names)), 'elements of `names` must be distinct'
    return True


def _determine_variable_type(values, name=None):
    for v in values:
        if isinstance(v, str):
            raise ValueError(">=1 elements %s are of type str" % ("in '%s'" % name if name else ''))
    integer_valued = np.equal(np.mod(values, 1), 0).all()
    if integer_valued:
        if np.isin(values, (0, 1)).all():
            return bool
        else:
            return int
    else:
        return float


def _expand_values(value, m):

    if isinstance(value, np.ndarray):

        if len(value) == m:
            value_array = value
        elif value.size == 1:
            value_array = np.repeat(value, m)
        else:
            raise ValueError("length mismatch; need either 1 or %d values" % m)

    elif isinstance(value, list):
        if len(value) == m:
            value_array = value
        elif len(value) == 1:
            value_array = [value] * m
        else:
            raise ValueError("length mismatch; need either 1 or %d values" % m)

    elif isinstance(value, str):
        value_array = [str(value)] * m

    elif isinstance(value, bool):
        value_array = [bool(value)] * m

    elif isinstance(value, int):
        value_array = [int(value)] * m

    elif isinstance(value, float):
        value_array = [float(value)] * m

    else:
        raise ValueError("unknown variable type %s")

    return value_array


def tabulate_actions(action_set):
    """
    prints a table with information about each element in the action set
    :param action_set: ActionSet object
    :return:
    """
    assert isinstance(action_set, ActionSet)

    t = PrettyTable()
    t.add_column("name", action_set.name, align = "r")
    t.add_column("variable type", action_set.variable_type, align = "r")
    t.add_column("actionable", action_set.actionable, align = "r")
    t.add_column("compatible", action_set.compatible, align = "r")
    t.add_column("step direction", action_set.step_direction, align = "r")
    t.add_column("flip direction", action_set.flip_direction, align = "r")
    t.add_column("grid size", action_set.size, align = "r")
    t.add_column("step type", action_set.step_type, align = "r")
    t.add_column("step size", action_set.step_size, align = "r")
    t.add_column("lb", action_set.lb, align = "r")
    t.add_column("ub", action_set.ub, align = "r")
    return str(t)


def texify_action_set(action_set):
    """
    :param action_set: ActionSet object
    :return: formatted latex table summarizing the action set for publications
    """
    assert isinstance(action_set, ActionSet)

    tex_binary_str = '$\{0,1\}$'
    tex_integer_str = '$\mathbb{Z}$'
    tex_real_str = '$\mathbb{R}$'

    df = action_set.df
    df = df.drop(['compatible', 'flip_direction'], axis = 1)

    new_types = [tex_real_str] * len(df)
    new_ub = ['%1.1f' % v for v in df['ub'].values]
    new_lb = ['%1.1f' % v for v in df['lb'].values]

    for i, t in enumerate(df['variable_type']):
        ub, lb = df['ub'][i], df['lb'][i]
        if t == 'int':
            new_ub[i] = '%d' % int(ub)
            new_lb[i] = '%d' % int(lb)
            new_types[i] = tex_binary_str if lb == 0 and ub == 1 else tex_integer_str

    df['variable_type'] = new_types
    df['ub'] = new_ub
    df['lb'] = new_lb

    df['mutability'] = df['actionable'].map({False: 'no', True: 'yes'}) #todo change
    up_idx = df['actionable'] & df['step_direction'] == 1
    dn_idx = df['actionable'] & df['step_direction'] == -1
    df.loc[up_idx, 'mutability'] = 'only increases'
    df.loc[dn_idx, 'mutability'] = 'only decreases'

    df = df.drop(['actionable', 'step_direction'], axis = 1)

    df = df.rename(columns = {
        'name': 'Name',
        'grid_size': '\# Actions',
        'variable_type': 'Type',
        'actionability': 'Actionability',
        'lb': 'LB',
        'ub': 'UB',
        })

    table = df.to_latex(index = False, escape = False)
    return table