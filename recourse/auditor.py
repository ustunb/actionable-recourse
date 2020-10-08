import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from recourse.defaults import DEFAULT_SOLVER
from recourse.helper_functions import parse_classifier_args
from recourse.action_set import ActionSet
from recourse.builder import RecourseBuilder

__all__ = ['RecourseAuditor']

# todo add timer / print
class RecourseAuditor(object):
    """
    Compute feasibility and cost of recourse over a sample of points that were denied access.
    (i.e. this method will not be run on data points that are already qualifying, (eg. y_pred > 0).
    """

    _default_print_flag = True


    def __init__(self, action_set, **kwargs):
        """
        :param action_set: ActionSet  for features
        :param clf: scikit-learn linear classifier
        :param coefficients: vector of coefficients (only used when clf is not specified)
        :param intercept: set to 0.0 by default (only used when clf is not specified)
        :param solver: valid MIP solver
        """

        # action_set
        assert isinstance(action_set, ActionSet)
        self.action_set = action_set

        # attach coefficients
        self.coefficients, self.intercept = parse_classifier_args(**kwargs)

        # set_alignment coefficients to action set
        self.action_set.set_alignment(self.coefficients)

        # set solver
        self.solver = kwargs.get('solver', DEFAULT_SOLVER)

        # setup recourse problem
        self.builder = RecourseBuilder(coefficients = self.coefficients,
                                       intercept = self.intercept,
                                       action_set = self.action_set,
                                       solver = self.solver)

        self._print_flag = kwargs.get('print_flag', self._default_print_flag)


    @property
    def print_flag(self):
        return self._print_flag


    @print_flag.setter
    def print_flag(self, flag):
        if flag is None:
            self._print_flag = bool(self._default_print_flag)
        elif isinstance(flag, bool):
            self._print_flag = bool(flag)
        else:
            raise AttributeError('print_flag must be boolean or None')


    def audit(self, X, y_desired = 1):
        """
        evaluate cost and feasibility of recourse for for each point in X
        that is not assigned a desired outcome

        :param X: feature matrix (np.array or pd.DataFrame)
        :param y_desired: desired label (+1 by default)
        :return: pd.DataFrame containing the feasibility and cost of recourse for each point in X
                 rows that already attain desired outcome have entries: feasible = NaN & cost = NaN
                 rows that are certified to have no recourse have entries: feasible = False & cost = Inf
        """

        if isinstance(X, pd.DataFrame):
            raw_index = X.index.tolist()
            X = X.values
        else:
            raw_index = list(range(X.shape[0]))

        assert isinstance(X, np.ndarray)
        assert X.ndim == 2
        assert X.shape[0] >= 1
        assert X.shape[1] == len(self.coefficients)
        assert np.isfinite(X).all()
        assert float(y_desired) in {1.0, -1.0, 0.0}

        U, distinct_idx = np.unique(X, axis = 0, return_inverse = True)
        scores = U.dot(self.coefficients)
        if y_desired > 0:
            audit_idx = np.less(scores, -self.intercept)
        else:
            audit_idx = np.greater_equal(scores, -self.intercept)
        audit_idx = np.flatnonzero(audit_idx)

        # solve recourse problem
        output = []
        pbar = tqdm(total=len(audit_idx)) ## stop tqdm from playing badly in ipython notebook.
        for idx in audit_idx:
            self.builder.x = U[idx, :]
            info = self.builder.fit()
            info['idx'] = idx
            output.append({k: info[k] for k in ['feasible', 'cost', 'idx']})
            pbar.update(1)
        pbar.close()

        # add in points that were not denied recourse
        df = pd.DataFrame(output)
        df = df.set_index('idx')

        # include unique points that attain desired label already
        df = df.reindex(range(U.shape[0]))

        # include duplicates of original points
        df = df.iloc[distinct_idx]
        df = df.reset_index(drop = True)
        df.index = raw_index
        return df

