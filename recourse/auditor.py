import numpy as np
import pandas as pd
from recourse.defaults import DEFAULT_SOLVER
from recourse.helper_functions import parse_classifier_args
from recourse.action_set import ActionSet
from recourse.builder import RecourseBuilder


class RecourseAuditor(object):

    """
    """
    def __init__(self, action_set, **kwargs):
        """
        :param action_set:
        :param solver:
        :param kwargs: either clf or coefficient
        """

        # action_set
        assert isinstance(action_set, ActionSet)
        self.action_set = action_set

        # attach coefficients
        self.coefficients, self.intercept = parse_classifier_args(**kwargs)

        # align coefficients to action set
        self.action_set.align(self.coefficients)
        self.solver = kwargs.get('solver', DEFAULT_SOLVER)


    def audit(self, X):

        ### TODO: bake decision threshold into the optimizer.
        if isinstance(X, pd.DataFrame):
            X = X.values

        assert X.shape[0] >= 1
        assert X.shape[1] == len(self.coefficients)

        U, sample_idx = np.unique(X, axis = 0, return_inverse = True)
        audit_idx = np.flatnonzero(np.less(U.dot(self.coefficients), self.intercept))

        ## run flipsets
        all_output = []
        for i in audit_idx:

            rb = RecourseBuilder(solver = self.solver,
                                 coefficients = self.coefficients,
                                 intercept = self.intercept,
                                 action_set = self.action_set,
                                 x = U[i, :])
            output = rb.fit()
            output['idx'] = i
            all_output.append({k: output[k] for k in ['feasible', 'cost', 'idx']})

        df = pd.DataFrame(all_output)
        self._df = pd.DataFrame(df)

        return df