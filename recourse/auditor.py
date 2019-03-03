import numpy as np
import pandas as pd
from recourse.defaults import _DEFAULT_SOLVER
from recourse.action_set import ActionSet
from recourse.builder import RecourseBuilder


class RecourseAuditor(object):
    """
    """

    def __init__(self, action_set, **kwargs):
        """

        :param X:
        :param action_set:
        :param solver:
        :param decision_threshold:
        :param clf_args:
        """

        # action_set
        assert isinstance(action_set, ActionSet)
        self.action_set = action_set

        # set clf and coefficients
        self.__parse_classifier_args(**kwargs)
        self.action_set.align(self.coefficients)
        self.solver = kwargs.get('solver', _DEFAULT_SOLVER)


    def __parse_classifier_args(self, **kwargs):
        """
        :param kwargs:
        :return:
        """
        assert 'clf' in kwargs or 'coefficients' in kwargs
        if 'clf' in kwargs:
            clf = kwargs.get('clf')
            w = np.array(clf.coef_)
            t = float(clf.intercept_)

        elif 'coefficients' in kwargs:
            w = kwargs.get('coefficients')
            t = kwargs.get('intercept', 0.0)

        self.intercept = float(t)
        self.coefficients = np.array(w).flatten()


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
            all_output.append(output)

        return all_output