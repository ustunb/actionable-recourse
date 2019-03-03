import time
import warnings
import numpy as np
from recourse.builder import RecourseBuilder, _SOLVER_TYPE_CPX, _SOLVER_TYPE_CBC
from recourse.action_set import ActionSet

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

        ## set clf and coefficients
        self.__parse_classifier_args(**kwargs)

        ### action_set
        assert isinstance(action_set, ActionSet)
        self.action_set = action_set

        if not self.action_set:
            if not self.X:
                raise("No action_set or X provided.")
            self.action_set = ActionSet(X = self.X)

        self.action_set.align(self.coefficients)
        self.optimizer = kwargs.get('solver', _SOLVER_TYPE_CPX)


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

        audit_idx = np.less(self.coefficients.dot(X), self.intercept)


        ## run flipsets
        idx = 0
        flipsets = {}
        now = time.time()
        for i in denied_individuals:
            if idx % 50 == 0:
                print('finished %d points in %f...' % (idx, time.time() - now))
                now = time.time()

            x = self.X[i]
            fb = RecourseBuilder(
                optimizer = self.optimizer,
                coefficients = self.coefficients,
                intercept = self.intercept,
                action_set = self.action_set,
                x=x
            )

            output = fb.fit()
            flipsets[i] = output.get('total_cost') or output.get('max_cost')
            idx += 1

        return flipsets