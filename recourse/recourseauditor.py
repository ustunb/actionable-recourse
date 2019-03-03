import time
import warnings
import numpy as np
from recourse.builder import RecourseBuilder, _SOLVER_TYPE_CPX, _SOLVER_TYPE_CBC
from recourse.action_set import ActionSet


class RecourseAuditor(object):

    def __init__(self, X, y = None, actionset = None, optimizer = _SOLVER_TYPE_CPX, decision_threshold = None, **clf_args):
        """
        Run an audit on a classifier.

        :param optimizer:
        :param clf:
        :param coefficients:
        :param intercept:
        :param actionset:
        """
        ## set clf and coefficients
        self.__parse_clf_args(clf_args)

        ### actionset
        self.actionset = actionset
        self.X = X

        if not self.actionset:
            warnings.warn("No actionset provided, instantiating with defaults: all features mutable, all features percentile.")
            if not self.X:
                raise("No actionset or X provided.")
            self.actionset = ActionSet(X = self.X)
            self.actionset.align(self.coefficients)

        self.optimizer = optimizer
        self.decision_threshold = decision_threshold


    def __parse_clf_args(self, args):

        assert 'clf' in args or ('coefficients' in args)

        if 'clf' in args:

            clf = args['clf']
            self.coefficients = np.array(clf.coef_).flatten()
            self.intercept = float(clf.intercept_)

        elif 'coefficients' in args:
            self.coefficients = args['coefficients']
            self.intercept = args['intercept'] if 'intercept' in args else 0.0


    def get_negative_points(self):
        scores = self.clf.predict_proba(self.X)[:, 1]
        return np.where(scores < self.decision_threshold)[0]


    def audit(self, num_cases = None):

        ### TODO: bake decision threshold into the optimizer.

        denied_individuals = self.get_negative_points()
        ## downsample
        if num_cases and num_cases < len(denied_individuals):
            denied_individuals = np.random.choice(denied_individuals, num_cases)

        if not any(self.actionset.aligned):
            self.actionset.align(self.coefficients)

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
                action_set = self.actionset,
                x=x
            )

            output = fb.fit()
            flipsets[i] = output.get('total_cost') or output.get('max_cost')
            idx += 1

        return flipsets