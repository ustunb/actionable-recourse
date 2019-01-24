import numpy as np
from recourse.flipset import FlipsetBuilder
from recourse.action_set import ActionSet
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
import warnings
import time
import pandas as pd

class Audit(object):
    def __init__(self, optimizer="cplex",
                 clf=None, coefficients=None, intercept=None,
                 actionset=None, dataset=None, decision_threshold=None
                 ):
        """
        Run an audit on a classifier.

        :param optimizer:
        :param clf:
        :param coefficients:
        :param intercept:
        :param actionset:
        """
        ## set clf and coefficients
        self.clf = clf
        if clf==None:
            if coefficients==None and intercept==None:
                warnings.warn("No model provided at compiletime, please provide at runtime.")
            else:
                self.coefficients = coefficients
                self.intercept = intercept

        if coefficients==None and intercept==None and clf != None:
                if isinstance(clf, (LogisticRegression, SVC)):
                    self.coefficients = clf.coef_[0]
                    self.intercept = clf.intercept_[0]
                elif isinstance(clf, SVC):
                    try:
                        self.coefficients = clf.coef_[0]
                        self.intercept = clf.intercept_[0]
                    except AttributeError:
                        raise("Please run SVC with linear kernel.")
                elif isinstance(clf, LinearRegression):
                    self.coefficients = clf.coef_
                    self.intercept = clf.intercept_

        ### actionset
        self.actionset = actionset
        self.dataset=dataset
        if not self.actionset:
            warnings.warn("No actionset provided, instantiating with defaults: all features mutable, all features percentile.")
            if not self.dataset:
                raise("No actionset or dataset provided.")
            self.actionset = ActionSet(X = self.dataset)
            self.actionset.align(self.coefficients)

        self.optimizer=optimizer
        self.decision_threshold = decision_threshold


    def get_negative_points(self):
        scores = self.clf.predict_proba(self.dataset)[:, 1]
        return np.where(scores < self.decision_threshold)[0]


    def run_audit(self, num_cases=None):
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

            x = self.dataset[i]
            fb = FlipsetBuilder(
                optimizer=self.optimizer,
                coefficients=self.coefficients,
                intercept=self.intercept,
                action_set=self.actionset,
                x=x
            )

            output = fb.fit()
            flipsets[i] = output.get('total_cost') or output.get('max_cost')
            idx += 1

        return flipsets