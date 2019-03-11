import pandas as pd
from sklearn.linear_model import LogisticRegression
from recourse.paths import *
from recourse.builder import RecourseBuilder, _SOLVER_TYPE_CBC, _SOLVER_TYPE_CPX
from recourse.action_set import ActionSet
import numpy as np

data_name = 'german'
data_file = test_dir / ('%s_processed.csv' % data_name)

## load dataset
data_df = pd.read_csv(data_file)
outcome_name = data_df.columns[0]
y = data_df[outcome_name]
X = data_df.drop([outcome_name, 'Gender', 'PurposeOfLoan', 'OtherLoansAtStore'], axis=1)

# setup actionset
action_set = ActionSet(X = X)
immutable_attributes = ['Age', 'Single', 'JobClassIsSkilled', 'ForeignWorker', 'OwnsHouse', 'RentsHouse']
action_set[immutable_attributes].mutable = False
action_set['CriticalAccountOrLoansElsewhere'].step_direction = -1
action_set['CheckingAccountBalance_geq_0'].step_direction = 1

# fit classifier, get median score, and get denied individuals.
clf = LogisticRegression(max_iter=1000, solver = 'lbfgs')
clf.fit(X, y)
coefficients = clf.coef_[0]
intercept = clf.intercept_[0]
scores = pd.Series(clf.predict_proba(X)[:, 1])
p = scores.median()
denied_individuals = scores.loc[lambda s: s <= p].index

idx = denied_individuals[0]
x = X.values[idx]

## CPLEX
fb_cplex = RecourseBuilder(
    solver=_SOLVER_TYPE_CPX,
    coefficients=coefficients,
    intercept=intercept - (np.log(p / (1. - p))),
    action_set=action_set,
    x=x
)
fb_cplex.fit()

## CBC
fb_cbc = RecourseBuilder(
    solver=_SOLVER_TYPE_CBC,
    coefficients=coefficients,
    intercept=intercept - (np.log(p / (1. - p))),
    action_set=action_set,
    x=x
)
fb_cbc.fit()