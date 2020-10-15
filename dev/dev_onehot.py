import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from recourse import ActionSet, RecourseBuilder
from recourse.defaults import _SOLVER_TYPE_CPX, _SOLVER_TYPE_PYTHON_MIP
from recourse.paths import test_dir

data_name = 'german'
data_file = test_dir / ('%s_processed.csv' % data_name)

## load dataset
data_df = pd.read_csv(data_file)
outcome_name = data_df.columns[0]
y = data_df[outcome_name]
X = data_df.drop([outcome_name, 'Gender', 'PurposeOfLoan', 'OtherLoansAtStore'], axis=1)

# one-hot encoding for categorical variables
X_onehot = pd.get_dummies(data_df[['PurposeOfLoan']], prefix_sep = '_is_')
names_onehot = X_onehot.columns.to_list()
X = X.join(X_onehot)

# setup actionset
a = ActionSet(X = X)
names_immutable = ['Age', 'Single', 'JobClassIsSkilled', 'ForeignWorker', 'OwnsHouse', 'RentsHouse']
a[names_immutable].actionable = False
a['CriticalAccountOrLoansElsewhere'].step_direction = -1
a['CheckingAccountBalance_geq_0'].step_direction = 1


# only allow changes in the one-hot variable
a.actionable = False
a[names_onehot].actionable = True
id = a.add_constraint(constraint_type = 'subset_limit', names = names_onehot, lb = 1, ub = 1)


# fit classifier, get median score, and get denied individuals.
clf = LogisticRegression(max_iter=1000, solver = 'lbfgs')
clf.fit(X, y)
coefficients = clf.coef_[0]
intercept = clf.intercept_[0]

a.y_desired = 1
yhat = clf.predict(X)
x = X[yhat != a.y_desired].values[0]
assert clf.predict(x[None, :]) != a.y_desired

## CPLEX
fb_cplex = RecourseBuilder(
    solver=_SOLVER_TYPE_CPX,
    coefficients=coefficients,
    intercept=intercept,
    action_set = a,
    x=x
)
fb_cplex.fit()


## CBC
fb_pymip = RecourseBuilder(
    solver=_SOLVER_TYPE_PYTHON_MIP,
    coefficients=coefficients,
    intercept=intercept,
    action_set = a,
    x=x
)
fb_pymip.fit()