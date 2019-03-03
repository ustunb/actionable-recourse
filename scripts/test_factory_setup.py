import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scripts.paths import *
from recourse.auditor import RecourseAuditor
from recourse.action_set import ActionSet

data_name = 'german'
data_file = data_dir / ('%s_processed.csv' % data_name)

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

# fit classifier
clf = LogisticRegression(max_iter=1000, solver = 'lbfgs')
clf.fit(X, y)

#
scores = pd.Series(clf.predict_proba(X)[:, 1])
coefficients = clf.coef_[0]
intercept = clf.intercept_[0]
action_set.align(coefficients=coefficients)
# p = .8
p = scores.median()
denied_individuals = scores.loc[lambda s: s<=p].index

### load cache
coefficients = clf.coef_[0]
intercept = clf.intercept_[0]
p = scores.median()
denied_individuals = scores.loc[lambda s: s <= p].index

idx = denied_individuals[5]



x = X.iloc[idx].values
t_cplex = RecourseBuilder(
    optimizer='cplex',
    action_set=action_set,
    coefficients=coefficients,
    intercept=intercept- (np.log(p / (1. - p))),
    x=x
)
cplex_output = t_cplex.fit()
cplex_output_df = pd.DataFrame(cplex_output)[['actions', 'costs']]
print(cplex_output_df)

t_pyomo = RecourseBuilder(
    optimizer='cbc',
    action_set=action_set,
    coefficients=coefficients,
    intercept=intercept - (np.log(p / (1. - p))),
    x=x
)
pyo_output = t_pyomo.fit()
max_cost = pyo_output.pop('max_cost', None)
pyo_output_df = (pd.DataFrame
 .from_dict(pyo_output, orient='index')
 .loc[lambda df: df['u'] == 1]
)
print(pyo_output_df)


