import pandas as pd
import numpy as np
 # import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import time
import os

import pickle
from recourse.path import *
from recourse.action_set import ActionSet
from recourse.flipset import FlipsetBuilder

# os.listdir('data')
data_dir = 'data'
data_name = 'german'
data_file = os.path.join(data_dir, '%s_processed.csv' % data_name)
demo_results_dir = os.path.join(results_dir, data_name)
file_header = '%s%s' % (demo_results_dir, data_name)
cache_dir = os.path.join(repo_dir, "scripts", "results_intermediate", data_name)

def as_result_file(name, extension = 'pdf', header = file_header):
    return os.path.join(demo_results_dir, '%s.%s' % (name, extension))

## load and process data
german_df = pd.read_csv(data_file).reset_index(drop=True)
# german_df = german_df.assign(isMale=lambda df: (df['Gender']=='Male').astype(int))#.drop(['PurposeOfLoan', 'Gender', 'OtherLoansAtStore'], axis=1)
y = german_df['GoodCustomer']
X = (german_df.drop('GoodCustomer', axis=1)
     .drop(['PurposeOfLoan', 'Gender', 'OtherLoansAtStore'], axis=1)
     )



## set up actionset
gender_weight = german_df.assign(c=1).groupby('Gender')['c'].transform(lambda s: s*1./len(s))
X_gender_balanced = X.sample(n = len(X)*3, replace=True, weights=gender_weight)
action_set = ActionSet(X = X_gender_balanced)
action_set['Age'].mutable = False
action_set['Single'].mutable = False
action_set['JobClassIsSkilled'].mutable = False
action_set['ForeignWorker'].mutable = False
action_set['OwnsHouse'].mutable = False
action_set['RentsHouse'].mutable = False
action_set['CriticalAccountOrLoansElsewhere'].step_direction = -1
action_set['CheckingAccountBalance_geq_0'].step_direction = 1
# action_set['isMale'].mutable = False


## dummy model
clf = LogisticRegression(max_iter=1000, solver='lbfgs')
# grid = GridSearchCV(
#     clf, param_grid={'C': np.logspace(-4, 3)},
#     cv=10,
#     scoring='roc_auc',
#     return_train_score=True
# )
clf.fit(X, y)

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
t_cplex = FlipsetBuilder(
    optimizer='cplex',
    action_set=action_set,
    coefficients=coefficients,
    intercept=intercept- (np.log(p / (1. - p))),
    x=x
)
cplex_output = t_cplex.fit()
cplex_output_df = pd.DataFrame(cplex_output)[['actions', 'costs']]
print(cplex_output_df)

t_pyomo = FlipsetBuilder(
    optimizer='cbc',
    action_set=action_set,
    coefficients=coefficients,
    intercept=intercept- (np.log(p / (1. - p))),
    x=x
)
pyo_output = t_pyomo.fit()
max_cost = pyo_output.pop('max_cost', None)
pyo_output_df = (pd.DataFrame
 .from_dict(pyo_output, orient='index')
 .loc[lambda df: df['u'] == 1]
)
print(pyo_output_df)


