import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from recourse.paths import *

from recourse.action_set import ActionSet
from recourse.builder import RecourseBuilder
from recourse.flipset import Flipset

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

# fit classifier
clf = LogisticRegression(max_iter=1000, solver = 'lbfgs')
clf.fit(X, y)

# run audit
denied_idx = np.flatnonzero(clf.predict(X) < 0)
x = X.values[denied_idx[0]]
rb = RecourseBuilder(clf = clf, action_set = action_set, x = x, mip_cost_type = 'local')

#fb.max_items = 4
flipset = Flipset(x = rb.x, variable_names = action_set._names, clf = clf)
flipset.add(rb.populate(enumeration_type = 'distinct_subsets', total_items = 14))
print(flipset.to_latex()) #creates latex table for paper
print(flipset.view()) # displays to screen

auditor = Flipset(clf = clf, action_set = action_set)
df = auditor.audit(X = X)
