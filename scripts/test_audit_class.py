import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scripts.paths import *
from recourse.auditor import Auditor
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
audit = Auditor(clf=clf, dataset=X.values, actionset=action_set, decision_threshold = 0.8)
audit.run_audit(num_cases=10)