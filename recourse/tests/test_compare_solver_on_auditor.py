import pandas as pd
from sklearn.linear_model import LogisticRegression
from recourse.paths import *
from recourse.auditor import RecourseAuditor
from recourse.action_set import ActionSet

data_name = 'german'
data_file = test_dir / ('%s_processed.csv' % data_name)
n_test_points = 10
test_seed = 2338

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

# create sample points for test
denied_idx = clf.predict(X) < 0
X_test = X.iloc[denied_idx].sample(n = n_test_points, random_state = test_seed)


auditor_cpx = RecourseAuditor(clf=clf, action_set = action_set, solver = 'cplex')
df_cpx = auditor_cpx.audit(X = X_test)

auditor_cbc = RecourseAuditor(clf=clf, action_set = action_set, solver = 'cbc')
df_cbc = auditor_cbc.audit(X = X_test)