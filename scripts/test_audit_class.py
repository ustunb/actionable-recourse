from recourse.audit import Audit
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle

from recourse.path import *
from recourse.action_set import ActionSet
from recourse.flipset import FlipsetBuilder

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

clf = LogisticRegression(max_iter=1000, solver='lbfgs')
grid = GridSearchCV(
    clf, param_grid={'C': np.logspace(-4, 3)},
    cv=10,
    scoring='roc_auc',
    return_train_score=True
)
grid.fit(X, y)
clf = grid.best_estimator_

pickle.dump(action_set, open('scripts/temp_results/audit_action_set.pkl', 'wb'))
pickle.dump(clf, open('scripts/temp_results/audit_clf.pkl', 'wb'))


###
action_set = pickle.load(open('scripts/temp_results/audit_action_set.pkl', 'rb'))
clf = pickle.load(open('scripts/temp_results/audit_clf.pkl', 'rb'))

##############
# run audit
audit = Audit(optimizer="cplex", clf=clf, dataset=X.values, actionset=action_set, decision_threshold=.8)
audit.run_audit(num_cases=10)