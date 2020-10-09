`actionable-recourse` is a python library for recourse verification and reporting. 

## Recourse in Machine Learning?


*Recourse* is the ability of a person to change the prediction of a machine learning model by altering *actionable* input variables – e.g., `income` and `n_credit_cards` as opposed to `age` or `alma_mater`.

Recourse is an essential aspect of procedural fairness in consumer-facing applications of machine learning. When a consumer is denied a loan by a machine learning model, for example, they should be able to change the input variables of the model in a way that guarantees approval. Otherwise, this person will be denied the loan so long as the model is deployed, and stripped of the ability to influence a decision that affects their livelihood. 

## Verification & Reporting

This library provides protect consumers against this harm through verification and reporting. These tools can be used to answer questions like:

- What can a person do to obtain a favorable prediction from a given model?
- How many people can change their prediction?
- How difficult for people to change their prediction?
 
Specific functionality includes:

- Customize the set of feasible action for each input variable of an machine learning model.
- Produce a list of actionable changes for a person to flip the prediction of a model.
- Estimate the feasibility of recourse of a model on a population of interest.
- Evaluate the difficulty of recourse of a model on a population of interest.

The tools are currently designed to support linear classification models, and will be extended to cover other kinds of models over time. 

----

## Installation

You can install the library via `pip`.
```
$ pip install actionable-recourse
```

### Requirements

- Python 3
- Python-MIP or CPLEX  

#### CPLEX

CPLEX is fast optimization solver with a Python API. It is commercial software, but free to download for students and faculty at accredited institutions. To obtain CPLEX:

1. Register for [IBM Academic Initiative](https://www.ibm.com/academic/technology/data-science)
2. Download the *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://www-03.ibm.com/isc/esd/dswdown/searchPartNumber.wss?partNumber=CJ6BPML)
3. Install the CPLEX Optimization Studio (on MacOS, run `./<cplex-path>/Contents/MacOS`).
4. Setup the CPLEX Python API [as described here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

If you have problems installing CPLEX, please check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059). 


### Usage
```
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import recourse as rs

# import data
url = 'https://raw.githubusercontent.com/ustunb/actionable-recourse/master/examples/paper/data/credit_processed.csv'
df = pd.read_csv(url)
y, X = df.iloc[:, 0], df.iloc[:, 1:]

# train a classifier
clf = LogisticRegression(max_iter = 1000)
clf.fit(X, y)
yhat = clf.predict(X)

# customize the set of actions
A = rs.ActionSet(X)  ## matrix of features. ActionSet will set bounds and step sizes by default

# specify immutable variables
A['Married'].mutable = False

# can only specify properties for multiple variables using a list
A[['Age_lt_25', 'Age_in_25_to_40', 'Age_in_40_to_59', 'Age_geq_60']].mutable = False

# education level
A['EducationLevel'].step_direction = 1  ## force conditional immutability.
A['EducationLevel'].step_size = 1  ## set step-size to a custom value.
A['EducationLevel'].step_type = "absolute"  ## force conditional immutability.
A['EducationLevel'].bounds = (0, 3)

A['TotalMonthsOverdue'].step_size = 1  ## set step-size to a custom value.
A['TotalMonthsOverdue'].step_type = "absolute"  ## discretize on absolute values of feature rather than percentile values
A['TotalMonthsOverdue'].bounds = (0, 100)  ## set bounds to a custom value.

## get model coefficients and align
A.align(clf)  ## tells `ActionSet` which directions each feature should move in to produce positive change.

# Get one individual
i = np.flatnonzero(yhat <= 0).astype(int)[0]

# build a flipset for one individual
fs = rs.Flipset(x = X.values[i], action_set = A, clf = clf)
fs.populate(enumeration_type = 'distinct_subsets', total_items = 10)
fs.to_latex()
fs.to_html()

# Run Recourse Audit on Training Data
auditor = rs.RecourseAuditor(A, coefficients = clf.coef_, intercept = clf.intercept_)
audit_df = auditor.audit(X)  ## matrix of features over which we will perform the audit.

## print mean feasibility and cost of recourse
print(audit_df['feasible'].mean())
print(audit_df['cost'].mean())
print_recourse_audit_report(X, audit_df, y)
# or produce additional information of cost of recourse by other variables
# print_recourse_audit_report(X, audit_df, y, group_by = ['y', 'Married', 'EducationLevel'])

```

## Contributing

We're actively working to improve this package and make it more useful. If you come across bugs, have comments, or want to help, let us know. We welcome any and all contributions! For more info on how to contribute, check out [these guidelines](https://github.com/ustunb/actionable-recourse/blob/master/CONTRIBUTING.md). Thank you community!


## Resources

For more about recourse, check out our paper:

[Actionable Recourse in Linear Classification](https://arxiv.org/abs/1809.06514)

 If you use this library in your research, we would appreciate a citation!