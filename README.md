recourse
=================== 

`recourse` is a python library for recourse verification and reporting. 

### Recourse in Machine Learning

Recourse is the ability to change the prediction of a machine learning model by altering *actionable* input variables – e.g., `savings` and `n_credit_cards` as opposed to `age`. Recourse is an essential aspect of fairness in consumer-facing ML. When a person is denied a loan by a machine learning model, for example, they should be able to change its input variables in order to be approved. Otherwise, they will lack the ability to influence a decision that affects their livelihood.

#### Overview

This library provides tools to protect recourse by reporting and verification. The tools current support linear classifiers. We will extend the toolkit to support other kinds of models over time.

**Reporting**: The goal of recourse reporting is to present individuals who receive an unfavorable prediction from a machine learning models with an actions that they can prediction from a given machine learning model. What can a person do to obtain a favorable prediction from a ML model? Generate a list of actions that can be used to flip the prediction of a model.

**Verification**: The goal of recourse verification is to ensure that a model will provide its decision-subjects with a way to flip their predictions. We wish to answer how many people can change their prediction? How difficult for people to change their prediction? Estimate the feasibility and difficulty of recourse of a model on a population of interest.

Customize the set of feasible action for each input variable of a machine learning model.

----

## Installation

Install with `pip` from PyPI:

```
pip install recourse
```

It requires Python 3 and one of the following mixed-integer programming packages: 
 
- [Python-MIP](http://python-mip.com/) – Python-MIP is an open-source MIP library that supports multiple solvers. It uses the [CBC]() solver by default, and can support commercial solvers like Gurobi. 

- CPLEX – is a commercial MIP solver with a Python API that is free for academic use. See [here](docs/cplex_installation.md) for instructions on how to obtain and install CPLEX.

## Usage

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

### Contributing

We're actively working to improve this package. If you find bugs, have comments, or want to help, let us know. We welcome any and all contributions! For more info, see [our guidelines](https://github.com/ustunb/actionable-recourse/blob/master/CONTRIBUTING.md). Thank you community!

### Resources

For more about recourse, check out our paper [Actionable Recourse in Linear Classification](https://arxiv.org/abs/1809.06514). If you use actionable-recourse in your research, we would appreciate a citation ([bibtex](/docs/references/ustun2019recourse.bibtex))!