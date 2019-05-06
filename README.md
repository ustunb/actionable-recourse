`actionable-recourse` is a python library to evaluate recourse in linear classification models. 

## Overview

*Recourse* is the ability to change the decision of a model by altering *actionable* input variables (e.g., income, # of bank accounts, # of credit cards vs. age, gender, ethnicity). 

This package includes tools to audit recourse in linear classification models.

### Highlights

- Generate a list of actionable changes for a person to obtain a specific outcome from a linear model
- Measure the feasibility and difficulty of recourse for model over a population of interest

## Installation

Please install from source by running:

```
$ git clone git@github.com:ustunb/actionable-recourse.git
$ python setup.py
```

#### Requirements:

- Python 3
- CPLEX or CBC+Pyomo
 
#### CPLEX 

CPLEX is cross-platform commercial optimization tool with a Python API. It is free for students and faculty at accredited institutions. To get CPLEX:

1. Register for [IBM OnTheHub](https://ibm.onthehub.com/WebStore/Account/VerifyEmailDomain.aspx)
2. Download the *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://ibm.onthehub.com/WebStore/ProductSearchOfferingList.aspx?srch=CPLEX)
3. Install the CPLEX Optimization Studio.
4. Setup the CPLEX Python API [as described here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

If you have problems installing CPLEX, check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059). 

#### CBC + Pyomo

* Download COIN-OR and CBC from: https://www.coin-or.org/
* Install Pyomo using `pip` or `conda` and then run the Pyomo installer in the command line: `pyomo install-extras`
* If you're on Windows, `conda install -c conda-forge pyomo.extras` is a safer way to go

## Development Roadmap

- Contributing.md
- Support for categorical variables in `ActionSet`
- Support for rule-based such as decision lists and rule lists
- [scikit-learn](http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator) compatability
- [Integration into AI360 Fairness Toolkit](https://www.ibm.com/blogs/research/2018/09/ai-fairness-360/)

## Reference

For more about recourse or how to use these tools, check out our paper:

[Actionable Recourse in Linear Classification](http://www.berkustun.com/docs/actionable_recourse.pdf)
     
```
inproceedings{ustun2019recourse,
     title = {Actionable Recourse in Linear Classification},
     author = {Ustun, Berk and Spangher, Alexander and Liu, Yang},
     booktitle = {Proceedings of the Conference on Fairness, Accountability, and Transparency},
     series = {FAT* '19},
     year = {2019},
     isbn = {978-1-4503-6125-5},
     location = {Atlanta, GA, USA},
     pages = {10--19},
     numpages = {10},
     url = {http://doi.acm.org/10.1145/3287560.3287566},
     doi = {10.1145/3287560.3287566},
     publisher = {ACM},
}
```
