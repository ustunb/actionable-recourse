`actionable-recourse` is a python library to check recourse in linear classification models. 

----

**Recourse?**

*Recourse* is the ability to flip the prediction of a machine learning model by change *actionable* input variables (e.g., `income` instead of `age` or `gender`). 

**Package Highlights**

- Specify a custom set of feasible actions for each input to a machine learning model.

- Generate a list of actionable changes to flip the prediction of a linear classifier (i.e., what can a person do to obtain a favorable outcome from a model).

- Evaluate the *feasibility of recourse* of a linear classifier on a population of interest (i.e., how many people will be able to alter their predictions?).

- Measure the *difficulty of recourse* for a linear classifier on a population of interest (i.e., how difficult is it for individuals on change their prediction).

## Installation

Please install from source by running

```
$ git clone git@github.com:ustunb/actionable-recourse.git
$ python setup.py
```

#### Requirements

- Python 3
- MIP solver: either CPLEX or Pyomo + CBC
 
#### Installing CPLEX 

CPLEX is fast optimization solver with a Python API. It is commercial software, but worth downloading since it is free to students and faculty at accredited institutions. To get CPLEX:

1. Register for [IBM OnTheHub](https://ibm.onthehub.com/)
2. Download the *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://ibm.onthehub.com/WebStore/ProductSearchOfferingList.aspx?srch=CPLEX)
3. Install the CPLEX Optimization Studio.
4. Setup the CPLEX Python API [as described here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

If you have problems installing CPLEX, please check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059). 

#### Installing CBC and Pyomo

If you are unable to obtain CPLEX, you can also work with an open-source solver. This requires the following steps (which you can do *before* you) 

1. Download and install [CBC](https://github.com/coin-or/Cbc) from [Bintray](https://bintray.com/coin-or/download/Cbc)
2. Download and install `pyomo` *and* `pyomo-extras` [(instructions)](http://www.pyomo.org/installation)

## Development Roadmap

- pip installation
- Contributing.md
- Support for categorical variables in `ActionSet`
- Support for rule-based models such as decision lists and rule lists
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
