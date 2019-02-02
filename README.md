`recourse` is a python library to evaluate recourse in classification models. 

Letur tools let users to:

1. Measure the feasibility and difficulty of recourse in a target population
2. Generate a list of actionable changes for an individual to obtain a desired outcome

## Background

*Recourse* is the ability to change the decision of a machine learning model by manipulating *actionable* input variables (e.g., income vs. age, ethnicity, marital status). 

Classification models are often used to make decisions that affect humans: whether to approve a loan application, extend a job offer, or provide insurance. In such applications, individuals should have the ability to change the decision of the model. When a person is denied a loan by a credit scoring model, for example, they should be able to change its input variables in order to be approval. Otherwise, they will be denied the loan so long as the model is deployed, and lack agency over a decision that affects their livelihood. 

#### Paper

[Actionable Recourse in Linear Classification](https://arxiv.org/abs/1809.06514)
     
```
@article{ustun2018actionable,
  title={Actionable Recourse in Linear Classification},
  author={Ustun, Berk and Spangher, Alexander and Liu, Yang},
  journal={arXiv preprint arXiv:1809.06514},
  year={2018}
}
```

## Installation

Please install from source, as this project is still in development and updating:

```
$ git clone git@github.com:ustunb/actionable-recourse.git
$ python setup.py
```

### Requirements

- Python 3.5+ 
- CPLEX 12.6+
 
The code may still work with older versions of Python and CPLEX, but this will not be tested or supported. If you're hesitant about switching to Python 3, check out this [migration guide](https://github.com/arogozhnikov/python3_with_pleasure)  

#### CPLEX 

CPLEX is cross-platform commercial optimization tool with a Python API. It is freely available to students and faculty members at accredited institutions. To get CPLEX:

1. Register for [IBM OnTheHub](https://ibm.onthehub.com/WebStore/Account/VerifyEmailDomain.aspx)
2. Download the *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://ibm.onthehub.com/WebStore/ProductSearchOfferingList.aspx?srch=CPLEX)
3. Install the CPLEX Optimization Studio.
4. Setup the CPLEX Python API [as described here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

If you have problems installing CPLEX, check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059). 

## Development Roadmap

**NOTE: THIS PACKAGE IS CURRENTLY UNDER ACTIVE DEVELOPMENT. THE CODE MAY CHANGE WITH EACH COMMIT.** 

- Support for open-source MIP solver (either [CBC](https://projects.coin-or.org/Cbc) or [MIPCL](http://www.mipcl-cpp.appspot.com/))
- Support for categorical variables in `ActionSet`
- Refactoring for future development 
- Support for Boolean models such as decision lists, rule lists etc.
- Comparatabilty with [scikit-learn](http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator)
- [Integration into AI360 Fairness Toolkit](https://www.ibm.com/blogs/research/2018/09/ai-fairness-360/)


### Pyomo and CBC

* Run the Pyomo installer in the command line: pyomo install-extras 
* Or, if you're on windows, `conda install -c conda-forge pyomo.extras` is a safer way to go
* Download COIN-OR from: https://www.coin-or.org/download/binary/OptimizationSuite/
* Make sure to update your `$PATH` variable to point to the correct location:
* For example: `export PATH=PATH:/usr/local/bin/Cbc-2.8.5`
(* The windows installer should do this for you.)