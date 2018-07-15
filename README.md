# Actionable Recourse

Recourse Audits provide a mechanism for determining basic fairness properties in decision-making algorithms. The following package provides an optimized auditing function for linear classifiers. The basic principles are given in the paper introducing the concept and the auditing algorithm at the bottom of this document.

## Installation

**NOTE: THIS PACKAGE IS CURRENTLY UNDER ACTIVE DEVELOPMENT. THE CODE WILL CHANGE SUBSTANTIALLY WITH EACH COMMIT.** 

Please install from source, as this project is still in development and updating:

```
$ git clone git@github.com:ustunb/actionable-recourse.git
$ python setup.py
```

## Basic Usage

The primary purpose of this package is to determine how to optimally change a negatively classified datapoint into a positively classified datapoint.

__Step 1.__ Initialize the `ActionSet` for the dataset:
```
action_set = ActionSet(
    X = data['X'],
    custom_bounds = custom_bounds, 
    default_bounds = default_bounds
)
action_set[immutable_variables].mutable = False

action_set['var1'].step_type = 'absolute'
action_set['var1'].step_size = 100
action_set['var1'].update_grid()

action_set['var2'].step_type = 'absolute'
action_set['var2'].step_size = 100
action_set['var2'].update_grid()
```

An `ActionSet` provides the basic blueprint on how variables can be changed to flip a prediction. A self-contained module, it can be instantialized with default properties applied to all features, and custom properties for specific features. It is also highly iterative, and one can easily tweak each variable's properties after instantiaion.

Properties for each variable include:

a. _Immutability_: whether the variable can be changed. `Age`, `gender` and `ethnicity`, for instance, are examples of variables that might be considered immutable in a dataset. (By default, all variables are considered mutable).

b. _Bounds_: How much each variable can change. Bounds can be either `bound_type={ "percentile", "relative", "absolute" }`, where `percentile` bounds would be set to the kth percentile of the observed data, `relative` is relative to the range of the data, and `absolute` is a fixed number.

c. _Step size_: the level of discretization for the optimizer to scan. Smaller step sizes will provide a more granular `flipset` while larger step sizes will run faster. `Step_size` can also be set via `{"percentile", "relative" and "absolute"}` specifications.

__Step 2:__ Initialize a `FlipsetBuilder` object for each negatively-classified datapoint:

```
fb = FlipsetBuilder(
	coefficients=coefficients,
	intercept=intercept,
	action_set=action_set,
	x=x, mip_cost_type=mip_cost_type
)

output = fb.fit()
```

The flipset builder takes the scaler intercept and 1-d vector of coefficients of the linear classifier that is being audited (the decision boundary is assumed to be 0). It also takes an `action_set` defined in the previous step, a cost_type `{total, or max}` (see paper for more details). And, finally, the datapoint, `x`. Calling `.fit()` will run the optimization and provide the steps to take to flip the prediction.

## Package Details

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

## Paper

[Actionable Recourse in Linear Classification](http://www.berkustun.com/docs/actionable_recourse_fatml_2018.pdf)
     
```
@inproceedings{recourse2018fatml,
	Author = {Spangher, Alexander and Ustun, Berk},
	Booktitle = {Proceedings of the 5th Workshop on Fairness, Accountability and Transparency in Machine Learning},
	Title = {{Actionable Recourse in Linear Classification}},
	Year = {2018}}
}
```
