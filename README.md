# Actionable Recourse

In machine learning, recourse describes the ability for the individual to achieve a desired outcome under a fixed prediction model. Consider, for example, a classifier built to automate lending decisions. If this model does not provide recourse to a person who is denied a loan, then this person cannot change any of the input variables of the model to be approved for a loan, and will be denied credit so long as the model is deployed.


## Installation

**NOTE: THIS PACKAGE IS CURRENTLY UNDER ACTIVE DEVELOPMENT. THE CODE WILL CHANGE SUBSTANTIALLY WITH EACH COMMIT.** 

Please install from source, as this project is still in development and updating:

```
$ git clone git@github.com:ustunb/actionable-recourse.git
$ python setup.py
```


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
