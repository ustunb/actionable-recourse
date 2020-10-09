import numpy as np


def is_sklearn_linear_classifier(obj):
    """
    Checks if object is a sklearn linear classifier for a binary outcome
    :param obj: object
    """
    binary_flag = hasattr(obj, 'classes_') and len(obj.classes_) == 2
    linear_flag = hasattr(obj, 'coef_') and hasattr(obj, 'intercept_')
    return binary_flag and linear_flag


def parse_classifier_args(*args, **kwargs):
    """
    helper function to parse coefficients and intercept from linear classifier arguments

    *args and **kwargs can contain either:
        - sklearn classifiers with 'coef_' and 'intercept_' fields (keyword: 'clf', 'classifier')
        - vector of coefficients (keyword: 'coefficients')
        - intercept: set to 0 by default (keyword: 'intercept')

    returns:
        w - np.array containing coefficients of linear classifier (finite, flattened)
        t - float containing intercept of linear classifier (finite, float)

    raises:
        ValueError if fails to parse classifier arguments

    :return:
    """
    w, t = None, None

    if 'clf' in kwargs:

        assert is_sklearn_linear_classifier(kwargs['clf'])
        w = kwargs['clf'].coef_
        t = kwargs['clf'].intercept_

    elif 'classifier' in kwargs:

        assert is_sklearn_linear_classifier(kwargs['classifier'])
        w = kwargs['classifier'].coef_
        t = kwargs['classifier'].intercept_

    elif 'coefficients' in kwargs:

        w = kwargs.get('coefficients')
        t = kwargs.get('intercept', 0.0)

    elif len(args) == 1:

        if is_sklearn_linear_classifier(args[0]):

            w = args[0].coef_
            t = args[0].intercept_

        elif isinstance(args[0], (list, np.ndarray)):

            w = np.array(args[0]).flatten()
            t = 0.0

    elif len(args) == 2:

        w = args[0]
        t = float(args[1])

    else:
        raise ValueError('failed to match classifier arguments')

    w = np.array(w).flatten()
    t = float(t)
    assert np.isfinite(w).all()
    assert np.isfinite(t)
    return w, t

import seaborn as sns

# X is the data to audit on
def print_recourse_audit_report(X, audit_df, y, group_by = ['y']):
    processed_data = (audit_df
                            .merge(X, right_index=True, left_index=True)
                            .merge(y.to_frame('y'), right_index=True, left_index=True)
                            .replace([np.inf, -np.inf], np.nan)
#                             .dropna()
                           )
    
#     We run the audit for everyone where \hat{Y}=0.
#     We break down these people by the true value of Y.
#     For each group defined by Y = (0, 1). For, Y = we want to know:
#     (0) how many people are in the group
#     (1) how many group members have recourse
#     (2) what is the distribution of the cost of recourse amost those with recourse
#     more generally, we'll want the function to take in different ways to slice and dice the population into mutually exclusive groups, one variable that can be used to do that is just "Y", but you could also do it by Age Group, etc.

#     rows that already attain desired outcome have entries: feasible = NaN & cost = NaN
#     rows that are certified to have no recourse have entries: feasible = False & cost = Inf`

    # DELIVERABLES

    print("Stats: ")
    print("Audit Dataset Size: %s"%(processed_data.shape[0]))
    
    def subset_data_and_plot(y, data, group_by=['y']):
        
        # individuals who already have the desired outcome
        df1 = data.loc[(data['y']==y)&
                                 (data['feasible'].isnull())&
                                 (data['cost'].isnull())]
        print("Number of datapoints that have Y=%s and already have the desired outcome: %s"%(y, df1.shape[0]))
        
        # individuals who are certified to have no recourse
        df2 = data.loc[
            (data['y']==y)&
            (data['feasible']==False)&
            (np.isfinite(data['cost']))]
        print("Number of datapoints that have Y=%s and DO NOT have recourse: %s"%(y, df2.shape[0]))
        
        # individuals who have feasible cost of recourse
        df3 = data.loc[(data['y']==y)&
                       (data['feasible']==True)]

        print(df3[['cost']+group_by].groupby(group_by).agg(['mean','median','count']))
        print(df3[['feasible']+group_by].groupby(group_by).agg(['count']))
        
        sns.set_context("paper", 
                        rc={"font.size":15,"axes.titlesize":10,"axes.labelsize":15})
        
        if group_by[0] == 'y':
            sns.histplot(df3['cost']).set(title='Histogram of Cost of Recourse for Y=%s'%(y),
                           xlabel='Cost',
                           ylabel='Counts')

        else:
            sns.violinplot(
                x=group_by[0], y='cost',
                data=df3,
                linewidth = 0.5, cut=0,
                scale='width', color="lightskyblue",  inner='quartile').set(title="Histogram of Cost of Recourse by %s for Y=%s"%(group_by[0], y),xlabel=group_by[0])
        
        plt.show()
    
    y_val = list(set(y))
    for y in y_val:
        subset_data_and_plot(y, processed_data)
    
    # assuming that the other inputs are categorial variables
    if len(group_by)>1:
        for i in range(1, len(group_by)):
            for y in y_val:
                print(group_by[i])
                subset_data_and_plot(y, processed_data, group_by=[group_by[i]])

