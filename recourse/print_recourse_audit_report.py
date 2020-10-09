import seaborn as sns
import numpy as np

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

    y_val = list(set(y))
    for y in y_val:
        subset_data_and_plot(y, processed_data)
    
    # assuming that the other inputs are categorial variables
    if len(group_by)>1:
        for i in range(1, len(group_by)):
            for y in y_val:
                print(group_by[i])
                subset_data_and_plot(y, processed_data, group_by=[group_by[i]])

def subset_data_and_plot(y, data, group_by = ['y']):

        # individuals who already have the desired outcome
        df1 = data.loc[(data['y'] == y) &
                       (data['feasible'].isnull()) &
                       (data['cost'].isnull())]
        print(
            "Number of datapoints that have Y=%s and already have the desired outcome: %s" % (
            y, df1.shape[0]))

        # individuals who are certified to have no recourse
        df2 = data.loc[
            (data['y'] == y) &
            (data['feasible'] == False) &
            (np.isfinite(data['cost']))]
        print(
            "Number of datapoints that have Y=%s and DO NOT have recourse: %s" % (
            y, df2.shape[0]))

        # individuals who have feasible cost of recourse
        df3 = data.loc[(data['y'] == y) &
                       (data['feasible'] == True)]

        print(df3[['cost'] + group_by].groupby(group_by).agg(
                ['mean', 'median', 'count']))
        print(df3[['feasible'] + group_by].groupby(group_by).agg(['count']))

        sns.set_context("paper",
                        rc = {
                            "font.size": 15, "axes.titlesize": 10,
                            "axes.labelsize": 15})

        if group_by[0] == 'y':
            sns.histplot(df3['cost']).set(
                title = 'Histogram of Cost of Recourse for Y=%s' % (y),
                xlabel = 'Cost',
                ylabel = 'Counts')

        else:
            sns.violinplot(
                    x = group_by[0], y = 'cost',
                    data = df3,
                    linewidth = 0.5, cut = 0,
                    scale = 'width', color = "lightskyblue",
                    inner = 'quartile').set(
                title = "Histogram of Cost of Recourse by %s for Y=%s" % (
                group_by[0], y), xlabel = group_by[0])

        plt.show()