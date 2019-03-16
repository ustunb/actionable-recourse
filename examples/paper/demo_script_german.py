from examples.paper.initialize import *
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score



data_name = 'german'
data_file = data_dir / '%s_processed.csv' % data_name
output_dir = results_dir /  data_name

## load and process data
data_df = pd.read_csv(data_file).reset_index(drop=True)
# german_df = german_df.assign(isMale=lambda df: (df['Gender']=='Male').astype(int))#.drop(['PurposeOfLoan', 'Gender', 'OtherLoansAtStore'], axis=1)
y = data_df['GoodCustomer']
X = (data_df.drop('GoodCustomer', axis=1)
     .drop(['PurposeOfLoan', 'Gender', 'OtherLoansAtStore'], axis=1)
     )

## set up actionset
gender_weight = data_df.assign(c = 1).groupby('Gender')['c'].transform(lambda s: s * 1. / len(s))
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

## plot the demographic composition
# print("...plotting histogram of all X values")
# plt.rc("font", size=20)
# X.hist(figsize = (32, 16))
# plt.tight_layout()
# plt.savefig(as_result_file('dist_by_group'), bbox_inches='tight')
# plt.close()

# ### Train and generate AUCs
## kfold cross-val
print("...training model over indices")
manual_grid_search = False
if manual_grid_search:
    clf = LogisticRegressionCV(cv=100, max_iter=1000, solver='lbfgs')
    scores_train = []
    scores_test = []
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X.index):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        clf.fit(X_train,y_train)

        y_pred_train = clf.predict_proba(X_train)[:, 1]
        y_pred_test = clf.predict_proba(X_test)[:, 1]

        score_train = roc_auc_score(y_train, y_pred_train)
        score_test = roc_auc_score(y_test, y_pred_test)
        scores_train.append(score_train)
        scores_test.append(score_test)
    print(np.mean(scores_train))
    print(np.mean(scores_test))

else:
    ## grid search
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    grid = GridSearchCV(
            clf, param_grid={'C': np.logspace(-4, 3)},
            cv=10,
            scoring='roc_auc',
            return_train_score=True
            )
    grid.fit(X, y)
    clf = grid.best_estimator_

    ## get the distribution over train and test scores across model hyperparameters
    grid_df = pd.DataFrame(grid.cv_results_)
    grid_df = (grid_df
        .assign(lambda_=lambda df: 1./df['param_C'])
        .set_index('lambda_')
    [['mean_train_score', 'mean_test_score', 'std_train_score', 'std_test_score', ]]
        )


    ## plot distribution over train/test scores
    # (grid_df
    #  .pipe(lambda df: df['mean_train_score'].plot(yerr=df['std_train_score'], alpha=.7, label='train auc'),)
    #  )
    # (grid_df
    #  .pipe(lambda df: df['mean_test_score'].plot(yerr=df['std_test_score'], alpha=.7, label='test auc'), )
    #  )
    # plt.legend()
    # plt.title("10-fold CV, Train and Test Scores: \n German Dataset")
    # plt.xlabel('$\ell_{1}$ penalty')
    # plt.semilogx()
    # plt.savefig(output_dir, '2018-08-12__train-test-auc.png', bbox_inches='tight')
    # plt.close()


# import pyperclip
# print((pd.Series(clf.coef_[0], index=X.columns)
#  .to_frame("Coefficient")
#  .assign(Actionable="Y")
#  .to_latex()
#  ))


pickle.dump(clf, open(output_dir, '2018-08-12__demo-2-clf.pkl'), 'wb')
scores = pd.Series(clf.predict_proba(X)[:, 1])
scores.to_csv(output_dir /   '2018-08-12__demo-2-scores.csv')

# cache classifier
use_cached_flipset = False
if not use_cached_flipset:
    coefficients = clf.coef_[0]
    intercept = clf.intercept_[0]
    action_set.align(coefficients=coefficients)
    # p = .8
    p = scores.median()
    denied_individuals = scores.loc[lambda s: s<=p].index

    ## run flipsets
    idx = 0
    flipsets = {}
    now = time.time()
    for i in denied_individuals:
        if idx % 50 == 0:
            print('finished %d points in %f...' %  (idx, time.time() - now))
            now = time.time()

        x = X.values[i]
        # p = scores.median()
        fb = RecourseBuilder(
                coefficients=coefficients,
                intercept = intercept - (np.log(p / (1. - p))),
                action_set = action_set,
                x = x
                )
        ## CPLEX
        cplex_output = fb.fit()
        flipsets[i] = cplex_output
        idx += 1

    flipset_df = pd.DataFrame.from_dict(flipsets, orient="index")
    flipset_df.to_csv(output_dir /  "2018-11-11_flipset-german-demo-cache.csv")

else:
    ## Examine histogram of actions
    def convert_array_col_cache_to_col(col):
        return (col.str.replace('[','')
                .str.replace(']','')
                .str.split()
                .apply(lambda x: map(float, x))
                )

    flipset_df = pd.read_csv(output_dir /  "2018-11-11_flipset-german-demo-cache.csv")
    flipset_df = (flipset_df
                  .assign(actions=lambda df: df['actions'].pipe(convert_array_col_cache_to_col))
                  .assign(costs=lambda df: df['costs'].pipe(convert_array_col_cache_to_col))
                  )
    scores = pd.read_csv(output_dir /   '2018-08-12__demo-2-scores.csv', index_col=0, squeeze=True)

flipset_df['sum_cost'] = flipset_df['costs'].apply(sum)
matching_df = pd.concat([
    data_df[['GoodCustomer', 'Gender']],
    flipset_df[['total_cost']],
    scores.to_frame('y_pred')
    ], axis=1).replace(np.inf, np.nan).dropna()


####### Matching 2: Control for Y=+/- 1
print("matching 2: control for y=+/- 1...")
matching_df['y_pred_bin'] = pd.cut(
        matching_df['y_pred'],
        bins=np.arange(0, .9, .2)
        )

bins = matching_df['y_pred_bin'].unique()
# (matching_df
#     .loc[lambda df: df['GoodCustomer'] == 1]
#     .loc[lambda df: df['y_pred_bin'] == bins[0]]
# )

max_cost = matching_df['total_cost'].max()
plt.rc("font", size=20)
for y_true in [-1, 1]:
    plt.figure(figsize=(4, 4))
    ax = sns.violinplot(x='y_pred_bin',  y='total_cost', hue='Gender',
                        data=matching_df.loc[lambda df: df['GoodCustomer'] == y_true].sort_values('Gender'),
                        linewidth = 0.5, cut=0, background='white',
                        scale = 'width', color="gold",  inner = 'quartile')
    # , inner = 'quartile', color = "gold", scale = 'width'
    ax.set_xticklabels(["10%", "30%", "50%", "70%"])#, rotation='vertical')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylim((0, max_cost))
    # plt.title('Total Cost By Gender')
    plt.ylabel('Cost of Recourse')
    plt.xlabel('Predicted Risk' )

    # plt.legend(bbox_to_anchor=(0.1, .03, .65, .14), ncol=2, mode="expand", borderaxespad=0.)
    # plt.legend(fontsize=14, ncol=2, mode='expand',bbox_to_anchor=(.18, .51, .875, .51), )
    if y_true == -1:
        plt.legend(fontsize=12., loc='upper right', bbox_to_anchor=(1.1, 1))
    else:
        plt.legend(fontsize=12., loc='lower left')#, bbox_to_anchor=(1.1, 1))
    for l in ax.lines:
        l.set_linewidth(2.)
        l.set_linestyle('-')
        l.set_solid_capstyle('butt')
    # plt.legend(mode='expandoutput_dir + ')
    plt.savefig(output_dir / 'matched-cost-by-y-pred-y-%d.png' % y_true, bbox_inches='tight')
    plt.savefig(output_dir /  'matched-cost-by-y-pred-y-%d.pdf' % y_true, bbox_inches='tight')
    plt.close()

coef_df = pd.Series(clf.coef_[0], index=X.columns).to_frame('Coefficient')
coef_df['Actionable'] = "Yes"

##### Plot Lift
agg_dfs = {}
for y_true in [-1, +1]:
    agg_df = (matching_df
              .loc[lambda df: df['GoodCustomer'] == y_true]
              .pipe(
            lambda df: df.groupby(pd.cut(df['y_pred'], bins=np.arange(0, scores.median(), .1)))
            )
              .apply(lambda df:
                     df.loc[lambda df: df['Gender']=='Female']['total_cost'].median() /
                     df.loc[lambda df: df['Gender']=='Male']['total_cost'].median()
                     )
              .fillna(1)
              )
    agg_dfs[y_true] = agg_df

min_, max_ = pd.concat(agg_dfs.values()).pipe(lambda s: (s.min(), s.max()))
for y_true in [-1, +1]:
    # ax = plt.plot(np.arange(len(agg_df)), agg_df.values)
    ax = agg_dfs[y_true].plot(color="black", figsize=(4,4))
    plt.scatter(range(len(agg_dfs[y_true])), agg_dfs[y_true].values, color='black')
    ax.set_facecolor("white")
    plt.ylabel('Cost Lift, Females / Males')
    plt.ylim((min_, max_))
    plt.xlabel('Predicted Risk' )
    plt.hlines(1, *plt.xlim(), linestyles='dashed')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    xticks = plt.xticks()
    # plt.xticks(xticks[0], np.arange(0,1,.2).round(2),)
    plt.xticks(np.arange(0, len(agg_df)+.6, (len(agg_df) +.6) /5.), [0., .2, .4, .6, .8])
    plt.savefig(output_dir / 'lift-by-gender-by-y-pred-y-true-%d.png' % y_true, bbox_inches='tight')
    plt.savefig(output_dir / 'lift-by-gender-by-y-pred-y-true-%d.pdf' % y_true, bbox_inches='tight')
    plt.show()
    plt.close()



### get flipset
# gender = 'Female'
gender = "Male"
individuals = (matching_df
    # .loc[lambda df: df['Gender'] == gender]
    .loc[lambda df: df['y_pred'] < .4]
    .loc[lambda df: df['y_pred'] > .2]
    .loc[lambda df: df['GoodCustomer'] == 1]
    )
i = individuals.index[0]

# i = 815
i = 818
action_set = ActionSet(X=X_gender_balanced)
action_set['Age'].mutable = False
action_set['Single'].mutable = False
action_set['JobClassIsSkilled'].mutable = False
action_set['ForeignWorker'].mutable = False
action_set['OwnsHouse'].mutable = False
action_set['RentsHouse'].mutable = False
action_set['CriticalAccountOrLoansElsewhere'].step_direction = -1
action_set['CheckingAccountBalance_geq_0'].step_direction = 1

action_set['NoCurrentLoan'].mutable = False
action_set['YearsAtCurrentJob_lt_1'].mutable = False
action_set['YearsAtCurrentHome'].mutable = False
action_set['SavingsAccountBalance_geq_500'].mutable = False
action_set['YearsAtCurrentJob_geq_4'].mutable = False

# actions to switch up actionset
action_set['HasGuarantor'].mutable = False
action_set['HasCoapplicant'].mutable = False
action_set['HasTelephone'].mutable = False
action_set['OtherLoansAtBank'].mutable = False
# action_set['LoanAmount'].mutable = False
action_set['LoanRateAsPercentOfIncome'].mutable = False

# action_set['LoanDuration'].mutable = False
# action_set['CheckingAccountBalance_geq_200'].mutable = False
action_set['NumberOfLiableIndividuals'].mutable = False
# action_set['SavingsAccountBalance_geq_500'].mutable = False
# action_set['SavingsAccountBalance_geq_100'].mutable = False
action_set['Unemployed'].mutable = False

action_set.align(coefficients=coefficients)

p = scores.median()
x = X.values[i]
rb = RecourseBuilder(
        coefficients=coefficients,
        intercept=intercept - (np.log(p / (1. - p))),
        action_set=action_set,
        x=x
        )

cplex_output = rb.fit()
if cplex_output['feasible']:
    full_actionset = []
    for feature in range(len(x)):
        orig_val = x[feature]
        changed_val = (x[feature] + cplex_output['actions'][feature])
        if not np.isclose(orig_val, changed_val):
            output = "& \\textit{%s} & " %  X.columns[feature]
            output += "%f" % orig_val  + " & \\longrightarrow & %f \\\\" % changed_val
            full_actionset.append(output)
    print("\n".join(full_actionset))
else:
    print("infeasible")




## Generate and plot score (y_pred) distributions
# overall
clf = pickle.load(output_dir / '2018-08-12__demo-2-clf.pkl', 'rb')
plt.rc("font", size = 20)
scores.hist(bins=50)
plt.title('Score Distribution')
ylim = plt.ylim()
plt.vlines(scores.median(), *plt.ylim(), linestyles='dashed')
plt.ylim(ylim)
plt.text(scores.median() * .99, 42, 'Median Score', horizontalalignment='right', fontsize=18)
plt.grid(False)
plt.savefig(output_dir / 'score_distribution_female.png', bbox_inches = 'tight')
plt.close()



print("...plotting cost")
# 'Cost Distribution'
ax = flipset_df.loc[denied_individuals].loc[lambda df: df['feasible'] == True]['total_cost'].hist(bins=50, figsize=(4,4))
plt.grid(False)
plt.text(.0545, 290, 'HasGuarantor\n(0->1)', fontsize=15)
plt.text(.067, 80, 'CheckingAcct. \nBal.$\geq$200\n(0->1)', fontsize=15)
plt.ylabel('Count of Individuals')
plt.xlabel('Cost of Recourse')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(output_dir / '2018-08-12__cost-distribution.pdf', bbox_inches='tight')
plt.close()

# min_action_grid = pd.DataFrame(np.array(flipset_df['actions'].tolist()), columns=X.columns)
# min_cost_grid = pd.DataFrame(np.array(flipset_df['costs'].tolist()), columns=X.columns)
# t = min_action_grid.loc[german_df['Gender'] == 'Male']

### Matching 1: Control for Score

print("matching 1: control for score...")

## plot matched cost by true label
plt.figure(figsize=(4, 4))
ax = sns.violinplot(
        x='GoodCustomer',
        y='total_cost',
        hue='Gender',
        data=matching_df.sort_values('Gender'),
        linewidth = 0.5, cut=0, background='white',
        scale = 'width', color="gold",  inner = 'quartile',
        )

ax.set_facecolor("white")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('True Label: Good Customer')
ax.set_ylabel('Cost of Recourse')
ax.set_ylim((0, 0.4))
for l in ax.lines:
    l.set_linewidth(2.)
    l.set_linestyle('-')
    l.set_solid_capstyle('butt')

plt.legend(fontsize=16, ncol=2, mode='expand', loc="upper right", )# bbox_to_anchor=(.18, .51, .875, .51), )
plt.savefig(output_dir / '2018-08-12__matched-cost-by-true-label.pdf',bbox_inches='tight')
plt.close()

#plot distribution of scores
for g in ['Male', 'Female']:

    plt.rc("font", size = 20)
    plt.grid(False)

    ax = scores.loc[data_df.loc[lambda df: df['Gender'] == g].index].hist(bins=25, color= "gold", alpha=.4, figsize=(4, 4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('$P(y | Gender = %s)$' % g)
    ax.set_ylabel('Count of Individuals')

    plt.savefig(output_dir / '%s_score_distribution_%s.pdf' % (data_name, g), bbox_inches = 'tight')
    plt.close()
