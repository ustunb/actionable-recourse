from examples.paper.initialize import *
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


data_name = 'givemecredit'
data_file = data_dir / '%s/%s_training.csv' % (data_name, data_name)
output_dir = results_dir / data_name
raw_df = pd.concat([pd.read_csv(data_file, index_col=0), ])

# # 1. Out of Sample Costs
## sample the dataset
## preprocess and prepare
sample_weights = raw_df['age'].apply(lambda x: x < 35).map({True: 0, False: 1.})
downsampled_givemecredit = raw_df.sample(n = 100000, weights = sample_weights.values)
y = 1 - raw_df['SeriousDlqin2yrs'].reset_index(drop=True)
X = raw_df.drop('SeriousDlqin2yrs', axis=1).fillna(0).reset_index(drop=True)

## split datasets for classifier
X_clf, X_audit_holdout, y_clf, y_audit_holdout = train_test_split(X, y, test_size=.25)

# X_train = X.loc[lambda df: ~df.index.isin(flipset_full.index)]
# X_audit_holdout = X.loc[lambda df: df.index.isin(flipset_full.index)]
# y_train = y.loc[lambda df: ~df.index.isin(flipset_full.index)]
# y_test = y.loc[lambda df: df.index.isin(flipset_full.index)]
# X_downsampled_train, X_downsampled_test, y_downsampled_train, y_downsampled_test = (
#     train_test_split(X, y, test_size=.25)
# )
# X_downsampled = (downsampled_givemecredit
#                  .drop('SeriousDlqin2yrs', axis=1)
#                  .fillna(0)
#                  .reset_index(drop=True)
#                  )
#

baseline_train_aucs, baseline_test_aucs = [], []
biased_train_aucs, biased_test_aucs = [], []
for i in range(1):
    print("Iteration %d..." % i)
    X_train, X_test, y_train, y_test = train_test_split(
            X_clf, y_clf, test_size=.25
            )
    X_biased_train = X_train.loc[lambda df: df['age'] >= 35] #.drop('age', axis=1)
    y_biased_train = y_train.loc[X_biased_train.index]
    X_biased_test = X_test.loc[lambda df: df['age'] >= 35] #.drop('age', axis=1)
    y_biased_test = y_test.loc[X_biased_test.index]

    ## train
    clf_full = (
        LogisticRegressionCV(max_iter=1000, Cs=100)
            .fit(X_train, y_train)
    )
    clf_age_limited = (
        LogisticRegressionCV(max_iter=1000, Cs=100)
            .fit(X_biased_train, y_biased_train)
    )

    ## baseline classifier
    y_baseline_train_pred = clf_full.predict_proba(X_train)[:, 1]
    baseline_train_auc = roc_auc_score(y_train, y_baseline_train_pred)
    baseline_train_aucs.append(baseline_train_auc)

    y_baseline_test_pred = clf_full.predict_proba(X_test)[:, 1]
    baseline_test_auc = roc_auc_score(y_test, y_baseline_test_pred)
    baseline_test_aucs.append(baseline_test_auc)

    ## biased classifier
    y_biased_train_pred = clf_age_limited.predict_proba(X_biased_train)[:, 1]
    biased_train_auc = roc_auc_score(y_biased_train, y_biased_train_pred)
    biased_train_aucs.append(biased_train_auc)

    y_biased_test_pred = clf_age_limited.predict_proba(X_biased_test)[: , 1]
    biased_test_auc = roc_auc_score(y_biased_test, y_biased_test_pred)
    biased_test_aucs.append(biased_test_auc)



# original dataset. put aside 10,000 as holdout.
# remaining 40,000. split into age groups.
# over 35 should be included in both
# for the other add in the other.
# don't include age as a feature

# print("sample, run y_pred...")
# X_sample = X.sample(5000)
# y_pred_all_full_model = clf_full.predict_proba(X_sample)[:, 1]
# y_pred_all_downsampled = clf_age_limited.predict_proba(X_sample)[:, 1]

print("run on full holdout...")
# X_sample = X.sample(5000)
y_pred_all_full_model = clf_full.predict_proba(X_audit_holdout)[:, 1]
y_pred_all_downsampled = clf_age_limited.predict_proba(X_audit_holdout)[:, 1]

exp_df = pd.DataFrame(columns=['y_true', 'y_full_score', 'y_downsampled_score', 'age', 'y_full_cost', 'y_downsampled_cost'],
                      index = X_audit_holdout.index)


exp_df['y_full_score'] = y_pred_all_full_model
exp_df['y_downsampled_score'] = y_pred_all_downsampled
exp_df['age'] = X_audit_holdout['age']
exp_df['y_true'] = y.loc[X_audit_holdout.index]
exp_df.to_csv(output_dir / '2018-11-19__demo-1__exp-df.csv')

X_sample = (X_audit_holdout
            # .sample(10000)
            .copy()
            )
exp_df_sample = exp_df.loc[X_sample.index]

coefficients = {}
intercept = {}
coefficients['full'] = clf_full.coef_[0]
intercept['full'] = clf_full.intercept_[0]
coefficients['downsampled'] = clf_age_limited.coef_[0]
intercept['downsampled'] = clf_age_limited.intercept_[0]

# utilization bounded
# RealEstate should be positive
# numtimes90 days > 0
# monthly income >0
# debt
p = 0.98

# run audit
for dataset in ['full', 'downsampled']:
    y_col = 'y_%s_score' % dataset
    scores = exp_df_sample[y_col]
    # p = scores.median()
    denied_individuals = scores.loc[lambda s: s<=p].index
    # actionset
    action_set = ActionSet(X=X_audit_holdout)
    action_set['age'].mutable = False
    action_set['NumberOfDependents'].mutable = False
    action_set['DebtRatio'].step_direction = -1
    # action_set['NumberOfTime60-89DaysPastDueNotWorse'].step_direction = -1
    action_set.align(coefficients=coefficients[dataset])

    idx = 0
    flipsets = {}
    import time
    now = time.time()
    for i in denied_individuals:
        if idx % 100 == 0:
            print('finished %d points in %f...' %  (idx, time.time() - now))
            now = time.time()

        x = X.values[i]
        fb = RecourseBuilder(coefficients=coefficients[dataset],
                             intercept=intercept[dataset] - (np.log(p / (1. - p))),
                             action_set=action_set,
                             x=x)
        ## CPLEX
        cplex_output = fb.fit()
        flipsets[i] = cplex_output
        idx += 1

    ## plot cost
    flipset_df = pd.DataFrame.from_dict(flipsets, orient="index")
    flipset_df.to_csv(output_dir / '2018-11-19__demo-1__flipsets-%s.csv' % dataset)


flipset_full = pd.read_csv(output_dir / '2018-11-19__demo-1__flipsets-full.csv', index_col=0)
flipset_age = pd.read_csv(output_dir / '2018-11-19__demo-1__flipsets-downsampled.csv', index_col=0)
exp_df = pd.read_csv(output_dir / '2018-11-19__demo-1__exp-df.csv', index_col=0)

exp_df = exp_df.loc[flipset_full.index]
exp_df['y_downsampled_cost'] = flipset_age['total_cost']
exp_df['y_full_cost'] = flipset_full['total_cost']
flipset_full['y_true'] = exp_df['y_true'].loc[flipset_full.index]
flipset_age['y_true'] = exp_df['y_true'].loc[flipset_age.index]

### get flipset

## min-cost:
dataset = 'downsampled'
# dataset = 'full'
young_individuals = (exp_df
    .loc[lambda df: df['age'] < 30]
    .loc[lambda df: df['y_full_score'] > .96]
    .loc[lambda df: df['y_full_score'] < .98 ]
    # .loc[lambda df: df['y_downsampled_score']<.9]
    )
i = young_individuals.index[0]
action_set = ActionSet(X=X_audit_holdout)
action_set['DebtRatio'].step_direction = -1
action_set['age'].mutable = False
action_set['NumberOfDependents'].mutable = False
# action_set['MonthlyIncome'].mutable = False
action_set['NumberOfTime60-89DaysPastDueNotWorse'].mutable = False
# action_set['RevolvingUtilizationOfUnsecuredLines'].mutable = False
# action_set['NumberOfOpenCreditLinesAndLoans'].mutable = False
# action_set['NumberRealEstateLoansOrLines'].mutable = False
action_set.align(coefficients=coefficients[dataset])

p = .97
x = X.values[i]
fb = RecourseBuilder(
        coefficients=coefficients[dataset],
        intercept=intercept[dataset] - (np.log(p / (1. - p))),
        action_set=action_set,
        x=x
        )
cplex_output = fb.fit()
if cplex_output['feasible']:
    full_actionset = []
    for feature in range(len(x)):
        orig_val = x[feature]
        changed_val = (x[feature] + cplex_output['actions'][feature])
        if not np.isclose(orig_val, changed_val):
            output = "& \\textit{%s} & " %  X.columns[feature]
            output += "%f" % orig_val  + "\\longrightarrow & %f \\\\" % changed_val
            full_actionset.append(output)
    print("\n".join(full_actionset))
else:
    print("infeasible")


# &   \textit{NumberOfTime60 - 89DaysPastDueNotWorse} & 0 &  $\longrightarrow$ & 2 \\

full_stats_unstacked = (
    exp_df[[ 'y_full_score', 'age', 'y_full_cost']]
        .dropna()
        .rename(columns={
        'y_full_score':'score',
        'y_full_cost': 'cost'
        })
        .assign(**{"Training Run": "Full dataset"})
)

age_stats_unstacked = (
    exp_df[[ 'y_downsampled_score', 'age', 'y_downsampled_cost']]
        .dropna()
        .rename(columns={
        'y_downsampled_score': 'score',
        'y_downsampled_cost': 'cost'
        })
        .assign(**{"Training Run": "Age-limited dataset"})
)

unstacked_combined_set = (pd.concat([
    age_stats_unstacked,
    full_stats_unstacked
    ])
                          # .loc[lambda df: df['score'] < .96]
                          .replace([np.inf, -np.inf], np.nan)
                          .assign(age_cut=lambda df: pd.cut(df['age'], np.arange(25, 75, 5)))
                          .dropna()
                          )

combined_data_df = (pd.concat([
    (exp_df
     .assign(**{"Training Run": "Full Dataset" })
     [['y_full_cost', 'age', 'Training Run', 'y_true', 'y_full_score']]
     .rename(columns={'y_full_cost': 'total_cost', 'y_full_score': 'y_pred'})
     ),
    (exp_df
     .assign(**{"Training Run": "Age-downsampled"})
     [['y_downsampled_cost', 'age', 'Training Run', 'y_true', 'y_downsampled_score']]
     .rename(columns={'y_downsampled_cost': 'total_cost', 'y_downsampled_score': 'y_pred'})
     )]).replace([np.inf, -np.inf], np.nan)
                    .assign(age_cut=lambda df: pd.cut(df['age'], np.arange(25, 95, 5)))
                    .dropna()
                    )

max_total_cost = combined_data_df['total_cost'].max()
mapper = {
    0: "-1",
    1: "+1",
    'Age-downsampled': "Sample Pop.",
    'Full Dataset': "Target Pop.",
    }

plt.rc("font", size=20)
for y_true in [0, 1]:

    for training_run in ['Downsampled', 'Full Dataset']:

        plt.figure(figsize=(4, 4))
        ax = sns.violinplot(
                x='age_cut', y='total_cost',
                data=(combined_data_df
                    .loc[lambda df: df['Training Run'] == training_run]
                    .loc[lambda df: df['y_true'] == y_true]
                    ),
                linewidth = 0.5, cut=0,
                scale='width', color="gold",  inner='quartile'
                )

        plt.ylim((0, max_total_cost))
        plt.ylabel("Cost of Recourse")
        plt.xlabel("Age")
        ax.set_ylim((0, 1))
        ax.set_xticks(np.arange(0, 14, 2)- 1,)
        ax.set_xticklabels( np.arange(20, 90, 10))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for l in ax.lines:
            l.set_linewidth(2.)
            l.set_linestyle('-')
            l.set_solid_capstyle('butt')
        ax.set_facecolor("white")
        plt.savefig(output_dir / '2018-11-18__age-bins-by-sample-%s-y-%s.png' % (training_run.replace(' ', '-').lower(), y_true), bbox_inches="tight")
        plt.savefig(output_dir / '2018-11-18__age-bins-by-sample-%s-y-%s.pdf' % (training_run.replace(' ', '-').lower(), y_true), bbox_inches="tight")
        plt.close()





### extra EDA
### exploration...
age_cutoff_exploration = False
if age_cutoff_exploration:
    # ### Find the right age cutoffs for training
    # test different AUC cutoff ranges
    grid_search_age_cutoff = False
    if grid_search_age_cutoff:
        train_cutoffs = range(30, 50, 5)
        holdout_cutoffs = range(22, 35, 3)
        auc_df = pd.DataFrame(index=train_cutoffs, columns=holdout_cutoffs, )
        for train_cutoff, holdout_cutoff in itertools.product(train_cutoffs, holdout_cutoffs):
            ## split dataset
            # training dataset
            X_train = X.loc[lambda df: df['age'] > train_cutoff]
            y_train = y.loc[X_train.index]
            # holdout dataset
            X_holdout = X.loc[lambda df: df['age'] <= holdout_cutoff]
            y_holdout = y.loc[X_holdout.index]

            # fit/predict
            clf = LogisticRegression().fit(X_train, y_train)
            y_pred = clf.predict_proba(X_holdout)[:, 1]
            auc = roc_auc_score(y_holdout, y_pred)
            # cache auc
            auc_df.at[train_cutoff, holdout_cutoff] = auc

        plt.figure(figsize=(6, 3))
        sns.heatmap(auc_df.fillna(0).loc[auc_df.index[::-1]], )
        plt.ylabel('Training set > age')
        plt.xlabel('Testing set < age')
        plt.title('AUCs of Models Trained on\nVarious dataset splits')
        plt.savefig(output_dir / '2018-08-11__aucs-of-age-based-dataset-splits.png')


    # ### Generate AUC scores for vanilla vs. hardweighted models
    def cross_val_scores_weighted(model, X, y, weights, cv=5, metrics=[sklearn.metrics.roc_auc_score]):
        kf = KFold(n_splits=cv)
        kf.get_n_splits(X)
        training_scores = []
        test_scores = []
        for train_index, test_index in kf.split(X.index):
            model_clone = sklearn.base.clone(model)
            X_train, X_audit_holdout = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            weights_train, weights_test = weights[train_index], weights[test_index]
            model_clone.fit(X_train, y_train, sample_weight=weights_train)
            y_pred = model_clone.predict_proba(X_audit_holdout)[:, 1]
            aucs = []

            y_pred_train = model_clone.predict_proba(X_train)[:, 1]
            y_pred_test = model_clone.predict_proba(X_audit_holdout)[:, 1]
            for i, metric in enumerate(metrics):
                test_score = metric(y_test, y_pred_test, sample_weight=np.abs(1 - weights_test))
                test_scores.append(score)

                training_score = metric(y_train, y_pred_train, sample_weight=np.abs(1 - weights_train))
                training_scores.append(training_score)
        return training_scores, test_scores


    weights = X['age'].pipe(lambda s: s > train_cutoff).map({True: .9, False: .1}).values
    logistic_model = LogisticRegression()
    weighted_training_scores, weighted_test_scores = cross_val_scores_weighted(logistic_model, X, y, weights, cv=10)

    unweighted_test_scores = []
    unweighted_train_scores = []
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X.index):
        X_train, X_audit_holdout = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_audit_holdout)[:, 1]
        score = roc_auc_score(y_test, y_pred)
        unweighted_test_scores.append(score)
        y_pred = clf.predict_proba(X_train)[:, 1]
        score = roc_auc_score(y_train, y_pred)
        unweighted_train_scores.append(score)

    hard_weighted_test_scores = []
    hard_weighted_train_scores = []
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    X_s = X.loc[lambda df: df['age'] <= 35]
    y_s = y.loc[X_s.index].reset_index(drop=True)
    X_s = X_s.reset_index(drop=True)
    for train_index, test_index in kf.split(X_s.index):
        X_train, X_audit_holdout = X_s.loc[train_index], X_s.loc[test_index]
        y_train, y_test = y_s.loc[train_index], y_s.loc[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_audit_holdout)[:, 1]
        score = roc_auc_score(y_test, y_pred)
        hard_weighted_test_scores.append(score)
        y_pred = clf.predict_proba(X_train)[:, 1]
        score = roc_auc_score(y_train, y_pred)
        hard_weighted_train_scores.append(score)

    aucs = []
    for age_split in range(25, 65, 5):
        y_pred = exp_df.loc[lambda df: df['age'] >= age_split]['score']
        y_true = y.loc[y_pred.index]
        auc = roc_auc_score(y_true, y_pred)
        aucs.append(auc)
    pd.Series(aucs, index=range(25, 65, 5)).plot()
    plt.title('AUCs')
    plt.xlabel('AUC on Age > x')
    plt.vlines(train_cutoff, *plt.ylim(), linestyles='dashed')
    plt.text(train_cutoff * .98, .725, 'training\ncutoff', horizontalalignment='right')

    # ### Another look at age-cutoff AUC: Aucs over trained across age
    train_cutoff = 30
    holdout_cutoff = 28
    exp_df = pd.DataFrame(columns=['score', 'age', 'inc_in_train'], index=X.index)
    exp_df.loc[X_train.index, 'inc_in_train'] = True
    exp_df.loc[X_holdout.index, 'inc_in_train'] = False
    X_train = X.loc[lambda df: df['age'] > train_cutoff]
    y_train = y.loc[X_train.index]
    # holdout dataset
    X_holdout = X.loc[lambda df: df['age'] <= holdout_cutoff]
    y_holdout = y.loc[X_holdout.index]

    aucs = []
    for age_split in range(25, 65, 5):
        y_pred = exp_df.loc[lambda df: df['age'] >= age_split]['score']
        y_true = y.loc[y_pred.index]
        auc = roc_auc_score(y_true, y_pred)
        aucs.append(auc)

    pd.Series(aucs, index=range(25, 65, 5)).plot()
    plt.title('AUCs')
    plt.xlabel('AUC on Age > x')
    plt.vlines(train_cutoff, *plt.ylim(), linestyles='dashed')
    plt.text(train_cutoff * 1.01, .705, 'training\ncutoff', horizontalalignment='left')
    plt.savefig(output_dir / '2018-08-12__cv-auc-by-age.png', bbox_inches='tight')


    ## extra
    # ### How does average score differ across age?
    score_and_age = (
        exp_df
            .assign(c=1)
            .groupby(pd.cut(exp_df['score'], 50)).aggregate({'c':'sum', 'age':'mean'})
            .reset_index()             # redo index
            .assign(score=lambda df: df['score'].apply(lambda x: x.left))
            .set_index('score')
    )

    cmap = plt.get_cmap('RdYlGn')
    ax = score_and_age['c'].plot(kind='bar', color=cmap(score_and_age['age'].pipe(lambda s: (s - s.min())/(s.max() - s.min()))))
    ax.set_xticks(range(0, 50, 10))
    ax.set_xticklabels(['%.01f' % s for s in np.arange(0, 1, .2)])
    ax.semilogy()

    plt.rc("font", size=20)
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    train_cutoff = 35
    flipset_df = pd.concat([flipset_df, exp_df.loc[flipset_df.index]], axis=1)
    flipset_df = flipset_df.assign(total_cost=lambda df: df['total_cost'].replace([-np.inf, np.inf], np.nan)).dropna()

    color_var = 'total_cost'
    x_axis_var = 'age'

    flipset_cost_and_age = (
        flipset_df.assign(c=1).pipe(lambda df:
                                    df.groupby(pd.cut(df[x_axis_var], 50)).aggregate({'c': 'sum', color_var: 'mean'})
                                    # redo index
                                    .reset_index()
                                    .assign(**{x_axis_var: lambda df: df[x_axis_var].apply(lambda x: x.left)})
                                    .set_index(x_axis_var)
                                    )
    )

    cmap = plt.get_cmap('RdYlGn')
    ax = (flipset_cost_and_age['c']
          .plot(kind='bar', color=cmap(
            flipset_cost_and_age[color_var].pipe(lambda s: (s - s.min()) / (s.max() - s.min()))
            )))

    ylim = ax.get_ylim()
    train_cutoff_x = float(np.digitize(train_cutoff, flipset_cost_and_age.index))
    ax.vlines(train_cutoff_x, *ylim, linestyles='dashed')
    ax.set_ylim(ylim)
    ax.text(train_cutoff_x * 1.05, 40, 'Training age\ncutoff: %d' % train_cutoff)

    ax.set_xticks(range(0, 50, 10))
    ax.set_xticklabels(flipset_cost_and_age.index[::10])

    # ax.semilogy()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.15)
    norm = mpl.colors.Normalize(
            vmin=flipset_cost_and_age[color_var].min(),
            vmax=flipset_cost_and_age[color_var].max()
            )

    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('Average Cost of Bucket')

    ax.set_title('Average Age Across Cost Range\nAge-Holdout: %d' % train_cutoff)
    ax.set_ylabel('Number of datapoints')

    plt.savefig(output_dir / '2018-08-12__hist-over-ages-and-costs__age-holdout-%d.png' % train_cutoff, bbox_inches="tight")