from sklearn.model_selection import GridSearchCV
from copy import copy

# import scripts to setup experiments
from scripts.experimental_setup import *
from scripts.plotting import *

# user settings
settings = {
    #
    'data_name': 'credit',
    'method_name': 'logreg',
    'n_folds': 10,
    'cv_type': 'stratified',
    #
    # script flags
    'train_classifiers': True,
    'normalize_data': True,
    'print_flag': True,
    'save_flag': True,
    'random_seed': 2338,
    #
    # data-related plots
    'plot_data_summary': False,
    'plot_model_error_path': True,
    'plot_model_size_path': True,
    'plot_model_coef_path': True,
    #
    # placeholders
    'method_suffixes': [''],
    }

# file names
output_dir = '%s/%s' % (results_dir, settings['data_name'])
os.makedirs(output_dir, exist_ok = True)

if settings['normalize_data']:
    settings['method_suffixes'].append('normalized')

# set file names
settings['dataset_file'] = '%s/%s_processed.csv' % (data_dir, settings['data_name'])
settings['file_header'] = '%s/%s_%s%s' % (output_dir, settings['data_name'], settings['method_name'], '_'.join(settings['method_suffixes']))
settings['model_file'] = '%s_models.pkl' % settings['file_header']

#### Initialize Dataset ####

data_df = pd.read_csv(settings['dataset_file'])

data = {
    'outcome_name': data_df.columns[0],
    'variable_names': data_df.columns[1:].tolist(),
    'X': data_df.iloc[:, 1:],
    'y': data_df.iloc[:, 0],
    'scaler': None,
    'immutable_variable_names': [],
    }

if settings['data_name'] == 'credit':

    immutable_names = ['Female', 'Single', 'Married'] + list(filter(lambda x : 'Age' in x or 'Overdue' in x, data['variable_names']))
    data['immutable_variable_names'] = [n for n in immutable_names if n in data['variable_names']]


if settings['normalize_data']:
    from sklearn.preprocessing import StandardScaler
    data['scaler'] = StandardScaler(copy = True, with_mean = True, with_std = True)
    data['X_train'] = pd.DataFrame(data['scaler'].fit_transform(data['X'], data['y']), columns = data['X'].columns)
else:
    data['X_train'] = data['X']

if settings['plot_data_summary']:
    f, _ = create_data_summary_plot(data_df)
    f.savefig('%s_data_summary.pdf' % settings['file_header'])

##### Train Models ####

if settings['train_classifiers']:

    # choose method
    get_coef = lambda clf: clf.coef_[0]
    get_intp = lambda clf: clf.intercept_[0]
    param_grid = {}

    if settings['method_name'] == 'logreg':
        from sklearn.linear_model import LogisticRegression as Classifier
        param_grid = {'penalty': ['l1'],
                      'C': [1.0 / l for l in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]],
                      # 'verbose': [True],
                      'solver': ['saga'], 'tol': [1e-8], 'max_iter': [1000]}

    elif settings['method_name'] == 'svm':
        from sklearn.svm import LinearSVC as Classifier
        param_grid = {'C': [1.0 / l for l in [0.01, 0.1, 1, 10, 100]], 'dual': [False], 'tol': [1e-8]}
    else:
        raise NameError('method %s not supported' % settings['method_name'])

    clf = Classifier()

    # setup cross validation
    if settings['cv_type'] == 'stratified':
        from sklearn.model_selection import StratifiedKFold as CVGenerator
    else:
        from sklearn.model_selection import KFold as CVGenerator
    cv_generator = CVGenerator(n_splits = settings['n_folds'], random_state = settings['random_seed'])

    # the remainder of this code is a general purpose training function using GridSearchCV
    gridsearch = GridSearchCV(clf,
                              param_grid,
                              return_train_score = True,
                              cv = cv_generator,
                              scoring = 'neg_mean_squared_error')

    #todo add AUC to models

    gridsearch.fit(data['X_train'], data['y'])
    grid_search_df = pd.DataFrame(gridsearch.cv_results_)

    # cache a model for each parameter combination, trained on all data
    grid_search_df['key'] = pd.np.nan
    final_models = {}
    for idx, p in grid_search_df.params.iteritems():
        key = '__'.join(map(lambda x: '%s_%s' % x, p.items()))
        model = copy(clf.set_params(**p)).fit(data['X_train'], data['y'])
        final_models[key] = model
        grid_search_df.loc[idx, 'key'] = key

    # get stats df out of gridsearch
    model_stats_df = format_gridsearch_df(grid_search_df,
                                          settings = settings,
                                          n_coefficients = data['X'].shape[1],
                                          invert_C = settings['method_name'] == 'logreg')

    model_stats = {
        'all_models': final_models,
        'model_stats_df': model_stats_df,
        'best_model': gridsearch.best_estimator_,
        'raw_coef_df': get_coefficient_df(final_models, variable_names = data['variable_names']),
        'coef_df': get_coefficient_df(final_models, variable_names = data['variable_names'], scaler = data['scaler']),
        }

    if settings['save_flag']:
        pickle.dump(model_stats, file = open(settings['model_file'], 'wb'), protocol=2)
        print_log('saved model file to %s' % settings['model_file'])

else:

    model_stats = pickle.load(open(settings['model_file'], 'rb'))


#### Model Analysis ####

assert model_stats is not None
coef_df = model_stats['coef_df']
raw_coef_df = model_stats['raw_coef_df']

if settings['method_name'] == 'logreg':
    coef_df = coef_df.rename(columns={col: 1. / float(col.split('_')[1]) for col in coef_df.columns})
    raw_coef_df = raw_coef_df.rename(columns={col: 1. / float(col.split('_')[1]) for col in raw_coef_df.columns})
    xlabel = '$\ell_1$-penalty (log scale)'
else:
    coef_df = coef_df.rename(columns={col: float(col.split('_')[1]) for col in coef_df.columns})
    raw_coef_df = raw_coef_df.rename(columns = {col: 1. / float(col.split('_')[1]) for col in raw_coef_df.columns})
    xlabel = '$C$-penalty (log scale)'

coef_df.to_csv('%s_coefficient_df.csv' % settings['file_header'], float_format = '%1.6f')

if settings['plot_model_size_path']:

    f, ax = create_figure()

    # plot # of non zero coefficients
    nnz_coef_df = raw_coef_df.apply(lambda x: ~np.isclose(x, 0.0, rtol = 1e-4))
    non_zero_sum = nnz_coef_df.sum()
    non_zero_sum_actionable = (nnz_coef_df.pipe(lambda df: df.loc[~df.index.isin(data['immutable_variable_names'])]).sum())
    non_zero_sum.plot(ax = ax, marker='o', label = 'All Features')
    non_zero_sum_actionable.plot(ax = ax, marker='o', label = 'Actionable Features')
    # todo: fix bug -> if a variable has a coefficient with flip_direction != step_direction then it's also not actionable
    # todo: need to compute this using the action_set

    # formatting
    max_nnz = np.max(np.sum(nnz_coef_df, axis = 0))
    ax.semilogx()
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Non-Zero Coefficients')
    ax.set_ylim((0, max_nnz + 0.5))
    ax.set_yticks(ticks = np.arange(0, max_nnz + 2, 2).tolist())
    ax.legend(frameon = True, prop = {'size': 20})
    ax = fix_font_sizes(ax)
    f.savefig('%s_model_size_path.pdf' % settings['file_header'], bbox_inches='tight')
    plt.close()


if settings['plot_model_error_path']:

    train_error = model_stats['model_stats_df'].groupby('param 0: C')['train_score'].aggregate(['mean', 'var'])
    test_error = model_stats['model_stats_df'].groupby('param 0: C')['test_score'].aggregate(['mean', 'var'])

    f, ax = create_figure()
    test_error['mean'] = -test_error['mean']
    test_error['mean'].plot(ax = ax, label='test error', color='black')
    ax.errorbar(test_error.index, test_error['mean'], yerr = test_error['var'], fmt='o', color='black')

    # formatting
    plt.semilogx()
    ax.set_xlabel(xlabel)
    ax.set_ylabel('%d-CV Mean Test Error' % settings['n_folds'])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals = 1))
    ax.set_ylim((0.19, 0.20))
    ax = fix_font_sizes(ax)
    f.savefig('%s_model_error_path.pdf' % settings['file_header'], bbox_inches='tight')
    plt.close()


if settings['plot_model_coef_path']:
    f, ax = create_coefficient_path_plot(coefs_df = coef_df, fig_size = (20, 20), label_coefs = True, fontsize = 12)
    ax = fix_font_sizes(ax)
    f.savefig('%s_model_coefficient_path.pdf' % settings['file_header'])
    plt.close()

