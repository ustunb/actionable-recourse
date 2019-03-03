from scripts.paths import *
from scripts.experimental_setup import *
from scripts.plotting import *
from recourse.action_set import ActionSet
from recourse.flipset import Flipset

# user settings
settings = {
    'data_name': 'credit',
    'method_name': 'logreg',
    #
    'method_suffixes': [''],
    'audit_suffixes': [''],
    #
    'normalize_data': True,
    'print_flag': True,
    'randomseed': 2338,
    #
    # specific plots for logistic regression
    'audit_recourse': False,
    'save_audit_results': True,
    'force_rational_actions': False,
    'plot_audits': True,
    }
# results_dir = '~/Projects/berk-research/recourse/tex/figure'

# file names
output_dir = '%s/%s' % (results_dir, settings['data_name'])
os.makedirs(output_dir, exist_ok = True)

if settings['normalize_data']:
    settings['method_suffixes'].append('normalized')
else:
    settings['method_suffixes'].append('unnormalized')

if settings['force_rational_actions']:
    settings['audit_suffixes'].append('rational')

# set file header
settings['dataset_file'] = '%s/%s_processed.csv' % (data_dir, settings['data_name'])
settings['file_header'] = '%s/%s_%s%s' % (output_dir, settings['data_name'], settings['method_name'], '_'.join(settings['method_suffixes']))
settings['audit_file_header'] = '%s%s' % (settings['file_header'], '_'.join(settings['audit_suffixes']))
settings['model_file'] = '%s_models.pkl' % settings['file_header']
settings['audit_file'] = '%s_audit_results.pkl' % settings['audit_file_header']

pp.pprint(settings)

#### Initialize Dataset ####

data_df = pd.read_csv(settings['dataset_file'])

data = {
    'outcome_name': data_df.columns[0],
    'variable_names': data_df.columns[1:].tolist(),
    'X': data_df.iloc[:, 1:],
    'y': data_df.iloc[:, 0],
    'scaler': None,
    }

if settings['normalize_data']:
    from sklearn.preprocessing import StandardScaler
    data['scaler'] = StandardScaler(copy = True, with_mean = True, with_std = True)
    data['X_train'] = pd.DataFrame(data['scaler'].fit_transform(data['X'], data['y']), columns = data['X'].columns)
else:
    data['X_train'] = data['X']

#### Initialize Actionset ####

default_bounds = (0.1, 99.9, 'percentile')
custom_bounds = None
immutable_variables = []


if settings['data_name'] == 'credit':

    immutable_names = ['Female', 'Single', 'Married']
    immutable_names += list(filter(lambda x: 'Age' in x or 'Overdue' in x, data['variable_names']))
    default_bounds = (0.1, 99.9, 'percentile')
    custom_bounds = {'Female': (0, 100, 'p'),  'Married': (0, 100, 'p')}
    data['immutable_variable_names'] = [n for n in immutable_names if n in data['variable_names']]

    action_set = ActionSet(X = data['X'], custom_bounds = custom_bounds, default_bounds = default_bounds)
    action_set[data['immutable_variable_names']].mutable = False

    payment_fields = list(filter(lambda x: 'Amount' in x, data['variable_names']))
    action_set[payment_fields].step_type = 'absolute'
    action_set[payment_fields].step_size = 5
    for p in payment_fields:
        action_set[p].update_grid()

    action_set['EducationLevel'].step_direction = 1
    action_set['MaxBillAmountOverLast6Months'].step_direction = -1
    action_set['MaxPaymentAmountOverLast6Months'].step_direction = 1



#### Initialize Model Files ####
model_stats = pickle.load(open(settings['model_file'], 'rb'))

mean_test_error = -model_stats['model_stats_df'].groupby('param 0: C')['test_score'].aggregate(['mean'])
#min_idx = mean_test_error == mean_test_error.min()
#min_error = mean_test_error.index(min_idx)


clf = model_stats['all_models']['C_0.02__max_iter_1000__penalty_l1__solver_saga__tol_1e-08']
yhat = clf.predict(X = data['X_train'])
coefficients, intercept = undo_coefficient_scaling(clf, scaler = data['scaler'])
action_set.align(coefficients)

predicted_neg = np.flatnonzero(yhat < 1)
U = data['X'].iloc[predicted_neg].values

# produce flipset
k = 4
fb = FlipsetBuilder(coefficients = coefficients, intercept = intercept, action_set = action_set, x = U[k], mip_cost_type = 'local')
flipset = Flipset(x = fb.x, coefficients = coefficients, intercept = intercept, variable_names = data['variable_names'])

#fb.max_items = 4
items = fb.populate(enumeration_type = 'distinct_subsets', total_items = 14)
flipset.add(items)
print(flipset.to_latex()) #creates latex table for paper
print(flipset.view()) # displays to screen

#