from examples.paper.experimental_setup import *
from examples.paper.plotting import *
from recourse.action_set import ActionSet
from recourse.builder import RecourseBuilder
from recourse.flipset import Flipset
import pickle

# user settings
settings = {
    #
    # audit settings
    'data_name': 'credit',
    'method_name': 'logreg',
    'normalize_data': True,
    'force_rational_actions': False,
    #
    # script flags
    'audit_recourse': True,
    'plot_audits': True,
    'print_flag': True,
    'save_flag': True,
    'randomseed': 2338,
    #
    # placeholders
    'method_suffixes': [''],
    'audit_suffixes': [''],
    }

# file names
output_dir = results_dir / settings['data_name']
output_dir.mkdir(exist_ok = True)

if settings['normalize_data']:
    settings['method_suffixes'].append('normalized')

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
    'y': data_df.iloc[:, 0]
    }

scaler = None
if settings['normalize_data']:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(copy = True, with_mean = True, with_std = True)
    data['X_scaled'] = pd.DataFrame(scaler.fit_transform(data['X'], data['y']), columns = data['X'].columns)

#### Initialize Actionset ####

default_bounds = (1.0, 99.0, 'percentile')
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

    action_set['EducationLevel'].step_direction = 1

    payment_fields = list(filter(lambda x: 'Amount' in x, data['variable_names']))
    action_set[payment_fields].step_type = 'absolute'
    action_set[payment_fields].step_size = 5

    for p in payment_fields:
        action_set[p].update_grid()


#### Initialize Model Files ####
model_stats = pickle.load(open(settings['model_file'], 'rb'))
model_stats['all_models'].keys()

audit = None
settings['print_flag'] = False
if settings['audit_recourse']:

    audit_results = {}
    for key, model in model_stats['all_models'].items():

        if settings['method_name'] == 'logreg':
            model_name = 1. / float(key.split('_')[1])
        else:
            model_name = float(key.split('_')[1])

        # run the audit_results.
        model_results = get_flipset_solutions(model = model, data = data, action_set = action_set, scaler = scaler, print_flag = settings['print_flag'])
        if len(model_results) > 0:
            audit_results[model_name] = model_results

    if settings['save_flag']:
        pickle.dump(audit_results, file = open(settings['audit_file'], 'wb'), protocol=2)

else:
    audit_results = pickle.load(file = open(settings['audit_file'], 'rb'))

#### Audit Analysis ####

if settings['method_name'] == 'logreg':
    xlabel = '$\ell_1$-penalty (log scale)'
else:
    xlabel = '$C$-penalty (log scale)'


if settings['plot_audits']:

    # percent of points without recourse
    feasibility_df = {}
    obj_val = {}

    for model_name in sorted(audit_results):
        recourse_df = pd.DataFrame(audit_results[model_name])
        recourse_cost = recourse_df.loc[lambda df: df.feasible].loc[:, 'total_cost']
        feasibility_df[model_name] = recourse_df['feasible'].mean()
        obj_val[model_name] = recourse_cost.mean()

    # feasibility plot
    f, ax = create_figure(fig_size = (6, 6))
    t_found = pd.Series(feasibility_df)
    t_found.plot(ax = ax, color = 'black', marker='o')
    plt.semilogx()
    ax.set_xlabel(xlabel)
    ax.set_ylabel('% of Individuals with Recourse')
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals = 0))
    ax = fix_font_sizes(ax)
    f.savefig('%s_recourse_feasibility.pdf' % settings['audit_file_header'], bbox_inches = 'tight')
    plt.close()

    cost_df = {k: pd.DataFrame(v) for k, v in audit_results.items()}
    cost_df = pd.concat([cost_df[k]['total_cost'].to_frame('%f' % k) for k in sorted(cost_df.keys())], axis=1).replace([-np.inf, np.inf], np.nan)

    # plot cost distribution
    f, ax = create_figure(fig_size = (6, 6))
    sns.violinplot(data = cost_df, ax = ax, linewidth = 0.5, cut = 0, inner = 'quartile', color = "gold", scale = 'width')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cost of Recourse')
    ax.set_ylim(bottom = 0, top = 1)
    xtick_labels = []
    # for xt in ax.get_xticklabels():
    #     v = np.log10(float(xt.get_text()))
    #     label = '$10^{%.0f}$' % v if v == np.round(v, 0) else ' '
    #     xtick_labels.append(label)
    ax.set_xticklabels(xtick_labels)

    for l in ax.lines:
        l.set_linewidth(3.0)
        l.set_linestyle('-')
        l.set_solid_capstyle('butt')

    ax = fix_font_sizes(ax)
    f.savefig('%s_recourse_cost_distribution.pdf' % settings['audit_file_header'], bbox_inches = 'tight')
    plt.close()

    # store median cost
    cost_df.median(axis = 0).to_csv('%s_median_cost_df.csv' % settings['audit_file_header'])

    # plot the mean cost of recourse
    f, ax = create_figure(fig_size = (6, 6))
    ts_m = pd.Series(obj_val)
    ax = ts_m.plot(ax = ax, color = 'black', marker = 'o')
    plt.semilogx()
    plt.xlabel(xlabel)
    plt.ylabel('Mean Cost of Recourse')
    ax = fix_font_sizes(ax)
    f.savefig('%s_recourse_cost.pdf' % settings['audit_file_header'], bbox_inches = 'tight')
    plt.close()



#### Flipset Generation
clf = model_stats['all_models']['C_0.02__max_iter_1000__penalty_l1__solver_saga__tol_1e-08']
yhat = clf.predict(X = data['X_train'])
coefficients, intercept = undo_coefficient_scaling(clf, scaler = data['scaler'])
action_set.align(coefficients)

predicted_neg = np.flatnonzero(yhat < 1)
U = data['X'].iloc[predicted_neg].values

# produce flipset
k = 4
fb = RecourseBuilder(coefficients = coefficients, intercept = intercept, action_set = action_set, x = U[k], mip_cost_type = 'local')
flipset = Flipset(x = fb.x, coefficients = coefficients, intercept = intercept, variable_names = data['variable_names'])

#fb.max_items = 4
items = fb.populate(enumeration_type = 'distinct_subsets', total_items = 14)
flipset.add(items)
print(flipset.to_latex()) #creates latex table for paper
print(flipset.view()) # displays to screen
