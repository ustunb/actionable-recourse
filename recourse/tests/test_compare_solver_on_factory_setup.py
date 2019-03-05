from recourse.tests.test_classes import *
from recourse.builder import RecourseBuilder

@pytest.fixture
def german_recourse_builder_cplex(german_scores, german_actionset_aligned, german_coef, german_intercept, german_denied_individual, german_p):
    return RecourseBuilder(
        solver='cplex',
        action_set=german_actionset_aligned,
        coefficients=german_coef,
        intercept=german_intercept - (np.log(german_p / (1. - german_p))),
        x=german_denied_individual
    )

@pytest.fixture
def german_recourse_builder_pyomo(german_actionset_aligned, german_coef, german_intercept, german_denied_individual, german_p):
    return RecourseBuilder(
        solver='cbc',
        action_set=german_actionset_aligned,
        coefficients=german_coef,
        intercept=german_intercept - (np.log(german_p / (1. - german_p))),
        x=german_denied_individual
    )


def test_german_recourse_cplex_fit(german_recourse_builder_cplex):
    """Test fitting on a denied individual, CPLEX."""
    cplex_output = german_recourse_builder_cplex.fit()
    cplex_output_df = pd.DataFrame(cplex_output)[['actions', 'costs']]
    print(cplex_output_df)


def test_german_recourse_pyomo_fit(german_recourse_builder_pyomo):
    """Test fitting on a denied individual, PYOMO."""
    pyo_output = german_recourse_builder_pyomo.fit()
    max_cost = pyo_output.pop('max_cost', None)
    pyo_output_df = (pd.DataFrame
     .from_dict(pyo_output, orient='index')
     .loc[lambda df: df['u'] == 1]
    )
    print(pyo_output_df)


