from recourse.tests.fixtures import *


def test_rb_fit(data, recourse_builder):
    """Test fitting on a denied individual, CPLEX."""

    # pick a denied individual
    output = recourse_builder.fit()
    output = pd.DataFrame(output)[['actions', 'costs']]

    # todo check all costs are positives
    print(output)

