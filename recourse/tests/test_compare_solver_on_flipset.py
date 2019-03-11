from recourse.tests.fixtures import *
import pytest
import numpy as np
import pandas as pd
from recourse.builder import _SOLVER_TYPE_CBC, _SOLVER_TYPE_CPX
import itertools


@pytest.mark.parametrize('flipset', [_SOLVER_TYPE_CPX, _SOLVER_TYPE_CBC], indirect=True)
def test_flipset_distinct_subsets(flipset):
    # generate flipset for person i
    items = flipset.populate(total_items=5, enumeration_type='distinct_subsets').items
    actions = np.array(list(map(lambda x: x['actions'], items))).astype(int)
    on_actions = (pd.DataFrame(actions)
                  .pipe(lambda df: pd.DataFrame(~np.isclose(df, 0)))
                  .astype(int)
                  )
    num_actions_on = on_actions.sum(axis=1)

    ## check that the overlap between different actions is less than the max of either actionset
    for i, j in itertools.combinations( on_actions.index, 2):
        num_overlap = on_actions.loc[i].dot(on_actions.loc[j])
        assert num_overlap < max(num_actions_on[i], num_actions_on[j])


@pytest.mark.parametrize('flipset', [_SOLVER_TYPE_CPX, _SOLVER_TYPE_CBC], indirect=True)
def test_flipset_mutually_exclusive(flipset):
    # generate flipset for person i
    items = flipset.populate(total_items=5, enumeration_type='mutually_exclusive').items
    actions = list(map(lambda x: x['actions'], items))
    ## check that the overlap between different actions is less than the max of either actionset
    for actionset_i, actionset_j in itertools.combinations( actions, 2):
        assert actionset_i.dot(actionset_j) == 0


def test_compare_flipsets(flipset_cpx, flipset_cbc):
    cpx_items = flipset_cbc.populate(total_items=5).items
    cbc_items = flipset_cbc.populate(total_items=5).items

    assert len(cpx_items) == len(cbc_items)
    assert np.isclose([x['cost'] for x in cpx_items], [x['cost'] for x in cbc_items]).all()
