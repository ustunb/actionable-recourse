from recourse.tests.fixtures import *
import itertools

# Test Strategy
# --------------------------------------------------------
# cost function:        local
# variable types:       all binary, mix
# # of variables in w:  1, >1
# recourse:             exists, does not exist
# action_set:           all actionable, all conditionally actionable, all immutable, mix

# fit
# populate

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


def test_flipset_mutually_exclusive(flipset):
    # generate flipset for person i
    items = flipset.populate(total_items=5, enumeration_type='mutually_exclusive').items
    actions = list(map(lambda x: x['actions'], items))
    ## check that the overlap between different actions is less than the max of either actionset
    for actionset_i, actionset_j in itertools.combinations( actions, 2):
        assert actionset_i.dot(actionset_j) == 0
