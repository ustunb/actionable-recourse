from recourse.flipset import Flipset
from recourse.tests.test_classes import *
import pytest

@pytest.fixture
def german_flipset(german_denied_individual, german_actionset_aligned, german_clf):
    return Flipset(x=german_denied_individual, action_set=german_actionset_aligned, clf=german_clf)


def test_flipset_cplex(german_flipset):
    # generate flipset for person i
    german_flipset.populate(total_items = 5)
    german_flipset.to_latex()
    german_flipset.view()

# todo Alex