import numpy as np

def is_sklearn_linear_classifier(obj):
    """
    Checks if object is a sklearn linear classifier for a binary outcome
    :param obj: object
    """
    binary_flag = hasattr(obj, 'classes_') and len(obj.classes_) == 2
    linear_flag = hasattr(obj, 'coef_') and hasattr(obj, 'intercept_')
    return binary_flag and linear_flag


def parse_classifier_args(*args, **kwargs):
    """
    helper function to parse coefficients and intercept from linear classifier arguments

    *args and **kwargs can contain either:
        - sklearn classifiers with 'coef_' and 'intercept_' fields (keyword: 'clf', 'classifier')
        - vector of coefficients (keyword: 'coefficients')
        - intercept: set to 0 by default (keyword: 'intercept')

    returns:
        w - np.array containing coefficients of linear classifier (finite, flattened)
        t - float containing intercept of linear classifier (finite, float)

    raises:
        ValueError if fails to parse classifier arguments

    :return:
    """
    w, t = None, None

    if 'clf' in kwargs:

        assert is_sklearn_linear_classifier(kwargs['clf'])
        w = kwargs['clf'].coef_
        t = kwargs['clf'].intercept_

    elif 'classifier' in kwargs:

        assert is_sklearn_linear_classifier(kwargs['classifier'])
        w = kwargs['classifier'].coef_
        t = kwargs['classifier'].intercept_

    elif 'coefficients' in kwargs:

        w = kwargs.get('coefficients')
        t = kwargs.get('intercept', 0.0)

    elif len(args) == 1:

        if is_sklearn_linear_classifier(args[0]):

            w = args[0].coef_
            t = args[0].intercept_

        elif isinstance(args[0], (list, np.ndarray)):

            w = np.array(args[0]).flatten()
            t = 0.0

    elif len(args) == 2:

        w = args[0]
        t = float(args[1])

    else:
        raise ValueError('failed to match classifier arguments')

    w = np.array(w).flatten()
    t = float(t)
    assert np.isfinite(w).all()
    assert np.isfinite(t)
    return w, t


def check_variable_names(names):
    """
    checks variable names
    :param names: list of names for each feature in a dataset.
    :return:
    """
    assert isinstance(names, list), '`names` must be a list'
    assert all([isinstance(n, str) for n in names]), '`names` must be a list of strings'
    assert len(names) >= 1, '`names` must contain at least 1 element'
    assert all([len(n) > 0 for n in names]), 'elements of `names` must have at least 1 character'
    assert len(names) == len(set(names)), 'elements of `names` must be distinct'
    return True


def determine_variable_type(values, name=None):
    for v in values:
        if isinstance(v, str):
            raise ValueError(">=1 elements %s are of type str" % ("in '%s'" % name if name else ''))
    integer_valued = np.equal(np.mod(values, 1), 0).all()
    if integer_valued:
        if np.isin(values, (0, 1)).all():
            return bool
        else:
            return int
    else:
        return float


def expand_values(value, m):

    if isinstance(value, np.ndarray):

        if len(value) == m:
            value_array = value
        elif value.size == 1:
            value_array = np.repeat(value, m)
        else:
            raise ValueError("length mismatch; need either 1 or %d values" % m)

    elif isinstance(value, list):
        if len(value) == m:
            value_array = value
        elif len(value) == 1:
            value_array = [value] * m
        else:
            raise ValueError("length mismatch; need either 1 or %d values" % m)

    elif isinstance(value, str):
        value_array = [str(value)] * m

    elif isinstance(value, bool):
        value_array = [bool(value)] * m

    elif isinstance(value, int):
        value_array = [int(value)] * m

    elif isinstance(value, float):
        value_array = [float(value)] * m

    else:
        raise ValueError("unknown variable type %s")

    return value_array