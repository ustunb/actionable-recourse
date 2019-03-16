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
