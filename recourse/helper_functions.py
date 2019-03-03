import numpy as np

def parse_classifier_args(**kwargs):
    """
    :param kwargs:
    :return:
    """
    assert 'clf' in kwargs or 'coefficients' in kwargs

    if 'clf' in kwargs:
        clf = kwargs.get('clf')
        w = clf.coef_
        t = clf.intercept_

    elif 'coefficients' in kwargs:
        w = kwargs.get('coefficients')
        t = kwargs.get('intercept', 0.0)

    w = np.array(w).flatten()
    t = float(t)
    assert np.isfinite(w).all()
    assert np.isfinite(t)
    return w, t
