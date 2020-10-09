import numpy as np
import pandas as pd
from recourse.helper_functions import parse_classifier_args
from recourse.action_set import ActionSet
from recourse.builder import RecourseBuilder
from recourse.defaults import VALID_MIP_COST_TYPES, VALID_ENUMERATION_TYPES, DEFAULT_SOLVER

pd.set_option('display.max_columns', 10)
__all__ = ['Flipset']

class Flipset(object):
    """
    List of actions that will flip the predicted value of a classifier from x
    """
    _valid_enumeration_types = VALID_ENUMERATION_TYPES
    _valid_cost_types = VALID_MIP_COST_TYPES

    df_column_names = ['cost',
                       'size',
                       'features',
                       'feature_idx',
                       'x',
                       'x_new',
                       'score_new',
                       'yhat_new',
                       'feasible',
                       'flipped']


    def __init__(self, x, action_set, solver = DEFAULT_SOLVER, **kwargs):

        # attach action set
        assert isinstance(action_set, ActionSet)
        self.action_set = action_set
        self._n_variables = len(action_set)
        self._variable_names = action_set.name
        self._solver = solver

        # attach feature vector
        assert isinstance(x, (list, np.ndarray))
        self._x = np.array(x, dtype = np.float_).flatten()

        # attach coefficients
        self._coefs, self._intercept = parse_classifier_args(**kwargs)

        # initialize Flipset attributes
        self._builder = kwargs.get('builder')
        self._items = []
        self._df = pd.DataFrame(columns = Flipset.df_column_names, dtype = object)
        self._sort_args = {'by': ['size', 'cost', 'score_new'], 'inplace': True, 'axis': 0}


    def __len__(self):
        """
        :return: # of items in the flipset
        """
        return len(self._items)


    def __str__(self):
        return str(self._df[['cost', 'size', 'features', 'x', 'x_new']])


    def __repr__(self):
        s = ['Flipset with %d Items' % len(self),
             'x: %r' % self._x,
             'w: (%s)' % self._coefs,
             'items: %r' % self._items]
        return '\n'.join(s)


    ### properties ###
    @property
    def x(self):
        """
        :return: feature vector
        """
        return self._x


    @property
    def solutions_info(self):
        """
        Dictionary representation of Flipset
        This is a list containing mildly processed output from recourse.builder
        """
        return self._items


    @property
    def items(self):
        return list(map(lambda x: dict(zip(self.action_set._names, x['actions'].tolist())), self._items))


    @property
    def actions(self):
        return list(map(lambda x: x['actions'], self._items))


    @property
    def df(self):
        """
        Pandas DataFrame representation of Flipset
        Each row represents a different action to flip the prediction
        Rows are sorted according to the last arguments passed to the Flipset.sort()

        DESCRIPTION OF COLUMNS
        ----------------------
        cost:           cost of action
        size:           # of altered features
        features:       names of altered variables
        feature_idx:    column indices of altered features
        x:              values of x[j] for j in feature_idx
        x_new:          values of x[j] + a[j]) for j in feature_idx
        score_new:      values of score function at x_new = x + a = w0 + w.dot(x + a)
        yhat_new:       value of prediction at x_new = x + a = f(score_new)
        feasible:       True if x + a is a feasible action
        flipped:        True if the f(x) != f(x+a)
        """
        return self._df


    @property
    def yhat(self):
        return self._intercept + np.dot(self._coefs, self._x)


    def predict(self, actions = None):
        return np.sign(self.score(actions = actions))


    def score(self, actions = None):
        if actions is not None:
            return self._intercept + np.dot(self._coefs, self._x + actions)
        else:
            return self._intercept + np.dot(self._coefs, self._x)


    #### API functions ####

    def populate(self, total_items = 10, enumeration_type = 'distinct_subsets', cost_type = 'local', time_limit = None, node_limit = None, display_flag = None):

        """
        Generates a list of actions to flip the predicted value of the linear classifier from feature vector x.

        :param total_items: maximum # of actions to generate
                            set as float('inf') to enumerate all possible actions

        :param enumeration_type: enumeration algorithm to use for the flipset
                            must be a string in Flipset.valid_enumeration_types
                            - 'distinct_subsets'
                            - 'mutually_exclusive'

        :param cost_type: cost function to use for Flipset generation
                          must be a string in Flipset.valid_cost_types
                            options include:
                            - local

        :param time_limit: max # of seconds to spend before stopping the solver at each iteration

        :param node_limit: max # of branch and bound nodes to process before stopping the solver at each iteration

        :param display_flag: True to display solver progress during enumeration

        :return:
        """
        assert enumeration_type in self._valid_enumeration_types, \
            'enumeration_type must be one of %r' % self._valid_enumeration_types

        assert cost_type in self._valid_cost_types, \
            'cost_type must be one of %r' % self._valid_cost_types

        if self._builder is None:
            self._builder = RecourseBuilder(action_set = self.action_set, x = self.x, coefficients = self._coefs, intercept = self._intercept, mip_cost_type = cost_type, solver=self._solver)

        items = self._builder.populate(total_items = total_items, enumeration_type = enumeration_type, time_limit = time_limit, node_limit = node_limit, display_flag = display_flag)
        self._add(items)
        return self


    def sort(self, **kwargs):
        """
        Reorders the items in the Flipset dataframe
        Arguments used to sort are saved
        :param sort_args: list of fields to use in sort, or arguments passed to pd.DataFrame.sort_values
        :return:
        """
        if len(kwargs) == 0:
            self._df.sort_values(**self._sort_args)
            return

        if 'by' in kwargs:
            sort_names = kwargs['by']
        else:
            sort_names = list(kwargs.keys())

        assert isinstance(sort_names, list)
        assert len(sort_names) > 0
        for s in sort_names:
            assert isinstance(s, str)
            assert s in self._df.columns

        sort_args = {
            'by': sort_names,
            'inplace': kwargs.get('inplace', True),
            'axis': kwargs.get('axis', 0),
            }
        self._df.sort_values(**sort_args)
        self._sort_args = sort_args


    def view(self):
        """
        prints Flipset as Pandas dataframe
        :return:
        """
        return self._df


    def to_flat_df(self):
        """Flatten out the actionsets in the flipset to product either a latex or HTML representation."""
        self.sort()
        tex_columns = ['features', 'x', 'x_new']
        tex_df = self._df[tex_columns]
        if len(tex_df) == 0:
            return []

        # split components for each item
        tex_df = tex_df.reset_index().rename(columns = {'index': 'item_id'})
        df_list = []
        for n in tex_columns:
            tmp = tex_df.set_index(['item_id'])[n].apply(pd.Series).stack()
            tmp = tmp.reset_index().rename(columns = {'level_1': 'var_id'})
            tmp_name = tmp.columns[-1]
            tmp = tmp.rename(columns = {tmp_name: n})
            df_list.append(tmp)

        # combine into a flattened list
        flat_df = df_list[0]
        for k in range(1, len(df_list)):
            flat_df = flat_df.merge(df_list[k])

        # drop the merge index
        flat_df = flat_df.drop(columns = ['var_id'])

        # index items by item_id
        flat_df = flat_df.sort_values(by = 'item_id')
        flat_df = flat_df.rename(columns = {
            'item_id': 'item',
            'features': 'Features to Change',
            'x':'Current Value',
            'x_new': 'Required Value'
        })
        return flat_df.set_index('item')


    def to_latex(self, name_formatter = '\\textit'):
        """
        converts current Flipset to Latex table
        :param name_formatter:
        :return:
        """
        flat_df = self.to_flat_df()

        # add another column for the latex arrow symbol
        idx = flat_df.columns.tolist().index('Required Value')
        flat_df.insert(loc = idx, column = 'to', value = ['longrightarrow'] * len(flat_df))

        # name headers
        flat_df = flat_df.rename(columns = {
            'features': '\textsc{Feature Subset}',
            'Current Value': '\textsc{Current Values}',
            'Required Value': '\textsc{Required Values}'})

        # get raw tex table
        table = flat_df.to_latex(multirow = True, index = True, escape = False, na_rep = '-', column_format = 'rlccc')

        # manually wrap names with a formatter function
        if name_formatter is not None:
            for v in self._variable_names:
                table = table.replace(v, '%s{%s}' % (name_formatter, v))

        # add the backslash for the arrow
        table = table.replace('longrightarrow', '$\\longrightarrow$')

        # minor embellishments
        table = table.split('\n')
        table[2] = table[2].replace('to', '')
        table[2] = table[2].replace('{}', '')
        table.pop(3)
        table.pop(3)
        return '\n'.join(table)


    def to_html(self):
        # remove the numbering on the left if possible?
        cfpb_color = '#e2efd8'
        def _color_white_or_gray(row):
            first_item = row.name if isinstance(row.name, int) else row.name[0]
            color = 'white' if first_item % 2 == 1 else cfpb_color
            res = 'background-color: %s' % color
            return [res] * len(row)

        flat_df = self.to_flat_df()
        if len(flat_df) == 0:
            style = "text-shadow: 0px 1px 1px #4d4d4d; color: 'black'; font: 30px 'LeagueGothicRegular'; background-color:" + cfpb_color
            ## style 1
            html = (pd.DataFrame([{'outcome': 'No Recourse'}]).style
                    .set_table_styles([{"selector": "tr", "props": [('background-color', 'white')]}])
                    .apply(_color_white_or_gray, axis=1)
                    .hide_index()
                    .render()
                    )
            ## style 2
            html = '<span style="' + style + '">No Recourse</span>'
            return html

        # add another column for the latex arrow symbol
        idx = flat_df.columns.tolist().index('Required Value')

        flat_df.insert(loc = idx, column = 'to', value = ['&#8594;'] * len(flat_df))

        idx = (pd.DataFrame(flat_df.index)
               .assign(row=lambda df: df.groupby('item').cumcount().pipe(lambda s: s + 1))
               .pipe(lambda df: list(zip(df['item'], df['row'])))
               )

        idx = pd.MultiIndex.from_tuples(idx)
        flat_df.index = idx
        flat_df['Current Value'] = flat_df['Current Value'].apply(lambda x: str(int(x)) if int(x) == x else str(x))
        flat_df['Required Value'] = flat_df['Required Value'].apply(lambda x: str(int(x)) if int(x) == x else str(x))
        html = (flat_df.style
                .set_table_styles([{"selector": "tr", "props": [('background-color', 'white')]}])
                .apply(_color_white_or_gray, axis=1)
                .hide_index()
                .render()
                )
        return html


    #### item management ####
    def _add(self, items):
        """
        :param items: adds new items to flipset
        :return:
        """
        if isinstance(items, dict):
            items = [items]
        assert isinstance(items, list)
        items = list(map(lambda i: self._validate_item(i), items))
        self._items.extend(items)
        self._add_to_df(items)


    def _validate_item(self, item):
        """
        checks item to be added to the current Flipset
        :param item: raw flipset item
        :return: item in correct format
        """
        assert isinstance(item, dict)
        required_fields = ['feasible', 'actions', 'cost']
        for k in required_fields:
            assert k in item, 'item missing field %s' % k
        item['actions'] = self._validate_action(item['actions'])
        assert item['cost'] > 0.0, 'total cost must be positive'
        assert item['feasible'], 'item must be feasible'
        return item


    def _validate_action(self, a):
        """
        checks action vector to the added to the current Flipset
        :param a: action vector
        :return: a or AssertionError
        """
        a = np.array(a, dtype = np.float_).flatten()
        assert len(a) == self._n_variables, 'action vector must have %d elements' % self.n_variables
        assert np.isfinite(a).all(), 'actions must be finite'
        assert np.count_nonzero(a) >= 1, 'at least one action element must be non zero'
        assert np.not_equal(self.yhat, self.predict(a)), 'actions do not flip the prediction from %d' % self.yhat
        return a


    def _add_to_df(self, items):
        if len(items) > 0:
            row_data = list(map(lambda item: self._item_to_df_row(item), items))
            self._df = self._df.append(row_data, ignore_index = True, sort = True)[self._df.columns.tolist()]
            self.sort()


    def _item_to_df_row(self, item):
        """
        converts item to a row in the data frame
        :param item:
        :return:
        """
        x = self.x
        a = item['actions']
        h = self.predict(a)
        nnz_idx = np.flatnonzero(a)
        row = {
            'cost': float(item['cost']),
            'size': len(nnz_idx),
            'features': [self._variable_names[j] for j in nnz_idx],
            'feature_idx': nnz_idx,
            'x': x[nnz_idx],
            'x_new': x[nnz_idx] + a[nnz_idx],
            'score_new': self.score(a),
            'yhat_new': h,
            'feasible': item['feasible'],
            'flipped': np.not_equal(h, self.yhat),
            }

        return row
