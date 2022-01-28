#%%conda o pip
import os
import numpy as np
#graficadores
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#dataframe
import pandas as pd
from pathlib import Path
import datetime as dt
import math
import warnings
from collections import Counter
import openpyxl
#entrenar modelos estadistico
import scipy
import scipy.stats as ss
import scipy.cluster.hierarchy as sch
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.formula.api import logit
#solo en pip es un graficador
import plot_likert


from data_utils import identify_columns_by_type
from _private import (
    convert, remove_incomplete_samples, replace_nan_with_value
)

#%%extra functions

__all__ = [
    'associations',
    'cluster_correlations',
    'compute_associations',
    'conditional_entropy',
    'correlation_ratio',
    'cramers_v',
    'identify_nominal_columns',
    'identify_numeric_columns',
    'numerical_encoding',
    'theils_u'
]

_REPLACE = 'replace'
_DROP = 'drop'
_DROP_SAMPLES = 'drop_samples'
_DROP_FEATURES = 'drop_features'
_SKIP = 'skip'
_DEFAULT_REPLACE_VALUE = 0.0

def results_summary_to_dataframe(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    odds = np.exp(results.params)
    conf_lower = np.exp(results.conf_int(0.05)[0])
    conf_higher = np.exp(results.conf_int(0.05)[1])

    results_df = pd.DataFrame({"ODDs":odds,
                               "pvals":pvals,
                               "coeff":coeff,
                               "0.05":conf_lower,
                               "0.95":conf_higher
                                })
    #Reordering...
    results_df = results_df[["ODDs","coeff","pvals","0.05","0.95"]]
    return results_df

def is_categorical(array_like):
    return array_like.dtype.name == 'category'

def plot_heatmap(cross_table, fmt='g'):
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(cross_table,
                annot=True,
                annot_kws={"size": 8},
                fmt=fmt,
                cmap='rocket_r',
                linewidths=.25,
                ax=ax)
    plt.show()


def _inf_nan_str(x):
    if np.isnan(x):
        return 'NaN'
    elif abs(x) == np.inf:
        return 'inf'
    else:
        return ''


def conditional_entropy(x,
                        y,
                        nan_strategy=_REPLACE,
                        nan_replace_value=_DEFAULT_REPLACE_VALUE,
                        log_base: float = math.e):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    Parameters:
    -----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop' to remove samples
        with missing values, or 'replace' to replace all missing values with
        the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'.
    log_base: float, default = e
        specifying base for calculating entropy. Default is base e.
    Returns:
    --------
    float
    """
    if nan_strategy == _REPLACE:
        x, y = replace_nan_with_value(x, y, nan_replace_value)
    elif nan_strategy == _DROP:
        x, y = remove_incomplete_samples(x, y)
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy, log_base)
    return entropy


def cramers_v(x,
              y,
              bias_correction=True,
              nan_strategy=_REPLACE,
              nan_replace_value=_DEFAULT_REPLACE_VALUE):
    """
    Calculates Cramer's V statistic for categorical-categorical association.
    This is a symmetric coefficient: V(x,y) = V(y,x)
    Original function taken from: https://stackoverflow.com/a/46498792/5863503
    Wikipedia: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Parameters:
    -----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
        Use bias correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328.
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop' to remove samples
        with missing values, or 'replace' to replace all missing values with
        the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'.
    Returns:
    --------
    float in the range of [0,1]
    """
    if nan_strategy == _REPLACE:
        x, y = replace_nan_with_value(x, y, nan_replace_value)
    elif nan_strategy == _DROP:
        x, y = remove_incomplete_samples(x, y)
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    if bias_correction:
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        if min((kcorr - 1), (rcorr - 1)) == 0:
            warnings.warn(
                "Unable to calculate Cramer's V using bias correction. Consider using bias_correction=False",
                RuntimeWarning)
            return np.nan
        else:
            return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    else:
        return np.sqrt(phi2 / min(k - 1, r - 1))


def theils_u(x,
             y,
             nan_strategy=_REPLACE,
             nan_replace_value=_DEFAULT_REPLACE_VALUE):
    """
    Calculates Theil's U statistic (Uncertainty coefficient) for categorical-
    categorical association. This is the uncertainty of x given y: value is
    on the range of [0,1] - where 0 means y provides no information about
    x, and 1 means y provides full information about x.
    This is an asymmetric coefficient: U(x,y) != U(y,x)
    Wikipedia: https://en.wikipedia.org/wiki/Uncertainty_coefficient
    Parameters:
    -----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop' to remove samples
        with missing values, or 'replace' to replace all missing values with
        the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'.
    Returns:
    --------
    float in the range of [0,1]
    """
    if nan_strategy == _REPLACE:
        x, y = replace_nan_with_value(x, y, nan_replace_value)
    elif nan_strategy == _DROP:
        x, y = remove_incomplete_samples(x, y)
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def correlation_ratio(categories,
                      measurements,
                      nan_strategy=_REPLACE,
                      nan_replace_value=_DEFAULT_REPLACE_VALUE):
    """
    Calculates the Correlation Ratio (sometimes marked by the greek letter Eta)
    for categorical-continuous association.
    Answers the question - given a continuous value of a measurement, is it
    possible to know which category is it associated with?
    Value is in the range [0,1], where 0 means a category cannot be determined
    by a continuous measurement, and 1 means a category can be determined with
    absolute certainty.
    Wikipedia: https://en.wikipedia.org/wiki/Correlation_ratio
    Parameters:
    -----------
    categories : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    measurements : list / NumPy ndarray / Pandas Series
        A sequence of continuous measurements
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop' to remove samples
        with missing values, or 'replace' to replace all missing values with
        the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'.
    Returns:
    --------
    float in the range of [0,1]
    """
    if nan_strategy == _REPLACE:
        categories, measurements = replace_nan_with_value(
            categories, measurements, nan_replace_value)
    elif nan_strategy == _DROP:
        categories, measurements = remove_incomplete_samples(
            categories, measurements)
    categories = convert(categories, 'array')
    measurements = convert(measurements, 'array')
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg),
                                      2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def identify_nominal_columns(dataset):
    """
    Given a dataset, identify categorical columns.
    Parameters:
    -----------
    dataset : NumPy ndarray / Pandas DataFrame
    Returns:
    --------
    A list of categorical columns names
    Example:
    --------
    >> df = pd.DataFrame({'col1': ['a', 'b', 'c', 'a'], 'col2': [3, 4, 2, 1]})
    >> identify_nominal_columns(df)
    ['col1']
    """
    return identify_columns_by_type(dataset, include=['object', 'category'])


def identify_numeric_columns(dataset):
    """
    Given a dataset, identify numeric columns.
    Parameters:
    -----------
    dataset : NumPy ndarray / Pandas DataFrame
    Returns:
    --------
    A list of numerical columns names
    Example:
    --------
    >> df = pd.DataFrame({'col1': ['a', 'b', 'c', 'a'], 'col2': [3, 4, 2, 1], 'col3': [1., 2., 3., 4.]})
    >> identify_numeric_columns(df)
    ['col2', 'col3']
    """
    return identify_columns_by_type(dataset, include=['int64', 'float64'])


def _comp_assoc(dataset, nominal_columns, mark_columns, theil_u, clustering,
                bias_correction, nan_strategy, nan_replace_value):
    """
    This is a helper function for compute_associations and associations
    """
    dataset = convert(dataset, 'dataframe')
    if nan_strategy == _REPLACE:
        dataset.fillna(nan_replace_value, inplace=True)
    elif nan_strategy == _DROP_SAMPLES:
        dataset.dropna(axis=0, inplace=True)
    elif nan_strategy == _DROP_FEATURES:
        dataset.dropna(axis=1, inplace=True)
    columns = dataset.columns
    if nominal_columns is None:
        nominal_columns = list()
    elif nominal_columns == 'all':
        nominal_columns = columns
    elif nominal_columns == 'auto':
        nominal_columns = identify_nominal_columns(dataset)

    corr = pd.DataFrame(index=columns, columns=columns)
    single_value_columns = []
    inf_nan = pd.DataFrame(data=np.zeros_like(corr),
                           columns=columns,
                           index=columns)
    for c in columns:
        if dataset[c].unique().size == 1:
            single_value_columns.append(c)
    for i in range(0, len(columns)):
        if columns[i] in single_value_columns:
            corr.loc[:, columns[i]] = 0.0
            corr.loc[columns[i], :] = 0.0
            continue
        for j in range(i, len(columns)):
            if columns[j] in single_value_columns:
                continue
            elif i == j:
                corr.loc[columns[i], columns[j]] = 1.0
            else:
                if columns[i] in nominal_columns:
                    if columns[j] in nominal_columns:
                        if theil_u:
                            ji = theils_u(
                                dataset[columns[i]],
                                dataset[columns[j]],
                                nan_strategy=_SKIP)
                            ij = theils_u(
                                dataset[columns[j]],
                                dataset[columns[i]],
                                nan_strategy=_SKIP)
                        else:
                            cell = cramers_v(dataset[columns[i]],
                                             dataset[columns[j]],
                                             bias_correction=bias_correction,
                                             nan_strategy=_SKIP)
                            ij = cell
                            ji = cell
                    else:
                        cell = correlation_ratio(dataset[columns[i]],
                                                 dataset[columns[j]],
                                                 nan_strategy=_SKIP)
                        ij = cell
                        ji = cell
                else:
                    if columns[j] in nominal_columns:
                        cell = correlation_ratio(dataset[columns[j]],
                                                 dataset[columns[i]],
                                                 nan_strategy=_SKIP)
                        ij = cell
                        ji = cell
                    else:
                        cell, _ = ss.pearsonr(dataset[columns[i]],
                                              dataset[columns[j]])
                        ij = cell
                        ji = cell
                corr.loc[columns[i], columns[j]] = ij if not np.isnan(ij) and abs(ij) < np.inf else 0.0
                corr.loc[columns[j], columns[i]] = ji if not np.isnan(ji) and abs(ji) < np.inf else 0.0
                inf_nan.loc[columns[i], columns[j]] = _inf_nan_str(ij)
                inf_nan.loc[columns[j], columns[i]] = _inf_nan_str(ji)
    corr.fillna(value=np.nan, inplace=True)
    if mark_columns:
        marked_columns = [
            '{} (nom)'.format(col)
            if col in nominal_columns else '{} (con)'.format(col)
            for col in columns
        ]
        corr.columns = marked_columns
        corr.index = marked_columns
        inf_nan.columns = marked_columns
        inf_nan.index = marked_columns
    if clustering:
        corr, _ = cluster_correlations(corr)
        columns = corr.columns
    return corr, columns, nominal_columns, inf_nan, single_value_columns


def compute_associations(dataset,
                         nominal_columns='auto',
                         mark_columns=False,
                         theil_u=False,
                         clustering=False,
                         bias_correction=True,
                         nan_strategy=_REPLACE,
                         nan_replace_value=_DEFAULT_REPLACE_VALUE,
                         ):
    """
    Calculate the correlation/strength-of-association of features in data-set
    with both categorical and continuous features using:
     * Pearson's R for continuous-continuous cases
     * Correlation Ratio for categorical-continuous cases
     * Cramer's V or Theil's U for categorical-categorical cases
    It is equivalent to run `associations(data, plot=False, ...)['corr']`, only
    it skips entirely on the drawing phase of the heat-map (See
    https://github.com/shakedzy/dython/issues/49)
    Parameters:
    -----------
    dataset : NumPy ndarray / Pandas DataFrame
        The data-set for which the features' correlation is computed
    nominal_columns : string / list / NumPy ndarray
        Names of columns of the data-set which hold categorical values. Can
        also be the string 'all' to state that all columns are categorical,
        'auto' (default) to try to identify nominal columns, or None to state
        none are categorical
    mark_columns : Boolean, default = False
        if True, output's columns' names will have a suffix of '(nom)' or
        '(con)' based on there type (eda_tools or continuous), as provided
        by nominal_columns
    theil_u : Boolean, default = False
        In the case of categorical-categorical feaures, use Theil's U instead
        of Cramer's V
    clustering : Boolean, default = False
        If True, hierarchical clustering is applied in order to sort
        features into meaningful groups
    bias_correction : Boolean, default = True
        Use bias correction for Cramer's V from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328.
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop_samples' to remove
        samples with missing values, 'drop_features' to remove features
        (columns) with missing values, or 'replace' to replace all missing
        values with the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'
    Returns:
    --------
    A DataFrame of the correlation/strength-of-association between all features
    """
    corr, _, _, _, _ = _comp_assoc(dataset, nominal_columns, mark_columns, theil_u, clustering,
                                   bias_correction, nan_strategy, nan_replace_value)
    return corr


def associations(dataset,
                 nominal_columns='auto',
                 mark_columns=False,
                 theil_u=False,
                 plot=True,
                 clustering=False,
                 bias_correction=True,
                 nan_strategy=_REPLACE,
                 nan_replace_value=_DEFAULT_REPLACE_VALUE,
                 ax=None,
                 figsize=None,
                 annot=True,
                 fmt='.2f',
                 cmap=None,
                 sv_color='silver',
                 cbar=True,
                 vmax=1.0,
                 vmin=None,
                 title=None,
                 filename=None
                 ):
    """
    Calculate the correlation/strength-of-association of features in data-set
    with both categorical and continuous features using:
     * Pearson's R for continuous-continuous cases
     * Correlation Ratio for categorical-continuous cases
     * Cramer's V or Theil's U for categorical-categorical cases
    Parameters:
    -----------
    dataset : NumPy ndarray / Pandas DataFrame
        The data-set for which the features' correlation is computed
    nominal_columns : string / list / NumPy ndarray
        Names of columns of the data-set which hold categorical values. Can
        also be the string 'all' to state that all columns are categorical,
        'auto' (default) to try to identify nominal columns, or None to state
        none are categorical
    mark_columns : Boolean, default = False
        if True, output's columns' names will have a suffix of '(nom)' or
        '(con)' based on their type (nominal or continuous), as provided
        by nominal_columns
    theil_u : Boolean, default = False
        In the case of categorical-categorical feaures, use Theil's U instead
        of Cramer's V. If selected, heat-map rows are the provided information
        (U = U(row|col))
    plot : Boolean, default = True
        Plot a heat-map of the correlation matrix
    clustering : Boolean, default = False
        If True, hierarchical clustering is applied in order to sort
        features into meaningful groups
    bias_correction : Boolean, default = True
        Use bias correction for Cramer's V from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328.
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop_samples' to remove
        samples with missing values, 'drop_features' to remove features
        (columns) with missing values, or 'replace' to replace all missing
        values with the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'
    ax : matplotlib ax, default = None
        Matplotlib Axis on which the heat-map will be plotted
    figsize : (int,int) or None, default = None
        A Matplotlib figure-size tuple. If `None`, falls back to Matplotlib's
        default. Only used if `ax=None`.
    annot : Boolean, default = True
        Plot number annotations on the heat-map
    fmt : string, default = '.2f'
        String formatting of annotations
    cmap : Matplotlib colormap or None, default = None
        A colormap to be used for the heat-map. If None, falls back to Seaborn's
        heat-map default
    sv_color : string, default = 'silver'
        A Matplotlib color. The color to be used when displaying single-value
        features over the heat-map
    cbar: Boolean, default = True
        Display heat-map's color-bar
    vmax: float, default = 1.0
        Set heat-map vmax option
    vmin: float or None, default = None
        Set heat-map vmin option. If set to None, vmin will be chosen automatically
        between 0 and -1, depending on the types of associations used (-1 if Pearson's R
        is used, 0 otherwise)
    title : string or None, default = None
        Plotted graph title
    filename : string or None, default = None
        If not None, plot will be saved to the given file name
    Returns:
    --------
    A dictionary with the following keys:
    - `corr`: A DataFrame of the correlation/strength-of-association between
    all features
    - `ax`: A Matplotlib `Axe`
    Example:
    --------
    See examples under `dython.examples`
    """
    corr, columns, nominal_columns, inf_nan, single_value_columns = _comp_assoc(dataset, nominal_columns, mark_columns,
                                                                                theil_u, clustering, bias_correction,
                                                                                nan_strategy, nan_replace_value)
    if ax is None:
        plt.figure(figsize=figsize)
    if inf_nan.any(axis=None):
        inf_nan_mask = np.vectorize(lambda x: not bool(x))(inf_nan.values)
        ax = sns.heatmap(inf_nan_mask,
                         cmap=['white'],
                         annot=inf_nan if annot else None,
                         fmt='',
                         center=0,
                         square=True,
                         ax=ax,
                         mask=inf_nan_mask,
                         cbar=False)
    else:
        inf_nan_mask = np.ones_like(corr)
    if len(single_value_columns) > 0:
        sv = pd.DataFrame(data=np.zeros_like(corr),
                          columns=columns,
                          index=columns)
        for c in single_value_columns:
            sv.loc[:, c] = ' '
            sv.loc[c, :] = ' '
            sv.loc[c, c] = 'SV'
        sv_mask = np.vectorize(lambda x: not bool(x))(sv.values)
        ax = sns.heatmap(sv_mask,
                         cmap=[sv_color],
                         annot=sv if annot else None,
                         fmt='',
                         center=0,
                         square=True,
                         ax=ax,
                         mask=sv_mask,
                         cbar=False)
    else:
        sv_mask = np.ones_like(corr)
    mask = np.vectorize(lambda x: not bool(x))(inf_nan_mask) + np.vectorize(lambda x: not bool(x))(sv_mask)
    vmin = vmin or (-1.0 if len(columns) - len(nominal_columns) >= 2 else 0.0)
    ax = sns.heatmap(corr,
                     cmap=cmap,
                     annot=annot,
                     fmt=fmt,
                     center=0,
                     vmax=vmax,
                     vmin=vmin,
                     square=True,
                     mask=mask,
                     ax=ax,
                     cbar=cbar)
    plt.title(title)
    if filename:
        plt.savefig(filename)
    if plot:
        plt.show()
    return {'corr': corr,
            'ax': ax}


def numerical_encoding(dataset,
                       nominal_columns='auto',
                       drop_single_label=False,
                       drop_fact_dict=True,
                       nan_strategy=_REPLACE,
                       nan_replace_value=_DEFAULT_REPLACE_VALUE):
    """
    Encoding a data-set with mixed data (numerical and categorical) to a
    numerical-only data-set using the following logic:
    * categorical with only a single value will be marked as zero (or dropped,
        if requested)
    * categorical with two values will be replaced with the result of Pandas
        `factorize`
    * categorical with more than two values will be replaced with the result
        of Pandas `get_dummies`
    * numerical columns will not be modified
    Parameters:
    -----------
    dataset : NumPy ndarray / Pandas DataFrame
        The data-set to encode
    nominal_columns : sequence / string. default = 'all'
        A sequence of the nominal (categorical) columns in the dataset. If
        string, must be 'all' to state that all columns are nominal. If None,
        nothing happens. If 'auto', categorical columns will be identified
        based on dtype.
    drop_single_label : Boolean, default = False
        If True, nominal columns with a only a single value will be dropped.
    drop_fact_dict : Boolean, default = True
        If True, the return value will be the encoded DataFrame alone. If
        False, it will be a tuple of the DataFrame and the dictionary of the
        binary factorization (originating from pd.factorize)
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop_samples' to remove
        samples with missing values, 'drop_features' to remove features
        (columns) with missing values, or 'replace' to replace all missing
        values with the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when nan
        _strategy is set to 'replace'
    Returns:
    --------
    DataFrame or (DataFrame, dict). If `drop_fact_dict` is True,
    returns the encoded DataFrame.
    else, returns a tuple of the encoded DataFrame and dictionary, where each
    key is a two-value column, and the value is the original labels, as
    supplied by Pandas `factorize`. Will be empty if no two-value columns are
    present in the data-set
    """
    dataset = convert(dataset, 'dataframe')
    if nan_strategy == _REPLACE:
        dataset.fillna(nan_replace_value, inplace=True)
    elif nan_strategy == _DROP_SAMPLES:
        dataset.dropna(axis=0, inplace=True)
    elif nan_strategy == _DROP_FEATURES:
        dataset.dropna(axis=1, inplace=True)
    if nominal_columns is None:
        return dataset
    elif nominal_columns == 'all':
        nominal_columns = dataset.columns
    elif nominal_columns == 'auto':
        nominal_columns = identify_nominal_columns(dataset)
    converted_dataset = pd.DataFrame()
    binary_columns_dict = dict()
    for col in dataset.columns:
        if col not in nominal_columns:
            converted_dataset.loc[:, col] = dataset[col]
        else:
            unique_values = pd.unique(dataset[col])
            if len(unique_values) == 1 and not drop_single_label:
                converted_dataset.loc[:, col] = 0
            elif len(unique_values) == 2:
                converted_dataset.loc[:, col], binary_columns_dict[
                    col] = pd.factorize(dataset[col])
            else:
                dummies = pd.get_dummies(dataset[col], prefix=col)
                converted_dataset = pd.concat([converted_dataset, dummies],
                                              axis=1)
    if drop_fact_dict:
        return converted_dataset
    else:
        return converted_dataset, binary_columns_dict


def cluster_correlations(corr_mat, indices=None):
    """
    Apply agglomerative clustering in order to sort
    a correlation matrix.
    Based on https://github.com/TheLoneNut/CorrelationMatrixClustering/blob/master/CorrelationMatrixClustering.ipynb
    Parameters:
    -----------
    - corr_mat : a square correlation matrix (pandas DataFrame)
    - indices : cluster labels [None]; if not provided we'll do
        an aglomerative clustering to get cluster labels.
    Returns:
    --------
    - corr : a sorted correlation matrix
    - indices : cluster indexes based on the original dataset
    Example:
    --------
    >> assoc = associations(
        customers,
        plot=False
    )
    >> correlations = assoc['corr']
    >> correlations, _ = cluster_correlations(correlations)
    """
    if indices is None:
        X = corr_mat.values
        d = sch.distance.pdist(X)
        L = sch.linkage(d, method='complete')
        indices = sch.fcluster(L, 0.5 * d.max(), 'distance')
    columns = [corr_mat.columns.tolist()[i]
               for i in list((np.argsort(indices)))]
    corr_mat = corr_mat.reindex(columns=columns).reindex(index=columns)
    return corr_mat, indices


#%% open dataset 
# Set working directory file cell
cwd = Path.cwd()
file_I = Path("Data/Data_I.csv")
file_collabels = Path("Data/Colnames.csv")
file_defunciones = Path("Data/Defunciones.csv")
file_confirmados = Path("Data/Confirmados.csv")
#read files
data_I_HGM = pd.read_csv(cwd / file_I)
data_collabels = pd.read_csv(cwd / file_collabels)
data_defunciones = pd.read_csv(cwd / file_defunciones)
data_confirmados = pd.read_csv(cwd / file_confirmados)

#%% filter rows with more than one answer in hesitanci as 0
columnset = np.r_[0,40:47]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_pivot = dataset_test.melt('R0').pivot_table(index='R0', columns='value', aggfunc='size', fill_value=0) 
dataset_pivot = dataset_pivot.reset_index()
truelist = dataset_pivot[0] <= 1
data_I_HGM = data_I_HGM.loc[truelist,:] 

#%% Change date columns to the proper type
data_I_HGM['R2'] = pd.to_datetime(data_I_HGM['R2'], format='%d/%m/%Y')
data_I_HGM['R12'] = pd.to_datetime(data_I_HGM['R12'], format='%d/%m/%Y')
data_defunciones['Date'] = pd.to_datetime(data_defunciones['Date'], format='%d/%m/%Y')
data_confirmados['Date'] = pd.to_datetime(data_confirmados['Date'], format='%d/%m/%Y')

#%%filter by time
data_I_HGM = data_I_HGM[data_I_HGM.R2.between('2021-03', '2021-10')]

#%% Add confirmados y defunciones ass colums in data_I_HGM
data_I_HGM['R168'] = np.NAN
data_collabels.loc[len(data_collabels.index)] = ["R168","DefuncionesNacionales"]
data_I_HGM['R169'] = np.NAN
data_collabels.loc[len(data_collabels.index)] = ["R169","ConfirmadosNacionales"]
for index, row in data_I_HGM.iterrows():
    data_I_HGM.loc[index,'R168'] = data_defunciones.loc[data_defunciones['Date'] == row['R2'],'Nacional'].values
    data_I_HGM.loc[index,'R169'] = data_confirmados.loc[data_confirmados['Date'] == row['R2'],'Nacional'].values

#%% create groups of data 
# age groups
CutAge = pd.cut(data_I_HGM.R13, bins=[18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, np.inf])
data_I_HGM['R170']=CutAge
data_collabels.loc[len(data_collabels.index)] = ["R170","CutAge"]
# date groups
time_cats = 12
CutTime =  pd.cut(data_I_HGM.R2,time_cats)
data_I_HGM['R171']=CutTime
data_collabels.loc[len(data_collabels.index)] = ["R171","CutDate"]
temp=data_I_HGM.R171.astype(str).str.extract(', (.+?)]').astype('datetime64[ns]')
data_I_HGM['R171c']=pd.to_datetime(temp[0]).dt.date
#education groups
CutEdu = pd.cut(data_I_HGM.R16, bins=[0, 6, 9, 12, 16, 22, np.inf])
data_I_HGM['R172']=CutEdu
data_collabels.loc[len(data_collabels.index)] = ["R172","CutEdu"]
#add media of incertidumbre
data_I_HGM['R173']=data_I_HGM.loc[:,['R70 - R77','R71 - R78', 'R72 - R79', 'R73 - R80', 'R74 - R83','R75 - R82', 'R76 - R81']].mean(axis=1)
data_collabels.loc[len(data_collabels.index)] = ["R173","HesitancyMean"]
#dichotomic media
data_I_HGM['R174'] = 0
data_I_HGM.loc[data_I_HGM['R173'] >= data_I_HGM.R173.quantile(0.75),'R174'] = 1

#%%dummies to categorical error no funciona
data_I_HGM['R175']=data_I_HGM.iloc[:,10:17].idxmax(axis=1)
data_collabels.loc[len(data_collabels.index)] = ["R175","SeguridadSocial"]
data_I_HGM['R176']=data_I_HGM.iloc[:,17:30].idxmax(axis=1)
data_collabels.loc[len(data_collabels.index)] = ["R176","EnfermedadReuma"]
data_I_HGM['R177']=data_I_HGM.iloc[:,73:84].idxmax(axis=1)
data_collabels.loc[len(data_collabels.index)] = ["R177","Cormo"]


#%%clean columns
data_I_HGM['R70 - R77c'] = data_I_HGM['R70 - R77'].map({0:"No sé", 5:"Definitivamente no", 4:"Probablemente no", 3:"Tal vez si o tal vez no", 2:"Probablemente", 1:"Seguramente"})
data_I_HGM['R71 - R78c'] = data_I_HGM['R71 - R78'].map({0:"No sé", 5:"Me negaré a aplicármela", 4:"Pospondré (retrasaré) su aplicación", 3:"No estoy seguro(a) de lo que haré", 2:"Me la aplicaría cuando me la ofrezcan", 1:"Me gustaría aplicármela lo antes posible"})
data_I_HGM['R72 - R79c'] = data_I_HGM['R72 - R79'].map({0:"No sé", 5:"En contra de la vacuna", 4:"Bastante preocupado(a)", 3:"Neutra", 2:"Bastante positiva", 1:"Muy entusiasta"})
data_I_HGM['R73 - R80c'] = data_I_HGM['R73 - R80'].map({0:"No sé", 5:"Nunca me la aplicaría", 4:"Evitaría aplicármela durante el mayor tiempo posible", 3:"Retrasaría su aplicación", 2:"Me la aplicaría cuando tenga tiempo", 1:"Me la aplicaría tan pronto como pueda"})
data_I_HGM['R74 - R83c'] = data_I_HGM['R74 - R83'].map({0:"No sé", 5:"Les sugeriría que no se vacunen", 4:"Les pediría que retrasen la vacuna", 3:"No les diría nada al respecto", 2:"Los animaría", 1:"Los animaría con entusiasmo"})
data_I_HGM['R75 - R82c'] = data_I_HGM['R75 - R82'].map({0:"No sé", 5:"En contra de la vacuna para la COVID-19", 4:"No dispuesto(a) a recibir la vacuna para la COVID-19", 3:"No preocupado(a) por recibir la vacuna para la COVID-19", 2:"Dispuesto(a) a recibir la vacuna para la COVID-19", 1:"Entusiasmado(a), por recibir la vacuna para la COVID-19"})
data_I_HGM['R76 - R81c'] = data_I_HGM['R76 - R81'].map({0:"No sé", 5:"Realmente no es importante", 4:"No es importante", 3:"Ni importante ni no importante", 2:"Importante", 1:"Realmente importante"})
#error no funciona
data_I_HGM['R175'] = data_I_HGM['R175'].map({'R20':'IMSS', 'R19':'Ninguna', 'R25':'Otro', 'R21':'ISSSTE', 'R24':'INSABI', 'R22':'Otro'})
data_I_HGM['R176'] = data_I_HGM['R176'].map({'R31':'Gota', 'R27':'ArtritisReumatoide', 'R38':'Otro', 'R30':'Esclerodermia', 'R33':'MiopatíaInflamatoria', 'R36':'Otro', 'R26':'Lupus', 'R32':'Sjögren', 'R28':'VasculitisANCA','R29':'EspondilitisAnquilosante', 'R34':'AAntifosfolípidos', 'R37':'Osteoartrosis', 'R35':'ArtrititsIdiopáticaJ'})
data_I_HGM['R177'] = data_I_HGM['R177'].map({'R157':'EnfermedadPulmonar', 'R161':'HipertensiónArterialSistémica', 'R165':'Otro', 'R164':'Depresión', 'R159':'OtraCardiovascular', 'R162':'HipertensiónArterialSistémica', 'R167':'OtrasGástricas', 'R160':'VascularCerebral','R163':'Otro', 'R166':'Ulcera', 'R158':'Otro'})

#%%likert scales plot
columnset = np.r_[40:47]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test = dataset_test.replace({0:'None',1: 'Strongly agree', 2:'Agree', 3: 'Neither agree nor disagree', 4:'Disagree', 5:'Strongly disagree'})
another_scale = \
    ['Strongly disagree',
    'Disagree',
    'Neither agree nor disagree',
    'Agree',
    'Strongly agree',
    'None']
colours5 = ['white','#f9665e','#fec9c9','#949494','#95b4cc','#799fcb'] 
plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours5) #another_scale plot_likert.scales.agree
plt.close()
image_format = 'svg'
file_O = Path("Data/outimg/likert_hesitancy.svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200)
plt.close()

columnset = np.r_[40:47,134]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test = dataset_test[dataset_test['R173'] >= dataset_test.R173.quantile(0.75)]
dataset_test.drop('R173', axis=1, inplace=True)
dataset_test = dataset_test.replace({0:'None',1: 'Strongly agree', 2:'Agree', 3: 'Neither agree nor disagree', 4:'Disagree', 5:'Strongly disagree'})
another_scale = \
    ['Strongly disagree',
    'Disagree',
    'Neither agree nor disagree',
    'Agree',
    'Strongly agree',
    'None']
plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours5) #another_scale plot_likert.scales.agree
image_format = 'svg'
file_O = Path("Data/outimg/likert_hesitancy_75per.svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200)
plt.close()

columnset = np.r_[40:47]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test=dataset_test[(dataset_test != 0).all(1)]
dataset_test=dataset_test[(dataset_test != 3).all(1)]
dataset_test = dataset_test.replace({1: 'Strongly agree', 2:'Agree', 4:'Disagree', 5:'Strongly disagree'})
another_scale = \
    ['Strongly disagree',
    'Disagree',
    'Agree',
    'Strongly agree']
colours4 = ['white','#f9665e','#fec9c9','#95b4cc','#799fcb'] 
plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours4) #another_scale plot_likert.scales.agree
image_format = 'svg'
file_O = Path("Data/outimg/likert_hesitancy4v.svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200)
plt.close()

columnset = np.r_[40:47,134]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test = dataset_test[dataset_test['R173'] >= dataset_test.R173.quantile(0.75)]
dataset_test.drop('R173', axis=1, inplace=True)
dataset_test=dataset_test[(dataset_test != 0).all(1)]
dataset_test=dataset_test[(dataset_test != 3).all(1)]
dataset_test = dataset_test.replace({1: 'Strongly agree', 2:'Agree', 4:'Disagree', 5:'Strongly disagree'})
another_scale = \
    ['Strongly disagree',
    'Disagree',
    'Agree',
    'Strongly agree']
plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours4) #another_scale plot_likert.scales.agree
image_format = 'svg'
file_O = Path("Data/outimg/likert_hesitancy_75per4v.svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200)
plt.close()

#%%factors
columnset = np.r_[47:65,66,69]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test = dataset_test.replace({0:'None',1: 'Strongly agree', 2:'Agree', 3: 'Neither agree nor disagree', 4:'Disagree', 5:'Strongly disagree'})
another_scale = \
    ['Strongly disagree',
    'Disagree',
    'Neither agree nor disagree',
    'Agree',
    'Strongly agree',
    'None']
colours5 = ['white','#f9665e','#fec9c9','#949494','#95b4cc','#799fcb'] 
plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours5) #another_scale plot_likert.scales.agree
image_format = 'svg'
file_O = Path("Data/outimg/likert_factors.svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200)

#%%final
columnset = np.r_[1,4:7,133,136,8,73:84,17:30,31:34,35,47:49,67:70,139:146]#,138:146]#17:89]#   ,]
dataset_test= data_I_HGM.iloc[:,columnset]
#dataset_test.columns.tolist
dataset_test.rename(columns={"R1":"Hospital", "R13": "Edad", "R14":"Sexo","R15":"Estado civil", "R172":"Educacion", "R17": "Ocupación", "R175":"Seguridad Social", "R176":"Diagnóstico", "R157":"Enfermedad pulmonar","R158":"Infarto agudo de miocardio","R159":"Otra enfermedad cardiovascular","R160":"Enfermedad vascular cerebral","R161":"Hipertensión arterial sistémica","R162":"Diabetes mellitus","R163":"Fractura de cadera/columna o pierna","R164":"Depresión","R165":"Cancer","R166":"Ulcera gastrointestinal","R167":"Otras enfermedades gástrica","R168":"Otro", "R60":"¿Usted requirió hospitalización?","R62":"¿Ya fue vacunado?","R63":"¿Ha tenido la oportunidad de vacunarse?","R65":"¿Cuenta con el esquema completo?","R70 - R77c":"¿Aceptaría alguna de las vacunas si se le ofreciera?","R71 - R78c":"Existen varias vacunas. Usted considera que","R72 - R79c":"Mi actitud con respecto a recibir la vacuna","R73 - R80c":"Si ya estuviera disponible, ¿qué haría?","R74 - R83c":"Si mi familia o amigos estuvieran pensando en vacunarse","R75 - R82c":"Con respecto a recibir la vacuna, yo me describiría como","R76 - R81c":"Recibir una vacuna es","R84":"¿Cree que se infectará?","R85":"Probabilidad que me funcione","R105":"Vacunado influenza","R148":"¿Cómo se siente en este momento?","R149":"Sistema inmune","R26":"Lupus eritematoso generalizado","R27":"Artritis reumatoide","R28":"Vasculitis ANCA","R29":"Espondilitis anquilosante","R30":"Esclerodermia","R31":"Gota","R32":"Síndrome de Sjögren","R33":"Miopatía inflamatoria","R34":"Síndrome de anticuerpos antifosfolípidos","R35":"Artritits idiopática juvenil","R36":"Enfermedad mixta del tejido conectivo","R37":"Osteoartrosis","R38":"Otro"}, inplace = True)
dataset_test.columns.tolist

dataset_test["Edad"].mean()
dataset_test["Edad"].std()
data_I_HGM["R16"].mean()
data_I_HGM["R16"].std()

dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
        dataset_test.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
dataset_test.loc[:, dataset_test.isin([0,1]).all()] = dataset_test.loc[:, dataset_test.isin([0,1]).all()].apply(lambda x: x.astype('category'))
dataset_test.info()
ColNames = dataset_test.columns
ColGrouped = ColNames[0]
ColNames = np.delete(ColNames, [np.r_[0]], 0)
ColTitles = {'CHI', 'HGM', 'MTY', 'NUT'}
CQ  = pd.DataFrame(columns = ['Variable', 'CHI', 'HGM', 'MTY', 'NUT','p'])
temp_totales = dataset_test[ColGrouped].value_counts()
for coltitle in ColTitles:
    CQ.loc['Totales',coltitle] = temp_totales[coltitle]
for cols in ColNames:
    title_data = [cols]
    title = pd.DataFrame(title_data, index=[0])
    if(is_categorical(dataset_test[cols])):
        ctable = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped])
        ctablen = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped], normalize='columns')*100.00
        ctablen = ctablen.applymap(lambda x: " ({:.2f})".format(x))
        stat, pvalue, dof, expected = ss.chi2_contingency(ctable)
        ctable = ctable.astype(str)
        ctable = ctable + ctablen
        if((ctable.index == 0).any()):
            ctable = ctable.drop(0)
            ctable.rename({1:"Sí"}, inplace = True) 
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue) 
        #ctable.loc[ctable.index[0], 'Valor estadistica'] ="{:.2f}".format(stat)
    else:
        dataset_numeric = pd.concat([dataset_test[cols], dataset_test[ColGrouped]], axis=1)
        #correlation, pvalue = ss.pointbiserialr(dataset_test[ColGrouped], dataset_test[cols])
        x = []
        for coltitle in ColTitles:
            x.append(dataset_numeric.loc[dataset_numeric[ColGrouped] == coltitle,cols])
        correlation, pvalue = ss.kruskal(*x)
        ctable = dataset_numeric.groupby(ColGrouped).mean().T
        ctable = ctable.applymap(lambda x: "{:.2f}".format(x))
        ctabled = dataset_numeric.groupby(ColGrouped).std().T
        ctabled = ctabled.applymap(lambda x: " ({:.2f})".format(x))
        ctable = ctable + ctabled
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue)  
    title.rename(columns={0:"Variable"}, inplace = True)
    ctable = title.append(ctable)
    CQ  =  CQ.append(ctable)
CQ = CQ.replace(np.nan,'',regex = True)
file_O = Path("Data/out/crosstab.xlsx")
file_save = cwd / file_O
CQ.to_excel(file_save) 

columnset = np.r_[32,1,4:7,133,136,8,137:139,31:32,33:34,35,47:49,67:70,139:146]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test['R62'] = dataset_test['R62'].map({1:"Sí", 0:"No"})
dataset_test.rename(columns={"R1":"Hospital", "R13": "Edad", "R14":"Sexo","R15":"Estado civil", "R172":"Educacion", "R17": "Ocupación", "R175":"Seguridad Social", "R176":"Diagnóstico", "R177":"Comorbilidad", "R60":"¿Usted requirió hospitalización?","R62":"¿Ya fue vacunado?","R63":"¿Ha tenido la oportunidad de vacunarse?","R65":"¿Cuenta con el esquema completo?","R70 - R77c":"¿Aceptaría alguna de las vacunas si se le ofreciera?","R71 - R78c":"Existen varias vacunas. Usted considera que","R72 - R79c":"Mi actitud con respecto a recibir la vacuna","R73 - R80c":"Si ya estuviera disponible, ¿qué haría?","R74 - R83c":"Si mi familia o amigos estuvieran pensando en vacunarse","R75 - R82c":"Con respecto a recibir la vacuna, yo me describiría como","R76 - R81c":"Recibir una vacuna es","R84":"¿Cree que se infectará?","R85":"Probabilidad que me funcione","R105":"Vacunado influenza","R148":"¿Cómo se siente en este momento?","R149":"Sistema inmune"}, inplace = True)
dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
        dataset_test.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
dataset_test.loc[:, dataset_test.isin([0,1]).all()] = dataset_test.loc[:, dataset_test.isin([0,1]).all()].apply(lambda x: x.astype('category'))
dataset_test.info()
ColNames = dataset_test.columns
ColGrouped = ColNames[0]
ColNames = np.delete(ColNames, [np.r_[0]], 0)
ColTitles = {'Sí', 'No'}
CQ  = pd.DataFrame(columns = ['Variable', 'Sí', 'No','p'])
temp_totales = dataset_test[ColGrouped].value_counts()
for coltitle in ColTitles:
    CQ.loc['Totales',coltitle] = temp_totales[coltitle]
for cols in ColNames:
    title_data = [cols]
    title = pd.DataFrame(title_data, index=[0])
    if(is_categorical(dataset_test[cols])):
        ctable = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped])
        ctablen = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped], normalize='columns')*100.00
        ctablen = ctablen.applymap(lambda x: " ({:.2f})".format(x))
        stat, pvalue, dof, expected = ss.chi2_contingency(ctable)
        ctable = ctable.astype(str)
        ctable = ctable + ctablen
        if((ctable.index == 0).any()):
            ctable = ctable.drop(0)
            ctable.rename({1:"Sí"}, inplace = True) 
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue) 
        #ctable.loc[ctable.index[0], 'Valor estadistica'] ="{:.2f}".format(stat)
    else:
        dataset_numeric = pd.concat([dataset_test[cols], dataset_test[ColGrouped]], axis=1)
        #correlation, pvalue = ss.pointbiserialr(dataset_test[ColGrouped], dataset_test[cols])
        x = []
        for coltitle in ColTitles:
            x.append(dataset_numeric.loc[dataset_numeric[ColGrouped] == coltitle,cols])
        correlation, pvalue = ss.kruskal(*x)
        ctable = dataset_numeric.groupby(ColGrouped).mean().T
        ctable = ctable.applymap(lambda x: "{:.2f}".format(x))
        ctabled = dataset_numeric.groupby(ColGrouped).std().T
        ctabled = ctabled.applymap(lambda x: " ({:.2f})".format(x))
        ctable = ctable + ctabled
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue)  
    title.rename(columns={0:"Variable"}, inplace = True)
    ctable = title.append(ctable)
    CQ  =  CQ.append(ctable)
CQ = CQ.replace(np.nan,'',regex = True)
file_O = Path("Data/out/crosstab.xlsx")
file_save = cwd / file_O
CQ.to_excel(file_save) 

columnset = np.r_[40:47]
dataset_dependents= data_I_HGM.iloc[:,columnset]
list_del = (dataset_dependents != 0).all(1)
dataset_dependents = dataset_dependents.replace({2:1, 3:0, 4:0, 5:0})
columnset = np.r_[1,4:7,133,136,8,137:139,31:34,35,47:49,67:70,139:146]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test.rename(columns={"R1":"Hospital", "R13": "Edad", "R14":"Sexo","R15":"Estado civil", "R172":"Educacion", "R17": "Ocupación", "R175":"Seguridad Social", "R176":"Diagnóstico", "R177":"Comorbilidad", "R60":"¿Usted requirió hospitalización?","R62":"¿Ya fue vacunado?","R63":"¿Ha tenido la oportunidad de vacunarse?","R65":"¿Cuenta con el esquema completo?","R70 - R77c":"¿Aceptaría alguna de las vacunas si se le ofreciera?","R71 - R78c":"Existen varias vacunas. Usted considera que","R72 - R79c":"Mi actitud con respecto a recibir la vacuna","R73 - R80c":"Si ya estuviera disponible, ¿qué haría?","R74 - R83c":"Si mi familia o amigos estuvieran pensando en vacunarse","R75 - R82c":"Con respecto a recibir la vacuna, yo me describiría como","R76 - R81c":"Recibir una vacuna es","R84":"¿Cree que se infectará?","R85":"Probabilidad que me funcione","R105":"Vacunado influenza","R148":"¿Cómo se siente en este momento?","R149":"Sistema inmune"}, inplace = True)
dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
        dataset_test.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
dataset_test.loc[:, dataset_test.isin([0,1]).all()] = dataset_test.loc[:, dataset_test.isin([0,1]).all()].apply(lambda x: x.astype('category'))
dataset_test.info()
CBin = (dataset_dependents.sum(axis=1)>=4)*1
CBin= CBin.map({1:"Sí", 0:"No"})
dataset_test.insert(0, 'Aceptación', CBin)
ColNames = dataset_test.columns
ColGrouped = ColNames[0]
ColNames = np.delete(ColNames, [np.r_[0]], 0)
ColTitles = {'Sí', 'No'}
CQ  = pd.DataFrame(columns = ['Variable', 'Sí', 'No','p'])
dataset_test = dataset_test[list_del]
temp_totales = dataset_test[ColGrouped].value_counts()
for coltitle in ColTitles:
    CQ.loc['Totales',coltitle] = temp_totales[coltitle]
for cols in ColNames:
    title_data = [cols]
    title = pd.DataFrame(title_data, index=[0])
    if(is_categorical(dataset_test[cols])):
        ctable = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped])
        ctablen = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped], normalize='columns')*100.00
        ctablen = ctablen.applymap(lambda x: " ({:.2f})".format(x))
        stat, pvalue, dof, expected = ss.chi2_contingency(ctable)
        ctable = ctable.astype(str)
        ctable = ctable + ctablen
        if((ctable.index == 0).any()):
            ctable = ctable.drop(0)
            ctable.rename({1:"Sí"}, inplace = True) 
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue) 
        #ctable.loc[ctable.index[0], 'Valor estadistica'] ="{:.2f}".format(stat)
    else:
        dataset_numeric = pd.concat([dataset_test[cols], dataset_test[ColGrouped]], axis=1)
        #correlation, pvalue = ss.pointbiserialr(dataset_test[ColGrouped], dataset_test[cols])
        x = []
        for coltitle in ColTitles:
            x.append(dataset_numeric.loc[dataset_numeric[ColGrouped] == coltitle,cols])
        correlation, pvalue = ss.kruskal(*x)
        ctable = dataset_numeric.groupby(ColGrouped).mean().T
        ctable = ctable.applymap(lambda x: "{:.2f}".format(x))
        ctabled = dataset_numeric.groupby(ColGrouped).std().T
        ctabled = ctabled.applymap(lambda x: " ({:.2f})".format(x))
        ctable = ctable + ctabled
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue)  
    title.rename(columns={0:"Variable"}, inplace = True)
    ctable = title.append(ctable)
    CQ  =  CQ.append(ctable)
CQ = CQ.replace(np.nan,'',regex = True)
file_O = Path("Data/out/crosstab.xlsx")
file_save = cwd / file_O
CQ.to_excel(file_save)

columnsetsp = np.r_[46]
dataset_dependents= data_I_HGM.iloc[:,columnsetsp]
dataset_dependents = dataset_dependents.replace({2:1, 3:0, 4:0, 5:0})
columnset = np.r_[1,4:7,133,136,8,137:139,31:34,35,47:49,67:70,139:146]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test.rename(columns={"R1":"Hospital", "R13": "Edad", "R14":"Sexo","R15":"Estado civil", "R172":"Educacion", "R17": "Ocupación", "R175":"Seguridad Social", "R176":"Diagnóstico", "R177":"Comorbilidad", "R60":"¿Usted requirió hospitalización?","R62":"¿Ya fue vacunado?","R63":"¿Ha tenido la oportunidad de vacunarse?","R65":"¿Cuenta con el esquema completo?","R70 - R77c":"¿Aceptaría alguna de las vacunas si se le ofreciera?","R71 - R78c":"Existen varias vacunas. Usted considera que","R72 - R79c":"Mi actitud con respecto a recibir la vacuna","R73 - R80c":"Si ya estuviera disponible, ¿qué haría?","R74 - R83c":"Si mi familia o amigos estuvieran pensando en vacunarse","R75 - R82c":"Con respecto a recibir la vacuna, yo me describiría como","R76 - R81c":"Recibir una vacuna es","R84":"¿Cree que se infectará?","R85":"Probabilidad que me funcione","R105":"Vacunado influenza","R148":"¿Cómo se siente en este momento?","R149":"Sistema inmune"}, inplace = True)
dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
        dataset_test.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
dataset_test.loc[:, dataset_test.isin([0,1]).all()] = dataset_test.loc[:, dataset_test.isin([0,1]).all()].apply(lambda x: x.astype('category'))
dataset_test.info()
CBin = dataset_dependents.iloc[:,0]
CBin= CBin.map({1:"Sí", 0:"No"})
dataset_test.insert(0, 'Aceptación', CBin)
ColNames = dataset_test.columns
ColGrouped = ColNames[0]
ColNames = np.delete(ColNames, [np.r_[0]], 0)
ColTitles = {'Sí', 'No'}
CQ  = pd.DataFrame(columns = ['Variable', 'Sí', 'No','p'])
dataset_test = dataset_test[list_del]
temp_totales = dataset_test[ColGrouped].value_counts()
for coltitle in ColTitles:
    CQ.loc['Totales',coltitle] = temp_totales[coltitle]
for cols in ColNames:
    title_data = [cols]
    title = pd.DataFrame(title_data, index=[0])
    if(is_categorical(dataset_test[cols])):
        ctable = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped])
        ctablen = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped], normalize='columns')*100.00
        ctablen = ctablen.applymap(lambda x: " ({:.2f})".format(x))
        stat, pvalue, dof, expected = ss.chi2_contingency(ctable)
        ctable = ctable.astype(str)
        ctable = ctable + ctablen
        if((ctable.index == 0).any()):
            ctable = ctable.drop(0)
            ctable.rename({1:"Sí"}, inplace = True) 
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue) 
        #ctable.loc[ctable.index[0], 'Valor estadistica'] ="{:.2f}".format(stat)
    else:
        dataset_numeric = pd.concat([dataset_test[cols], dataset_test[ColGrouped]], axis=1)
        #correlation, pvalue = ss.pointbiserialr(dataset_test[ColGrouped], dataset_test[cols])
        x = []
        for coltitle in ColTitles:
            x.append(dataset_numeric.loc[dataset_numeric[ColGrouped] == coltitle,cols])
        correlation, pvalue = ss.kruskal(*x)
        ctable = dataset_numeric.groupby(ColGrouped).mean().T
        ctable = ctable.applymap(lambda x: "{:.2f}".format(x))
        ctabled = dataset_numeric.groupby(ColGrouped).std().T
        ctabled = ctabled.applymap(lambda x: " ({:.2f})".format(x))
        ctable = ctable + ctabled
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue)  
    title.rename(columns={0:"Variable"}, inplace = True)
    ctable = title.append(ctable)
    CQ  =  CQ.append(ctable)
CQ = CQ.replace(np.nan,'',regex = True)
file_O = Path("Data/out/crosstab.xlsx")
file_save = cwd / file_O
CQ.to_excel(file_save)

#AR lupus
columnset = np.r_[137,1,4:7,133,136,8,73:84,138:139,31:34,35,47:49,67:70,139:146]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test.rename(columns={"R176":"Diagnóstico","R1":"Hospital", "R13": "Edad", "R14":"Sexo","R15":"Estado civil", "R172":"Educacion", "R17": "Ocupación", "R175":"Seguridad Social", "R176":"Diagnóstico", "R157":"Enfermedad pulmonar","R158":"Infarto agudo de miocardio","R159":"Otra enfermedad cardiovascular","R160":"Enfermedad vascular cerebral","R161":"Hipertensión arterial sistémica","R162":"Diabetes mellitus","R163":"Fractura de cadera/columna o pierna","R164":"Depresión","R165":"Cancer","R166":"Ulcera gastrointestinal","R167":"Otras enfermedades gástrica","R168":"Otro", "R60":"¿Usted requirió hospitalización?","R62":"¿Ya fue vacunado?","R63":"¿Ha tenido la oportunidad de vacunarse?","R65":"¿Cuenta con el esquema completo?","R70 - R77c":"¿Aceptaría alguna de las vacunas si se le ofreciera?","R71 - R78c":"Existen varias vacunas. Usted considera que","R72 - R79c":"Mi actitud con respecto a recibir la vacuna","R73 - R80c":"Si ya estuviera disponible, ¿qué haría?","R74 - R83c":"Si mi familia o amigos estuvieran pensando en vacunarse","R75 - R82c":"Con respecto a recibir la vacuna, yo me describiría como","R76 - R81c":"Recibir una vacuna es","R84":"¿Cree que se infectará?","R85":"Probabilidad que me funcione","R105":"Vacunado influenza","R148":"¿Cómo se siente en este momento?","R149":"Sistema inmune"}, inplace = True)

list_subset = (dataset_test['R176'] == 'Lupus') | (dataset_test['R176'] == 'ArtritisReumatoide')
dataset_test=dataset_test[list_subset]
dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
        dataset_test.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
dataset_test.loc[:, dataset_test.isin([0,1]).all()] = dataset_test.loc[:, dataset_test.isin([0,1]).all()].apply(lambda x: x.astype('category'))
dataset_test.info()
ColNames = dataset_test.columns
ColGrouped = ColNames[0]
ColNames = np.delete(ColNames, [np.r_[0]], 0)
dataset_test[ColGrouped] = dataset_test[ColGrouped].replace("ArtritisReumatoide", "AR")
ColTitles = {'AR', 'Lupus'}
CQ  = pd.DataFrame(columns = ['Variable', 'AR', 'Lupus','p'])
temp_totales = dataset_test[ColGrouped].value_counts()
for coltitle in ColTitles:
    CQ.loc['Totales',coltitle] = temp_totales[coltitle]
for cols in ColNames:
    title_data = [cols]
    title = pd.DataFrame(title_data, index=[0])
    if(is_categorical(dataset_test[cols])):
        ctable = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped])
        ctablen = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped], normalize='columns')*100.00
        ctablen = ctablen.applymap(lambda x: " ({:.2f})".format(x))
        stat, pvalue, dof, expected = ss.chi2_contingency(ctable)
        ctable = ctable.astype(str)
        ctable = ctable + ctablen
        if((ctable.index == 0).any()):
            ctable = ctable.drop(0)
            ctable.rename({1:"Sí"}, inplace = True) 
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue) 
        #ctable.loc[ctable.index[0], 'Valor estadistica'] ="{:.2f}".format(stat)
    else:
        dataset_numeric = pd.concat([dataset_test[cols], dataset_test[ColGrouped]], axis=1)
        #correlation, pvalue = ss.pointbiserialr(dataset_test[ColGrouped], dataset_test[cols])
        x = []
        for coltitle in ColTitles:
            x.append(dataset_numeric.loc[dataset_numeric[ColGrouped] == coltitle,cols])
        correlation, pvalue = ss.kruskal(*x)
        ctable = dataset_numeric.groupby(ColGrouped).mean().T
        ctable = ctable.applymap(lambda x: "{:.2f}".format(x))
        ctabled = dataset_numeric.groupby(ColGrouped).std().T
        ctabled = ctabled.applymap(lambda x: " ({:.2f})".format(x))
        ctable = ctable + ctabled
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue)  
    title.rename(columns={0:"Variable"}, inplace = True)
    ctable = title.append(ctable)
    CQ  =  CQ.append(ctable)
CQ = CQ.replace(np.nan,'',regex = True)
file_O = Path("Data/out/crosstab.xlsx")
file_save = cwd / file_O
CQ.to_excel(file_save)

#numero de dosis
columnsetsp = np.r_[35]
dataset_temp1= data_I_HGM.iloc[:,columnsetsp]
columnsetsp = np.r_[32]
dataset_temp2= data_I_HGM.iloc[:,columnsetsp]
dataset_dependents = dataset_temp2.iloc[:,0] + dataset_temp1.iloc[:,0]
dataset_dependents = dataset_dependents.map({1:"Una dosis", 0:"No vacunado", 2:"Dos dosis"})
#columnset = np.r_[1,4:7,133,136,8,137:139,31,47:49,67:70,139:146]
columnset = np.r_[1,4:7,133,136,8,73:84,17:30,31,47:49,67:70,139:146]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test.rename(columns={"R1":"Hospital", "R13": "Edad", "R14":"Sexo","R15":"Estado civil", "R172":"Educacion", "R17": "Ocupación", "R175":"Seguridad Social", "R176":"Diagnóstico", "R157":"Enfermedad pulmonar","R158":"Infarto agudo de miocardio","R159":"Otra enfermedad cardiovascular","R160":"Enfermedad vascular cerebral","R161":"Hipertensión arterial sistémica","R162":"Diabetes mellitus","R163":"Fractura de cadera/columna o pierna","R164":"Depresión","R165":"Cancer","R166":"Ulcera gastrointestinal","R167":"Otras enfermedades gástrica","R168":"Otro", "R60":"¿Usted requirió hospitalización?","R62":"¿Ya fue vacunado?","R63":"¿Ha tenido la oportunidad de vacunarse?","R65":"¿Cuenta con el esquema completo?","R70 - R77c":"¿Aceptaría alguna de las vacunas si se le ofreciera?","R71 - R78c":"Existen varias vacunas. Usted considera que","R72 - R79c":"Mi actitud con respecto a recibir la vacuna","R73 - R80c":"Si ya estuviera disponible, ¿qué haría?","R74 - R83c":"Si mi familia o amigos estuvieran pensando en vacunarse","R75 - R82c":"Con respecto a recibir la vacuna, yo me describiría como","R76 - R81c":"Recibir una vacuna es","R84":"¿Cree que se infectará?","R85":"Probabilidad que me funcione","R105":"Vacunado influenza","R148":"¿Cómo se siente en este momento?","R149":"Sistema inmune","R26":"Lupus eritematoso generalizado","R27":"Artritis reumatoide","R28":"Vasculitis ANCA","R29":"Espondilitis anquilosante","R30":"Esclerodermia","R31":"Gota","R32":"Síndrome de Sjögren","R33":"Miopatía inflamatoria","R34":"Síndrome de anticuerpos antifosfolípidos","R35":"Artritits idiopática juvenil","R36":"Enfermedad mixta del tejido conectivo","R37":"Osteoartrosis","R38":"Otro"}, inplace = True)
dataset_test.columns.tolist
dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
        dataset_test.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
dataset_test.loc[:, dataset_test.isin([0,1]).all()] = dataset_test.loc[:, dataset_test.isin([0,1]).all()].apply(lambda x: x.astype('category'))
dataset_test.info()
dataset_test.insert(0, 'Dosis', dataset_dependents)

ColNames = dataset_test.columns
ColGrouped = ColNames[0]
ColNames = np.delete(ColNames, [np.r_[0]], 0)
ColTitles = {"No vacunado", "Una dosis","Dos dosis"}
CQ  = pd.DataFrame(columns = ['Variable', "No vacunado", "Una dosis","Dos dosis",'p'])
temp_totales = dataset_test[ColGrouped].value_counts()
for coltitle in ColTitles:
    CQ.loc['Totales',coltitle] = temp_totales[coltitle]
for cols in ColNames:
    title_data = [cols]
    title = pd.DataFrame(title_data, index=[0])
    if(is_categorical(dataset_test[cols])):
        ctable = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped])
        ctablen = pd.crosstab(dataset_test[cols],dataset_test[ColGrouped], normalize='columns')*100.00
        ctablen = ctablen.applymap(lambda x: " ({:.2f})".format(x))
        stat, pvalue, dof, expected = ss.chi2_contingency(ctable)
        ctable = ctable.astype(str)
        ctable = ctable + ctablen
        if((ctable.index == 0).any()):
            ctable = ctable.drop(0)
            ctable.rename({1:"Sí"}, inplace = True) 
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue) 
        #ctable.loc[ctable.index[0], 'Valor estadistica'] ="{:.2f}".format(stat)
    else:
        dataset_numeric = pd.concat([dataset_test[cols], dataset_test[ColGrouped]], axis=1)
        #correlation, pvalue = ss.pointbiserialr(dataset_test[ColGrouped], dataset_test[cols])
        x = []
        for coltitle in ColTitles:
            x.append(dataset_numeric.loc[dataset_numeric[ColGrouped] == coltitle,cols])
        correlation, pvalue = ss.kruskal(*x)
        ctable = dataset_numeric.groupby(ColGrouped).mean().T
        ctable = ctable.applymap(lambda x: "{:.2f}".format(x))
        ctabled = dataset_numeric.groupby(ColGrouped).std().T
        ctabled = ctabled.applymap(lambda x: " ({:.2f})".format(x))
        ctable = ctable + ctabled
        if(pvalue < 0.001):
            ctable.loc[ctable.index[0], 'p'] = '<0.001'
        else:
            ctable.loc[ctable.index[0], 'p'] = "{:.3f}".format(pvalue)  
    title.rename(columns={0:"Variable"}, inplace = True)
    ctable = title.append(ctable)
    CQ  =  CQ.append(ctable)
CQ = CQ.replace(np.nan,'',regex = True)
file_O = Path("Data/out/crosstab.xlsx")
file_save = cwd / file_O
CQ.to_excel(file_save)


#time
columnset = np.r_[40:47]#40:47
dataset_dependents= data_I_HGM.iloc[:,columnset]
list_del = (dataset_dependents != 0).all(1)
max_column = dataset_dependents.apply(pd.Series.value_counts, axis=1).idxmax(axis=1)
max_column.name = "MaxFrec"
columnset = np.r_[132,40:47]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents = pd.concat([dataset_dependents, max_column], axis=1)
dataset_dependents.loc[:, dataset_dependents.dtypes == 'object'] =\
        dataset_dependents.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
ColNames = dataset_dependents.columns
ColNames = np.delete(ColNames, [np.r_[0]], 0)
for cols in ColNames:
    datset_time = pd.DataFrame()
    dataset_sub = dataset_dependents[['R171c',cols]]
    for cat_id in range(0, time_cats):
        dataset_test = dataset_sub.loc[dataset_sub.R171c == dataset_sub.R171c.cat.categories[cat_id],cols]
        datset_time = pd.concat([datset_time, dataset_test.rename(str(dataset_sub.R171c.cat.categories[cat_id]))], axis=1)
    dataset_test = datset_time.fillna(0)
    dataset_test = dataset_test.replace({0:'None',1: 'Strongly agree', 2:'Agree', 3: 'Neither agree nor disagree', 4:'Disagree', 5:'Strongly disagree'})
    another_scale = \
        ['Strongly disagree',
        'Disagree',
        'Neither agree nor disagree',
        'Agree',
        'Strongly agree',
        'None']
    colours5 = ['white','#f9665e','#fec9c9','#949494','#95b4cc','#799fcb'] 
    ax = plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours5) #another_scale plot_likert.scales.agree
    ax.set_xlim([0, 30]);
    ax.xaxis.set_label_text('Porcentaje de respuestas: Población completa');
    ax.get_legend().remove()
    image_format = 'svg'
    file_O = Path("Data/outimg/likert_hesitancy_"+cols+".svg")
    file_save = cwd / file_O
    plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
    plt.close()

columnset = np.r_[132,32,35]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents.rename(columns={"R171c":"Date","R62":"1 dosis","R65":"2 dosis"}, inplace = True)
agg_dict = {
    '1 dosis':['sum'],
    '2 dosis':['sum']
}
dataset_dependents_count=dataset_dependents.groupby(by=["Date"]).agg(agg_dict)
sns.heatmap(dataset_dependents_count,annot = dataset_dependents_count, fmt=".0f", linewidths=.5, cmap="flare") 
image_format = 'svg'
file_O = Path("Data/outimg/dosis_totales"+".svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
plt.close()

columnset = np.r_[132,32,35,128:130]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents.rename(columns={"R171c":"Date","R62":"¿1 dosis?","R65":"¿2 dosis?","R168":"Muertes", "R169": "Casos"}, inplace = True)
dataset_dependents_norm= dataset_dependents.groupby(by=["Date"]).mean()/(1.0,1.0,dataset_dependents.iloc[:,3].max(),dataset_dependents.iloc[:,4].max())
agg_dict = {
    '1 dosis':['sum'],
    '2 dosis':['sum'],
    'Muertes':['mean'],
    'Casos':['mean'],
}
dataset_dependents_count=dataset_dependents.groupby(by=["Date"]).agg(agg_dict)
sns.heatmap(dataset_dependents_norm,annot = dataset_dependents_count, fmt=".0f", linewidths=.5, cmap="flare") 
image_format = 'svg'
file_O = Path("Data/outimg/Cases_totales"+".svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
plt.close()


#enfermedad ArtritisReumatoide Lupus
columnset = np.r_[137]#40:47
dataset_dependents= data_I_HGM.iloc[:,columnset]
list_subset = dataset_dependents['R176'] == 'ArtritisReumatoide'
columnset = np.r_[40:47]#40:47
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents=dataset_dependents[list_subset]
list_del = (dataset_dependents != 0).all(1)
max_column = dataset_dependents.apply(pd.Series.value_counts, axis=1).idxmax(axis=1)
max_column.name = "MaxFrec"
max_column=max_column[list_subset]
columnset = np.r_[132,40:47]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents=dataset_dependents[list_subset]
dataset_dependents = pd.concat([dataset_dependents, max_column], axis=1)
dataset_dependents.loc[:, dataset_dependents.dtypes == 'object'] =\
        dataset_dependents.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
ColNames = dataset_dependents.columns
ColNames = np.delete(ColNames, [np.r_[0]], 0)
for cols in ColNames:
    datset_time = pd.DataFrame()
    dataset_sub = dataset_dependents[['R171c',cols]]
    for cat_id in range(0, time_cats):
        dataset_test = dataset_sub.loc[dataset_sub.R171c == dataset_sub.R171c.cat.categories[cat_id],cols]
        datset_time = pd.concat([datset_time, dataset_test.rename(str(dataset_sub.R171c.cat.categories[cat_id]))], axis=1)
    dataset_test = datset_time.fillna(0)
    dataset_test = dataset_test.replace({0:'None',1: 'Strongly agree', 2:'Agree', 3: 'Neither agree nor disagree', 4:'Disagree', 5:'Strongly disagree'})
    another_scale = \
        ['Strongly disagree',
        'Disagree',
        'Neither agree nor disagree',
        'Agree',
        'Strongly agree',
        'None']
    colours5 = ['white','#f9665e','#fec9c9','#949494','#95b4cc','#799fcb'] 
    ax = plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours5) #another_scale plot_likert.scales.agree
    ax.set_xlim([0, 30]);
    ax.xaxis.set_label_text('Porcentaje de respuestas: Artritis Reumatoide');
    ax.get_legend().remove()
    image_format = 'svg'
    file_O = Path("Data/outimg/likert_hesitancy_ArtritisReumatoide_"+cols+".svg")
    file_save = cwd / file_O
    plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
    plt.close()

columnset = np.r_[132,32,35]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents.rename(columns={"R171c":"Date","R62":"1 dosis","R65":"2 dosis"}, inplace = True)
agg_dict = {
    '1 dosis':['sum'],
    '2 dosis':['sum']
}
dataset_dependents_count=dataset_dependents.groupby(by=["Date"]).agg(agg_dict)
sns.heatmap(dataset_dependents_count,annot = dataset_dependents_count, fmt=".0f", linewidths=.5, cmap="flare") 
image_format = 'svg'
file_O = Path("Data/outimg/dosis_artritis"+".svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
plt.close()
    
columnset = np.r_[132,32,35,128:130]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents=dataset_dependents[list_subset]
dataset_dependents.rename(columns={"R171c":"Date","R62":"¿1 dosis?","R65":"¿2 dosis?","R168":"Muertes", "R169": "Casos"}, inplace = True)
dataset_dependents_norm= dataset_dependents.groupby(by=["Date"]).mean()/(1.0,1.0,dataset_dependents.iloc[:,3].max(),dataset_dependents.iloc[:,4].max())
agg_dict = {
    '¿1 dosis?':['sum'],
    '¿2 dosis?':['sum'],
    'Muertes':['mean'],
    'Casos':['mean'],
}
dataset_dependents_count=dataset_dependents.groupby(by=["Date"]).agg(agg_dict)
sns.heatmap(dataset_dependents_norm,annot = dataset_dependents_count, fmt=".0f", linewidths=.5, cmap="flare") 
image_format = 'svg'
file_O = Path("Data/outimg/Cases_artritis"+".svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
plt.close()

columnset = np.r_[137]#40:47
dataset_dependents= data_I_HGM.iloc[:,columnset]
list_subset = dataset_dependents['R176'] == 'Lupus'
columnset = np.r_[40:47]#40:47
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents=dataset_dependents[list_subset]
list_del = (dataset_dependents != 0).all(1)
max_column = dataset_dependents.apply(pd.Series.value_counts, axis=1).idxmax(axis=1)
max_column.name = "MaxFrec"
max_column=max_column[list_subset]
columnset = np.r_[132,40:47]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents=dataset_dependents[list_subset]
dataset_dependents = pd.concat([dataset_dependents, max_column], axis=1)
dataset_dependents.loc[:, dataset_dependents.dtypes == 'object'] =\
        dataset_dependents.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
ColNames = dataset_dependents.columns
ColNames = np.delete(ColNames, [np.r_[0]], 0)
for cols in ColNames:
    datset_time = pd.DataFrame()
    dataset_sub = dataset_dependents[['R171c',cols]]
    for cat_id in range(0, time_cats):
        dataset_test = dataset_sub.loc[dataset_sub.R171c == dataset_sub.R171c.cat.categories[cat_id],cols]
        datset_time = pd.concat([datset_time, dataset_test.rename(str(dataset_sub.R171c.cat.categories[cat_id]))], axis=1)
    dataset_test = datset_time.fillna(0)
    dataset_test = dataset_test.replace({0:'None',1: 'Strongly agree', 2:'Agree', 3: 'Neither agree nor disagree', 4:'Disagree', 5:'Strongly disagree'})
    another_scale = \
        ['Strongly disagree',
        'Disagree',
        'Neither agree nor disagree',
        'Agree',
        'Strongly agree',
        'None']
    colours5 = ['white','#f9665e','#fec9c9','#949494','#95b4cc','#799fcb'] 
    ax = plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours5) #another_scale plot_likert.scales.agree
    ax.set_xlim([0, 30]);
    ax.xaxis.set_label_text('Porcentaje de respuestas: Lupus');
    ax.get_legend().remove()
    image_format = 'svg'
    file_O = Path("Data/outimg/likert_hesitancy_Lupus_"+cols+".svg")
    file_save = cwd / file_O
    plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
    plt.close()

columnset = np.r_[132,32,35]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents.rename(columns={"R171c":"Date","R62":"1 dosis","R65":"2 dosis"}, inplace = True)
agg_dict = {
    '1 dosis':['sum'],
    '2 dosis':['sum']
}
dataset_dependents_count=dataset_dependents.groupby(by=["Date"]).agg(agg_dict)
sns.heatmap(dataset_dependents_count,annot = dataset_dependents_count, fmt=".0f", linewidths=.5, cmap="flare") 
image_format = 'svg'
file_O = Path("Data/outimg/dosis_lupus"+".svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
plt.close()

columnset = np.r_[132,32,35,128:130]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents=dataset_dependents[list_subset]
dataset_dependents.rename(columns={"R171c":"Date","R62":"¿1 dosis?","R65":"¿2 dosis?","R168":"Muertes", "R169": "Casos"}, inplace = True)
dataset_dependents_norm= dataset_dependents.groupby(by=["Date"]).mean()/(1.0,1.0,dataset_dependents.iloc[:,3].max(),dataset_dependents.iloc[:,4].max())
agg_dict = {
    '¿1 dosis?':['sum'],
    '¿2 dosis?':['sum'],
    'Muertes':['mean'],
    'Casos':['mean'],
}
dataset_dependents_count=dataset_dependents.groupby(by=["Date"]).agg(agg_dict)
sns.heatmap(dataset_dependents_norm,annot = dataset_dependents_count, fmt=".0f", linewidths=.5, cmap="flare") 
image_format = 'svg'
file_O = Path("Data/outimg/Cases_lupus"+".svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
plt.close()

#time2
columnset = np.r_[40:47]#40:47
dataset_dependents= data_I_HGM.iloc[:,columnset]
list_del = (dataset_dependents != 0).all(1)
columnset = np.r_[132,40:47]
dataset_dependents= data_I_HGM.iloc[:,columnset]
dataset_dependents.loc[:, dataset_dependents.dtypes == 'object'] =\
        dataset_dependents.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
for cat_id in range(0, time_cats):
    dataset_test = dataset_dependents[dataset_dependents.R171c == dataset_dependents.R171c.cat.categories[cat_id]]
    date_str=str(dataset_test.iloc[0,0])
    dataset_test.drop(columns=dataset_test.columns[0], axis=1, inplace=True)
    dataset_test = dataset_test.replace({0:'None',1: 'Strongly agree', 2:'Agree', 3: 'Neither agree nor disagree', 4:'Disagree', 5:'Strongly disagree'})
    another_scale = \
        ['Strongly disagree',
        'Disagree',
        'Neither agree nor disagree',
        'Agree',
        'Strongly agree',
        'None']
    colours5 = ['white','#f9665e','#fec9c9','#949494','#95b4cc','#799fcb'] 
    plot_likert.plot_likert(dataset_test, another_scale,plot_percentage=True, colors=colours5) #another_scale plot_likert.scales.agree
    image_format = 'svg'
    file_O = Path("Data/outimg/likert_hesitancy_"+date_str+".svg")
    file_save = cwd / file_O
    plt.savefig(file_save, format=image_format, dpi=1200)
    plt.close()

#cosine simmilitued
cosine_similarity(df)

#matriz de asociacion de cramer correlacion chi-square
columnset = np.r_[139:146]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test.rename(columns={"R70 - R77c":"¿Aceptaría alguna de las vacunas si se le ofreciera?","R71 - R78c":"Existen varias vacunas. Usted considera que","R72 - R79c":"Mi actitud con respecto a recibir la vacuna","R73 - R80c":"Si ya estuviera disponible, ¿qué haría?","R74 - R83c":"Si mi familia o amigos estuvieran pensando en vacunarse","R75 - R82c":"Con respecto a recibir la vacuna, yo me describiría como","R76 - R81c":"Recibir una vacuna es"}, inplace = True)
dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
        dataset_test.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
dataset_test.info()        
dataset_test.loc[:, dataset_test.isin([0,1]).all()] = dataset_test.loc[:, dataset_test.isin([0,1]).all()].apply(lambda x: x.astype('category'))
ColNames = dataset_test.columns
datast_c = pd.DataFrame(columns=ColNames, index=ColNames,dtype="float")
datast_t = pd.DataFrame(columns=ColNames, index=ColNames,dtype="float")
datast_p = pd.DataFrame(columns=ColNames, index=ColNames,dtype="float")
for cols_x in ColNames:
    for cols_y in ColNames:
        c_value = cramers_v(dataset_test[cols_x],dataset_test[cols_y],bias_correction=False)
        t_value = theils_u(dataset_test[cols_x],dataset_test[cols_y])
        ctable = pd.crosstab(dataset_test[cols_x],dataset_test[cols_y])
        stat, p_value, dof, expected = ss.chi2_contingency(ctable)
        datast_c.loc[cols_x,cols_y] = c_value
        datast_t.loc[cols_x,cols_y] = t_value
        if(pvalue < 0.001):
            datast_p.loc[cols_x,cols_y] = '<0.001'
        else:
            datast_p.loc[cols_x,cols_y] = "{:.3f}".format

sns.heatmap(datast_c, annot = True, fmt=".2f", linewidths=.5, cmap="flare") 
image_format = 'svg'
file_O = Path("Data/outimg/cramer"+".svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
plt.close()

sns.heatmap(datast_t, annot = True, fmt=".2f", linewidths=.5, cmap="flare") 
image_format = 'svg'
file_O = Path("Data/outimg/theils"+".svg")
file_save = cwd / file_O
plt.savefig(file_save, format=image_format, dpi=1200, transparent=True)
plt.close()

#logistic regresion
#new set
columnset = np.r_[1,4:7,136,8,137:139,31:34,35,47:49,67:70]#,138:146]#17:89]#   ,]
dataset_test= data_I_HGM.iloc[:,columnset]
dataset_test['R62'] = dataset_test['R62'].map({1:"Sí", 0:"No"})
dataset_test.columns.tolist()
dataset_test.rename(columns={"R1":"Hospital", "R13": "Edad", "R14":"Sexo","R15":"EstadoCivil", "R17": "Ocupacion", "R175":"SeguridadSocial", "R176":"Diagnostico", "R177":"Comorbilidad", "R60":"RequirioHospitalización","R62":"Vacunado","R63":"OportunidadVacunarse","R65":"EsquemaCompleto","R84":"Infectara","R85":"ProbabilidadFuncione","R105":"VacunadoInfluenza","R148":"SienteMomento","R149":"SistemaInmune"}, inplace = True)
dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
        dataset_test.select_dtypes(['object'])\
        .apply(lambda x: x.astype('category'))
dataset_test.loc[:, dataset_test.isin([0,1]).all()] = dataset_test.loc[:, dataset_test.isin([0,1]).all()].apply(lambda x: x.astype('category'))
#
columnset = np.r_[40:47]
dataset_dependents= data_I_HGM.iloc[:,columnset]
list_del = (dataset_dependents != 0).all(1)
dataset_dependents = dataset_dependents.replace({2:1, 3:0, 4:0, 5:0})
dataset_dependents['bin'] = (dataset_dependents.sum(axis=1)>=4)*1
columnset = np.r_[134]
dataset_temp= data_I_HGM.iloc[:,columnset]
dataset_dependents['per'] = (dataset_temp['R173'] >= dataset_temp.R173.quantile(0.75))*1
dataset_dependents=dataset_dependents[list_del]
dataset_odds= pd.DataFrame(columns=['ODDs', 'coeff', 'pvals', '0.025', '0.975', 'name'])
dataset_odds= pd.DataFrame()
for name, values in dataset_dependents.iteritems():
    print(name)
    dataset_results_final = pd.DataFrame()
    dataset_dependent = dataset_dependents.loc[:,name]
    dataset_independent= dataset_test
    dataset_independent=dataset_independent[list_del]
    dataset_model = pd.concat([dataset_dependent, dataset_independent], axis=1)
    dataset_model.rename(columns={ dataset_model.columns[0]: "D" }, inplace = True)
    train_data, test_data = train_test_split(dataset_model, test_size=0.20, random_state= 42)
    # formula = ('D ~ Hospital + Edad + Sexo + EstadoCivil + SeguridadSocial + Ocupacion + Diagnostico + Comorbilidad + RequirioHospitalización + Vacunado + OportunidadVacunarse + EsquemaCompleto + Infectara + ProbabilidadFuncione + VacunadoInfluenza + SienteMomento + SistemaInmune')
    # formula = ('D ~ Hospital + Edad + Sexo + EstadoCivil + SeguridadSocial + Ocupacion + RequirioHospitalización + Vacunado + OportunidadVacunarse + EsquemaCompleto + Infectara + ProbabilidadFuncione + VacunadoInfluenza + SienteMomento + SistemaInmune')
    formula = ('D ~ Hospital + Edad + Sexo + EstadoCivil  + OportunidadVacunarse + EsquemaCompleto + Infectara + ProbabilidadFuncione + VacunadoInfluenza + SienteMomento + SistemaInmune')
    model = logit(formula = formula, data = train_data).fit()
    dataset_results = results_summary_to_dataframe(model)
    dataset_results_final['OR (95% CI two-sided)'] = dataset_results['ODDs'].fillna('').apply('{:.2f}'.format).astype(str) + ' (' + dataset_results['0.05'].fillna('').apply('{:.2f}'.format).astype(str) + ' - ' + dataset_results['0.95'].fillna('').apply('{:.2f}'.format).astype(str) +  ' )'
    dataset_results['ps'] = ''
    dataset_results.loc[dataset_results['pvals']< 0.05,'ps'] = '< 0.05' 
    dataset_results.loc[dataset_results['pvals']>= 0.05,'ps'] = dataset_results.loc[dataset_results['pvals']>= 0.05,'pvals'].fillna('').apply('{:.3f}'.format).astype(str)
    dataset_results_final['p'] = dataset_results['ps']
    dataset_results_final.columns = pd.MultiIndex.from_product([[name],dataset_results_final.columns])
    dataset_odds = pd.concat([dataset_odds, dataset_results_final], axis=1)
file_O = Path("Data/out/" + "OddsAll.xlsx")
file_save = cwd / file_O
dataset_odds.to_excel(file_save)












#%% Correlation
# columnset = np.r_[5:10, 11:89]
# columnset = np.r_[5:10, 11:85]
# columnset = np.r_[5:10,11:89]
# dataset_cor= data_I_HGM.iloc[:,columnset]
# dataset_cor['R170'] = dataset_cor.R170.astype(str)
# dataset_cor['R171'] = dataset_cor.R171.astype(str)

# corr_mat= associations(dataset_cor, annot=False,fmt='.1f')

# #%%Time cross tables
# cross = pd.crosstab(index = data_I_HGM['R84'], 
#             columns = data_I_HGM['R171c'],  
#             normalize='index')
# plot_heatmap(cross, fmt='.2%')
# cross = pd.crosstab(index = data_I_HGM['R84'], 
#             columns = data_I_HGM['R171c'])
# plot_heatmap(cross, fmt='3')

#%% descriptors per hospital
# columnset = np.r_[1,5:7,8,10:89]
# dataset_test= data_I_HGM.iloc[:,columnset]
# ColNames = dataset_test.columns
# ColNames = np.delete(ColNames, [np.r_[0]], 0)
# CQ  = pd.DataFrame(columns = ['CHI', 'HGM', 'MTY', 'NUT', 'p','name'])
# for cols in ColNames:
#     print(cols)
#     ctable = pd.crosstab(dataset_test[cols],dataset_test['R1'],)
#     stat, p, dof, expected = ss.chi2_contingency(ctable)
#     ctable['p']=p
#     ctable['name']=cols
#     CQ  =  CQ.append(ctable)
# dataset_test['R1'].value_counts()
# file_O = Path("Data/out/crosstab.xlsx")
# file_save = cwd / file_O
# CQ.to_excel(file_save) 

# #%% descriptors per vacunado
# columnset = np.r_[32,1,5:7,8,10:32,33:89]
# dataset_test= data_I_HGM.iloc[:,columnset]

# ColNames = dataset_test.columns
# ColNames = np.delete(ColNames, [np.r_[0]], 0)
# CQ  = pd.DataFrame(columns = ['0', '1', 'p','name'])
# for cols in ColNames:
#     print(cols)
#     ctable = pd.crosstab(dataset_test[cols],dataset_test['R62'],)
#     stat, p, dof, expected = ss.chi2_contingency(ctable)
#     ctable['p']=p
#     ctable['name']=cols
#     CQ  =  CQ.append(ctable)
# dataset_test['R1'].value_counts()
# dataset_test['R62'].value_counts()
# file_O = Path("Data/out/crosstab.xlsx")
# file_save = cwd / file_O
# CQ.to_excel(file_save) 

# #%% count people per answer in hecitanci
# columnset = np.r_[0,40:47]
# dataset_test= data_I_HGM.iloc[:,columnset]
# dataset_pivot = dataset_test.melt('R0').pivot_table(index='R0', columns='value', aggfunc='size', fill_value=0) 
# dataset_countpernum = pd.DataFrame({
#     '7e': dataset_pivot[dataset_pivot == 7.0].count(), 
#     '6e': dataset_pivot[dataset_pivot == 6.0].count(),
#     '5e': dataset_pivot[dataset_pivot == 5.0].count(),
#     '4e': dataset_pivot[dataset_pivot == 4.0].count(),
#     '3e': dataset_pivot[dataset_pivot == 3.0].count(),
#     '2e': dataset_pivot[dataset_pivot == 2.0].count(),
#     '1e': dataset_pivot[dataset_pivot == 1.0].count()
#     })
# #minimum 4 questions with the likerd value
# columnset = np.r_[0:4]
# dataset_likerdmax = dataset_countpernum.iloc[:,columnset].sum(axis=1)


# columnset = np.r_[0,40:47,90]
# dataset_test= data_I_HGM.iloc[:,columnset]
# dataset_quartiles = pd.DataFrame({
# 'min':[dataset_test.R173.quantile(0.0)],
# '1q':[dataset_test.R173.quantile(0.25)],
# 'median':[dataset_test.R173.quantile(0.5)],
# '3q':[dataset_test.R173.quantile(0.75)],
# 'max':[dataset_test.R173.quantile(1.0)]
# })
# dataset_test = dataset_test[dataset_test['R173'] >= dataset_test.R173.quantile(0.75)]
# dataset_test.drop('R173', axis=1, inplace=True)
# dataset_pivot = dataset_test.melt('R0').pivot_table(index='R0', columns='value', aggfunc='size', fill_value=0) 
# dataset_countpernumquartil = pd.DataFrame({
#     '7e': dataset_pivot[dataset_pivot == 7.0].count(), 
#     '6e': dataset_pivot[dataset_pivot == 6.0].count(),
#     '5e': dataset_pivot[dataset_pivot == 5.0].count(),
#     '4e': dataset_pivot[dataset_pivot == 4.0].count(),
#     '3e': dataset_pivot[dataset_pivot == 3.0].count(),
#     '2e': dataset_pivot[dataset_pivot == 2.0].count(),
#     '1e': dataset_pivot[dataset_pivot == 1.0].count()
#     })
# #minimum 4 questions with the likerd value
# columnset = np.r_[0:4]
# dataset_likerdmaxquartil = dataset_countpernumquartil.iloc[:,columnset].sum(axis=1)

#%%
#logistic regresion
# columnset = np.r_[40:47]
# dataset_dependents= data_I_HGM.iloc[:,columnset]
# list_del = (dataset_dependents != 0).all(1)
# dataset_dependents = dataset_dependents.replace({2:1, 3:0, 4:0, 5:0})
# dataset_dependents['bin'] = (dataset_dependents.sum(axis=1)>=4)*1
# columnset = np.r_[134]
# dataset_temp= data_I_HGM.iloc[:,columnset]
# dataset_dependents['per'] = (dataset_temp['R173'] >= dataset_temp.R173.quantile(0.75))*1
# dataset_dependents=dataset_dependents[list_del]

# for name, values in dataset_dependents.iteritems():
#     dataset_dependent = dataset_dependents.loc[:,name]
#     #dataset_dependent = dataset_dependents.iloc[:,0]
#     columnset = np.r_[4,5,6,7,92,93,94]
#     dataset_independent= data_I_HGM.iloc[:,columnset]
#     dataset_independent=dataset_independent[list_del]

#     dataset_test = pd.concat([dataset_dependent, dataset_independent], axis=1)
#     dataset_test.rename(columns={ dataset_test.columns[0]: "D" }, inplace = True)
#     dataset_test.rename(columns={"R13": "Edad", "R14": "Sexo", "R15": "EstadoCivil", "R16": "Escolaridad", "R175":"SeguridadSocial", "R176":"Enfermedad", "R177":"comorbilidad"}, inplace = True)
#     dataset_test.loc[:, dataset_test.dtypes == 'object'] =\
#         dataset_test.select_dtypes(['object'])\
#         .apply(lambda x: x.astype('category'))

#     dataset_test = pd.get_dummies(dataset_test, prefix=['EstadoCivil', 'SeguridadSocial','Enfermedad','comorbilidad'], columns=['EstadoCivil', 'SeguridadSocial','Enfermedad','comorbilidad'])
#     #dataset_test.info()
#     # sns.regplot(x="R13", y="D",
#     # y_jitter=0.03,
#     # data=dataset_test,
#     # logistic=True,
#     # ci=None)
#     # plt.show()
#     #dataset_test['R177'].value_counts()
#     train_data, test_data = train_test_split(dataset_test, test_size=0.20, random_state= 42)
#     # train_data['Enfermedad'].value_counts()
#     # train_data['comorbilidad'].value_counts()
#     # train_data['SeguridadSocial'].value_counts()
#     #formula = ('D ~ Edad + Sexo + EstadoCivil + Escolaridad + SeguridadSocial + Enfermedad + comorbilidad')

#     # formula = ('D ~ Edad + Sexo + Escolaridad + EstadoCivil_Casado +EstadoCivil_Separado + EstadoCivil_Soltero + EstadoCivil_UnionLibre +EstadoCivil_Viudo')
#     # formula = ('D ~ Edad + Sexo + Escolaridad + EstadoCivil_Casado +EstadoCivil_Separado + EstadoCivil_Soltero + EstadoCivil_UnionLibre +EstadoCivil_Viudo + SeguridadSocial_IMSS + SeguridadSocial_INSABI +SeguridadSocial_ISSSTE + SeguridadSocial_Ninguna +SeguridadSocial_Otro + Enfermedad_AAntifosfolípidos +Enfermedad_ArtritisReumatoide + Enfermedad_ArtrititsIdiopáticaJ +Enfermedad_Esclerodermia + Enfermedad_EspondilitisAnquilosante +Enfermedad_Gota + Enfermedad_Lupus +Enfermedad_MiopatíaInflamatoria + Enfermedad_Osteoartrosis +Enfermedad_Otro + Enfermedad_Sjögren + Enfermedad_VasculitisANCA +comorbilidad_Depresión + comorbilidad_EnfermedadPulmonar +comorbilidad_HipertensiónArterialSistémica +comorbilidad_OtraCardiovascular + comorbilidad_OtrasGástricas +comorbilidad_Otro + comorbilidad_Ulcera +comorbilidad_VascularCerebral')
#     formula = ('D ~ Enfermedad_AAntifosfolípidos +Enfermedad_ArtritisReumatoide + Enfermedad_ArtrititsIdiopáticaJ +Enfermedad_Esclerodermia + Enfermedad_EspondilitisAnquilosante +Enfermedad_Gota + Enfermedad_Lupus +Enfermedad_MiopatíaInflamatoria + Enfermedad_Osteoartrosis +Enfermedad_Otro + Enfermedad_Sjögren + Enfermedad_VasculitisANCA +comorbilidad_Depresión + comorbilidad_EnfermedadPulmonar +comorbilidad_HipertensiónArterialSistémica +comorbilidad_OtraCardiovascular + comorbilidad_OtrasGástricas +comorbilidad_Otro + comorbilidad_Ulcera +comorbilidad_VascularCerebral')
#     model = logit(formula = formula, data = train_data).fit()
#     dataset_results = results_summary_to_dataframe(model)
#     # results_summary = model.summary()
#     # results_as_html = results_summary.tables[1].as_html()
#     # pd.read_html(results_as_html, header=0, index_col=0)[0]
#     file_O = Path("Data/out/" + name + ".xlsx")
#     file_save = cwd / file_O
#     dataset_results.to_excel(file_save) 


# for name, values in dataset_dependents.iteritems():
#     dataset_dependent = dataset_dependents.loc[:,name]
#     #dataset_dependent = dataset_dependents.iloc[:,0]
#     columnset = np.r_[32,35]
#     dataset_independent= data_I_HGM.iloc[:,columnset]
#     dataset_independent=dataset_independent[list_del]
#     dataset_test = pd.concat([dataset_dependent, dataset_independent], axis=1)
#     dataset_test.rename(columns={ dataset_test.columns[0]: "D" }, inplace = True)
#     dataset_test.rename(columns={"R62": "Vacunado", "R65": "DosVacunas"}, inplace = True)
#     train_data, test_data = train_test_split(dataset_test, test_size=0.20, random_state= 42)
#     formula = ('D ~ Vacunado + DosVacunas')
#     model = logit(formula = formula, data = train_data).fit()
#     dataset_results = results_summary_to_dataframe(model)
#     file_O = Path("Data/out/" + name + ".xlsx")
#     file_save = cwd / file_O
#     dataset_results.to_excel(file_save) 

# for name, values in dataset_dependents.iteritems():
#     dataset_dependent = dataset_dependents.loc[:,name]
#     #dataset_dependent = dataset_dependents.iloc[:,0]
#     columnset = np.r_[67,69]
#     dataset_independent= data_I_HGM.iloc[:,columnset]
#     dataset_independent=dataset_independent[list_del]
#     dataset_test = pd.concat([dataset_dependent, dataset_independent], axis=1)
#     dataset_test.rename(columns={ dataset_test.columns[0]: "D" }, inplace = True)
#     dataset_test.rename(columns={"R105": "Influenza", "R149": "SistemaInmune"}, inplace = True)
#     dataset_test = pd.get_dummies(dataset_test, prefix=['Influenza', 'SistemaInmune'], columns=['Influenza', 'SistemaInmune'])
#     train_data, test_data = train_test_split(dataset_test, test_size=0.20, random_state= 42)
#     formula = ('D ~ Influenza_1_2_veces + Influenza_3_4_veces + Influenza_3_4_veces  + Influenza_NingunaVez + Influenza_TodoslosAños + SistemaInmune_1 + SistemaInmune_2 +  SistemaInmune_3 + SistemaInmune_4 + SistemaInmune_5')
#     model = logit(formula = formula, data = train_data).fit()
#     dataset_results = results_summary_to_dataframe(model)
#     file_O = Path("Data/out/" + name + ".xlsx")
#     file_save = cwd / file_O
#     dataset_results.to_excel(file_save) 