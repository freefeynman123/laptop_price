import numpy as np
import pandas as pd


def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input


def sub_psi(e_perc, a_perc, corr):
    '''Calculate the actual PSI value from comparing the values.
       Update the actual value to a very small number if equal to zero
    '''
    if a_perc == 0:
        a_perc = corr
    if e_perc == 0:
        e_perc = corr

    value = (e_perc - a_perc) * np.log(e_perc / a_perc)
    return(value)
    
def psi(expected_array, actual_array, buckets, corr):
    '''Calculate the PSI for a single variable
    Args:
       expected_array: numpy array of original values
       actual_array: numpy array of new values, same size as expected
       buckets: number of percentile ranges to bucket the values into
    Returns:
       psi_value: calculated PSI value
    '''


    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

    if buckettype == 'bins':
        breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
    elif buckettype == 'quantiles':
        breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

    psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i], corr) for i in range(0, len(expected_percents)))

    return psi_value



    expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
    actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

def calculate_psi(expected, actual, corr, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets, corr)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets, corr)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets, corr)

    return psi_values

def calculate_psi_category(
                data_expected: pd.DataFrame,
                data_actual: pd.DataFrame
): 
    expected = data_expected.value_counts()/len(data_expected)
    actual = data_actual.value_counts()/len(data_actual)
    if expected.index.equals(actual.index):
        psi_value = sum(sub_psi(expected[index], actual[index], corr) for index in expected.index)
    else:
        for index_null in expected.index.difference(actual.index):
            actual[index_null] = corr
        for index_null in actual.index.difference(expected.index):
            expted[index_null] = corr
        sum(sub_psi(expected[index], actual[index], corr) for index in expected.index)
    return psi_value

def calculate_psi_report(
                        expected, 
                        actual, 
                        columns_num: str, 
                        columns_cat: str,
                        buckettype: str='bins',
                        buckets=10,
                        axis=0
):
    psi_values_num = [[calculate_psi(expected.loc[:, col], actual.loc[:, col], buckettype, buckets, axis), col] for col in columns_num]
    psi_values_cat = [[calculate_psi_category(expected.loc[:, col], actual.loc[:, col], buckettype, buckets, axis), col] for col in columns_cat]
    return [*psi_values_num, *psi_values_cat]

