import numpy as np
import pandas as pd
from typing import Union

class PSI(object):

    def __init__(
                self, 
                data_expected: pd.DataFrame,
                data_actual: pd.DataFrame,
                corr = 1e-4
    ) -> None:

        self.data_expected = data_expected
        self.data_actual = data_actual
        self.corr = corr

    @staticmethod
    def _scale_range (
                    input: np.ndarray, 
                    min: np.ndarray, 
                    max: float
    ):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    @staticmethod
    def _sub_psi(
                e_perc: float, 
                a_perc: float, 
                corr: float
    ):
        '''Calculate the actual PSI value from comparing the values.
           Update the actual value to a very small number if equal to zero
        '''
        if a_perc == 0:
            a_perc = corr
        if e_perc == 0:
            e_perc = corr

        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return(value)

    def psi_numeric(
                    self,
                    column: str,
                    buckets: int = 10, 
                    buckettype: str = 'bins'
    ):
        '''Calculate the PSI for a single numerical variable
        Args:
           column:
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''
        expected = self.data_expected.loc[:, column]
        actual = self.data_actual.loc[:, column]

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = self._scale_range(breakpoints, np.min(expected), np.max(expected))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected, b) for b in breakpoints])

        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

        psi_value = np.sum(self._sub_psi(expected_percents[i], actual_percents[i], self.corr) for i in range(0, len(expected_percents)))

        return psi_value


    def psi_category(
                    self,
                    column: str
    ): 
        expected = self.data_expected.loc[:, column].value_counts()/len(self.data_expected.loc[:, column])
        actual = self.data_actual.loc[:, column].value_counts()/len(self.data_actual.loc[:, column])
        if expected.index.equals(actual.index):
            psi_value = sum(self._sub_psi(expected[index], actual[index], self.corr) for index in expected.index)
        else:
            for index_null in expected.index.difference(actual.index):
                actual[index_null] = self.corr
            for index_null in actual.index.difference(expected.index):
                expted[index_null] = self.corr
            sum(self._sub_psi(expected[index], actual[index], self.corr) for index in expected.index)
        return psi_value

    def calculate_psi_report(
                            self,
                            columns_num: list, 
                            columns_cat: list=[],
                            buckettype: str='bins',
                            buckets=10
    ):
        """
        Calculate PSI values for each variable given in arguments
             Args:
               expected: numpy array of original values
               actual: numpy array of new values, same size as expected
               buckets: number of percentile ranges to bucket the values into
            Returns:
               psi_value: calculated PSI value
        """
        try:
            psi_values_num = [[col, self.psi_numeric(col, buckets, buckettype)] for col in columns_num]
            psi_values_cat = [[col, self.psi_category(col)] for col in columns_cat]
        except TypeError:
            print("Passed column name refers to wrong data type")
            raise
        psi_values = [*psi_values_num, *psi_values_cat]
        return pd.DataFrame(psi_values, columns=["Variable", "PSI"])

