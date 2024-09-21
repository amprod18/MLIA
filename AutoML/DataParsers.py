import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array

from pandas import DataFrame, Series
from numpy import ndarray

class Data2DtoDataFrame:
    def __init__(self) -> None:
        pass
    
    @classmethod
    def parse(self, data: DataFrame|Series|ndarray) -> DataFrame:
        """
        Parse input data into a pandas DataFrame.

        Args:
            data: Input data in one of the following formats:
                - pandas DataFrame
                - pandas Series
                - 2D NumPy array

        Returns:
            A pandas DataFrame representation of the input data.
        """
        if isinstance(data, DataFrame):
            return data
        elif isinstance(data, Series):
            return data.to_frame()
        elif isinstance(data, ndarray):
            data = check_array(data, ensure_2d=True)
            return pd.DataFrame(data)
        else:
            raise ValueError("Unsupported input type. Only pandas DataFrames, Series, and 2D NumPy arrays are supported.")
