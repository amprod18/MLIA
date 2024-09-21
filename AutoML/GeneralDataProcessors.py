import numpy as np
import pandas as pd
from AutoML.DataParsers import Data2DtoDataFrame
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, FunctionTransformer, QuantileTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.feature_selection import f_classif
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import shapiro, normaltest, skew, kurtosis, chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Typing annotations
from pandas import Series, DataFrame
from numpy import ndarray
from typing import Self, Union, Dict, Callable, Optional


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    FrequencyEncoder: Encodes categorical variables based on their frequency.

    Parameters:
    - None

    Attributes:
    - freq_dict_: Dictionary mapping categories to their frequencies
    """

    def __init__(self) -> None:
        self.parser:Data2DtoDataFrame = Data2DtoDataFrame()
        self.freq_dict_ = {}

    def fit(self, data:DataFrame|Series|ndarray) -> Self:
        """
        Fit the FrequencyEncoder to the data.

        Parameters:
        - X: pandas Series or DataFrame

        Returns:
        - self
        """
        data = self.parser.parse(data)
        for col in data.columns:
            self.freq_dict_[col] = data[col].value_counts(normalize=True).to_dict()

        return self

    def transform(self, data:DataFrame|Series|ndarray) -> DataFrame:
        """
        Transform the data using the frequency encoding.

        Parameters:
        - X: pandas Series or DataFrame

        Returns:
        - transformed data
        """
        data = self.parser.parse(data)
        for col in data.columns:
            data.loc[:, col] = data[col].map(self.freq_dict_[col])

        return data

    def fit_transform(self, X:DataFrame|Series|ndarray) -> DataFrame:
        """
        Fit the FrequencyEncoder to the data and transform it.

        Parameters:
        - X: pandas Series or DataFrame

        Returns:
        - transformed data
        """
        return self.fit(X).transform(X)


class GeneralImputer(BaseEstimator, TransformerMixin):
    def __init__(self, transformation=None, missing_threshold: float=0.2):
        self.parser:Data2DtoDataFrame = Data2DtoDataFrame()
        self.global_transform:str = transformation
        self.imputers_: dict[str, TransformerMixin] = {}
        self.missing_threshold = missing_threshold
    
    def _autofit(self, data:DataFrame, columns:list[str]) -> DataFrame:
        """
            Automatically decide the best imputation value for a column based on its properties.
            
            Parameters
            ----------
            series : pd.Series
                Column for which to decide the imputation value.
            
            Returns
            -------
            imputation_value : Union[float, str, pd.Timestamp]
                The value used for imputation.
        """
        for col in columns:
            missing_ratio = data[[col]].isna().mean().values[0]
            if missing_ratio == 0:
                self.imputers_[col] = None
                continue

            # Handle numerical columns
            if pd.api.types.is_numeric_dtype(data[col]):
                if missing_ratio > self.missing_threshold:
                    # If too many values are missing, use a constant (e.g., 0 or -1)
                    imputer = SimpleImputer(strategy='constant', fill_value=-1 if data[[col]].min() < 0 else 0)
                else:
                    skewness = skew(data[col])
                    if abs(skewness) > 0.5:
                        # Skewed data -> Use median (robust to outliers)
                        imputer = SimpleImputer(strategy='median')
                    else:
                        # Normally distributed -> Use mean
                        imputer = SimpleImputer(strategy='mean')

            # Handle categorical columns
            elif pd.api.types.is_object_dtype(data[col]):
                if missing_ratio > self.missing_threshold:
                    # Too many missing values -> use placeholder
                    imputer = SimpleImputer(strategy='constant', fill_value='Missing')
                else:
                    # Use mode (most frequent value)
                    imputer = SimpleImputer(strategy='most_frequent')

            # Handle datetime columns
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                if missing_ratio > self.missing_threshold:
                    # Default to epoch if too many missing values
                    imputer = SimpleImputer(strategy='constant', fill_value=pd.Timestamp("1970-01-01"))
                else:
                    # Use median date if reasonable
                    imputer = SimpleImputer(strategy='median')
            else:
                # Default to using a constant for any unknown types
                imputer = SimpleImputer(strategy='constant', fill_value='Unknown')

            imputer.fit(data[[col]])
            self.imputers_[col] = imputer
        return self
    
    def fit(self, data: DataFrame|Series|ndarray, column_transforms:dict[str, TransformerMixin]=None) -> Self:
        data = self.parser.parse(data)
        
        if column_transforms:
            # Apply column-specific transformations
            for column, imputer in column_transforms.items():
                imputer.fit(data[column])
                self.imputers_[column] = imputer
            untransformed_columns = [col for col in data.columns if col not in column_transforms]
            if self.global_transform == 'autofill':
                data[untransformed_columns] = self._autofit(data, untransformed_columns)
            else:
                if untransformed_columns:
                    print(f"Warning: Columns {untransformed_columns} do not have a transformation assigned.")
        else:
            data = self._autofit(data, data.columns)

        return self

    def transform(self, data: DataFrame|Series|ndarray) -> DataFrame:
        data = self.parser.parse(data)
        assert not not self.imputers_, "You must fit the scaler before transforming"
        
        imputed_data = data.copy()
        for col, imputer in self.imputers_.items():
            if imputer is None:
                continue
            imputed_data.loc[:, col] = imputer.transform(data[[col]])
        
        untransformed_columns = [col for col in data.columns if (col not in self.imputers_.keys()) and (data[col].isna().sum() > 0)]
        if untransformed_columns:
            print(f"Warning: Columns {untransformed_columns} do not have a transformation assigned.")
        
        return imputed_data

    def fit_transform(self, data: DataFrame|Series|ndarray) -> DataFrame:
        return self.fit(data).transform(data)
 

class GeneralEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, transform:str|None=None):
        self.parser:Data2DtoDataFrame = Data2DtoDataFrame()
        self.encoders_ = {}
        self.global_transform = transform

    def _autofit(self, data:DataFrame, target_data:DataFrame, threshold:int=10, task_threshold:int=20) -> Self:
        """
        Automatically select the best encoder for a given categorical or datetime column based on its properties.
        
        Parameters
        ----------
        column : pd.Series
            The column for which to select the encoder.
        target : pd.Series, optional
            The target variable for supervised learning tasks. If None, encoding is based on column characteristics alone.
        task : str, optional
            The task type, either 'classification' or 'regression'. Used to select the statistical test. Default is 'classification'.
        
        Returns
        -------
        Callable
            The encoder class or function to be applied to the column.
        """
        unique_values = int(target_data.nunique().values[0])
    
        # Check data type and number of unique values
        if pd.api.types.is_integer_dtype(target_data) or pd.api.types.is_object_dtype(target_data):
            # Discrete values (likely classification)
            if unique_values <= task_threshold:
                task = 'classification'
            else:
                task = 'regression'
        elif pd.api.types.is_float_dtype(target_data) or (unique_values > 20):
            # Continuous values (likely regression)
            task = 'regression'
        else:
            # Fall back to classification if unsure
            task = 'classification'
    
        # Find categorical columns
        cat_cols = data.select_dtypes(include=['object', 'category', 'datetime64']).columns

        for col in cat_cols:
            # If the column is datetime, perform datetime feature extraction
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                encoder =  FunctionTransformer(lambda X: pd.concat([X.dt.year.rename(f'{col}_year'), X.dt.month.rename(f'{col}_month'),
                                                                    X.dt.day.rename(f'{col}_day'), X.dt.weekday.rename(f'{col}_weekday'),
                                                                    X.dt.hour.rename(f'{col}_hour') if hasattr(X.dt, f'{col}_hour') else None], axis=1), validate=False)

            # If the column is categorical, analyze the cardinality and relationship with target (if provided)
            unique_values = data[col].nunique()

            # 1. Low cardinality: Use OneHotEncoder
            if unique_values == 2:
                encoder = LabelEncoder()
                
            elif unique_values <= threshold:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

            # 2. High cardinality: Apply statistical tests to decide between LabelEncoder and FrequencyEncoder
            elif task == 'classification':
                # Perform chi-squared test for independence (categorical target)
                contingency_table = pd.crosstab(data[col].to_numpy().flatten(), target_data.to_numpy().flatten())
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                if p_value < 0.05:
                    # Significant relationship, prefer FrequencyEncoder
                    encoder = FrequencyEncoder()
                else:
                    # No significant relationship, use LabelEncoder
                    encoder = LabelEncoder()
            elif task == 'regression':
                # Perform ANOVA test (categorical feature vs continuous target)
                categories = [data[col] == category for category in data[col].unique()]
                f_value, p_value = f_classif(categories, target_data)
                if p_value < 0.05:
                    # Significant relationship, prefer FrequencyEncoder
                    encoder = FrequencyEncoder()
                else:
                    # No significant relationship, use LabelEncoder
                    encoder = LabelEncoder()

            encoder.fit(data[[col]])
            self.encoders_[col] = encoder
        return self
    
    def fit(self, data: DataFrame|Series|ndarray, target_data: DataFrame|Series|ndarray, column_transforms:dict[str, TransformerMixin]=None) -> Self:
        """
        Fit the encoder to the input data.

        Args:
            data: Input data in one of the following formats:
                - pandas DataFrame
                - pandas Series
                - 2D NumPy array

        Returns:
            self
        """
        data = self.parser.parse(data)
        target_data = self.parser.parse(target_data)

        if column_transforms:
            # Apply column-specific transformations
            for column, encoder in column_transforms.items():
                encoder.fit(data[column])
                self.encoders_[column] = encoder
            untransformed_columns = [col for col in data.select_dtypes(include=['object', 'category']).columns if col not in self.encoders_.keys()]
            if self.global_transform == 'autofill':
                data[untransformed_columns] = self._autofit(data, target_data, untransformed_columns)
            else:
                if untransformed_columns:
                    print(f"Warning: Columns {untransformed_columns} do not have a transformation assigned.")
        else:
            data = self._autofit(data, target_data)

        return self

    def transform(self, data: DataFrame|Series|ndarray) -> DataFrame:
        """
        Transform the input data using the fitted encoder.

        Args:
            data: Input data in one of the following formats:
                - pandas DataFrame
                - pandas Series
                - 2D NumPy array

        Returns:
            A pandas DataFrame with the encoded data.
        """
        data = self.parser.parse(data)
        assert not not self.encoders_, "You must fit the scaler before transforming"
        
        encoded_data = data.copy()
        for col, encoder in self.encoders_.items():
            if isinstance(encoder, OneHotEncoder):
                new_cols = encoder.get_feature_names_out().tolist()
                encoded_data.loc[:, new_cols] = encoder.transform(data[[col]])
            else:
                encoded_data.loc[:, col] = encoder.transform(data[[col]])
            
            encoded_data.drop(columns=col, inplace=True)
        untransformed_columns = data.select_dtypes(include=['object', 'category']).columns
        if not untransformed_columns.empty:
            print(f"Warning: Columns {untransformed_columns} do not have a transformation assigned.")
        
        return encoded_data

    def fit_transform(self, data: DataFrame|Series|ndarray, target_data: DataFrame|Series|ndarray) -> DataFrame:
        """
        Fit the encoder to the input data and transform it.

        Args:
            data: Input data in one of the following formats:
                - pandas DataFrame
                - pandas Series
                - 2D NumPy array

        Returns:
            A pandas DataFrame with the encoded data.
        """
        return self.fit(data, target_data).transform(data)
    

class GeneralScaler(BaseEstimator, TransformerMixin):
    def __init__(self, transformation=None):
        self.parser:Data2DtoDataFrame = Data2DtoDataFrame()
        self.global_transform:str = transformation
        self.scalers_: dict[str, TransformerMixin] = {}

    def _get_outliers_percent(self, data:Series) -> int:
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        return outliers
    
    def _autofit(self, data:DataFrame, columns:list[str]) -> DataFrame:
        """
        Automatically select the best scaler for a given column based on its statistical properties.

        Parameters
        ----------
        column : Series
            The column for which to select the scaler.

        Returns
        -------
        Callable
            The scaler class to be applied to the column.
        """
        for col in columns:
            # 1. Check for normality using Shapiro-Wilk Test
            stat, p_value = shapiro(data[col])
            is_normal = p_value > 0.05
            # 2. Check for skewness
            data_skewness = skew(data[col])
            is_skewed = np.abs(data_skewness) > 0.5  # Threshold for considering data skewed
            # 3. Check for outliers using IQR (Interquartile Range)
            outliers = self._get_outliers_percent(data[col])
            has_many_outliers = outliers > 0.05 * len(data[col])  # Consider if more than 5% are outliers
            kurto = abs(kurtosis(data[col]))
            has_heavy_tails = kurto > 3  # Consider if kurtosis > 3

            # 4. Selection Logic:
            if is_normal and not has_many_outliers:
                # If normally distributed and no many outliers, use StandardScaler
                scaler = StandardScaler()
            elif is_skewed and (data[col] > 0).all():
                # If data is skewed and all values are positive, use Log transformation
                scaler = FunctionTransformer(np.log1p, validate=True)
            elif data_skewness > 0.5:
                # If positively skewed and data contains non-positive values, shift data then log-transform
                scaler =  FunctionTransformer(lambda x: np.log1p(x - x.min() + 1), validate=True)
            elif data_skewness < -0.5:
                # If negatively skewed, reflect and apply log transformation or use PowerTransformer
                scaler =  FunctionTransformer(lambda x: np.log1p(x.max() - x + 1), validate=True)
            elif has_heavy_tails or has_many_outliers:
                # If the data has heavy tails (high kurtosis) or many outliers, use QuantileTransformer
                scaler =  QuantileTransformer()
            elif has_many_outliers:
                # If many outliers, use RobustScaler
                scaler =  RobustScaler()
            else:
                # Default to MinMaxScaler for everything else
                scaler = MinMaxScaler()

            scaler.fit(data[[col]])
            self.scalers_[col] = scaler
        return self
    
    def fit(self, data: DataFrame|Series|ndarray, column_transforms:dict[str, TransformerMixin]=None) -> Self:
        data = self.parser.parse(data)
        
        if column_transforms:
            # Apply column-specific transformations
            for column, scaler in column_transforms.items():
                scaler.fit(data[column])
                self.scalers_[column] = scaler
            untransformed_columns = [col for col in data.columns if col not in column_transforms]
            if self.global_transform == 'autofill':
                data[untransformed_columns] = self._autofit(data, untransformed_columns)
            else:
                if untransformed_columns:
                    print(f"Warning: Columns {untransformed_columns} do not have a transformation assigned.")
        else:
            data = self._autofit(data, data.columns)

        return self

    def transform(self, data: DataFrame|Series|ndarray) -> DataFrame:
        data = self.parser.parse(data)
        assert not not self.scalers_, "You must fit the scaler before transforming"
        
        scaled_data = data.copy()
        for col, scaler in self.scalers_.items():
            scaled_data.loc[:, f'{col}_float'] = scaler.transform(data[[col]])
            scaled_data.drop(columns=col, inplace=True) 
            scaled_data.rename(columns={f'{col}_float':col}, inplace=True) 
        
        untransformed_columns = [col for col in data.columns if col not in self.scalers_.keys()]
        if untransformed_columns:
            print(f"Warning: Columns {untransformed_columns} do not have a transformation assigned.")
        
        return scaled_data

    def fit_transform(self, data: DataFrame|Series|ndarray) -> DataFrame:
        return self.fit(data).transform(data)