import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from sklearn.model_selection import train_test_split, LearningCurveDisplay, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, root_mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .GeneralDataProcessors import TaskType, GeneralDataProcessor

# Typing annotations
from pandas import Series, DataFrame
from numpy import ndarray
from typing import Self, Union, Dict, Callable, Optional

class AutoML:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def preprocess_data(self):
        # Data preprocessing tasks
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)

    def engineer_features(self):
        # Feature engineering tasks
        selector = SelectKBest(k=5)
        self.data = selector.fit_transform(self.data, self.target)

    def select_model(self):
        # Model selection tasks
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model

    def tune_hyperparameters(self, model):
        # Hyperparameter tuning tasks
        # Implement hyperparameter tuning using Optuna or other libraries
        pass

    def evaluate_model(self, model):
        # Model evaluation tasks
        y_pred = model.predict(self.data)
        accuracy = accuracy_score(self.target, y_pred)
        return accuracy
    
    
# FOR FUTURE USE

def create_imputation_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """
    Create a ColumnTransformer pipeline that imputes missing values based on data type.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with potential missing values.
    
    Returns
    -------
    ColumnTransformer
        A ColumnTransformer object that applies the appropriate imputation strategy
        to each column.
    """
    # Separate columns by data type
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # Define numeric imputation strategy (mean, median, etc.)
    numeric_transformer = Pipeline(steps=[
        ('imputer', GeneralImputer(strategy='auto'))
    ])
    
    # Define categorical imputation strategy (mode or constant like "Missing")
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Define datetime imputation strategy (default to constant date)
    datetime_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=pd.Timestamp('1970-01-01')))
    ])
    
    # Combine all transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
            ('date', datetime_transformer, datetime_cols)
        ],
        remainder='passthrough'  # Keep any other columns as they are
    )
    
    return preprocessor

class ModelTester:
    def __init__(self, train_data:DataFrame|ndarray|Series, target_column:list[str]|str, index_column:list[str]|str|None=None, eval_data:DataFrame|ndarray|Series|None=None, test_size:float=0.2) -> None:
        self.test_size = test_size
        self.data_processor = GeneralDataProcessor()
        self.TARGET = target_column
        self.INDEX = index_column
        if eval_data is None:
            train_data, eval_data = train_test_split(train_data, test_size=self.test_size)
            
        self.X_train, self.y_train = self.data_processor.fit_transform(train_data, self.TARGET, index_column=self.INDEX)
        self.X_eval, self.y_eval = self.data_processor.transform(eval_data)
        if isinstance(self.TARGET, str):
            self.y_train = self.y_train.to_numpy().flatten()
            self.y_eval = self.y_eval.to_numpy().flatten()
        elif isinstance(self.TARGET, list):
            print("Error: Multitarget problems are stil under construction")
            raise
        else:
            print("Error: where tf you going with that?")
            
            
    
    def train_models(self) -> Self:
        if self.data_processor.task_type == TaskType.CLASSIFICATION:
            self.logistic_regression_model, self.logistic_regression_score = self.train_logistic_regression()
        elif self.data_processor.task_type == TaskType.REGRESSION:
            self.linear_regression_model, self.linear_regression_score = self.train_linear_regression()
        return self
    
    # ----------| Regression Tasks |----------
    def train_linear_regression(self) -> tuple[LinearRegression, float]:
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        score = model.score(self.X_eval, self.y_eval)
        return model, score
    
    def train_decission_tree_regressor(self) -> tuple[DecisionTreeRegressor, float]:
        model = DecisionTreeRegressor()
        model.fit(self.X_train, self.y_train)
        score = model.score(self.X_eval, self.y_eval)
        return model, score
    
    def train_random_forest_regressor(self) -> tuple[RandomForestRegressor, float]:
        model = RandomForestRegressor()
        model.fit(self.X_train, self.y_train)
        score = model.score(self.X_eval, self.y_eval)
        return model, score
    
    def train_xgboost_regressor(self) -> tuple[XGBRegressor, float]:
        model = XGBRegressor()
        model.fit(self.X_train, self.y_train)
        score = model.score(self.X_eval, self.y_eval)
        return model, score
    
    def train_catboost_regressor(self) -> tuple[CatBoostRegressor, float]: # FIXME:
        model = CatBoostRegressor()
        model.fit(self.X_train, self.y_train)
        score = model.score(self.X_eval, self.y_eval)
        return model, score
    
    # ----------| Classification Tasks |----------
    def train_logistic_regression(self) -> tuple[LogisticRegression, float]:
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        score = model.score(self.X_eval, self.y_eval)
        return model, score
    
    def train_decision_tree_classifier(self) -> tuple[DecisionTreeClassifier, float]:
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        score = model.score(self.X_eval, self.y_eval)
        return model, score
    
    def train_random_forest_classifier(self) -> tuple[RandomForestClassifier, float]:
        model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)
        score = model.score(self.X_eval, self.y_eval)
        return model, score
    
    def train_xgboost_classifier(self) -> tuple[XGBClassifier, float]:
        model = XGBClassifier()
        model.fit(self.X_train, self.y_train)
        score = model.score(self.X_eval, self.y_eval)
        return model, score
    
    def train_catboost_classifier(self) -> tuple[CatBoostClassifier, float]: # FIXME:
        model = CatBoostClassifier()
        model.fit(self.X_train, self.y_train)
        score = model.score(self.X_eval, self.y_eval)
        return model, score