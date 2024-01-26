from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class weekday_imputter(BaseEstimator, TransformerMixin):
    """weekday column Imputer"""

    def __init__(self, variables: str):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")
        if variables != "weekday":
            raise ValueError("incorrect column used")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        wkday_null_idx = X[X['weekday'].isnull() == True].index
        X.loc[wkday_null_idx, 'weekday'] = X.loc[wkday_null_idx, 'dteday'].dt.day_name().apply(lambda x: x[:3])
        X.drop(labels=['dteday'], axis=1, inplace=True)
   
        return X

class weathersit_imputter(BaseEstimator, TransformerMixin):
    """weathersit column Imputer"""

    def __init__(self, variables: str):
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")
        if variables != "weathersit":
            raise ValueError("incorrect column used")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables].fillna('Clear', inplace=True)
   
        return X


class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""

    def __init__(self, variables: str, mappings: dict):
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X

class get_year_and_month(BaseEstimator, TransformerMixin):
    """Convert string to date and add year, month feature"""
    
    def __init__(self, variables: str):
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")
        if variables != "dteday":
            raise ValueError("incorrect column used")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # convert 'dteday' column to Datetime datatype
        X['dteday'] = pd.to_datetime(X['dteday'], format='%Y-%m-%d')
        # Add new features 'yr' and 'mnth
        X['yr'] = X['dteday'].dt.year
        X['mnth'] = X['dteday'].dt.month_name()

        return X

class handle_outliers(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
    
    def fit(self, df: pd.DataFrame, y: pd.Series = None):
        q1 = df.describe()[self.variables].loc['25%']
        q3 = df.describe()[self.variables].loc['75%']
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for i in df.index:
            if df.loc[i,self.variables] > self.upper_bound:
                df.loc[i,self.variables]= self.upper_bound
            if df.loc[i,self.variables] < self.lower_bound:
                df.loc[i,self.variables]= self.lower_bound    
        
        return df
        
class one_hot_encoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables: str):
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")
        self.variables = variables
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.ohe = OneHotEncoder(sparse_output=False)
        self.ohe.fit(X[[self.variables]])
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_trans = self.ohe.transform(X[[self.variables]])
        enc_features = self.ohe.get_feature_names_out([self.variables])
        X[enc_features] = X_trans
        X.drop(labels=[self.variables], axis=1, inplace=True)
        
        return X