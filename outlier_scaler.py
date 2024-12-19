import pandas as pd
import numpy as np

class OutlierRemover:
    def __init__(self, start=None, stop=None):
    """
    An object to handle outliers in dataframe.

    args:
        start: index of the column from which outliers needs to be removed
        end: index of the column upto which outliers needs to be removed
    """
        self.upper_bounds = None
        self.lower_bounds = None
        self.start = start
        self.stop = stop
        self.columns_to_scale = None

    def define_columns_to_scale(self, X):
        # Ensure the start and stop indices are within bounds
        if self.start is None or self.start < 0:
            self.start = 0
        if self.stop is None or self.stop > X.shape[1]:
            self.stop = X.shape[1]

        self.columns_to_scale = X.columns[self.start: self.stop]

    def fit(self, X):
        self.define_columns_to_scale(X)
        iqr = X[self.columns_to_scale].quantile(0.75) - X[self.columns_to_scale].quantile(0.25)
        self.upper_bounds = X[self.columns_to_scale].quantile(0.75) + 1.5 * iqr
        self.lower_bounds = X[self.columns_to_scale].quantile(0.25) - 1.5 * iqr

    def transform(self, X):
        if self.upper_bounds is None or self.lower_bounds is None:
            raise Exception("Fit the dataframe first!")

        # Copy the dataframe to avoid modifying original data
        X_scaled = X.copy()

        # Clip the selected columns within bounds
        X_scaled[self.columns_to_scale] = X_scaled[self.columns_to_scale].clip(self.lower_bounds, self.upper_bounds, axis=1)

        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def __str__(self):
        return f"""
        Removes outliers using IQR-percentile method within columns {self.start} to {self.stop}.
        """
