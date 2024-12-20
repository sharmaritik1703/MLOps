import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_predict

class OOFClassifier:
    def __init__(self, base_models: list, meta_model):
        """
        Initializes the OOFClassifier with base models and a meta model.
        
        Args:
            base_models (list): List of base models (must support `fit` and `predict_proba`), e.g.  [Random Forest, Gradient Boosting]
            meta_model: Meta model (must support `fit` and `predict`).   e.g. [Logistic Regression, SVM]
        """
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        """
        Trains the base models using cross-validation and the meta model on the out-of-fold predictions.
        
        Args:
            X (numpy.ndarray): Feature matrix.
            y (numpy.ndarray): Target vector.
        """
        meta_features = []

        for model in self.base_models:
            model.fit(X, y)  # Fit the base model on the full dataset
            oof_preds = cross_val_predict(model, X, y, cv=5, method="predict_proba", n_jobs=-1)  # Generate out-of-fold predictions
            meta_features.append(oof_preds)
            

        # Concatenate all meta features
        meta_features = np.hstack([X], meta_features)
        self.meta_model.fit(meta_features, y)  # Train the meta model

    def predict(self, X):
        """
        Predicts the target using the trained base models and meta model.
        
        Args:
            X (numpy.ndarray): Feature matrix.
        
        Returns:
            numpy.ndarray: Predicted labels.
        """
        meta_features = []

        for model in self.base_models:
            preds = model.predict_proba(X)  # Use trained base models to generate predictions
            meta_features.append(preds)

        # Concatenate all meta features
        meta_features = np.hstack([X], meta_features)
        return self.meta_model.predict(meta_features)



class OOFRegressor:
    def __init__(self, base_models: list, meta_model):
        """
        Initializes the OOFRegressor with base models and a meta model.
        
        Args:
            base_models (list): List of base models (must support `fit` and `predict`), e.g. [Random Forest, GBoost, etc.]
            meta_model: Meta-model (must support `fit` and `predict`), e.g. [Linear Regression, Ridge, Lasso, SVM]
        """
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        """
        Trains the base models using cross-validation and the meta model on the out-of-fold predictions.
        
        Args:
            X (numpy.ndarray): Feature matrix.
            y (numpy.ndarray): Target vector.
        """
        meta_features = []  # List to store OOF predictions from each base model

        for model in self.base_models:
            model.fit(X, y)    # Train the original base model on the entire dataset
            oof_preds = cross_val_predict(cloned_model, X, y, cv=5, method="predict", n_jobs=-1)   # Generate out-of-fold predictions
            meta_features.append(oof_preds.reshape(-1, 1))  # Reshape for concatenation
            
        # Formation of meta-dataset!
        meta_features = np.hstack([X] + meta_features)
        self.meta_model.fit(meta_features, y)

    def predict(self, X):
        """
        Predicts the target using the trained base models and meta model.
        
        Args:
            X (numpy.ndarray): Feature matrix.
        
        Returns:
            numpy.ndarray: Predicted values.
        """
        meta_features = []  # List to store predictions from each base model

        for model in self.base_models:
            preds = model.predict(X)  # Use trained base models to generate predictions
            meta_features.append(preds.reshape(-1, 1))  # Reshape for concatenation

        # Concatenate the original features with the meta-features
        meta_features = np.hstack([X] + meta_features)
        return self.meta_model.predict(meta_features)  # Predict using the meta model
