from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_predict
import numpy as np

class OOFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, meta_model):
        """
        Initializes the OOFClassifier with base models and a meta model.
        
        Args:
            base_models (list): List of base models (must support `fit` and `predict_proba`).
            meta_model: Meta model (must support `fit` and `predict`).
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
            model.fit(X, y)    # Train the base model on the entire dataset
            oof_preds = cross_val_predict(model, X, y, cv=5, method="predict_proba", n_jobs=-1)  # Generate out-of-fold predictions
            meta_features.append(oof_preds)

        # Concatenate meta features with original features
        meta_features = np.hstack([X] + meta_features)
        self.meta_model.fit(meta_features, y)    # Train the meta model on the combined dataset

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

        meta_features = np.hstack([X] + meta_features)    # Concatenate meta features with original features
        return self.meta_model.predict(meta_features)    # Predict using the meta model
        

class OOFRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model):
        """
        Initializes the OOFRegressor with base models and a meta model.
        
        Args:
            base_models (list): List of base models (must support `fit` and `predict`).
            meta_model: Meta-model (must support `fit` and `predict`).
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
            model.fit(X, y)    # Train the base model on the entire dataset
            oof_preds = cross_val_predict(model, X, y, cv=5, method="predict", n_jobs=-1)  # Generate out-of-fold predictions
            meta_features.append(oof_preds.reshape(-1, 1))

        # Concatenate meta features with original features
        meta_features = np.hstack([X] + meta_features)
        self.meta_model.fit(meta_features, y)    # Train the meta model on the combined dataset

    def predict(self, X):
        """
        Predicts the target using the trained base models and meta model.
        
        Args:
            X (numpy.ndarray): Feature matrix.
        
        Returns:
            numpy.ndarray: Predicted values.
        """
        meta_features = []

        for model in self.base_models:
            preds = model.predict(X)  # Use trained base models to generate predictions
            meta_features.append(preds.reshape(-1, 1))

        # Concatenate meta features with original features
        meta_features = np.hstack([X] + meta_features)
        return self.meta_model.predict(meta_features)    # Predict using the meta model
        
