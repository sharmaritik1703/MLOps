import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import r2_score, mean_absolute_error
from imblearn.metrics import specificity_score

# -------------- Outlier Scaling ---------------------
class OutlierScaler:
    def __init__(self, start=None, stop=None):
        """
        An object to handle outliers in the data frame.

        Args:
            start: index of the column from which outliers need to be removed
            end: index of the column up to which outliers need to be removed
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

        X_scaled = X.copy()    # Copy the data frame to avoid modifying the original data
        X_scaled[self.columns_to_scale] = X_scaled[self.columns_to_scale].clip(self.lower_bounds, self.upper_bounds, axis=1) # Clip the selected columns within bounds

        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def __str__(self):
        return f"""
        Removes outliers using IQR-percentile method within columns {self.start} to {self.stop}.
        """

# ---------------- Saving and Loading Models ---------------------
def save_model(model, file_path):
    """
    Saves the machine learning or preprocessing model in a directory.

    Args:
        model: object instance
        file_path: directory file path
    Returns:
        None
    """
    with open(file_path, mode='wb') as my_file:
        pickle.dump(model, my_file)


def load_model(file_path):
    """
    Loads the ML model or preprocessing model from the directory.

    Args:
        file_path: directory file path
    Returns:
        model
    """
    with open(file_path, mode='rb') as my_file:
        model = pickle.load(my_file)

    return model


# ------------ Hypertuning ML models ------------------
def get_hypertuned_model(X, y, model, param_grid, score_metric, imbalance: bool = False):
    """
    Get the optimal version of a machine learning model using its hyperparameters.

    Args:
        X: features of the training set
        y: labels corresponding to features in the training set
        model: ML model instance (e.g., logistic regression, linear regression, etc.)
        param_grid: A dictionary containing parameters and their possible values.
        score_metric: The evaluation metrics for the task (e.g., accuracy, f1 score, MAE, RMSE)
        imbalance: False (for regression and balanced classes). You can set true for imbalance

    Returns:
        The hyper-tuned model, along with the optimal parameters
    """
    # Selecting cross-validation type based upon data imbalance
    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=10) if imbalance else KFold(n_splits=5, shuffle=True, random_state=10)
    
    # Hyperparameter search instance! 
    search = RandomizedSearchCV(model, param_grid, cv=k_fold, scoring=score_metric, random_state=10, n_jobs=-1)
    search.fit(X, y)    # Training
    return search.best_estimator_, search.best_params_


# --------- Evaluating ML models -----------------------
def get_cross_validation_scores(X, y, model, task: str):
    """
    Calculates the performance of the ML model in each fold (cross-validation). It is usually evaluated for the training set in which the model is trained.

    Args:
        X: input features
        y: output labels
        model: trained model instance (scikit-learn)
        task: 'binary-class', 'multi-class', or 'regression'

    Returns:
        A dictionary containing all metrics
    """
    if task == 'binary-class':
        accuracy_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        precision_scores = cross_val_score(model, X, y, cv=5, scoring='precision', n_jobs=-1)
        recall_scores = cross_val_score(model, X, y, cv=5, scoring='recall', n_jobs=-1)
        f1_scores = cross_val_score(model, X, y, cv=5, scoring='f1', n_jobs=-1)
        roc_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
        return {"Mean Accuracy": np.mean(accuracy_scores), "Mean Precision": np.mean(precision_scores), 
                "Mean Recall": np.mean(recall_scores), "Mean F1 Score": np.mean(f1_scores), "Mean ROC Area": np.mean(roc_scores)}
        
    elif task == 'multi-class':
        accuracy_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        precision_scores = cross_val_score(model, X, y, cv=5, scoring='precision_macro', n_jobs=-1)
        recall_scores = cross_val_score(model, X, y, cv=5, scoring='recall_macro', n_jobs=-1)
        f1_scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro', n_jobs=-1)
        return {"Mean Accuracy": np.mean(accuracy_scores), "Mean Precision": np.mean(precision_scores), 
                "Mean Recall": np.mean(recall_scores), "Mean F1 Score": np.mean(f1_scores)}

    else:
        r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
        mae_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        return {"Mean R2 Score": np.mean(r2_scores), "Mean MAE": np.mean(-mae_scores)}


def get_evaluation_metrics(X, y, model, task):
    """
    Calculates average performance of ML model. It is usually preferred for the testing set.

    Args:
        X: Features
        y: labels
        model: trained model instance (scikit-learn)
        task: 'binary-class', 'multi-class', or 'regression'

    Returns:
        A dictionary containing all metrics
    """
    
    if task == 'binary-class':
        accuracy = accuracy_score(y, model.predict(X))
        precision = precision_score(y, model.predict(X))
        recall = recall_score(y, model.predict(X))
        specificity = specificity_score(y, model.predict(X))
        f1 = f1_score(y, model.predict(X))
        roc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        return {"Accuracy": accuracy, "Precision": precision, "Sensitivity": recall, "Specificity": specificity, "F1 Score": f1, "ROC Area": roc}
        
    elif task == 'multi-class':
        accuracy = accuracy_score(y, model.predict(X))
        precision = precision_score(y, model.predict(X), average='macro')
        recall = recall_score(y, model.predict(X), average='macro')
        specificity = specificity_score(y, model.predict(X), average='macro')
        f1 = f1_score(y, model.predict(X), average='macro')
        return {"Accuracy": accuracy, "Precision": precision, "Sensitivity": recall, "Specificity": specificity, "F1 Score": f1}

    else:
        r2 = r2_score(y, model.predict(X))
        mae = mean_absolute_error(y, model.predict(X))
        return {"R2 Score": r2, "MAE": mae}
