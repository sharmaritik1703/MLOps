from sklearn.mixture import GaussianMixture
from sklearn.utils import resample
import numpy as np

class GMM(GaussianMixture):
    def __init__(self, n_components=1, covariance_type='full', random_state=None):
        super().__init__(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
    
    def fit(self, X, y=None):
        """
        Fit the Gaussian Mixture model to the data.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : Ignored
            Not used, present here for API consistency by convention.
        """
        super().fit(X)
    
    def predict(self, X):
        """
        Predict the labels for the data samples in X using trained model.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            The input data to predict.
            
        Returns:
        labels : array, shape (n_samples,)
            Component labels.
        """
        return super().predict(X)
    
    def fit_resample(self, X, y):
        """
        Perform oversampling to balance the classes.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The class labels.
            
        Returns:
        X_resampled : array, shape (n_samples_new, n_features)
            The resampled input data.
        y_resampled : array, shape (n_samples_new,)
            The resampled class labels.
        """
        # Training the GMM model!
        self.fit(X)
        
        # Get the unique classes and their counts
        classes, counts = np.unique(y, return_counts=True)
        
        # Determine the maximum class count
        max_count = np.max(counts)
        
        # Initialize the resampled arrays
        X_resampled = []
        y_resampled = []
        
        # Resample each class to the maximum count
        for cls in classes:
            X_class = X[y == cls]
            y_class = y[y == cls]
            X_class_resampled, y_class_resampled = resample(X_class, y_class, 
                                                            replace=True, 
                                                            n_samples=max_count, 
                                                            random_state=self.random_state)
            X_resampled.append(X_class_resampled)
            y_resampled.append(y_class_resampled)
        
        # Concatenate the resampled arrays
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        
        return X_resampled, y_resampled
