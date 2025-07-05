import numpy as np
import pandas as pd


def min_max_normalize(series, feature_range=(0, 1)):
    """Min-Max normalization to scale values to a specific range."""
    min_val = series.min()
    max_val = series.max()
    min_range, max_range = feature_range
    
    # Handle constant values
    if max_val == min_val:
        return pd.Series(np.full(len(series), (min_range + max_range) / 2))
    
    normalized = (series - min_val) / (max_val - min_val)
    return normalized * (max_range - min_range) + min_range


def z_score_normalize(series):
    """Z-score normalization (standardization) to have mean=0 and std=1."""
    mean = series.mean()
    std = series.std()
    
    # Handle zero variance
    if std == 0:
        return pd.Series(np.zeros(len(series)))
    
    return (series - mean) / std


def robust_scale_normalize(series):
    """Robust scaling using median and IQR (less sensitive to outliers)."""
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    # Handle zero IQR
    if iqr == 0:
        return pd.Series(np.zeros(len(series)))
    
    return (series - median) / iqr


def max_abs_normalize(series):
    """Scale by dividing by the maximum absolute value (preserves sparsity)."""
    max_abs = series.abs().max()
    
    if max_abs == 0:
        return series
    
    return series / max_abs


def decimal_scaling_normalize(series):
    """Normalize by moving decimal point based on max absolute value."""
    max_abs = series.abs().max()
    
    if max_abs == 0:
        return series
    
    # Find number of digits in max value
    j = len(str(int(max_abs)))
    return series / (10 ** j)


def log_normalize(series):
    """Log transformation for right-skewed data."""
    # Add small constant to handle zeros
    return np.log1p(series)


def sigmoid_normalize(series):
    """Sigmoid/logistic normalization to map values to (0, 1)."""
    return 1 / (1 + np.exp(-series))


def tanh_normalize(series):
    """Hyperbolic tangent normalization to map values to (-1, 1)."""
    return np.tanh(series)


def unit_vector_normalize(series):
    """Normalize to unit length (L2 norm = 1)."""
    norm = np.linalg.norm(series)
    
    if norm == 0:
        return series
    
    return series / norm


def quantile_normalize(series, n_quantiles=100):
    """Map values to uniform distribution based on quantiles."""
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='uniform')
    return pd.Series(qt.fit_transform(series.values.reshape(-1, 1)).flatten())


def power_transform_normalize(series, method='yeo-johnson'):
    """Power transformation (Box-Cox or Yeo-Johnson) to make data more Gaussian-like."""
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method=method)
    return pd.Series(pt.fit_transform(series.values.reshape(-1, 1)).flatten())