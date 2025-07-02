from scipy.stats import pearsonr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.signal import hilbert
from scipy.signal import find_peaks


def compute_plv(signal1, signal2):
    """
    Computes the Phase Locking Value (PLV) between two 1D signals.

    Args:
        signal1 (numpy.ndarray): First signal (e.g., IMU-based wrist motion).
        signal2 (numpy.ndarray): Second signal (e.g., video-based wrist motion).

    Returns:
        float: PLV value (0 to 1), where 1 means perfect phase synchronization.
    """
    # Compute the analytic signal using Hilbert Transform
    signal1, signal2 = align_signals(signal1, signal2)
    analytic_signal1 = hilbert(signal1)
    analytic_signal2 = hilbert(signal2)

    # Extract instantaneous phase angles
    phase1 = np.angle(analytic_signal1)
    phase2 = np.angle(analytic_signal2)

    # Compute the phase difference
    phase_diff = phase1 - phase2

    # Compute PLV
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))  # Mean of complex exponentials

    return plv


def event_synchronization(signal1, signal2, max_delay=5):
    """Computes Event Synchronization Index (ESI)."""
    signal1, signal2 = align_signals(signal1, signal2)
    peaks1, _ = find_peaks(signal1, distance=40)  # Detect peaks
    peaks2, _ = find_peaks(signal2, distance=40)

    count = 0
    for p1 in peaks1:
        if any(abs(p1 - p2) <= max_delay for p2 in peaks2):
            count += 1

    esi = count / max(len(peaks1), len(peaks2))  # Normalize
    return esi


def phase_synchrony(signal1, signal2):
    """Computes phase synchronization using the Hilbert Transform."""
    signal1, signal2 = align_signals(signal1, signal2)
    phase1 = np.angle(hilbert(signal1))  # Extract phase
    phase2 = np.angle(hilbert(signal2))

    phase_diff = np.abs(phase1 - phase2)  # Phase difference
    return 1 - (np.mean(phase_diff) / np.pi)  # Normalize between 0 and 1


def compute_rmse(signal1, signal2):
    """Computes RMSE between two signals."""
    # Align signals to same length
    signal1, signal2 = align_signals(signal1, signal2)
    return np.sqrt(mean_squared_error(signal1, signal2))


def align_signals(signal1, signal2):
    """
    Aligns two signals to have the same length by trimming the longer one.
    """
    min_length = min(len(signal1), len(signal2))
    return signal1[:min_length], signal2[:min_length]


def normalized_cross_correlation(signal1, signal2):
    """Computes normalized cross-correlation between two signals."""
    # Align signals to same length
    signal1, signal2 = align_signals(signal1, signal2)
    signal1 = (signal1 - np.mean(signal1)) / np.std(signal1)
    signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)
    return np.correlate(signal1, signal2, mode="valid")[0] / len(signal1)


def calculate_pearson_correlation(signal1, signal2):
    """
    Calculates the Pearson Correlation Coefficient between two 1D signals.
    """
    try:
        # Align signals to same length
        signal1, signal2 = align_signals(signal1, signal2)

        # Check for constant signals
        if np.all(signal1 == signal1[0]) or np.all(signal2 == signal2[0]):
            print("Warning: One of the signals is constant. Pearson correlation is undefined.")
            return np.nan

        signal1 = (signal1 - np.mean(signal1)) / np.std(signal1)
        signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)

        correlation, _ = pearsonr(signal1, signal2)
        return correlation
    except ValueError:
        print("Warning: Could not calculate Pearson correlation. Signals may be invalid.")
        return np.nan


def calculate_dtw_distance(signal1, signal2, distance_metric=euclidean):
    """
    Calculates the Dynamic Time Warping (DTW) distance between two 1D signals.
    """
    try:
        # Align signals before computing DTW
        signal1, signal2 = align_signals(signal1, signal2)
        distance, _ = fastdtw(signal1.reshape(-1, 1), signal2.reshape(-1, 1), dist=distance_metric)
        return distance
    except Exception as e:
        print(f"Error calculating DTW distance: {e}")
        return np.nan