import numpy as np
import scipy.sparse
from scipy.signal import medfilt, detrend
from scipy.signal import butter, lfilter, filtfilt, freqz
from scipy import signal
from scipy.signal import find_peaks, stft, lfilter, butter, welch, hilbert, firwin


def BPfilter(x, minHz, maxHz, fs, order=6):
    """Band Pass filter (using BPM band)"""

    # nyq = fs * 0.5
    # low = minHz/nyq
    # high = maxHz/nyq

    # print(low, high)
    # -- filter type
    # print('filtro=%f' % minHz)
    b, a = butter(order, Wn=[minHz, maxHz], fs=fs, btype='bandpass')

    # TODO verificare filtfilt o lfilter
    y = lfilter(b, a, x)
    # y = filtfilt(b, a, x)

    # w, h = freqz(b, a)

    # import matplotlib.pyplot as plt
    # fig, ax1 = plt.subplots()
    # ax1.set_title('Digital filter frequency response')
    # ax1.plot((fs * 0.5 / np.pi) * w, abs(h), 'b')
    # ax1.set_ylabel('Amplitude [dB]', color='b')
    # plt.show()
    return y


def zeroMeanSTDnorm(x):
    # -- normalization along rows (1-3 channels)
    mx = x.mean(axis=1).reshape(-1, 1)
    sx = x.std(axis=1).reshape(-1, 1)
    y = (x - mx) / sx
    return y


def zeroMeanSTDnorm1CH(x):
    # -- normalization along rows (1-3 channels)
    mx = x.mean().reshape(-1, 1)
    sx = x.std().reshape(-1, 1)
    y = (x - mx) / sx
    return y


def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False, window=window, scale=False)
    return taps


def band_pass_filter(signal, bandpass, fs, order=5):
    # Bandpass filter
    [c, d] = butter(order, bandpass, 'bandpass', fs=fs)
    # [e, f] = signal.butter(5, 0.18, 'highpass', fs=fs)
    pulse_signal = lfilter(c, d, signal)
    return pulse_signal


def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20 * np.log10(abs(np.where(sd == 0, 0, m / sd)))


def detrend_advanced(input_signal, detLambda=10, method='scipy'):
    """
    Advanced detrending with Tarvainen method option.

    Parameters:
    - input_signal: Input signal to detrend
    - detLambda: Lambda parameter for Tarvainen method
    - method: 'scipy' or 'Tarvainen'
    """
    if method == 'Tarvainen':
        # Smoothness prior approach as in the paper appendix:
        # "An advanced detrending method with application to HRV analysis"
        # by Tarvainen, Ranta-aho and Karjaalainen
        t = input_signal.shape[0]
        l = t / detLambda  # lambda
        I = np.identity(t)
        D2 = scipy.sparse.diags([1, -2, 1], [0, 1, 2], shape=(t - 2, t)).toarray()
        detrended_signal = (I - np.linalg.inv(I + l ** 2 * (np.transpose(D2).dot(D2)))).dot(input_signal)
    else:
        detrended_signal = detrend(input_signal)

    return detrended_signal


def notch_filter(input_signal, notch_freq, sampling_rate):
    """
    Apply a notch filter to remove a specific frequency.

    Parameters:
    - input_signal: The input signal to be filtered
    - notch_freq: The frequency to be removed (in Hz)
    - sampling_rate: The sampling rate of the input signal
    """
    Q = 30.0  # Quality factor
    b, a = signal.iirnotch(notch_freq, Q, sampling_rate)
    output_signal = signal.lfilter(b, a, input_signal)
    return output_signal


def highpass_filter(input_signal, cutoff_freq, sampling_rate, order=5):
    """
    Apply a high-pass filter to an input signal.

    Parameters:
    - input_signal: The input signal to be filtered
    - cutoff_freq: The cutoff frequency of the high-pass filter in Hz
    - sampling_rate: The sampling rate of the input signal in Hz
    - order: Filter order (default: 5)
    """
    nyquist_freq = 0.5 * sampling_rate
    cutoff_norm = cutoff_freq / nyquist_freq
    b, a = butter(order, cutoff_norm, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, input_signal)
    return filtered_signal


def lowpass_filter(input_signal, cutoff_freq, sampling_rate, order=5):
    """
    Apply a low-pass filter to an input signal.

    Parameters:
    - input_signal: The input signal to be filtered
    - cutoff_freq: The cutoff frequency of the low-pass filter in Hz
    - sampling_rate: The sampling rate of the input signal in Hz
    - order: Filter order (default: 5)
    """
    nyquist_freq = 0.5 * sampling_rate
    cutoff_norm = cutoff_freq / nyquist_freq
    b, a = butter(order, cutoff_norm, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, input_signal)
    return filtered_signal
