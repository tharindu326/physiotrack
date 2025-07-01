import numpy as np
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


def band_pass_filter(signal, bandpass, fs, order=3):
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
