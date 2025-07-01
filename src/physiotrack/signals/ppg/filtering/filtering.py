import numpy as np
import scipy.sparse
from scipy.signal import medfilt, detrend
from scipy.signal import find_peaks, stft, lfilter, butter, welch, hilbert, firwin, lfilter_zi


class Filtering:

    def __init__(self, fps=30):
        self.frameRate = fps

    def detrend(self, input_signal, detLambda=10, method='scipy'):

        if method == 'Tarvainen':
            # Smoothness prior approach as in the paper appendix:
            # "An advanced detrending method with application to HRV analysis"
            # by Tarvainen, Ranta-aho and Karjaalainen
            t = input_signal.shape[0]
            l = t / detLambda  # lambda
            I = np.identity(t)
            D2 = scipy.sparse.diags([1, -2, 1], [0, 1, 2], shape=(t - 2, t)).toarray()  # this works better than spdiags in python
            detrended_signal = (I - np.linalg.inv(I + l ** 2 * (np.transpose(D2).dot(D2)))).dot(input_signal)
        else:
            detrended_signal = detrend(input_signal)

        return detrended_signal

    def BPfilter(self, input_signal, minHz=0.75, maxHz=4.0, order=6):
        """Band Pass filter (using BPM band)"""

        # nyq = fs * 0.5
        # low = minHz/nyq
        # high = maxHz/nyq

        # print(low, high)
        # -- filter type
        # print('filtro=%f' % minHz)
        b, a = butter(order, [minHz, maxHz], fs=self.frameRate, btype='bandpass')
        # TODO verificare filtfilt o lfilter
        y = lfilter(b, a, input_signal)
        # y = filtfilt(b, a, input_signal)

        # w, h = freqz(b, a)

        # import matplotlib.pyplot as plt
        # fig, ax1 = plt.subplots()
        # ax1.set_title('Digital filter frequency response')
        # ax1.plot((fs * 0.5 / np.pi) * w, abs(h), 'b')
        # ax1.set_ylabel('Amplitude [dB]', color='b')
        # plt.show()
        return y

    def zeroMeanSTDnorm(self, input_signal):
        # -- normalization along rows (1-3 channels)
        mx = input_signal.mean(axis=1).reshape(-1, 1)
        sx = input_signal.std(axis=1).reshape(-1, 1)
        y = (input_signal - mx) / sx
        return y

    def bandpass_firwin(self, ntaps, lowcut, highcut, fs, window='hamming'):
        # print(f" [BANDPASS] >>>  fs={fs}, hc={highcut}")
        nyq = 0.5 * fs
        taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False, window=window, scale=False)
        return taps
