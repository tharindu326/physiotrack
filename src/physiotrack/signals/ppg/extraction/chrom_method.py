from scipy import signal
import numpy as np


class chrom_method:
    """ This method is described in the following paper:
        "Remote heart rate variability for emotional state monitoring"
        by Y. Benezeth, P. Li, R. Macwan, K. Nakamura, R. Gomez, F. Yang
    """
    methodName = 'CHROM'

    def __init__(self, fps):
        self.frameRate = fps

    def apply(self, signal):

        # calculation of new X and Y
        Xcomp = 3 * signal[0] - 2 * signal[1]
        Ycomp = (1.5 * signal[0]) + signal[1] - (1.5 * signal[2])

        # standard deviations
        sX = np.std(Xcomp)
        sY = np.std(Ycomp)

        if sY != 0.0:
            alpha = sX / sY
        else:
            alpha = 1.0

        # -- rPPG signal
        bvp = Xcomp - alpha * Ycomp

        return bvp

