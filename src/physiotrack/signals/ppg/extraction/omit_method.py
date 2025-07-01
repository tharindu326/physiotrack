from scipy import signal
import numpy as np


class omit_method:
    """ This method is described in the following paper:
        "Face2PPG: Towards a reliable and unobtrusive blood volume pulse extraction from faces using RGB cameras"
        by Álvarez Casado, C and Bordallo López, M
    """
    methodName = 'OMIT'

    def __init__(self, fps):
        self.frameRate = fps

    def apply(self, signal):
        Q, R = np.linalg.qr(signal)

        S = Q[:, 0].reshape(1, -1)  # array 2D shape (1,3)
        P = np.identity(3) - np.matmul(S.T, S)
        Y = np.dot(P, signal)
        bvp = Y[1, :]

        # bvp = -bvp

        return bvp
