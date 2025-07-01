from scipy import signal
import numpy as np


class lgi_method:
    """
        LGI method on CPU using Numpy.
        Pilz, C. S., Zaunseder, S., Krajewski, J., & Blazek, V. (2018). Local group invariance for heart rate estimation from face videos in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 1254-1262).
    """
    methodName = 'LGI'
    projection = np.array([[1, 0, -1], [0, 1, 0], [-1, 0, 1]])

    def __init__(self, fps):
        self.frameRate = fps

    def apply(self, signal):

        U, _, _ = np.linalg.svd(signal)

        S = U[:, 0].reshape(1, -1)  # array 2D shape (1,3)
        P = np.identity(3) - np.matmul(S.T, S)

        Y = np.dot(P, signal)
        bvp = Y[1, :]

        return bvp