"""
========
pre_proc
========

This module is for some pre-processing of data
"""

import numpy as np


class Pre_Proc:  # noqa: N801
    """
    The Pre_Proc class is used for performing some
    pre-processing task for the data

    :examples:

    >>> from pystog import Pre_Proc
    >>> pre_proc = Pre_Proc()
    >>> q, sq = numpy.loadtxt("my_sofq_file.txt", unpack=True)
    >>> q_rb, sq_rb = pre_proc.rebin(q, sq)
    """

    def __init__(self):
        pass

    @staticmethod
    def rebin(x, y, xmin, xdiv, xmax, dy=None):
        """
        Rebin input data to the specified grid

        :param x: input X array
        :type x: numpy.array or list
        :param y: input Y array
        :type y: numpy.array or list
        :param xmin: Xmin
        :type xmin: float
        :param xdiv: X interval
        :type xdiv: float
        :param xmax: Xmax
        :type xmax: float
        :param dy: optional Y errors
        :type dy: numpy.array or list, optional

        :return: (output X array, output Y array, output Y errors array)
        :rtype: (list, list, list)
        """

        numpts = int((xmax - xmin) / xdiv) + 1
        xout = list()
        for loop in range(numpts):
            xout.append(xmin + loop * xdiv)
        yout = [0.0 for _ in range(len(xout) + 1)]
        ynorm = [0.0 for _ in range(len(xout) + 1)]
        if dy is not None:
            dyout = [0.0 for _ in range(len(xout) + 1)]

        for i, x_tmp in enumerate(x):
            if xmin <= x_tmp <= xmax:
                bin_index = int((x_tmp - xmin) / xdiv)
                scale1 = 1 - (x_tmp - xout[bin_index]) / xdiv
                scale2 = 1 - scale1

                yout[bin_index] += y[i] * scale1
                yout[bin_index + 1] += y[i] * scale2
                ynorm[bin_index] += scale1
                ynorm[bin_index + 1] += scale2
                if dy is not None:
                    dyout[bin_index] += (dy[i] * scale1) ** 2
                    dyout[bin_index + 1] += (dy[i] * scale2) ** 2

        for i in range(len(yout) - 1):
            yout[i] /= ynorm[i]
            if dy is not None:
                dyout[i] = np.sqrt(dyout[i]) / ynorm[i]

        if dy is not None:
            return (np.asarray(xout), np.asarray(yout[:-1]), np.asarray(dyout[:-1]))
        else:
            return (np.asarray(xout), np.asarray(yout[:-1]))
