"""
=============
Transformer
=============

This module defines the Transformer class
that performs the Fourier transforms
"""

import numpy as np

from pystog.converter import Converter

# -------------------------------------------------------#
# Transforms between Reciprocal and Real Space Functions


class Transformer:
    """
    The Transformer class is used to Fourier transform
    between the difference spaces. Either:
    a reciprocal space function -> real space function
    or a real space function -> reciprocal space function

    :examples:

    >>> import numpy
    >>> from pystog import Transformer
    >>> transformer = Transformer()
    >>> q, sq = numpy.loadtxt("my_sofq_file.txt",unpack=True)
    >>> r = numpy.linspace(0., 25., 2500)
    >>> r, gr, dgr = transformer.S_to_G(q, sq, r)
    >>> q = numpy.linspace(0., 25., 2500)
    >>> q, sq, dsq = transformer.G_to_S(r, gr, q)
    """

    def __init__(self):
        self.converter = Converter()

    def _low_x_correction(self, xin, yin, xout, yout, **kwargs):
        """
        Omitted low-x range correction performed in the
        space you have transformed to. Does so by assuming a
        linear extrapolation to zero.
        Original author: Jack Carpenter

        :param xin: domain vector for space to be transformed from
        :type xin: numpy.array or list
        :param yin: range vector for space to be transformed from
        :type yin: numpy.array or list
        :param xout: domain vector for space to be transformed to
        :type xout: numpy.array or list
        :param yin: range vector for space to be transformed to
        :type yin: numpy.array or list
        :return: range vector for space transformed to with correction applied
        :rtype: numpy.array
        """

        lorch_flag = kwargs.get("lorch", False)

        xnew = xin

        mask = xin != 0
        yin[mask] = yin[mask] / xin[mask] + 1.0

        for NR in range(len(yout)):
            R = xout[NR]
            V = xin[0] * R
            if lorch_flag:
                A = np.pi / xin[-1]
                VM = xnew[0] * (R - A)
                VP = xnew[0] * (R + A)
                F1 = (
                    (
                        (VM * np.sin(VM) + np.cos(VM) - 1.0) / (R - A) ** 2
                        - (VP * np.sin(VP) + np.cos(VP) - 1.0) / (R + A) ** 2
                    )
                    / 2.0
                    / A
                )
                F2 = (np.sin(VM) / (R - A) - np.sin(VP) / (R + A)) / 2.0 / A
            else:
                if R != 0.0:
                    F1 = (2.0 * V * np.sin(V) - (V * V - 2.0) * np.cos(V) - 2.0) / R / R / R
                    F2 = (np.sin(V) - V * np.cos(V)) / R / R
                else:
                    F1 = 0.0
                    F2 = 0.0

            if xin[0] == 0:
                yDS = (2.0 / np.pi) * (F1 * yin[1] / xin[1] - F2)
            else:
                yDS = (2.0 / np.pi) * (F1 * yin[0] / xin[0] - F2)

            yout[NR] += yDS * np.pi / 2.0

        return yout

    def apply_cropping(self, x, y, xmin, xmax, dy=None):
        """
        Utility to crop x and y based on xmin and xmax along x.
        Provides the capability to specify the (Qmin,Qmax)
        or (Rmin,Rmax) in the Fourier transform

        :param x: domain vector
        :type x: numpy.array or list
        :param y: range vector
        :type y: numpy.array or list
        :param xmin: minimum x-value for crop
        :type xmin: float
        :param xmax: maximum x-value for crop
        :type xmax: float
        :param dy: uncertainty vector
        :type dy: numpy.array or list
        :return: vector pair (x,y) with cropping applied
        :rtype: (numpy.array, numpy.array, numpy.array)
        """
        if dy is not None:
            err = np.asarray(dy)
        else:
            err = np.zeros_like(y)
        indices = np.logical_and(x >= xmin, x <= xmax)
        return x[indices], y[indices], err[indices]

    def fourier_transform(self, xin, yin, xout, xmin=None, xmax=None, dy_in=None, **kwargs):
        """
        The Fourier transform function. The kwargs
        argument allows for different modifications:
        Lorch dampening, omitted low-x range correction,

        :param xin: domain vector for space to be transformed from
        :type xin: numpy.array or list
        :param yin: range vector for space to be transformed from
        :type yin: numpy.array or list
        :param xout: domain vector for space to be transformed to
        :type xout: numpy.array or list
        :param xmin: minimum x-value for crop
        :type xmin: float
        :param xmax: maximum x-value for crop
        :type xmax: float
        :param dy_in: uncertainty vector for yin
        :type dy_in: numpy.array or list
        :return: vector pair of transformed domain, range vectors,
                 and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """

        if xmax is None:
            xmax = max(xin)
        if xmin is None:
            xmin = min(xin)

        xin, yin, err = self.apply_cropping(xin, yin, xmin, xmax, dy_in)

        factor = np.ones_like(yin)
        if kwargs.get("lorch", False):
            PiOverXmax = np.pi / xmax
            num = np.sin(PiOverXmax * xin)
            denom = PiOverXmax * xin
            factor = np.divide(num, denom, where=denom != 0)

        yout = np.zeros_like(xout)
        eout = np.zeros_like(xout)
        for i, x in enumerate(xout):
            kernel = factor * yin * np.sin(xin * x)
            ekernel = np.square(factor * err * np.sin(xin * x))
            yout[i] = np.trapezoid(kernel, x=xin)
            eout[i] = np.sqrt((np.diff(xin) ** 2 * (ekernel[1:] + ekernel[:-1]) / 2).sum())

        if kwargs.get("OmittedXrangeCorrection", False):
            self._low_x_correction(xin, yin, xout, yout, **kwargs)

        return xout, yout, eout

    # Reciprocal -> Real Space Transforms  #

    def F_to_G(self, q, fq, r, dfq=None, **kwargs):
        """
        Transforms from reciprocal space :math:`Q[S(Q)-1]`
        to real space :math:`G_{PDFFIT}(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`Q[S(Q)-1]` vector
        :type fq: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param dfq: uncertainty on :math:`Q[S(Q)-1]`
        :type dfq: numpy.array or list

        :return: :math:`r`, :math:`G_{PDFFIT}(r)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        r, gr, dgr = self.fourier_transform(q, fq, r, dy_in=dfq, **kwargs)
        gr *= 2.0 / np.pi
        dgr *= 2.0 / np.pi
        return r, gr, dgr

    def F_to_GK(self, q, fq, r, dfq=None, **kwargs):
        """
        Transforms from reciprocal space :math:`Q[S(Q)-1]`
        to real space :math:`G_{Keen Version}(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`Q[S(Q)-1]` vector
        :type fq: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param dfq: uncertainty on :math:`Q[S(Q)-1]`
        :type dfq: numpy.array or list

        :return: :math:`r`, :math:`G_{Keen Version}(r)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        r, gr, dgr = self.F_to_G(q, fq, r, dfq, **kwargs)
        gr, dgr = self.converter.G_to_GK(r, gr, dgr, **kwargs)
        return r, gr, dgr

    def F_to_g(self, q, fq, r, dfq=None, **kwargs):
        """
        Transforms from reciprocal space :math:`Q[S(Q)-1]`
        to real space :math:`g(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`Q[S(Q)-1]` vector
        :type fq: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param dfq: uncertainty on :math:`Q[S(Q)-1]`
        :type dfq: numpy.array or list

        :return: :math:`r`, :math:`g(r)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        r, gr, dgr = self.F_to_G(q, fq, r, dfq, **kwargs)
        gr, dgr = self.converter.G_to_g(r, gr, dgr, **kwargs)
        return r, gr, dgr

    # S(Q)
    def S_to_G(self, q, sq, r, dsq=None, **kwargs):
        """
        Transforms from reciprocal space :math:`S(Q)`
        to real space :math:`G_{PDFFIT}(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param sq: :math:`S(Q)` vector
        :type sq: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param dsq: :math:`S(Q)` uncertainties
        :type dsq: numpy.array or list

        :return: :math:`r`, :math:`G_{PDFFIT}(r)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        fq, dfq = self.converter.S_to_F(q, sq, dsq)
        return self.F_to_G(q, fq, r, dfq, **kwargs)

    def S_to_GK(self, q, sq, r, dsq=None, **kwargs):
        """
        Transforms from reciprocal space :math:`S(Q)`
        to real space :math:`G_{Keen Version}(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param sq: :math:`S(Q)` vector
        :type sq: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param dsq: :math:`S(Q)` uncertainties
        :type dsq: numpy.array or list

        :return: :math:`r`, :math:`G_{Keen Version}(r)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        fq, dfq = self.converter.S_to_F(q, sq, dsq)
        return self.F_to_GK(q, fq, r, dfq, **kwargs)

    def S_to_g(self, q, sq, r, dsq=None, **kwargs):
        """
        Transforms from reciprocal space :math:`S(Q)`
        to real space :math:`g(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param sq: :math:`S(Q)` vector
        :type sq: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param dsq: :math:`S(Q)` uncertainties
        :type dsq: numpy.array or list

        :return: :math:`r`, :math:`g(r)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        fq, dfq = self.converter.S_to_F(q, sq, dsq)
        return self.F_to_g(q, fq, r, dfq, **kwargs)

    # Keen's F(Q)
    def FK_to_G(self, q, fq_keen, r, dfq_keen=None, **kwargs):
        """
        Transforms from reciprocal space :math:`F(Q)`
        to real space :math:`G_{PDFFIT}(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq_keen: :math:`F(Q)` vector
        :type fq_keen: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param dfq_keen: :math:`F(Q)` vector uncertainties
        :type dfq_keen: numpy.array or list

        :return: :math:`r`, :math:`G_{PDFFIT}(r)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        fq, dfq = self.converter.FK_to_F(q, fq_keen, dfq_keen, **kwargs)
        return self.F_to_G(q, fq, r, dfq, **kwargs)

    def FK_to_GK(self, q, fq_keen, r, dfq_keen=None, **kwargs):
        """
        Transforms from reciprocal space :math:`F(Q)`
        to real space :math:`G_{Keen Version}(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq_keen: :math:`F(Q)` vector
        :type fq_keen: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param dfq_keen: :math:`F(Q)` vector uncertainties
        :type dfq_keen: numpy.array or list

        :return: :math:`r`, :math:`G_{Keen Version}(r)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        fq, dfq = self.converter.FK_to_F(q, fq_keen, dfq_keen, **kwargs)
        return self.F_to_GK(q, fq, r, dfq, **kwargs)

    def FK_to_g(self, q, fq_keen, r, dfq_keen=None, **kwargs):
        """
        Transforms from reciprocal space :math:`F(Q)`
        to real space :math:`g(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq_keen: :math:`F(Q)` vector
        :type fq_keen: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param dfq_keen: :math:`F(Q)` vector uncertainties
        :type dfq_keen: numpy.array or list

        :return: :math:`r`, :math:`g(r)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        fq, dfq = self.converter.FK_to_F(q, fq_keen, dfq_keen, **kwargs)
        return self.F_to_g(q, fq, r, dfq, **kwargs)

    # Differential cross-section = d_simga / d_Omega
    def DCS_to_G(self, q, dcs, r, ddcs=None, **kwargs):
        """
        Transforms from reciprocal space
        :math:`\\frac{d \\sigma}{d \\Omega}(Q)`
        to real space :math:`G_{PDFFIT}(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dcs: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :type dcs: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param ddcs: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` uncertainties
        :type ddcs: numpy.array or list

        :return: :math:`r`, :math:`G_{PDFFIT}(r)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        fq, dfq = self.converter.DCS_to_F(q, dcs, ddcs, **kwargs)
        return self.F_to_G(q, fq, r, dfq, **kwargs)

    def DCS_to_GK(self, q, dcs, r, ddcs=None, **kwargs):
        """
        Transforms from reciprocal space
        :math:`\\frac{d \\sigma}{d \\Omega}(Q)`
        to real space :math:`G_{Keen Version}(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dcs: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :type dcs: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param ddcs: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` uncertainties
        :type ddcs: numpy.array or list

        :return: :math:`r`, :math:`G_{Keen Version}(r)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        fq, dfq = self.converter.DCS_to_F(q, dcs, ddcs, **kwargs)
        return self.F_to_GK(q, fq, r, dfq, **kwargs)

    def DCS_to_g(self, q, dcs, r, ddcs=None, **kwargs):
        """
        Transforms from reciprocal space
        :math:`\\frac{d \\sigma}{d \\Omega}(Q)`
        to real space :math:`g(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dcs: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :type dcs: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param ddcs: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` uncertainties
        :type ddcs: numpy.array or list

        :return: :math:`r`, :math:`g(r)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        fq, dfq = self.converter.DCS_to_F(q, dcs, ddcs, **kwargs)
        return self.F_to_g(q, fq, r, dfq, **kwargs)

    # Real -> Reciprocal Space Transforms  #

    # G(R) = PDF
    def G_to_F(self, r, gr, q, dgr=None, **kwargs):
        """
        Transforms from real space :math:`G_{PDFFIT}(r)`
        to reciprocal space :math:`Q[S(Q)-1]`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{PDFFIT}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dgr: :math:`G_{PDFFIT}(r)` uncertainties
        :type dgr: numpy.array or list

        :return: :math:`Q`, :math:`Q[S(Q)-1]`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        return self.fourier_transform(r, gr, q, dy_in=dgr, **kwargs)

    def G_to_S(self, r, gr, q, dgr=None, **kwargs):
        """
        Transforms from real space :math:`G_{PDFFIT}(r)`
        to reciprocal space :math:`S(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{PDFFIT}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dgr: :math:`G_{PDFFIT}(r)` uncertainties
        :type dgr: numpy.array or list

        :return: :math:`Q`, :math:`S(Q)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        q, fq, dfq = self.G_to_F(r, gr, q, dgr=dgr, **kwargs)
        sq, dsq = self.converter.F_to_S(q, fq, dfq=dfq)
        return q, sq, dsq

    def G_to_FK(self, r, gr, q, dgr=None, **kwargs):
        """
        Transforms from real space :math:`G_{PDFFIT}(r)`
        to reciprocal space :math:`F(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{PDFFIT}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dgr: :math:`G_{PDFFIT}(r)` uncertainties
        :type dgr: numpy.array or list

        :return: :math:`Q`, :math:`F(Q)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        q, fq, dfq = self.G_to_F(r, gr, q, dgr=dgr, **kwargs)
        fq, dfq = self.converter.F_to_FK(q, fq, dfq=dfq, **kwargs)
        return q, fq, dfq

    def G_to_DCS(self, r, gr, q, dgr=None, **kwargs):
        """
        Transforms from real space :math:`G_{PDFFIT}(r)`
        to reciprocal space
        :math:`\\frac{d \\sigma}{d \\Omega}(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{PDFFIT}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dgr: :math:`G_{PDFFIT}(r)` uncertainties
        :type dgr: numpy.array or list

        :return: :math:`Q`, :math:`\\frac{d \\sigma}{d \\Omega}(Q)`,
                 and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        q, fq, dfq = self.G_to_F(r, gr, q, dgr=dgr, **kwargs)
        dcs, ddcs = self.converter.F_to_DCS(q, fq, dfq=dfq, **kwargs)
        return q, dcs, ddcs

    # Keen's G(r)
    def GK_to_F(self, r, gr, q, dgr=None, **kwargs):
        """
        Transforms from real space :math:`G_{Keen Version}(r)`
        to reciprocal space :math:`Q[S(Q)-1]`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{Keen Version}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dgr: :math:`G_{Keen Version}(r)` uncertainties
        :type dgr: numpy.array or list

        :return: :math:`Q`, :math:`Q[S(Q)-1]`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        _gr, _dgr = self.converter.GK_to_G(r, gr, dgr=dgr, **kwargs)
        return self.G_to_F(r, _gr, q, dgr=_dgr, **kwargs)

    def GK_to_S(self, r, gr, q, dgr=None, **kwargs):
        """
        Transforms from real space :math:`G_{Keen Version}(r)`
        to reciprocal space :math:`S(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{Keen Version}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dgr: :math:`G_{Keen Version}(r)` uncertainties
        :type dgr: numpy.array or list

        :return: :math:`Q`, :math:`S(Q)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        _gr, _dgr = self.converter.GK_to_G(r, gr, dgr=dgr, **kwargs)
        return self.G_to_S(r, _gr, q, dgr=_dgr, **kwargs)

    def GK_to_FK(self, r, gr, q, dgr=None, **kwargs):
        """
        Transforms from real space :math:`G_{Keen Version}(r)`
        to reciprocal space :math:`F(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{Keen Version}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dgr: :math:`G_{Keen Version}(r)` uncertainties
        :type dgr: numpy.array or list

        :return: :math:`Q`, :math:`F(Q)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        _gr, _dgr = self.converter.GK_to_G(r, gr, dgr=dgr, **kwargs)
        return self.G_to_FK(r, _gr, q, dgr=_dgr, **kwargs)

    def GK_to_DCS(self, r, gr, q, dgr=None, **kwargs):
        """
        Transforms from real space :math:`G_{Keen Version}(r)`
        to reciprocal space :math:`\\frac{d \\sigma}{d \\Omega}(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{Keen Version}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dgr: :math:`G_{Keen Version}(r)` uncertainties
        :type dgr: numpy.array or list

        :return: :math:`Q`, :math:`\\frac{d \\sigma}{d \\Omega}(Q)`,
                 and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        _gr, _dgr = self.converter.GK_to_G(r, gr, dgr=dgr, **kwargs)
        return self.G_to_DCS(r, _gr, q, dgr=_dgr, **kwargs)

    # g(r)
    def g_to_F(self, r, gr, q, dgr=None, **kwargs):
        """
        Transforms from real space :math:`g(r)`
        to reciprocal space :math:`Q[S(Q)-1]`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`g(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dgr: :math:`g(r)` uncertainties
        :type dgr: numpy.array or list

        :return: :math:`Q`, :math:`Q[S(Q)-1]`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        _gr, _dgr = self.converter.g_to_G(r, gr, dgr=dgr, **kwargs)
        return self.G_to_F(r, _gr, q, dgr=_dgr, **kwargs)

    def g_to_S(self, r, gr, q, dgr=None, **kwargs):
        """
        Transforms from real space :math:`g(r)`
        to reciprocal space :math:`S(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`g(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dgr: :math:`g(r)` uncertainties
        :type dgr: numpy.array or list

        :return: :math:`Q`, :math:`S(Q)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        _gr, _dgr = self.converter.g_to_G(r, gr, dgr=dgr, **kwargs)
        return self.G_to_S(r, _gr, q, dgr=_dgr, **kwargs)

    def g_to_FK(self, r, gr, q, dgr=None, **kwargs):
        """
        Transforms from real space :math:`g(r)`
        to reciprocal space :math:`F(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`g(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dgr: :math:`g(r)` uncertainties
        :type dgr: numpy.array or list

        :return: :math:`Q`, :math:`F(Q)`, and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        _gr, _dgr = self.converter.g_to_G(r, gr, dgr=dgr, **kwargs)
        return self.G_to_FK(r, _gr, q, dgr=_dgr, **kwargs)

    def g_to_DCS(self, r, gr, q, dgr=None, **kwargs):
        """
        Transforms from real space :math:`g(r)`
        to reciprocal space :math:`\\frac{d \\sigma}{d \\Omega}(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`g(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dgr: :math:`g(r)` uncertainties
        :type dgr: numpy.array or list

        :return: :math:`Q`, :math:`\\frac{d \\sigma}{d \\Omega}(Q)`,
                 and uncertainties
        :rtype: numpy.array, numpy.array, numpy.array
        """
        _gr, _dgr = self.converter.g_to_G(r, gr, dgr=dgr, **kwargs)
        return self.G_to_DCS(r, _gr, q, dgr=_dgr, **kwargs)
