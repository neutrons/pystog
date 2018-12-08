"""
=============
Transformer
=============

This module defines the Transformer class
that performs the Fourier transforms
"""


from __future__ import (absolute_import, division, print_function)
import numpy as np

from pystog.utils import create_domain
from pystog.converter import Converter

# -------------------------------------------------------#
# Transforms between Reciprocal and Real Space Functions


class Transformer:
    """The Transformer class is used to Fourier transform
    between the difference spaces. Either:
    a reciprocal space function -> real space function
    or a real space function -> reciprocal space function

    :examples:

    >>> import numpy
    >>> from pystog import Transformer
    >>> transformer = Transformer()
    >>> q, sq = numpy.loadtxt("my_sofq_file.txt",unpack=True)
    >>> r = numpy.linspace(0., 25., 2500)
    >>> r, gr = transformer.S_to_G(q, sq, r)
    >>> q = numpy.linspace(0., 25., 2500)
    >>> q, sq = transformer.G_to_S(r, gr, q)
    """

    def __init__(self):
        self.converter = Converter()

    def _extend_axis_to_low_end(self, x, decimals=4):
        """Utility to setup axis for the forward transform space.

        :param x: vector
        :type x: numpy.array or list
        :param decimals: max decimals in output
        :type decimals: integer
        :return: vector
        :rtype: numpy.array
        """
        return create_domain(min(x), max(x), x[1] - x[0])

    def _low_x_correction(self, xin, yin, xout, yout, **kwargs):
        """Omitted low-x range correction performed in the
        space you have transformed to. Does so by assumming a
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

        """TODO: Refactor correction to just
        1) peform linear extrapolation in space before transform
        2) transform with extrapolated function
        Compare that implementation to this one.
        Replace this if equal (within tolerance)
        """

        lorch_flag = False
        if 'lorch' in kwargs:
            if kwargs['lorch']:
                lorch_flag = True

        xmin = min(xin)
        xmax = max(xin)
        yin_xmin = yin[0]
        np.pi / xmax

        PiOverXmax = np.pi / xmax

        correction = np.zeros_like(yout)
        for i, x in enumerate(xout):
            v = xmin * x
            if lorch_flag:
                vm = xmin * (x - PiOverXmax)
                vp = xmin * (x + PiOverXmax)
                term1 = (vm * np.sin(vm) + np.cos(vm) - 1.) / \
                    (x - PiOverXmax)**2.
                term2 = (vp * np.sin(vp) + np.cos(vp) - 1.) / \
                    (x + PiOverXmax)**2.
                F1 = (term1 - term2) / (2. * PiOverXmax)
                F2 = (np.sin(vm) / (x - PiOverXmax) - np.sin(vp) /
                      (x + PiOverXmax)) / (2. * PiOverXmax)
            else:
                F1 = (2. * v * np.sin(v) - (v * v - 2.) *
                      np.cos(v) - 2.)
                F1 = np.divide(
                    F1,
                    x * x * x,
                    out=np.zeros_like(F1),
                    where=x != 0)

                F2 = (np.sin(v) - v * np.cos(v))
                F2 = np.divide(F2, x * x, out=np.zeros_like(F2), where=x != 0)

            num = F1 * yin_xmin
            factor = np.divide(num, xmin,
                               out=np.zeros_like(num),
                               where=xmin != 0)
            correction[i] = (2 / np.pi) * (factor - F2)

        yout += correction

        return yout

    def apply_cropping(self, x, y, xmin, xmax):
        """Utility to crop x and y based on xmin and xmax along x.
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
        :return: vector pair (x,y) with cropping applied
        :rtype: numpy.array pair
        """
        y = y[np.logical_and(x >= xmin, x <= xmax)]
        x = x[np.logical_and(x >= xmin, x <= xmax)]
        return x, y

    def fourier_transform(self, xin, yin, xout,
                          xmin=None, xmax=None, **kwargs):
        """The Fourier transform function. The kwargs
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
        :return: vector pair of transformed domain, range vectors
        :rtype: numpy.array
        """

        if xmax is None:
            xmax = max(xin)
        if xmin is None:
            xmin = min(xin)

        xin, yin = self.apply_cropping(xin, yin, xmin, xmax)

        xout = self._extend_axis_to_low_end(xout)

        factor = np.full_like(yin, 1.0)
        if 'lorch' in kwargs:
            if kwargs['lorch']:
                PiOverXmax = np.pi / xmax
                num = np.sin(PiOverXmax * xin)
                denom = PiOverXmax * xin
                factor = np.divide(num, denom,
                                   out=np.zeros_like(num),
                                   where=denom != 0)

        yout = np.zeros_like(xout)
        for i, x in enumerate(xout):
            kernel = factor * yin * np.sin(xin * x)
            yout[i] = np.trapz(kernel, x=xin)

        if 'OmittedXrangeCorrection' in kwargs:
            if kwargs["OmittedXrangeCorrection"]:
                self._low_x_correction(xin, yin, xout, yout, **kwargs)

        return xout, yout

    # Reciprocal -> Real Space Transforms  #

    def F_to_G(self, q, fq, r, **kwargs):
        """Transforms from reciprocal space :math:`Q[S(Q)-1]`
        to real space :math:`G_{PDFFIT}(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`Q[S(Q)-1]` vector
        :type fq: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list

        :return: :math:`r` and :math:`G_{PDFFIT}(r)` vector pair
        :rtype: numpy.array pair
        """
        r, gr = self.fourier_transform(q, fq, r, **kwargs)
        gr = 2. / np.pi * gr
        return r, gr

    def F_to_GK(self, q, fq, r, **kwargs):
        """Transforms from reciprocal space :math:`Q[S(Q)-1]`
        to real space :math:`G_{Keen Version}(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`Q[S(Q)-1]` vector
        :type fq: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list

        :return: :math:`r` and :math:`G_{Keen Version}(r)` vector pair
        :rtype: numpy.array pair
        """
        r, gr = self.F_to_G(q, fq, r, **kwargs)
        gr = self.converter.G_to_GK(r, gr, **kwargs)
        return r, gr

    def F_to_g(self, q, fq, r, **kwargs):
        """Transforms from reciprocal space :math:`Q[S(Q)-1]`
        to real space :math:`g(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`Q[S(Q)-1]` vector
        :type fq: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list

        :return: :math:`r` and :math:`g(r)` vector pair
        :rtype: numpy.array pair
        """
        r, gr = self.F_to_G(q, fq, r, **kwargs)
        gr = self.converter.G_to_g(r, gr, **kwargs)
        return r, gr

    # S(Q)
    def S_to_G(self, q, sq, r, **kwargs):
        """Transforms from reciprocal space :math:`S(Q)`
        to real space :math:`G_{PDFFIT}(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param sq: :math:`S(Q)` vector
        :type sq: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list

        :return: :math:`r` and :math:`G_{PDFFIT}(r)` vector pair
        :rtype: numpy.array pair
        """
        fq = self.converter.S_to_F(q, sq)
        r, gr = self.F_to_G(q, fq, r, **kwargs)
        return r, gr

    def S_to_GK(self, q, sq, r, **kwargs):
        """Transforms from reciprocal space :math:`S(Q)`
        to real space :math:`G_{Keen Version}(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param sq: :math:`S(Q)` vector
        :type sq: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list

        :return: :math:`r` and :math:`G_{Keen Version}(r)` vector pair
        :rtype: numpy.array pair
        """
        fq = self.converter.S_to_F(q, sq)
        r, gr = self.F_to_GK(q, fq, r, **kwargs)
        return r, gr

    def S_to_g(self, q, sq, r, **kwargs):
        """Transforms from reciprocal space :math:`S(Q)`
        to real space :math:`g(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param sq: :math:`S(Q)` vector
        :type sq: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list

        :return: :math:`r` and :math:`g(r)` vector pair
        :rtype: numpy.array pair
        """
        fq = self.converter.S_to_F(q, sq)
        r, gr = self.F_to_g(q, fq, r, **kwargs)
        return r, gr

    # Keen's F(Q)
    def FK_to_G(self, q, fq_keen, r, **kwargs):
        """Transforms from reciprocal space :math:`F(Q)`
        to real space :math:`G_{PDFFIT}(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`F(Q)` vector
        :type fq: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list

        :return: :math:`r` and :math:`G_{PDFFIT}(r)` vector pair
        :rtype: numpy.array pair
        """
        fq = self.converter.FK_to_F(q, fq_keen, **kwargs)
        r, gr = self.F_to_G(q, fq, r, **kwargs)
        return r, gr

    def FK_to_GK(self, q, fq_keen, r, **kwargs):
        """Transforms from reciprocal space :math:`F(Q)`
        to real space :math:`G_{Keen Version}(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`F(Q)` vector
        :type fq: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list

        :return: :math:`r` and :math:`G_{Keen Version}(r)` vector pair
        :rtype: numpy.array pair
        """
        fq = self.converter.FK_to_F(q, fq_keen, **kwargs)
        r, gr = self.F_to_GK(q, fq, r, **kwargs)
        return r, gr

    def FK_to_g(self, q, fq_keen, r, **kwargs):
        """Transforms from reciprocal space :math:`F(Q)`
        to real space :math:`g(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`F(Q)` vector
        :type fq: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list

        :return: :math:`r` and :math:`g(r)` vector pair
        :rtype: numpy.array pair
        """
        fq = self.converter.FK_to_F(q, fq_keen, **kwargs)
        r, gr = self.F_to_g(q, fq, r, **kwargs)
        return r, gr

    # Differential cross-section = d_simga / d_Omega
    def DCS_to_G(self, q, dcs, r, **kwargs):
        """Transforms from reciprocal space
        :math:`\\frac{d \\sigma}{d \\Omega}(Q)`
        to real space :math:`G_{PDFFIT}(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dcs: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :type dcs: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list

        :return: :math:`r` and :math:`G_{PDFFIT}(r)` vector pair
        :rtype: numpy.array pair
        """
        fq = self.converter.DCS_to_F(q, dcs, **kwargs)
        r, gr = self.F_to_G(q, fq, r, **kwargs)
        return r, gr

    def DCS_to_GK(self, q, dcs, r, **kwargs):
        """Transforms from reciprocal space
        :math:`\\frac{d \\sigma}{d \\Omega}(Q)`
        to real space :math:`G_{Keen Version}(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dcs: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :type dcs: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list

        :return: :math:`r` and :math:`G_{Keen Version}(r)` vector pair
        :rtype: numpy.array pair
        """
        fq = self.converter.DCS_to_F(q, dcs, **kwargs)
        r, gr = self.F_to_GK(q, fq, r, **kwargs)
        return r, gr

    def DCS_to_g(self, q, dcs, r, **kwargs):
        """Transforms from reciprocal space
        :math:`\\frac{d \\sigma}{d \\Omega}(Q)`
        to real space :math:`g(r)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dcs: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :type dcs: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list

        :return: :math:`r` and :math:`g(r)` vector pair
        :rtype: numpy.array pair
        """
        fq = self.converter.DCS_to_F(q, dcs, **kwargs)
        r, gr = self.F_to_g(q, fq, r, **kwargs)
        return r, gr

    # Real -> Reciprocal Space Transforms  #

    # G(R) = PDF
    def G_to_F(self, r, gr, q, **kwargs):
        """Transforms from real space :math:`G_{PDFFIT}(r)`
        to reciprocal space :math:`Q[S(Q)-1]`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{PDFFIT}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list

        :return: :math:`Q` and :math:`Q[S(Q)-1]` vector pair
        :rtype: numpy.array pair
        """
        q = self._extend_axis_to_low_end(q)
        q, fq = self.fourier_transform(r, gr, q, **kwargs)
        return q, fq

    def G_to_S(self, r, gr, q, **kwargs):
        """Transforms from real space :math:`G_{PDFFIT}(r)`
        to reciprocal space :math:`S(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{PDFFIT}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list

        :return: :math:`Q` and :math:`S(Q)` vector pair
        :rtype: numpy.array pair
        """
        q, fq = self.G_to_F(r, gr, q, **kwargs)
        sq = self.converter.F_to_S(q, fq)
        return q, sq

    def G_to_FK(self, r, gr, q, **kwargs):
        """Transforms from real space :math:`G_{PDFFIT}(r)`
        to reciprocal space :math:`F(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{PDFFIT}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list

        :return: :math:`Q` and :math:`F(Q)` vector pair
        :rtype: numpy.array pair
        """
        q, fq = self.G_to_F(r, gr, q, **kwargs)
        fq = self.converter.F_to_FK(q, fq, **kwargs)
        return q, fq

    def G_to_DCS(self, r, gr, q, **kwargs):
        """Transforms from real space :math:`G_{PDFFIT}(r)`
        to reciprocal space
        :math:`\\frac{d \\sigma}{d \\Omega}(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{PDFFIT}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list

        :return: :math:`Q` and :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector pair
        :rtype: numpy.array pair
        """
        q, fq = self.G_to_F(r, gr, q, **kwargs)
        dcs = self.converter.F_to_DCS(q, fq, **kwargs)
        return q, dcs

    # Keen's G(r)
    def GK_to_F(self, r, gr, q, **kwargs):
        """Transforms from real space :math:`G_{Keen Version}(r)`
        to reciprocal space :math:`Q[S(Q)-1]`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{Keen Version}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list

        :return: :math:`Q` and :math:`Q[S(Q)-1]` vector pair
        :rtype: numpy.array pair
        """
        gr = self.converter.GK_to_G(r, gr, **kwargs)
        return self.G_to_F(r, gr, q, **kwargs)

    def GK_to_S(self, r, gr, q, **kwargs):
        """Transforms from real space :math:`G_{Keen Version}(r)`
        to reciprocal space :math:`S(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{Keen Version}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list

        :return: :math:`Q` and :math:`S(Q)` vector pair
        :rtype: numpy.array pair
        """
        gr = self.converter.GK_to_G(r, gr, **kwargs)
        return self.G_to_S(r, gr, q, **kwargs)

    def GK_to_FK(self, r, gr, q, **kwargs):
        """Transforms from real space :math:`G_{Keen Version}(r)`
        to reciprocal space :math:`F(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{Keen Version}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list

        :return: :math:`Q` and :math:`F(Q)` vector pair
        :rtype: numpy.array pair
        """
        gr = self.converter.GK_to_G(r, gr, **kwargs)
        return self.G_to_FK(r, gr, q, **kwargs)

    def GK_to_DCS(self, r, gr, q, **kwargs):
        """Transforms from real space :math:`G_{Keen Version}(r)`
        to reciprocal space :math:`\\frac{d \\sigma}{d \\Omega}(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{Keen Version}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list

        :return: :math:`Q` and :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector pair
        :rtype: numpy.array pair
        """
        gr = self.converter.GK_to_G(r, gr, **kwargs)
        return self.G_to_DCS(r, gr, q, **kwargs)

    # g(r)
    def g_to_F(self, r, gr, q, **kwargs):
        """Transforms from real space :math:`g(r)`
        to reciprocal space :math:`Q[S(Q)-1]`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`g(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list

        :return: :math:`Q` and :math:`Q[S(Q)-1]` vector pair
        :rtype: numpy.array pair
        """
        gr = self.converter.g_to_G(r, gr, **kwargs)
        return self.G_to_F(r, gr, q, **kwargs)

    def g_to_S(self, r, gr, q, **kwargs):
        """Transforms from real space :math:`g(r)`
        to reciprocal space :math:`S(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`g(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list

        :return: :math:`Q` and :math:`S(Q)` vector pair
        :rtype: numpy.array pair
        """
        gr = self.converter.g_to_G(r, gr, **kwargs)
        return self.G_to_S(r, gr, q, **kwargs)

    def g_to_FK(self, r, gr, q, **kwargs):
        """Transforms from real space :math:`g(r)`
        to reciprocal space :math:`F(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`g(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list

        :return: :math:`Q` and :math:`F(Q)` vector pair
        :rtype: numpy.array pair
        """
        gr = self.converter.g_to_G(r, gr, **kwargs)
        return self.G_to_FK(r, gr, q, **kwargs)

    def g_to_DCS(self, r, gr, q, **kwargs):
        """Transforms from real space :math:`g(r)`
        to reciprocal space :math:`\\frac{d \\sigma}{d \\Omega}(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`g(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list

        :return: :math:`Q` and :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector pair
        :rtype: numpy.array pair
        """
        gr = self.converter.g_to_G(r, gr, **kwargs)
        return self.G_to_DCS(r, gr, q, **kwargs)
