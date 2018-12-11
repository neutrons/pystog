"""
==========
Converter
==========

This module defines the Converter class
that converts functions in the same space
"""

from __future__ import (absolute_import, division, print_function)
import numpy as np


class Converter:
    """The Converter class is used to convert between
    either different reciprocal space functions or
    different real space functions

    :examples:

    >>> import numpy
    >>> from pystog import Converter
    >>> converter = Converter()
    >>> q, sq = numpy.loadtxt("my_sofq_file.txt",unpack=True)
    >>> fq = converter.S_to_F(q, sq)
    >>> r, gr = numpy.loadtxt("my_gofr_file.txt",unpack=True)
    >>> kwargs = {'rho' : 1.0}
    >>> gr_keen = converter.g_to_GK(r, gr, **kwargs)
    """

    def __init__(self):
        pass

    def _safe_divide(self, numerator, denominator):
        mask = (denominator != 0.0)
        out = np.zeros_like(numerator)
        out[mask] = numerator[mask] / denominator[mask]
        return out

    # Reciprocal Space Conversions

    def F_to_S(self, q, fq, **kwargs):
        """Converts from :math:`Q[S(Q)-1]` to :math:`S(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`Q[S(Q)-1]` vector
        :type fq: numpy.array or list

        :return: :math:`S(Q)` vector
        :rtype: numpy.array
        """
        return self._safe_divide(fq, q) + 1.

    def F_to_FK(self, q, fq, **kwargs):
        """Converts from :math:`Q[S(Q)-1]` to :math:`F(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`Q[S(Q)-1]` vector
        :type fq: numpy.array or list

        :return: :math:`F(Q)` vector
        :rtype: numpy.array
        """
        mask = (q != 0.0)
        fq_new = np.zeros_like(fq)
        fq_new[mask] = fq[mask] / q[mask]
        return kwargs['<b_coh>^2'] * self._safe_divide(fq, q)

    def F_to_DCS(self, q, fq, **kwargs):
        """Converts from :math:`Q[S(Q)-1]` to :math:`\\frac{d \\sigma}{d \\Omega}(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`Q[S(Q)-1]` vector
        :type fq: numpy.array or list

        :return: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :rtype: numpy.array
        """
        fq = self.F_to_FK(q, fq, **kwargs)
        return self.FK_to_DCS(q, fq, **kwargs)

    # S(Q)
    def S_to_F(self, q, sq, **kwargs):
        """Convert :math:`S(Q)` to :math:`Q[S(Q)-1]`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param sq: :math:`S(Q)` vector
        :type sq: numpy.array or list

        :return: :math:`Q[S(Q)-1]` vector
        :rtype: numpy.array
        """
        return q * (sq - 1.)

    def S_to_FK(self, q, sq, **kwargs):
        """Convert :math:`S(Q)` to :math:`F(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param sq: :math:`S(Q)` vector
        :type sq: numpy.array or list

        :return: :math:`F(Q)` vector
        :rtype: numpy.array
        """
        fq = self.S_to_F(q, sq)
        return self.F_to_FK(q, fq, **kwargs)

    def S_to_DCS(self, q, sq, **kwargs):
        """Convert :math:`S(Q)` to :math:`\\frac{d \\sigma}{d \\Omega}(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param sq: :math:`S(Q)` vector
        :type sq: numpy.array or list

        :return: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :rtype: numpy.array
        """
        fq = self.S_to_FK(q, sq, **kwargs)
        return self.FK_to_DCS(q, fq, **kwargs)

    # Keen's F(Q)
    def FK_to_F(self, q, fq_keen, **kwargs):
        """Convert :math:`F(Q)` to :math:`Q[S(Q)-1]`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`F(Q)` vector
        :type fq: numpy.array or list

        :return: :math:`Q[S(Q)-1]` vector
        :rtype: numpy.array
        """
        return q * fq_keen / kwargs['<b_coh>^2']

    def FK_to_S(self, q, fq_keen, **kwargs):
        """Convert :math:`F(Q)` to :math:`S(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`F(Q)` vector
        :type fq: numpy.array or list

        :return: :math:`S(Q)` vector
        :rtype: numpy.array
        """
        fq = self.FK_to_F(q, fq_keen, **kwargs)
        return self.F_to_S(q, fq)

    def FK_to_DCS(self, q, fq, **kwargs):
        """Convert :math:`F(Q)` to :math:`\\frac{d \\sigma}{d \\Omega}(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`F(Q)` vector
        :type fq: numpy.array or list

        :return: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :rtype: numpy.array
        """
        return fq + kwargs['<b_tot^2>']

    # Differential cross-section = d_simga / d_Omega
    def DCS_to_F(self, q, dcs, **kwargs):
        """Convert :math:`\\frac{d \\sigma}{d \\Omega}(Q)` to :math:`Q[S(Q)-1]`

        :param q: Q-space vector
        :type q: numpy.array or list
        :param dcs: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :type dcs: numpy.array or list

        :return: :math:`Q[S(Q)-1]` vector
        :rtype: numpy.array
        """
        fq = self.DCS_to_FK(q, dcs, **kwargs)
        return self.FK_to_F(q, fq, **kwargs)

    def DCS_to_S(self, q, dcs, **kwargs):
        """Convert :math:`\\frac{d \\sigma}{d \\Omega}(Q)` to :math:`S(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dcs: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :type dcs: numpy.array or list

        :return: :math:`S(Q)` vector
        :rtype: numpy.array
        """
        fq = self.DCS_to_FK(q, dcs, **kwargs)
        return self.FK_to_S(q, fq, **kwargs)

    def DCS_to_FK(self, q, dcs, **kwargs):
        """Convert :math:`\\frac{d \\sigma}{d \\Omega}(Q)` to :math:`F(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :type fq: numpy.array or list

        :return: :math:`F(Q)` vector
        :rtype: numpy.array
        """
        return dcs - kwargs['<b_tot^2>']

    # Real Space Conversions

    # G(r) = PDF
    def G_to_GK(self, r, gr, **kwargs):
        """Convert :math:`G_{PDFFIT}(r)` to :math:`G_{Keen Version}(r)`

        :param r: r-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{PDFFIT}(r)` vector
        :type gr: numpy.array or list

        :return: :math:`G_{Keen Version}(r)` vector
        :rtype: numpy.array
        """
        factor = kwargs['<b_coh>^2'] / (4. * np.pi * kwargs['rho'])
        return factor * self._safe_divide(gr, r)

    def G_to_g(self, r, gr, **kwargs):
        """Convert :math:`G_{PDFFIT}(r)` to :math:`g(r)`

        :param r: r-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{PDFFIT}(r)` vector
        :type gr: numpy.array or list

        :return: :math:`g(r)` vector
        :rtype: numpy.array
        """
        factor = 4. * np.pi * kwargs['rho']
        return self._safe_divide(gr, factor * r) + 1.

    # Keen's G(r)
    def GK_to_G(self, r, gr, **kwargs):
        """Convert :math:`G_{Keen Version}(r)` to :math:`G_{PDFFIT}(r)`

        :param r: r-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{Keen Version}(r)` vector
        :type gr: numpy.array or list

        :return: :math:`G_{PDFFIT}(r)` vector
        :rtype: numpy.array
        """
        factor = (4. * np.pi * kwargs['rho']) / kwargs['<b_coh>^2']
        return factor * r * gr

    def GK_to_g(self, r, gr, **kwargs):
        """Convert :math:`G_{Keen Version}(r)` to :math:`g(r)`

        :param r: r-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{Keen Version}(r)` vector
        :type gr: numpy.array or list

        :return: :math:`g(r)` vector
        :rtype: numpy.array
        """
        gr = self.GK_to_G(r, gr, **kwargs)
        return self.G_to_g(r, gr, **kwargs)

    # g(r)
    def g_to_G(self, r, gr, **kwargs):
        """Convert :math:`g(r)` to :math:`G_{PDFFIT}(r)`

        :param r: r-space vector
        :type r: numpy.array or list
        :param gr: :math:`g(r)` vector
        :type gr: numpy.array or list

        :return: :math:`G_{PDFFIT}(r)` vector
        :rtype: numpy.array
        """
        factor = 4. * np.pi * r * kwargs['rho']
        return factor * (gr - 1.)

    def g_to_GK(self, r, gr, **kwargs):
        """Convert :math:`g(r)` to :math:`G_{Keen Version}(r)`

        :param r: r-space vector
        :type r: numpy.array or list
        :param gr: :math:`g(r)` vector
        :type gr: numpy.array or list

        :return: :math:`G_{Keen Version}(r)` vector
        :rtype: numpy.array
        """
        gr = self.g_to_G(r, gr, **kwargs)
        return self.G_to_GK(r, gr, **kwargs)
