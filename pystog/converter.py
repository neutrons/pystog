"""
==========
Converter
==========

This module defines the Converter class
that converts functions in the same space
"""

import numpy as np


class Converter:
    """
    The Converter class is used to convert between
    either different reciprocal space functions or
    different real space functions

    :examples:

    >>> import numpy
    >>> from pystog import Converter
    >>> converter = Converter()
    >>> q, sq = numpy.loadtxt("my_sofq_file.txt",unpack=True)
    >>> fq, dfq = converter.S_to_F(q, sq)
    >>> r, gr = numpy.loadtxt("my_gofr_file.txt",unpack=True)
    >>> kwargs = {'rho' : 1.0}
    >>> gr_keen, dgr_keen = converter.g_to_GK(r, gr, **kwargs)
    """

    def __init__(self):
        pass

    def _safe_divide(self, numerator, denominator):
        numerator = np.array(numerator)
        denominator = np.array(denominator)
        mask = (denominator > 0.0)
        out = np.zeros_like(numerator)
        out[mask] = numerator[mask] / denominator[mask]
        return out

    # Reciprocal Space Conversions

    def F_to_S(self, q, fq, dfq=None, **kwargs):
        """
        Converts from :math:`Q[S(Q)-1]` to :math:`S(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`Q[S(Q)-1]` vector
        :type fq: numpy.array or list
        :param dfq: uncertainty vector
        :type dfq: numpy.array or list

        :return: (:math:`S(Q)` vector, uncertainty vector)
        :rtype: (numpy.array, numpy.array)
        """
        if dfq is None:
            dfq = np.zeros_like(fq)
        return (self._safe_divide(fq, q) + 1., self._safe_divide(dfq, q))

    def F_to_FK(self, q, fq, dfq=None, **kwargs):
        """
        Converts from :math:`Q[S(Q)-1]` to :math:`F(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`Q[S(Q)-1]` vector
        :type fq: numpy.array or list
        :param dfq: uncertainty vector
        :type dfq: numpy.array or list

        :return: (:math:`F(Q)` vector, uncertainty vector)
        :rtype: (numpy.array, numpy.array)
        """
        if dfq is None:
            dfq = np.zeros_like(fq)
        return (kwargs['<b_coh>^2'] * self._safe_divide(fq, q),
                kwargs['<b_coh>^2'] * self._safe_divide(dfq, q))

    def F_to_DCS(self, q, fq, dfq=None, **kwargs):
        """
        Converts from :math:`Q[S(Q)-1]` to
        :math:`\\frac{d \\sigma}{d \\Omega}(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`Q[S(Q)-1]` vector
        :type fq: numpy.array or list
        :param dfq: uncertainty vector
        :type dfq: numpy.array or list

        :return: (:math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector,
                 uncertainty vector)
        :rtype: (numpy.array, numpy.array)
        """
        fq, dfq = self.F_to_FK(q, fq, dfq, **kwargs)
        return self.FK_to_DCS(q, fq, dfq, **kwargs)

    # S(Q)
    def S_to_F(self, q, sq, dsq=None, **kwargs):
        """
        Convert :math:`S(Q)` to :math:`Q[S(Q)-1]`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param sq: :math:`S(Q)` vector
        :type sq: numpy.array or list
        :param dfq: uncertainty vector
        :type dfq: numpy.array or list
        :param dsq: uncertainty vector
        :type dsq: numpy.array or list

        :return: (:math:`Q[S(Q)-1]` vector, uncertainty vector)
        :rtype: (numpy.array, numpy.array)
        """
        if dsq is None:
            dsq = np.zeros_like(sq)
        return (q * (sq - 1.), q * dsq)

    def S_to_FK(self, q, sq, dsq=None, **kwargs):
        """
        Convert :math:`S(Q)` to :math:`F(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param sq: :math:`S(Q)` vector
        :type sq: numpy.array or list
        :param dsq: uncertainty vector
        :type dsq: numpy.array or list

        :return: (:math:`F(Q)` vector, uncertainty vector)
        :rtype: (numpy.array, numpy.array)
        """
        fq, dfq = self.S_to_F(q, sq, dsq)
        return self.F_to_FK(q, fq, dfq, **kwargs)

    def S_to_DCS(self, q, sq, dsq=None, **kwargs):
        """
        Convert :math:`S(Q)` to :math:`\\frac{d \\sigma}{d \\Omega}(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param sq: :math:`S(Q)` vector
        :type sq: numpy.array or list
        :param dsq: uncertainty vector
        :type dsq: numpy.array or list

        :return: (:math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector,
                 uncertainty vector)
        :rtype: (numpy.array, numpy.array)
        """
        fq, dfq = self.S_to_FK(q, sq, dsq, **kwargs)
        return self.FK_to_DCS(q, fq, dfq, **kwargs)

    # Keen's F(Q)
    def FK_to_F(self, q, fq_keen, dfq_keen=None, **kwargs):
        """
        Convert :math:`F(Q)` to :math:`Q[S(Q)-1]`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq_keen: :math:`F(Q)` vector
        :type fq_keen: numpy.array or list
        :param dfq_keen: uncertainty vector
        :type dfq_keen: numpy.array or list

        :return: (:math:`Q[S(Q)-1]` vector, uncertainty vector)
        :rtype: (numpy.array, numpy.array)
        """
        if dfq_keen is None:
            dfq_keen = np.zeros_like(fq_keen)
        return (q * fq_keen / kwargs['<b_coh>^2'],
                q * dfq_keen / kwargs['<b_coh>^2'])

    def FK_to_S(self, q, fq_keen, dfq_keen=None, **kwargs):
        """
        Convert :math:`F(Q)` to :math:`S(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq_keen: :math:`F(Q)` vector
        :type fq_keen: numpy.array or list
        :param dfq_keen: uncertainty vector
        :type dfq_keen: numpy.array or list

        :return: (:math:`S(Q)` vector, uncertainty vector)
        :rtype: (numpy.array, numpy.array)
        """
        fq, dfq = self.FK_to_F(q, fq_keen, dfq_keen, **kwargs)
        return self.F_to_S(q, fq, dfq)

    def FK_to_DCS(self, q, fq, dfq=None, **kwargs):
        """
        Convert :math:`F(Q)` to :math:`\\frac{d \\sigma}{d \\Omega}(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`F(Q)` vector
        :type fq: numpy.array or list
        :param dfq: uncertainty vector
        :type dfq: numpy.array or list

        :return: (:math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector,
                 uncertainty vector)
        :rtype: (numpy.array, numpy.array)
        """
        return fq + kwargs['<b_tot^2>'], dfq

    # Differential cross-section = d_simga / d_Omega
    def DCS_to_F(self, q, dcs, ddcs=None, **kwargs):
        """
        Convert :math:`\\frac{d \\sigma}{d \\Omega}(Q)` to :math:`Q[S(Q)-1]`

        :param q: Q-space vector
        :type q: numpy.array or list
        :param dcs: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :type dcs: numpy.array or list
        :param ddcs: uncertainty vector
        :type ddcs: numpy.array or list

        :return: (:math:`Q[S(Q)-1]` vector, uncertainty vector)
        :rtype: (numpy.array, numpy.array)
        """
        fq, dfq = self.DCS_to_FK(q, dcs, ddcs, **kwargs)
        return self.FK_to_F(q, fq, dfq, **kwargs)

    def DCS_to_S(self, q, dcs, ddcs=None, **kwargs):
        """
        Convert :math:`\\frac{d \\sigma}{d \\Omega}(Q)` to :math:`S(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dcs: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :type dcs: numpy.array or list
        :param ddcs: uncertainty vector
        :type ddcs: numpy.array or list

        :return: (:math:`S(Q)` vector, uncertainty vector)
        :rtype: (numpy.array, numpy.array)
        """
        fq, dfq = self.DCS_to_FK(q, dcs, ddcs, **kwargs)
        return self.FK_to_S(q, fq, dfq, **kwargs)

    def DCS_to_FK(self, q, dcs, ddcs=None, **kwargs):
        """
        Convert :math:`\\frac{d \\sigma}{d \\Omega}(Q)` to :math:`F(Q)`

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :type fq: numpy.array or list
        :param ddcs: uncertainty vector
        :type ddcs: numpy.array or list

        :return: (:math:`F(Q)` vector, uncertainty vector)
        :rtype: (numpy.array, numpy.array)
        """
        return (dcs - kwargs['<b_tot^2>'], ddcs)

    # Real Space Conversions

    # G(r) = PDF
    def G_to_GK(self, r, gr, dgr=None, **kwargs):
        r"""
        Convert :math:`G_{PDFFIT}(r)` to :math:`G_{Keen Version}(r)`

        :param r: r-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{PDFFIT}(r)` vector
        :type gr: numpy.array or list
        :param dgr: uncertainty vector :math:`\Delta g(r)`
        :type dgr: numpy.array or list

        :return: :math:`G_{Keen Version}(r)` vector, uncertainty vector
        :rtype: (numpy.array, numpy.array)
        """
        factor = kwargs['<b_coh>^2'] / (4. * np.pi * kwargs['rho'])
        if dgr is None:
            dgr = np.zeros_like(gr)
        return (factor * self._safe_divide(gr, r),
                factor * self._safe_divide(dgr, r))

    def G_to_g(self, r, gr, dgr=None, **kwargs):
        r"""
        Convert :math:`G_{PDFFIT}(r)` to :math:`g(r)`

        :param r: r-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{PDFFIT}(r)` vector
        :type gr: numpy.array or list
        :param dgr: uncertainty vector :math:`\Delta g(r)`
        :type dgr: numpy.array or list

        :return: :math:`g(r)` vector, uncertainty vector
        :rtype: (numpy.array, numpy.array)
        """
        factor = 4. * np.pi * kwargs['rho']
        if dgr is None:
            dgr = np.zeros_like(gr)
        return (self._safe_divide(gr, factor * r) + 1.,
                self._safe_divide(dgr, factor * r))

    # Keen's G(r)
    def GK_to_G(self, r, gr, dgr=None, **kwargs):
        r"""
        Convert :math:`G_{Keen Version}(r)` to :math:`G_{PDFFIT}(r)`

        :param r: r-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{Keen Version}(r)` vector
        :type gr: numpy.array or list
        :param dgr: uncertainty vector :math:`\Delta g(r)`
        :type dgr: numpy.array or list

        :return: :math:`G_{PDFFIT}(r)` vector, uncertainty vector
        :rtype: (numpy.array, numpy.array)
        """
        factor = (4. * np.pi * kwargs['rho']) / kwargs['<b_coh>^2']
        if dgr is None:
            dgr = np.zeros_like(gr)
        return (factor * r * gr, factor * r * dgr)

    def GK_to_g(self, r, gr, dgr=None, **kwargs):
        r"""
        Convert :math:`G_{Keen Version}(r)` to :math:`g(r)`

        :param r: r-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{Keen Version}(r)` vector
        :type gr: numpy.array or list
        :param dgr: uncertainty vector :math:`\Delta g(r)`
        :type dgr: numpy.array or list

        :return: :math:`g(r)` vector, uncertainty vector
        :rtype: (numpy.array, numpy.array)
        """
        _gr, _dgr = self.GK_to_G(r, gr, dgr=dgr, **kwargs)
        return self.G_to_g(r, _gr, dgr=_dgr, **kwargs)

    # g(r)
    def g_to_G(self, r, gr, dgr=None, **kwargs):
        r"""
        Convert :math:`g(r)` to :math:`G_{PDFFIT}(r)`

        :param r: r-space vector
        :type r: numpy.array or list
        :param gr: :math:`g(r)` vector
        :type gr: numpy.array or list
        :param dgr: uncertainty vector :math:`\Delta g(r)`
        :type dgr: numpy.array or list

        :return: :math:`G_{PDFFIT}(r)` vector, uncertainty vector
        :rtype: (numpy.array, numpy.array)
        """
        factor = 4. * np.pi * r * kwargs['rho']
        if dgr is None:
            dgr = np.zeros_like(gr)
        return (factor * (gr - 1.), factor * dgr)

    def g_to_GK(self, r, gr, dgr=None, **kwargs):
        r"""
        Convert :math:`g(r)` to :math:`G_{Keen Version}(r)`

        :param r: r-space vector
        :type r: numpy.array or list
        :param gr: :math:`g(r)` vector
        :type gr: numpy.array or list
        :param dgr: uncertainty vector :math:`\Delta g(r)`
        :type dgr: numpy.array or list

        :return: :math:`G_{Keen Version}(r)` vector, uncertainty vector
        :rtype: (numpy.array, numpy.array)
        """
        _gr, _dgr = self.g_to_G(r, gr, dgr=dgr, **kwargs)
        return self.G_to_GK(r, _gr, dgr=_dgr, **kwargs)
