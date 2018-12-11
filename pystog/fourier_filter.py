"""
==============
FourierFilter
==============

This module defines the FourierFilter class
that performs the Fourier filter for a
given range to exclude.
"""


from __future__ import (absolute_import, division, print_function)

from pystog.converter import Converter
from pystog.transformer import Transformer


class FourierFilter:
    """The FourierFilter class is used to exlude a given
    range in the current function by a back Fourier Transform
    of that section, followed by a difference from the non-excluded
    function, and then a forward transform of the difference function
    Can currently do:
    a real space function -> reciprocal space function -> real space function

    :examples:

    >>> import numpy
    >>> from pystog import FourierFilter
    >>> ff = FourierFilter()
    >>> r, gr = numpy.loadtxt("my_gofr_file.txt",unpack=True)
    >>> q = numpy.linspace(0., 25., 2500)
    >>> q, sq = transformer.G_to_S(r, gr, q)
    >>> q_ft, sq_ft, q, sq, r, gr = ff.G_using_F(r, gr, q, sq, 1.5)
    """

    def __init__(self):
        self.converter = Converter()
        self.transformer = Transformer()

    # g(r)
    def g_using_F(self, r, gr, q, fq, cutoff, **kwargs):
        """Fourier filters real space :math:`g(r)`
        using the reciprocal space :math:`Q[S(Q)-1]`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`g(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`Q[S(Q)-1]` vector
        :type fq: numpy.array or list
        :param cutoff: The :math:`r_{max}` value to filter from 0. to this cutoff
        :type cutoff: float

        :return: A tuple of the :math:`Q` and :math:`Q[S(Q)-1]`
                 for the 0. to cutoff transform,
                 the original :math:`Q` and :math:`Q[S(Q)-1]`,
                 and the filtered :math:`r` and :math:`g(r)`.

                 Thus,
                 [:math:`Q_{FF}`, :math:`Q[S(Q)-1]_{FF}`,
                 :math:`Q`, :math:`Q[S(Q)-1]`, :math:`r_{FF}`, :math:`g(r)_{FF}]`
        :rtype: tuple of numpy.array
        """
        # setup qmin, qmax, and get low-r region to back transform
        qmin = min(q)
        qmax = max(q)
        r_tmp, gr_tmp_initial = self.transformer.apply_cropping(
            r, gr, 0.0, cutoff)

        # Shift low-r so it goes to 1 at "high-r" for this section. Reduces the
        # sinc function issue.
        gr_tmp = gr_tmp_initial + 1

        # Transform the shifted low-r region to F(Q) to get F(Q)_ft
        q_ft, fq_ft = self.transformer.g_to_F(r_tmp, gr_tmp, q, **kwargs)
        q_ft, fq_ft = self.transformer.apply_cropping(q_ft, fq_ft, qmin, qmax)

        # Subtract F(Q)_ft from original F(Q) = delta_F(Q)
        q, fq = self.transformer.apply_cropping(q, fq, qmin, qmax)
        fq = (fq - fq_ft)

        # Transform delta_F(Q) for g(r) with low-r removed
        r, gr = self.transformer.F_to_g(q, fq, r, **kwargs)

        return q_ft, fq_ft, q, fq, r, gr

    def g_using_S(self, r, gr, q, sq, cutoff, **kwargs):
        """Fourier filters real space :math:`g(r)`
        using the reciprocal space :math:`S(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`g(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param sq: :math:`S(Q)` vector
        :type sq: numpy.array or list
        :param cutoff: The :math:`r_{max}` value to filter from 0. to this cutoff
        :type cutoff: float

        :return: A tuple of the :math:`Q` and :math:`S(Q)`
                 for the 0. to cutoff transform,
                 the original :math:`Q` and :math:`S(Q)`,
                 and the filtered :math:`r` and :math:`g(r)`.

                 Thus,
                 [:math:`Q_{FF}`, :math:`S(Q)_{FF}`,
                 :math:`Q`, :math:`S(Q)`, :math:`r_{FF}`, :math:`g(r)_{FF}]`
        :rtype: tuple of numpy.array
        """
        fq = self.converter.S_to_F(q, sq)
        q_ft, fq_ft, q, fq, r, gr = self.g_using_F(
            r, gr, q, fq, cutoff, **kwargs)
        sq_ft = self.converter.F_to_S(q_ft, fq_ft)
        sq = self.converter.F_to_S(q, fq)
        return q_ft, sq_ft, q, sq, r, gr

    def g_using_FK(self, r, gr, q, fq, cutoff, **kwargs):
        """Fourier filters real space :math:`g(r)`
        using the reciprocal space :math:`F(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`g(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`F(Q)` vector
        :type fq: numpy.array or list
        :param cutoff: The :math:`r_{max}` value to filter from 0. to this cutoff
        :type cutoff: float

        :return: A tuple of the :math:`Q` and :math:`F(Q)`
                 for the 0. to cutoff transform,
                 the original :math:`Q` and :math:`F(Q)`,
                 and the filtered :math:`r` and :math:`g(r)`.

                 Thus,
                 [:math:`Q_{FF}`, :math:`F(Q)_{FF}`,
                 :math:`Q`, :math:`F(Q)`, :math:`r_{FF}`, :math:`g(r)_{FF}]`
        :rtype: tuple of numpy.array
        """
        fq = self.converter.FK_to_F(q, fq, **kwargs)
        q_ft, fq_ft, q, fq, r, gr = self.g_using_F(
            r, gr, q, fq, cutoff, **kwargs)
        fq_ft = self.converter.F_to_FK(q_ft, fq_ft, **kwargs)
        fq = self.converter.F_to_FK(q, fq, **kwargs)
        return q_ft, fq_ft, q, fq, r, gr

    def g_using_DCS(self, r, gr, q, dcs, cutoff, **kwargs):
        """Fourier filters real space :math:`g(r)`
        using the reciprocal space
        :math:`\\frac{d \\sigma}{d \\Omega}(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`g(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dcs: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :type dcs: numpy.array or list
        :param cutoff: The :math:`r_{max}` value to filter from 0. to this cutoff
        :type cutoff: float

        :return: A tuple of the :math:`Q` and
                 :math:`\\frac{d \\sigma}{d \\Omega}(Q)`
                 for the 0. to cutoff transform,
                 the original :math:`Q` and
                 :math:`\\frac{d \\sigma}{d \\Omega}(Q)`,
                 and the filtered :math:`r` and :math:`g(r)`.

                 Thus,
                 [:math:`Q_{FF}`, :math:`\\frac{d \\sigma}{d \\Omega}(Q)_{FF}`,
                 :math:`Q`, :math:`\\frac{d \\sigma}{d \\Omega}(Q)`,
                 :math:`r_{FF}`, :math:`g(r)_{FF}]`
        :rtype: tuple of numpy.array
        """
        fq = self.converter.DCS_to_F(q, dcs, **kwargs)
        q_ft, fq_ft, q, fq, r, gr = self.g_using_F(
            r, gr, q, fq, cutoff, **kwargs)
        dcs_ft = self.converter.F_to_DCS(q_ft, fq_ft, **kwargs)
        dcs = self.converter.F_to_DCS(q_ft, fq, **kwargs)
        return q_ft, dcs_ft, q, dcs, r, gr

    # G(R) = PDF
    def G_using_F(self, r, gr, q, fq, cutoff, **kwargs):
        """Fourier filters real space :math:`G_{PDFFIT}(r)`
        using the reciprocal space :math:`Q[S(Q)-1]`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{PDFFIT}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`Q[S(Q)-1]` vector
        :type fq: numpy.array or list
        :param cutoff: The :math:`r_{max}` value to filter from 0. to this cutoff
        :type cutoff: float

        :return: A tuple of the :math:`Q` and :math:`Q[S(Q)-1]`
                 for the 0. to cutoff transform,
                 the original :math:`Q` and :math:`Q[S(Q)-1]`,
                 and the filtered :math:`r` and :math:`G_{PDFFIT}(r)`.

                 Thus,
                 [:math:`Q_{FF}`, :math:`Q[S(Q)-1]_{FF}`,
                 :math:`Q`, :math:`Q[S(Q)-1]`, :math:`r_{FF}`, :math:`G_{PDFFIT}(r)_{FF}]`
        :rtype: tuple of numpy.array
        """
        gr = self.converter.G_to_g(r, gr, **kwargs)
        q_ft, fq_ft, q, fq, r, gr = self.g_using_F(
            r, gr, q, fq, cutoff, **kwargs)
        gr = self.converter.g_to_G(r, gr, **kwargs)
        return q_ft, fq_ft, q, fq, r, gr

    def G_using_S(self, r, gr, q, sq, cutoff, **kwargs):
        """Fourier filters real space :math:`G_{PDFFIT}(r)`
        using the reciprocal space :math:`S(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{PDFFIT}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`S(Q)` vector
        :type fq: numpy.array or list
        :param cutoff: The :math:`r_{max}` value to filter from 0. to this cutoff
        :type cutoff: float

        :return: A tuple of the :math:`Q` and :math:`S(Q)`
                 for the 0. to cutoff transform,
                 the original :math:`Q` and :math:`S(Q)`,
                 and the filtered :math:`r` and :math:`G_{PDFFIT}(r)`.

                 Thus,
                 [:math:`Q_{FF}`, :math:`S(Q)_{FF}`,
                 :math:`Q`, :math:`S(Q)`, :math:`r_{FF}`, :math:`G_{PDFFIT}(r)_{FF}]`
        :rtype: tuple of numpy.array
        """
        fq = self.converter.S_to_F(q, sq)
        q_ft, fq_ft, q, fq, r, gr = self.G_using_F(
            r, gr, q, fq, cutoff, **kwargs)
        sq_ft = self.converter.F_to_S(q_ft, fq_ft)
        sq = self.converter.F_to_S(q, fq)
        return q_ft, sq_ft, q, sq, r, gr

    def G_using_FK(self, r, gr, q, fq, cutoff, **kwargs):
        """Fourier filters real space :math:`G_{PDFFIT}(r)`
        using the reciprocal space :math:`F(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{PDFFIT}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`F(Q)` vector
        :type fq: numpy.array or list
        :param cutoff: The :math:`r_{max}` value to filter from 0. to this cutoff
        :type cutoff: float

        :return: A tuple of the :math:`Q` and :math:`F(Q)`
                 for the 0. to cutoff transform,
                 the original :math:`Q` and :math:`F(Q)`,
                 and the filtered :math:`r` and :math:`G_{PDFFIT}(r)`.

                 Thus,
                 [:math:`Q_{FF}`, :math:`F(Q)_{FF}`,
                 :math:`Q`, :math:`F(Q)`, :math:`r_{FF}`, :math:`G_{PDFFIT}(r)_{FF}]`
        :rtype: tuple of numpy.array
        """
        fq = self.converter.FK_to_F(q, fq, **kwargs)
        q_ft, fq_ft, q, fq, r, gr = self.G_using_F(
            r, gr, q, fq, cutoff, **kwargs)
        fq_ft = self.converter.F_to_FK(q_ft, fq_ft, **kwargs)
        fq = self.converter.F_to_FK(q, fq, **kwargs)
        return q_ft, fq_ft, q, fq, r, gr

    def G_using_DCS(self, r, gr, q, dcs, cutoff, **kwargs):
        """Fourier filters real space :math:`G_{PDFFIT}(r)`
        using the reciprocal space
        :math:`\\frac{d \\sigma}{d \\Omega}(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{PDFFIT}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dcs: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :type dcs: numpy.array or list
        :param cutoff: The :math:`r_{max}` value to filter from 0. to this cutoff
        :type cutoff: float

        :return: A tuple of the :math:`Q` and
                 :math:`\\frac{d \\sigma}{d \\Omega}(Q)`
                 for the 0. to cutoff transform,
                 the original :math:`Q` and
                 :math:`\\frac{d \\sigma}{d \\Omega}(Q)`,
                 and the filtered :math:`r` and :math:`G_{PDFFIT}(r)`.

                 Thus,
                 [:math:`Q_{FF}`, :math:`\\frac{d \\sigma}{d \\Omega}(Q)_{FF}`,
                 :math:`Q`, :math:`\\frac{d \\sigma}{d \\Omega}(Q)`,
                 :math:`r_{FF}`, :math:`G_{PDFFIT}(r)_{FF}]`
        :rtype: tuple of numpy.array
        """
        fq = self.converter.DCS_to_F(q, dcs, **kwargs)
        q_ft, fq_ft, q, fq, r, gr = self.G_using_F(
            r, gr, q, fq, cutoff, **kwargs)
        dcs_ft = self.converter.F_to_DCS(q_ft, fq_ft, **kwargs)
        dcs = self.converter.F_to_DCS(q, fq, **kwargs)
        return q_ft, dcs_ft, q, dcs, r, gr

    # Keen's G(r)
    def GK_using_F(self, r, gr, q, fq, cutoff, **kwargs):
        """Fourier filters real space :math:`G_{Keen Version}(r)`
        using the reciprocal space :math:`Q[S(Q)-1]`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{Keen Version}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`Q[S(Q)-1]` vector
        :type fq: numpy.array or list
        :param cutoff: The :math:`r_{max}` value to filter from 0. to this cutoff
        :type cutoff: float

        :return: A tuple of the :math:`Q` and :math:`Q[S(Q)-1]`
                 for the 0. to cutoff transform,
                 the original :math:`Q` and :math:`Q[S(Q)-1]`,
                 and the filtered :math:`r` and :math:`G_{Keen Version}(r)`.

                 Thus,
                 [:math:`Q_{FF}`, :math:`Q[S(Q)-1]_{FF}`,
                 :math:`Q`, :math:`Q[S(Q)-1]`, :math:`r_{FF}`, :math:`G_{Keen Version}(r)_{FF}]`
        :rtype: tuple of numpy.array
        """

        gr = self.converter.GK_to_g(r, gr, **kwargs)
        q_ft, fq_ft, q, fq, r, gr = self.g_using_F(
            r, gr, q, fq, cutoff, **kwargs)
        gr = self.converter.g_to_GK(r, gr, **kwargs)
        return q_ft, fq_ft, q, fq, r, gr

    def GK_using_S(self, r, gr, q, sq, cutoff, **kwargs):
        """Fourier filters real space :math:`G_{Keen Version}(r)`
        using the reciprocal space :math:`S(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{Keen Version}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`S(Q)` vector
        :type fq: numpy.array or list
        :param cutoff: The :math:`r_{max}` value to filter from 0. to this cutoff
        :type cutoff: float

        :return: A tuple of the :math:`Q` and :math:`S(Q)`
                 for the 0. to cutoff transform,
                 the original :math:`Q` and :math:`S(Q)`,
                 and the filtered :math:`r` and :math:`G_{Keen Version}(r)`.

                 Thus,
                 [:math:`Q_{FF}`, :math:`S(Q)_{FF}`,
                 :math:`Q`, :math:`S(Q)`, :math:`r_{FF}`, :math:`G_{Keen Version}(r)_{FF}]`
        :rtype: tuple of numpy.array
        """

        fq = self.converter.S_to_F(q, sq)
        q_ft, fq_ft, q, fq, r, gr = self.GK_using_F(
            r, gr, q, fq, cutoff, **kwargs)
        sq_ft = self.converter.F_to_S(q_ft, fq_ft)
        sq = self.converter.F_to_S(q, fq)
        return q_ft, sq_ft, q, sq, r, gr

    def GK_using_FK(self, r, gr, q, fq, cutoff, **kwargs):
        """Fourier filters real space :math:`G_{Keen Version}(r)`
        using the reciprocal space :math:`F(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{Keen Version}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param fq: :math:`F(Q)` vector
        :type fq: numpy.array or list
        :param cutoff: The :math:`r_{max}` value to filter from 0. to this cutoff
        :type cutoff: float

        :return: A tuple of the :math:`Q` and :math:`F(Q)`
                 for the 0. to cutoff transform,
                 the original :math:`Q` and :math:`F(Q)`,
                 and the filtered :math:`r` and :math:`G_{Keen Version}(r)`.

                 Thus,
                 [:math:`Q_{FF}`, :math:`F(Q)_{FF}`,
                 :math:`Q`, :math:`F(Q)`, :math:`r_{FF}`, :math:`G_{Keen Version}(r)_{FF}]`
        :rtype: tuple of numpy.array
        """

        fq = self.converter.FK_to_F(q, fq, **kwargs)
        q_ft, fq_ft, q, fq, r, gr = self.GK_using_F(
            r, gr, q, fq, cutoff, **kwargs)
        fq_ft = self.converter.F_to_FK(q_ft, fq_ft, **kwargs)
        fq = self.converter.F_to_FK(q, fq, **kwargs)
        return q_ft, fq_ft, q, fq, r, gr

    def GK_using_DCS(self, r, gr, q, dcs, cutoff, **kwargs):
        """Fourier filters real space :math:`G_{Keen Version}(r)`
        using the reciprocal space
        :math:`\\frac{d \\sigma}{d \\Omega}(Q)`

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: :math:`G_{Keen Version}(r)` vector
        :type gr: numpy.array or list
        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param dcs: :math:`\\frac{d \\sigma}{d \\Omega}(Q)` vector
        :type dcs: numpy.array or list
        :param cutoff: The :math:`r_{max}` value to filter from 0. to this cutoff
        :type cutoff: float

        :return: A tuple of the :math:`Q` and
                 :math:`\\frac{d \\sigma}{d \\Omega}(Q)`
                 for the 0. to cutoff transform,
                 the original :math:`Q` and
                 :math:`\\frac{d \\sigma}{d \\Omega}(Q)`,
                 and the filtered :math:`r` and :math:`G_{Keen Version}(r)`.

                 Thus,
                 [:math:`Q_{FF}`, :math:`\\frac{d \\sigma}{d \\Omega}(Q)_{FF}`,
                 :math:`Q`, :math:`\\frac{d \\sigma}{d \\Omega}(Q)`,
                 :math:`r_{FF}`, :math:`G_{Keen Version}(r)_{FF}]`
        :rtype: tuple of numpy.array
        """
        fq = self.converter.DCS_to_F(q, dcs, **kwargs)
        q_ft, fq_ft, q, fq, r, gr = self.GK_using_F(
            r, gr, q, fq, cutoff, **kwargs)
        dcs_ft = self.converter.F_to_DCS(q_ft, fq_ft, **kwargs)
        dcs = self.converter.F_to_DCS(q, fq, **kwargs)
        return q_ft, dcs_ft, q, dcs, r, gr
