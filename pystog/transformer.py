# import modules
from __future__ import (absolute_import, division, print_function)
import numpy as np

from pystog.converter import Converter

# -------------------------------------------------------#
# Transforms between Reciprocal and Real Space Functions


class Transformer:
    def __init__(self):
        self.converter = Converter()

    def _extend_axis_to_low_end(self, x, decimals=4):
        dx = x[1] - x[0]
        if x[0] == 0.0:
            x[0] = 1e-6
        x = np.linspace(dx, x[-1], int(x[-1] / dx), endpoint=True)
        return np.around(x, decimals=decimals)

    def apply_cropping(self, x, y, xmin, xmax):
        y = y[np.logical_and(x >= xmin, x <= xmax)]
        x = x[np.logical_and(x >= xmin, x <= xmax)]
        return x, y

    def fourier_transform(self, xin, yin, xout,
                          xmin=None, xmax=None, **kwargs):
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
            self._low_x_correction(xin, yin, xout, yout, **kwargs)

        return xout, yout

    def _low_x_correction(self, xin, yin, xout, yout, **kwargs):
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
                      np.cos(v) - 2.) / x / x / x
                F2 = (np.sin(v) - v * np.cos(v)) / x / x

            num = F1 * yin_xmin
            factor = np.divide(num, xmin,
                               out=np.zeros_like(num),
                               where=xmin != 0)
            correction[i] = (2 / np.pi) * (factor - F2)

        yout += correction

        return yout

    # Reciprocal -> Real Space Transforms  #

    # F(Q) = Q[S(Q) - 1]
    def F_to_G(self, q, fq, r, **kwargs):
        r, gr = self.fourier_transform(q, fq, r, **kwargs)
        gr = 2. / np.pi * gr
        return r, gr

    def F_to_GK(self, q, fq, r, **kwargs):
        r, gr = self.F_to_G(q, fq, r, **kwargs)
        gr = self.converter.G_to_GK(r, gr, **kwargs)
        return r, gr

    def F_to_g(self, q, fq, r, **kwargs):
        r, gr = self.F_to_G(q, fq, r, **kwargs)
        gr = self.converter.G_to_g(r, gr, **kwargs)
        return r, gr

    # S(Q)
    def S_to_G(self, q, sq, r, **kwargs):
        fq = self.converter.S_to_F(q, sq)
        r, gr = self.F_to_G(q, fq, r, **kwargs)
        return r, gr

    def S_to_GK(self, q, sq, r, **kwargs):
        fq = self.converter.S_to_F(q, sq)
        r, gr = self.F_to_GK(q, fq, r, **kwargs)
        return r, gr

    def S_to_g(self, q, sq, r, **kwargs):
        fq = self.converter.S_to_F(q, sq)
        r, gr = self.F_to_g(q, fq, r, **kwargs)
        return r, gr

    # Keen's F(Q)
    def FK_to_G(self, q, fq_keen, r, **kwargs):
        fq = self.converter.FK_to_F(q, fq_keen, **kwargs)
        r, gr = self.F_to_G(q, fq, r, **kwargs)
        return r, gr

    def FK_to_GK(self, q, fq_keen, r, **kwargs):
        fq = self.converter.FK_to_F(q, fq_keen, **kwargs)
        r, gr = self.F_to_GK(q, fq, r, **kwargs)
        return r, gr

    def FK_to_g(self, q, fq_keen, r, **kwargs):
        fq = self.converter.FK_to_F(q, fq_keen, **kwargs)
        r, gr = self.F_to_g(q, fq, r, **kwargs)
        return r, gr

    # Differential cross-section = d_simga / d_Omega
    def DCS_to_G(self, q, dcs, r, **kwargs):
        fq = self.converter.DCS_to_F(q, dcs, **kwargs)
        r, gr = self.F_to_G(q, fq, r, **kwargs)
        return r, gr

    def DCS_to_GK(self, q, dcs, r, **kwargs):
        fq = self.converter.DCS_to_F(q, dcs, **kwargs)
        r, gr = self.F_to_GK(q, fq, r, **kwargs)
        return r, gr

    def DCS_to_g(self, q, dcs, r, **kwargs):
        fq = self.converter.DCS_to_F(q, dcs, **kwargs)
        r, gr = self.F_to_g(q, fq, r, **kwargs)
        return r, gr

    # Real -> Reciprocal Space Transforms  #

    # G(R) = PDF
    def G_to_F(self, r, gr, q, **kwargs):
        q = self._extend_axis_to_low_end(q)
        q, fq = self.fourier_transform(r, gr, q, **kwargs)
        return q, fq

    def G_to_S(self, r, gr, q, **kwargs):
        q, fq = self.G_to_F(r, gr, q, **kwargs)
        sq = self.converter.F_to_S(q, fq)
        return q, sq

    def G_to_FK(self, r, gr, q, **kwargs):
        q, fq = self.G_to_F(r, gr, q, **kwargs)
        fq = self.converter.F_to_FK(q, fq, **kwargs)
        return q, fq

    def G_to_DCS(self, r, gr, q, **kwargs):
        q, fq = self.G_to_F(r, gr, q, **kwargs)
        dcs = self.converter.F_to_DCS(q, fq, **kwargs)
        return q, dcs

    # Keen's G(r)
    def GK_to_F(self, r, gr, q, **kwargs):
        gr = self.converter.GK_to_G(r, gr, **kwargs)
        return self.G_to_F(r, gr, q, **kwargs)

    def GK_to_S(self, r, gr, q, **kwargs):
        gr = self.converter.GK_to_G(r, gr, **kwargs)
        return self.G_to_S(r, gr, q, **kwargs)

    def GK_to_FK(self, r, gr, q, **kwargs):
        gr = self.converter.GK_to_G(r, gr, **kwargs)
        return self.G_to_FK(r, gr, q, **kwargs)

    def GK_to_DCS(self, r, gr, q, **kwargs):
        gr = self.converter.GK_to_G(r, gr, **kwargs)
        return self.G_to_DCS(r, gr, q, **kwargs)

    # g(r)
    def g_to_F(self, r, gr, q, **kwargs):
        gr = self.converter.g_to_G(r, gr, **kwargs)
        return self.G_to_F(r, gr, q, **kwargs)

    def g_to_S(self, r, gr, q, **kwargs):
        gr = self.converter.g_to_G(r, gr, **kwargs)
        return self.G_to_S(r, gr, q, **kwargs)

    def g_to_FK(self, r, gr, q, **kwargs):
        gr = self.converter.g_to_G(r, gr, **kwargs)
        return self.G_to_FK(r, gr, q, **kwargs)

    def g_to_DCS(self, r, gr, q, **kwargs):
        gr = self.converter.g_to_G(r, gr, **kwargs)
        return self.G_to_DCS(r, gr, q, **kwargs)
