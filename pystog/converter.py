# import modules
from __future__ import (absolute_import, division, print_function)
import numpy as np

# -----------------------------------------------#
# Converters Reciprocal or Real Space Functions


class Converter(object):

    def __init__(self):
        pass

    #----------------------------#
    # Reciprocal Space Conversions

    # F(Q) = Q[S(Q) - 1]
    def F_to_S(self, q, fq, **kwargs):
        return (fq / q) + 1.

    def F_to_FK(self, q, fq, **kwargs):
        mask = (q != 0.0)
        fq_new = np.zeros_like(fq)
        fq_new[mask] = fq[mask] / q[mask]
        return kwargs['<b_coh>^2'] * fq_new

    def F_to_DCS(self, q, fq, **kwargs):
        fq = self.F_to_FK(q, fq, **kwargs)
        return self.FK_to_DCS(q, fq, **kwargs)

    # S(Q)
    def S_to_F(self, q, sq, **kwargs):
        return q * (sq - 1.)

    def S_to_FK(self, q, sq, **kwargs):
        fq = self.S_to_F(q, sq)
        return self.F_to_FK(q, fq, **kwargs)

    def S_to_DCS(self, q, sq, **kwargs):
        fq = self.S_to_FK(q, sq, **kwargs)
        return self.FK_to_DCS(q, fq, **kwargs)

    # Keen's F(Q)
    def FK_to_F(self, q, fq_keen, **kwargs):
        return q * fq_keen / kwargs['<b_coh>^2']

    def FK_to_S(self, q, fq_keen, **kwargs):
        fq = self.FK_to_F(q, fq_keen, **kwargs)
        return self.F_to_S(q, fq)

    def FK_to_DCS(self, q, fq, **kwargs):
        return fq + kwargs['<b_tot^2>']

    # Differential cross-section = d_simga / d_Omega
    def DCS_to_F(self, q, dcs, **kwargs):
        fq = self.DCS_to_FK(q, dcs, **kwargs)
        return self.FK_to_F(q, fq, **kwargs)

    def DCS_to_S(self, q, dcs, **kwargs):
        fq = self.DCS_to_FK(q, dcs, **kwargs)
        return self.FK_to_S(q, fq, **kwargs)

    def DCS_to_FK(self, q, dcs, **kwargs):
        return dcs - kwargs['<b_tot^2>']

    #----------------------------#
    # Real Space Conversions

    # G(r) = PDF
    def G_to_GK(self, r, gr, **kwargs):
        factor = kwargs['<b_coh>^2'] / (4. * np.pi * kwargs['rho'])
        return factor * (gr / r)

    def G_to_g(self, r, gr, **kwargs):
        factor = 4. * np.pi * kwargs['rho']
        return gr / (factor * r) + 1.

    # Keen's G(r)
    def GK_to_G(self, r, gr, **kwargs):
        factor = (4. * np.pi * kwargs['rho']) / kwargs['<b_coh>^2']
        return factor * r * gr

    def GK_to_g(self, r, gr, **kwargs):
        gr = self.GK_to_G(r, gr, **kwargs)
        return self.G_to_g(r, gr, **kwargs)

    # g(r)
    def g_to_G(self, r, gr, **kwargs):
        factor = 4. * np.pi * r * kwargs['rho']
        return factor * (gr - 1.)

    def g_to_GK(self, r, gr, **kwargs):
        gr = self.g_to_G(r, gr, **kwargs)
        return self.G_to_GK(r, gr, **kwargs)
