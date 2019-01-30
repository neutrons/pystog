#!/usr/bin/env python
import numpy as np


# PDFFIT function
def gaussian(x, mu, sig, a=1.):
    factor = a/(sig*np.sqrt(2.*np.pi))
    return factor*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def Aij(c, b, bavg):
    return c*b/np.power(bavg, 2.)


def Tij(rk, rij, sig):
    return gaussian(rk, rij, sig)


def GofR(rk, r, sig, Np, c, b, bavg, rho=1.):
    GofR = np.zeros(len(rk))
    for r_ij, sig_ij, c_ij, b_ij, bi_avg in zip(r, sig, c, b, bavg):
        g_ij = Aij(c_ij, b_ij, bi_avg)*Tij(rk, r_ij, sig_ij)
        GofR += g_ij
    return (1./(rk*Np)) * GofR - 4. * np.pi * rk * rho


# Nickel
class nickel(object):
    '''
    A 4 x 4 x 4 Nickel for Fm3m  generated from ASE
    '''

    def __init__(self):
        self._a = 3.5238
        self._space_group = 225
        self._name = 'Ni'
        self._b = 1.03

        self._Np = 256
        self._volume = 2800.3591
        self._distances = [2.4917028755451565,
                           3.5238,
                           4.3157559778096815,
                           4.9834057510903129,
                           5.5716170094506676,
                           6.1034006357112096,
                           6.5924261497570074,
                           7.0476000000000001,
                           7.4751086266354685,
                           7.8794563391137595,
                           8.2640435272329018,
                           8.6315119556193629,
                           8.9839624809991268,
                           8.9839624809991285,
                           9.6503237406835218,
                           9.9668115021806258,
                           10.273554143527935,
                           10.571400000000001,
                           10.86108103183104,
                           11.143234018901335,
                           11.418417036524808,
                           11.41841703652481,
                           11.687122436254359,
                           11.94978719726841,
                           12.206801271422419,
                           12.458514377725781,
                           12.705241584480007,
                           12.947267933429043,
                           12.947267933429044,
                           12.947267933429046,
                           13.184852299514015,
                           13.418230635221621,
                           13.87321447322141,
                           13.873214473221411,
                           14.313743265128098,
                           14.3137432651281,
                           14.528999603551512,
                           14.528999603551513,
                           14.741113007503877,
                           14.950217253270937,
                           15.156436888002403,
                           15.156436888002405,
                           15.359888097248625,
                           15.359888097248627,
                           15.560679470383032,
                           15.560679470383034,
                           16.33918842721388,
                           16.528087054465804,
                           16.714851028352005,
                           17.08225428156366,
                           17.441920128816093,
                           17.794317750900145,
                           18.310201907133631,
                           18.478963095909901,
                           18.81194417225397,
                           20.395467039026101]
        self._sigmas = [0.01*r for r in self._distances]
        self._concentrations = np.full_like(self._distances, 1./self._Np)
        sum_term = np.sum(self._b*self._concentrations/self._Np)
        self._avg_scat_lengths = np.ones_like(self._distances) * sum_term
        self._scat_lengths = np.ones_like(self._distances) * self._b

    def get_distances(self):
        return self._distances

    def get_sigmas(self):
        return self._sigmas

    def get_scattering_lengths(self):
        return self._scat_lengths

    def get_average_scattering_length(self):
        return self._avg_scat_lengths

    def get_number_of_nickels(self):
        return self._Np

    def get_concentrations(self):
        return self._concentrations

    def get_volume(self):
        return self._volume

    def get_density(self):
        return self.get_number_of_nickels() / self.get_volume()

    def make_from_ase(self, supercell):
        from ase.spacegroup import crystal
        from ase.build import make_supercell
        a = self._a
        ni = crystal(self._name, [(0, 0, 0)],
                     spacegroup=self._space_group,
                     cellpar=[a, a, a, 90, 90, 90])

        if len(supercell) != 3:
            raise Exception("Supercell must be a vector [a b c]")
        make_supercell(ni, supercell)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ni = nickel()
    print("Distances for r_ij: ", ni.get_distances())
    print("Sigmas for sig_ij:", ni.get_sigmas())

    r = np.linspace(0.02, 8.0, 500)
    rij = ni.get_distances()
    sig_ij = ni.get_sigmas()
    Np = ni.get_number_of_nickels()
    cij = ni.get_concentrations()
    bij = ni.get_scattering_lengths()
    bavg = ni.get_average_scattering_length()
    rho = ni.get_density()

    G = GofR(r, rij, sig_ij, Np, cij, bij, bavg, rho=rho)
    plt.plot(r, G)
    plt.show()
