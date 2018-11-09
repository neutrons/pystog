#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

import inspect
import argparse
import numpy as np
import pandas as pd

ReciprocalSpaceChoices = { "S(Q)" : "S(Q)",
                           "F(Q)" : "=Q[S(Q) - 1]",
                           "FK(Q)" : "Keen's F(Q)",
                           "DCS(Q)" : "Differential Cross-Section" }
RealSpaceChoices = { "g(r)" : ' "little" g(r)',
                     "G(r)" : "Pair Distribution Function",
                     "GK(r)" : "Keen's G(r)" }


# -------------------------------------#
# Utilities

def get_data(filename, skiprows=0, skipfooter=0, xcol=0, ycol=1):
    # Setup domain 
    return pd.read_csv(filename,sep=r"\s*",
                       skiprows=skiprows, 
                       skipfooter=skipfooter, 
                       usecols=[xcol,ycol], 
                       names=['x','y'],
                       engine='python')

def create_domain(xmin, xmax, binsize):
    x = np.arange(xmin, xmax+binsize, binsize)
    return x

def write_out(filename, xdata, ydata, title=None):
    with open(filename,'w') as f:
        f.write("%d \n" % len(xdata))
        f.write("%s \n" % title)
        for x, y in zip(xdata, ydata):
            f.write('%f %f\n' % (x, y))

# -----------------------------------------------#
# Converters Reciprocal or Real Space Functions

class Converter(object):

    def __init__(self):
        pass

    #----------------------------#
    # Reciprocal Space Conversions

    # F(Q) = Q[S(Q) - 1]
    def F_to_S(self, q, fq):
        return (fq / q) + 1.

    def F_to_FK(self, q, fq, **kwargs):
        mask = ( q != 0.0)
        fq_new = np.zeros_like(fq)
        fq_new[mask] = fq[mask] / q[mask] 
        return kwargs['<b_coh>^2'] * fq_new

    def F_to_DCS(self, q, fq, **kwargs):
        fq = self.F_to_FK(q, fq, **kwargs)
        return self.FK_to_DCS(q, fq, **kwargs)

    # S(Q)
    def S_to_F(self, q, sq):
        return q*(sq - 1.)

    def S_to_FK(self, q, sq, **kwargs):
        fq = self.S_to_F(q,sq)
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
        return factor * ( gr / r )

    def G_to_g(self, r, gr, **kwargs):
        factor = 4. * np.pi * kwargs['rho']
        return gr / (factor * r) + 1.

    # Keen's G(r)
    def GK_to_G(self, r, gr, **kwargs):
        factor = (4. * np.pi * kwargs['rho']) / kwargs['<b_coh>^2']
        return  factor * r * gr

    def GK_to_g(self, r, gr, **kwargs):
        gr = self.GK_to_G(r, gr, **kwargs)
        return self.g_to_G(r, gr, **kwargs)

    # g(r)
    def g_to_G(self, r, gr, **kwargs):
        factor = 4. * np.pi * r * kwargs['rho']
        return factor * (gr - 1.)

    def g_to_GK(self, r, gr, **kwargs):
        gr = self.g_to_G(r, gr, **kwargs)
        return self.G_to_GK(r, gr, **kwargs)

# -------------------------------------------------------#
# Transforms between Reciprocal and Real Space Functions

class Transformer(object):
    def __init__(self):
        self.converter = Converter()

    def _extend_axis_to_low_end(self,x,decimals=4):
        dx = x[1] - x[0]
        if x[0] == 0.0:
            x[0] = 1e-6
        x = np.linspace(dx, x[-1], int(x[-1]/dx), endpoint=True)
        return np.around(x,decimals=decimals)


    def fourier_transform(self, xin, yin, xout, **kwargs):
        xmax = max(xin)
        xout = self._extend_axis_to_low_end(xout)

        factor = np.full_like(yin, 1.0)
        if kwargs['lorch']:
            PiOverXmax = np.pi / xmax
            factor = np.sin(PiOverXmax * xin) / (PiOverXmax * xin)

        yout = np.zeros_like(xout)
        for i, x  in enumerate(xout):
            kernel = factor * yin * np.sin(xin * x)
            yout[i] = np.trapz(kernel, x=xin)

        if 'OmittedXrangeCorrection' in kwargs:
            self._low_x_correction(xin, yin, xout, yout, **kwargs)

        return xout, yout

       
    def apply_cropping(self, x, y, xmin, xmax):
        y = y[np.logical_and(x >= xmin, x <= xmax)]
        x = x[np.logical_and(x >= xmin, x <= xmax)]
        return x, y


    def _low_x_correction(self, xin, yin, xout, yout, **kwargs):

        lorch_flag = False
        if 'lorch' in kwargs:
            if kwargs['lorch']:
                lorch_flag = True

        xmin = min(xin)
        xmax = max(xin)
        yin_xmin = yin[0]
        PiOverQmax = np.pi / xmax

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
                F2 = (np.sin(vm) / (x - PiOverXmax) - np.sin(vp) / \
                    (x + PiOverXmax) )/ (2. * PiOverXmax)
            else:
                F1 = (2. * v * np.sin(v) - (v * v - 2.) *
                      np.cos(v) - 2.) / x / x / x
                F2 = (np.sin(v) - v * np.cos(v)) / x / x

            correction[i] = (2 / np.pi) * (F1 * yin_xmin / xmin - F2)

        yout += correction

        return yout 

    #--------------------------------------#
    # Reciprocal -> Real Space Transforms  #
    #--------------------------------------#

    # F(Q) = Q[S(Q) - 1]
    def F_to_G(self, q, fq, r, **kwargs):
        r, gr = self.fourier_transform(q, fq, r, **kwargs)
        gr = 2./ np.pi * gr
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
        fq = self.converter.S_to_F(q,sq)
        r, gr = self.F_to_GK(q, fq, r, **kwargs)
        return r, gr
 
    def S_to_g(self, q, sq, r, **kwargs):
        fq = self.converter.S_to_F(q,sq)
        r, gr = self.F_to_g(q, fq, r, **kwargs)
        return r, gr

    # Keen's F(Q)
    def FK_to_G(self, q, fq_keen, r, **kwargs):
        fq = self.converter.FK_to_F(q, fq_keen, **kwargs)
        r, gr = self.F_to_G(q, fq, r, **kwargs)
        return r, gr

    def FK_to_GK(self, q, fq_keen, r, **kwargs):
        fq = self.converter.FK_to_F(q, fq_keen, **kwargs)
        r, gr =  self.F_to_GK(q, fq, r, **kwargs)
        return r, gr
         
    def FK_to_g(self, q, fq_keen, r, **kwargs):
        fq = self.converter.FK_to_F(q, fq_keen, **kwargs)
        r, gr =  self.F_to_g(q, fq, r, **kwargs)
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

    #--------------------------------------#
    # Real -> Reciprocal Space Transforms  #
    #--------------------------------------#

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
        
class FourierFilter(object):
    def __init__(self):
        self.converter = Converter()
        self.transformer = Transformer()

    # g(r)
    def g_using_F(self, r, gr, q, fq, cutoff, **kwargs):
        # setup qmin, qmax, and get low-r region to back transform
        qmin = min(q)
        qmax = max(q)
        r_tmp, gr_tmp_initial = self.transformer.apply_cropping(r, gr, 0.0, cutoff)

        # Shift low-r so it goes to 1 at "high-r" for this section. Reduces the sinc function issue.
        gr_tmp = gr_tmp_initial + 1

        # Transform the shifted low-r region to F(Q) to get F(Q)_ft
        q_ft = self.transformer._extend_axis_to_low_end(q)
        q_ft, fq_ft = self.transformer.g_to_F(r_tmp, gr_tmp, q_ft, **kwargs)
        q_ft, fq_ft = self.transformer.apply_cropping(q_ft, fq_ft, qmin, qmax)

        # Subtract F(Q)_ft from original F(Q) = delta_F(Q) 
        q, fq = self.transformer.apply_cropping(q, fq, qmin, qmax)
        fq_original = fq
        fq = (fq - fq_ft) 

        # Transform delta_F(Q) for g(r) with low-r removed
        r, gr = self.transformer.F_to_g(q, fq, r, **kwargs)

        return q_ft, fq_ft, q, fq, r, gr
       
    def g_using_S(self, r, gr, q, sq, cutoff, **kwargs):
        fq = self.converter.S_to_F(q, sq)
        q_ft, fq_ft, q, fq, r, gr = self.g_using_F(r, gr, q, fq, cutoff, **kwargs)
        sq_ft = self.converter.F_to_S(q_ft, fq_ft)
        sq    = self.converter.F_to_S(q, fq)
        return q_ft, sq_ft, q, sq, r, gr
       
    def g_using_FK(self, r, gr, q, fq, cutoff, **kwargs):
        fq = self.converter.FK_to_F(q, fq)
        q_ft, fq_ft, q, fq, r, gr = self.g_using_F(r, gr, q, fq, cutoff, **kwargs)
        fq_ft = self.converter.F_to_FK(q_ft, fq_ft)
        fq    = self.converter.F_to_FK(q, fq)
        return q_ft, fq_ft, q, fq, r, gr
       
    def g_using_DCS(self, r, gr, q, dcs, cutoff, **kwargs):
        fq = self.converter.DCS_to_F(q, dcs)
        q_ft, fq_ft, q, fq, r, gr = self.g_using_F(r, gr, q, fq, cutoff, **kwargs)
        dcs_ft = self.converter.F_to_DCS(q_ft, fq_ft)
        dcs    = self.converter.F_to_DCS(q_ft, fq)
        return q_ft, dcs_ft, q, dcs, r, gr

    # G(R) = PDF
    def G_using_F(self, r, gr, q, fq, cutoff, **kwargs):
        gr = self.converter.G_to_g(r, gr, **kwargs) 
        q_ft, fq_ft, q, fq, r, gr = self.g_using_F(r, gr, q, fq, cutoff, **kwargs)
        gr = self.converter.g_to_G(r, gr, **kwargs) 
        return q_ft, fq_ft, q, fq, r, gr

    def G_using_S(self, r, gr, q, sq, cutoff, **kwargs):
        fq = self.converter.S_to_F(q, sq)
        q_ft, fq_ft, q, fq, r, gr = self.G_using_F(r, gr, q, fq, cutoff, **kwargs)
        sq_ft = self.converter.F_to_S(q_ft, fq_ft)
        sq = self.converter.F_to_S(q, fq)
        return q_ft, sq_ft, q, sq, r, gr

    def G_using_FK(self, r, gr, q, fq, cutoff, **kwargs):
        fq = self.converter.FK_to_F(q, fq, **kwargs)
        q_ft, fq_ft, q, fq, r, gr = self.G_using_F(r, gr, q, fq, cutoff, **kwargs)
        fq_ft = self.converter.F_to_FK(q_ft, fq_ft, **kwargs)
        fq = self.converter.F_to_FK(q, fq, **kwargs)
        return q_ft, fq_ft, q, fq, r, gr

    def G_using_DCS(self, r, gr, q, dcs, cutoff, **kwargs):
        fq = self.converter.DCS_to_F(q, dcs, **kwargs)
        q_ft, fq_ft, q, fq, r, gr = self.G_using_F(r, gr, q, fq, cutoff, **kwargs)
        dcs_ft = self.converter.F_to_DCS(q_ft, fq_ft, **kwargs)
        dcs = self.converter.F_to_DCS(q, fq, **kwargs)
        return q_ft, dcs_ft, q, dcs, r, gr

    # Keen's G(r)
    def GK_using_F(self, r, gr, q, fq, cutoff, **kwargs):
        gr = self.converter.GK_to_g(r, gr, **kwargs) 
        q_ft, fq_ft, q, fq, r, gr = self.g_using_F(r, gr, q, fq, cutoff, **kwargs)
        gr = self.converter.g_to_GK(r, gr, **kwargs) 
        return q_ft, fq_ft, q, fq, r, gr

    def GK_using_S(self, r, gr, q, sq, cutoff, **kwargs):
        fq = self.converter.S_to_F(q, sq)
        q_ft, sq_ft, q, sq, r, gr = self.GK_using_F(r, gr, q, sq, cutoff, **kwargs)
        sq_ft = self.converter.F_to_S(q_ft, fq_ft)
        sq = self.converter.F_to_S(q, fq)
        return q_ft, sq_ft, q, sq, r, gr
 
    def GK_using_FK(self, r, gr, q, fq, cutoff, **kwargs):
        fq = self.converter.FK_to_F(q, sq)
        q_ft, fq_ft, q, fq, r, gr = self.GK_using_F(r, gr, q, fq, cutoff, **kwargs)
        fq_ft = self.converter.F_to_FK(q_ft, fq_ft, **kwargs)
        fq = self.converter.F_to_FK(q, fq, **kwargs)
        return q_ft, fq_ft, q, fq, r, gr
 
    def GK_using_DCS(self, r, gr, q, dcs, cutoff, **kwargs):
        fq = self.converter.DCS_to_F(q, dcs, **kwargs)
        q_ft, fq_ft, q, fq, r, gr = self.GK_using_F(r, gr, q, fq, cutoff, **kwargs)
        dcs_ft = self.converter.F_to_DCS(q_ft, fq_ft, **kwargs)
        dcs = self.converter.F_to_DCS(q, fq, **kwargs)
        return q_ft, dcs_ft, q, dcs, r, gr

# -------------------------------------#
# Converter / Transform Factory
tf = Transformer()
transform_functions =  inspect.getmembers(tf, predicate=inspect.ismethod)
transform_dict = { entry[0] : entry[1] for entry in transform_functions }
transform_dict.pop('fourier_transform')
transform_dict.pop('_extend_axis_to_low_end')
transform_dict.pop('_low_x_correction')
transform_dict.pop('__init__')

cv = Converter()
converter_functions =  inspect.getmembers(cv, predicate=inspect.ismethod)
converter_dict = { entry[0] : entry[1] for entry in converter_functions }
converter_dict.pop('__init__')

choices = transform_dict.copy()
choices.update(converter_dict)

def TransformationFactory(transform_name):
    if transform_name in transform_dict:
        return transform_dict[transform_name]
    elif transform_name in converter_dict:
        return converter_dict[transform_name]
        

# -------------------------------------#
# Main function CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('transformation', choices=choices.keys(),
                        help="Fourier transform G(r) -> S(Q)")
    parser.add_argument('-x', '--domain_range', nargs=3, default=(0.0, 100.0, 0.05), type=float,
                        help="The domain range (xmin, xmax, binsize) for the post-transformed function")
    parser.add_argument('-i', '--input', type=str, 
                        help='Input Filename')
    parser.add_argument('-o', '--output', type=str, 
                        help='Output Filename')
    parser.add_argument('-s', '--skiprows', type=int, default=2,
                        help='Number of rows to skip in datasets')
    parser.add_argument('-t', '--trim', type=int, default=0,
                        help='Number of rows to trim off end in datasets')
    parser.add_argument('--xcol', type=int, default=0,
                        help='Set x-col for filename')
    parser.add_argument('--ycol', type=int, default=1,
                        help='Set y-col for filename')
    parser.add_argument('--bcoh-sqrd', type=float, default=None, dest='bcoh_sqrd',
                        help='Squared mean coherent scattering length (units=fm^2)')
    parser.add_argument('--btot-sqrd', type=float, default=None, dest='btot_sqrd',
                        help='Mean squared total scattering length (units=fm^2)')
    parser.add_argument('--rho', type=float, default=None, dest='rho',
                        help='Number density (units=atoms/angstroms^3)')
    parser.add_argument('--plot', action='store_true',
                        help='Show plot of before and after transformation')
    parser.add_argument('--lorch', action='store_true', default=False,
                        help='Apply Lorch Modifcation')

    args = parser.parse_args()


    # Read in data
    data = get_data(args.input, 
                    skiprows=args.skiprows,
                    skipfooter=args.trim,
                    xcol=args.xcol,
                    ycol=args.ycol)
    # Setup domain 
    xmin, xmax, binsize = args.domain_range
    xnew = create_domain(xmin, xmax, binsize)

    # Add extra  key-word arguments needed for some conversions
    kwargs = dict()
    if args.bcoh_sqrd:
        kwargs['<b_coh>^2'] = args.bcoh_sqrd
    if args.rho:
        kwargs['rho'] = args.rho
    if args.btot_sqrd:
        kwargs['<b_tot^2>'] = args.btot_sqrd
    kwargs['lorch'] = args.lorch

    # Transform data to new form
    tf = TransformationFactory(args.transformation)
    if args.transformation in transform_dict:
        xnew, ynew = tf(data['x'].values, data['y'].values, xnew, **kwargs)
    elif args.transformation in converter_dict:
        xnew = data['x'].values
        ynew = tf(data['x'].values, data['y'].values, **kwargs)

    # Output file
    write_out(args.output, xnew, ynew, title=args.transformation)

    # Plot old and new data
    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(data['x'].values, data['y'].values)
        plt.plot(xnew, ynew)
        plt.show()
