#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

import inspect
import argparse
import numpy as np
import pandas as pd

ReciprocalSpaceChoices = { "S(Q)" : "S(Q)",
                           "F(Q)" : "=Q[S(Q) - 1]",
                           "FK(Q)" : "Keen's F(Q)" }
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

    # Reciprocal Space Conversions

    def F_to_S(self, q, fq):
        return (fq / q) + 1.

    def S_to_F(self, q, sq):
        return q*(sq - 1.)

    def FK_to_F(self, q, fq_keen, **kwargs):
        return q * fq_keen / kwargs['bcoh_sqrd']

    def F_to_FK(self, q, fq, **kwargs):
        return kwargs['bcoh_sqrd'] * fq / q 

    def S_to_FK(self, q, sq, **kwargs):
        fq = self.S_to_F(q,sq)
        return self.F_to_FK(q, fq, **kwargs)

    def FK_to_S(self, q, fq_keen, **kwargs):
        fq = self.FK_to_F(q, fq_keen, **kwargs)
        return self.F_to_S(q, fq)

    # Real Space Conversions

    # G(r) = PDF
    def G_to_GK(self, r, gr, **kwargs):
        factor = kwargs['bcoh_sqrd'] / (4. * np.pi * kwargs['rho']) 
        return factor * ( gr / r )

    def G_to_g(self, r, gr, **kwargs):
        factor = 4. * np.pi * kwargs['rho']
        return gr / (factor * r) + 1.

    # Keen's G(r)
    def GK_to_G(self, r, gr, **kwargs):
        factor = (4. * np.pi * kwargs['rho']) / kwargs['bcoh_sqrd']
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

    def _extend_axis_to_low_end(x,decimals=32):
        dx = x[1] - x[0]
        if x[0] == 0.0:
            x[0] = 1e-6
        x = np.linspace(x[0], x[-1], int(x[-1]/dx), endpoint=True)
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

        if kwargs['OmittedXrangeCorrection']:
            self.low_x_correction(xin, yin, xout, yout, **kwargs)

        return xout, yout

    #--------------------------------------#
    # Reciprocal -> Real Space Transforms  #
    #--------------------------------------#

    # F(Q) = Q(SQ-1)
    def F_to_G(self, q, fq, r, **kwargs):
        r, gr = self.fourier_transform(q, fq, r, **kwargs)
        gr = 2./ np.pi * gr
        return r, gr

    def F_to_GK(self, q, fq, r, **kwargs):
        r, gr = self.F_to_G(q, fq, r)
        gr = self.converter.G_to_GK(r, gr, **kwargs)
        return r, gr

    def F_to_g(self, q, fq, r, **kwargs):
        r, gr = self.F_to_G(q, fq, r)
        gr = self.converter.G_to_g(r, gr, **kwargs)
        return r, gr

    # S(Q)
    def S_to_G(self, q, sq, r, **kwargs):
        fq = self.converter.S_to_F(q, sq)
        r, gr = self.F_to_G(q, fq, r)
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
        r, gr = self.F_to_G(q, fq, r)
        return r, gr

    def FK_to_GK(self, q, fq_keen, r, **kwargs):
        fq = self.converter.FK_to_F(q, fq_keen, **kwargs)
        r, gr =  self.F_to_GK(q, fq, r, **kwargs)
        return r, gr
         
    def FK_to_g(self, q, fq_keen, r, **kwargs):
        fq = self.converter.FK_to_F(q, fq_keen, **kwargs)
        r, gr =  self.F_to_g(q, fq, r, **kwargs)
        return r, gr

    #--------------------------------------#
    # Real -> Reciprocal Space Transforms  #
    #--------------------------------------#

    # G(R) = PDF
    def G_to_F(self, r, gr, q, **kwargs):
        q = self._extend_axis_to_low_end(q)
        q, fq = self.fourier_transform(r, gr, q)
        return q, fq

    def G_to_S(self, r, gr, q, **kwargs):
        q, fq = self.G_to_F(r, gr, q)
        sq = self.converter.F_to_S(q, fq) 
        return q, sq

    def G_to_FK(self, r, gr, q, **kwargs):
        q, fq = self.G_to_F(r, gr, q)
        fq = self.converter.F_to_FK(q, fq, **kwargs)
        return q, fq

    # Keen's G(r)
    def GK_to_F(self, r, gr, q, **kwargs):
        gr = self.converter.GK_to_G(r, gr, **kwargs)
        return self.G_to_F(r, gr, q)

    def GK_to_S(self, r, gr, q, **kwargs):
        gr = self.converter.GK_to_G(r, gr, **kwargs)
        return self.G_to_S(r, gr, q)

    def GK_to_FK(self, r, gr, q, **kwargs):
        gr = self.converter.GK_to_G(r, gr, **kwargs)
        q, fq = self.G_to_FK(r, gr, q, **kwargs)
        return q, fq

    # g(r)
    def g_to_F(self, r, gr, q, **kwargs):
        gr = self.converter.g_to_G(r, gr, **kwargs)
        return self.G_to_F(r, gr, q)

    def g_to_S(self, r, gr, q, **kwargs):
        gr = self.converter.g_to_G(r, gr, **kwargs)
        return self.G_to_S(r, gr, q)

    def g_to_FK(self, r, gr, q, **kwargs):
        gr = self.converter.g_to_G(r, gr, **kwargs)
        return self.G_to_FK(r, gr, q, **kwargs)
        

# -------------------------------------#
# Converter / Transform Factory
tf = Transformer()
transform_functions =  inspect.getmembers(tf, predicate=inspect.ismethod)
transform_dict = { entry[0] : entry[1] for entry in transform_functions }
transform_dict.pop('fourier_transform')
transform_dict.pop('_extend_axis_to_low_end')
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
    parser.add_argument('-b', '--<bcoh>^2', type=float, default=None, dest='bcoh_sqrd',
                        help='Squared mean coherent scattering length (units=fm^2)')
    parser.add_argument('--rho', type=float, default=None, dest='rho',
                        help='Number density (units=atoms/angstroms^3)')
    parser.add_argument('--plot', action='store_true',
                        help='Show plot of before and after transformation')

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
        kwargs['bcoh_sqrd'] = args.bcoh_sqrd
    if args.rho:
        kwargs['rho'] = args.rho

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
