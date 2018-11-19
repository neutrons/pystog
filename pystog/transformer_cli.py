#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

import inspect
import argparse
import numpy as np
import pandas as pd

from pystog.utils import get_data, create_domain, write_out
from pystog.converter import Converter
from pystog.transformer import Transformer
from pystog.fourier_filter import FourierFilter

ReciprocalSpaceChoices = {"S(Q)": "S(Q)",
                          "F(Q)": "=Q[S(Q) - 1]",
                          "FK(Q)": "Keen's F(Q)",
                          "DCS(Q)": "Differential Cross-Section"}
RealSpaceChoices = {"g(r)": ' "little" g(r)',
                    "G(r)": "Pair Distribution Function",
                    "GK(r)": "Keen's G(r)"}


# -------------------------------------#
# Converter / Transform Factory
tf = Transformer()
transform_functions = inspect.getmembers(tf, predicate=inspect.ismethod)
transform_dict = {entry[0]: entry[1] for entry in transform_functions}
transform_dict.pop('fourier_transform')
transform_dict.pop('_extend_axis_to_low_end')
transform_dict.pop('_low_x_correction')
transform_dict.pop('__init__')

cv = Converter()
converter_functions = inspect.getmembers(cv, predicate=inspect.ismethod)
converter_dict = {entry[0]: entry[1] for entry in converter_functions}
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
    parser.add_argument(
        '-x',
        '--domain_range',
        nargs=3,
        default=(
            0.0,
            100.0,
            0.05),
        type=float,
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
    parser.add_argument(
        '--bcoh-sqrd',
        type=float,
        default=None,
        dest='bcoh_sqrd',
        help='Squared mean coherent scattering length (units=fm^2)')
    parser.add_argument(
        '--btot-sqrd',
        type=float,
        default=None,
        dest='btot_sqrd',
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
