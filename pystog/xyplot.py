#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser("Quick XY plotter")
parser.add_argument('-s', '--skiprows', type=int, default=0,
                    help='Number of rows to skip in datasets')
parser.add_argument('-t', '--trim', type=int, default=0,
                    help='Number of rows to trim off end in datasets')
parser.add_argument('-f', '--filename', nargs='+', action="append",default=list(),
                    help='Filename, x-col, y-col')
parser.add_argument('--filenames', nargs='*',default=list(),
                    help='Multiple filenames. The x and y col are set by --xcol and --ycol. Default x=0, y=1')
parser.add_argument('-x', '--xcol', type=int, default=0,
                    help='Set x-col for multiple filenames (--filenames <filenames>)')
parser.add_argument('-y', '--ycol', type=int, default=1,
                    help='Set y-col for multiple filenames (--filenames <filenames>)')
parser.add_argument('--shift', type=float, default=None,
                    help='Shift factor to add by')
parser.add_argument('--scale', type=float, default=None,
                    help='Scale factor to multiply by')
parser.add_argument('--error', action='store_true', default=False,
                    help='Plot error bars')
parser.add_argument('--yerr_col', type=int, default=2,
                    help='Y-error bar column')
parser.add_argument('--save', type=str, choices=['y','shifted','scaled'], default=None,
                    help='Save data type to file')
parser.add_argument('--title', type=str)

args = parser.parse_args()
kwargs = {"skiprows" : args.skiprows,
          "trim" : args.trim,
          "shift" : args.shift,
          "scale" : args.scale,
          "error" : args.error,
          "yerr_col" : args.yerr_col}

def getKey(f, datasets):
    # Initialize counter and key
    i = 0
    key = f
    while key in datasets:
        # if key in datasets, add counter to end
        i += 1 
        key = "%s_%d" %(f,i)
    return key


def addToDataSet(f,xcol,ycol,datasets,skiprows=0,trim=0,shift=None,scale=None,save=None,error=False,yerr_col=2):

    if error:
        names=['x','y','yerr']
        data = pd.read_csv(f,sep=r"\s*",skiprows=skiprows, skipfooter=trim, usecols=[xcol,ycol,yerr_col], names=names, engine='python')
        data = data[pd.notnull(data['y'])]
        if data.empty:
            return datasets
        key = getKey(f, datasets)
        datasets[key] = { 'x' : np.array(data.x.tolist()), 'y' : np.array(data.y.tolist()), 'yerr' : np.array(data.yerr.tolist()) }
    else:
        names=['x','y']
        data = pd.read_csv(f,sep=r"\s*",skiprows=skiprows, skipfooter=trim, usecols=[xcol,ycol], names=names, engine='python')
        data = data[pd.notnull(data['y'])]
        if data.empty:
            return datasets
        key = getKey(f, datasets)
        datasets[key] = { 'x' : np.array(data.x.tolist()), 'y' : np.array(data.y.tolist()) }
    
    if shift:
        y_shifted = np.add(shift, datasets[key]['y'])
        datasets[key].update( {'shifted' : y_shifted } )

    if scale:
        y_scaled = scale *  datasets[key]['y']
        datasets[key].update( {'scaled'  : y_scaled } )

    if save:
        with open(f+'_SQ.dat','w') as f_tmp:
            for x, y in zip(datasets[key]['x'], datasets[key][save]):
                f_tmp.write('%f %f \n' % (x, y) )

    return datasets

 
datasets = dict()

for f in args.filenames:
    datasets = addToDataSet(f,args.xcol,args.ycol,datasets,**kwargs)

for f in args.filename:
    if f < 3:
        raise Exception("ERROR: For filename input, need: -f <filename> <x-col> <y-col> (optional: y-err col).")
    xcol = int(f[1])
    ycol = int(f[2])
    if len(f) > 3:
        kwargs['yerr_col'] = int(f[3])

    datasets = addToDataSet(f[0],xcol,ycol,datasets,**kwargs)
  
all_filenames = [ f[0] for f in args.filename ]
all_filenames.extend(args.filenames)
 
import matplotlib.pyplot as plt

for key in sorted(datasets):
    dset = datasets[key]
    if args.error:
        plt.errorbar(dset['x'],dset['y'],yerr=dset['yerr'],label=key)
        if args.shift:
            plt.errorbar(dset['x'],dset['shifted'],yerr=dset['yerr'],label=key+'_shifted')
        if args.scale:
            plt.errorbar(dset['x'],dset['scaled'],yerr=dset['yerr'],label=key+'_scaled')

    else:
        plt.plot(dset['x'],dset['y'],'-', label=key)
        if args.shift:
            plt.plot(dset['x'],dset['shifted'],label=key+'_shifted')
        if args.scale:
            plt.plot(dset['x'],dset['scaled'],label=key+'_scaled')
        
    
plt.title(args.title)
plt.legend()
plt.show()
plt.close()
