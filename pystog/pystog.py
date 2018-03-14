#!/usr/bin/env python
import json
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from pytransformer import Converter, Transformer, ReciprocalSpaceChoices


def parser_cli_args(args):
    # Get each file's info and story in dictionary
    files_info = list() 
    for f in args.filenames:
        file_info = dict()
        file_info["Filename"] = f[0]
        file_info["Qmin"] = float(f[1])
        file_info["Qmax"] = float(f[2])
        file_info["Y"] = { "Offset" : float(f[3]), "Scale" : float(f[4]) }
        file_info["X"] = { "Offset" : float(f[5]) }
        file_info["ReciprocalFunction"] = f[6]
        files_info.append(file_info)

    # Get general StoG options
    kwargs = {  "Files" : files_info,
                "NumberDensity" : args.density,
                "Rmax" : args.Rmax,
                "Rpoints" : args.Rpoints, 
                "FourierFilter" : { "Cutoff" : args.fourier_filter_cutoff},
                "LorchFlag" : args.lorch_flag, 
                "PlotFlag" : args.plot, 
                "Outputs" : { "StemName" : args.stem_name },
                "<b_coh>^2" : args.final_scale,
    }
    if args.Rdelta:
        kwargs["Rdelta"] = args.Rdelta

    return kwargs



class PyStoG(object):

    def __init__(self, **kwargs):
        self.df_individuals = pd.DataFrame()

        self.df_sq_master = pd.DataFrame()
        self.sq_title = "S(Q) Merged"
        self.ft_title = "FT term"
        self.sq_ft_title = "S(Q) FT"
        self.fq_rmc_title = "F(Q) RMC"

        self.df_gr_master = pd.DataFrame()
        self.gr_ft_title = "g(r) FT"
        self.dr_ft_title = "D(r) FT"
        self.gr_lorch_title = "g(r) FT Lorched"
        self.gr_title = "g(r) Merged"
        self.gr_rmc_title = "G(r) RMC"

        self.files = kwargs["Files"]
        self.xmin = 100
        self.xmax = 0

        self.dr = None

        self.density = kwargs["NumberDensity"]
        self.stem_name = kwargs["Outputs"]["StemName"]
        self.Rmax = float(kwargs["Rmax"])
        self.Rdelta = self.Rmax / kwargs["Rpoints"]
        self.fourier_filter_cutoff = kwargs["FourierFilter"]["Cutoff"]
        if "Rdelta" in kwargs:
            self.Rdelta = kwargs["Rdelta"]
        self.lorch_flag = kwargs["LorchFlag"]
        self.plot_flag = kwargs["PlotFlag"]
        self.final_scale = kwargs["<b_coh>^2"]

    # -------------------------------------#
    # Reading and Merging Spectrum

    def read_all_data(self, **kwargs):
        if len(self.files) == 0:
            print("No files loaded for PyStog")
            return

        for i, file_info in enumerate(self.files):
            file_info['index'] = i
            self.read_dataset(file_info, **kwargs)
        return

    def read_dataset(self, info, xcol=0, ycol=1, sep=r"\s*", **kwargs):
        data = pd.read_csv(info['Filename'],
                           sep=sep,
                           usecols=[xcol, ycol],
                           names=['x', 'y'],
                           engine='python',
                           **kwargs)
        info['data'] = data
        data = self.add_dataset(info, **kwargs)
        return data

    def add_dataset(self, info, **kwargs):
        x = np.array(info['data']['x'])
        y = np.array(info['data']['y'])

        x, y = self.apply_cropping(x, y, info['Qmin'], info['Qmax'])
        x, y = self.apply_scales_and_offset(
            x, y, info['Y']['Scale'], info['Y']['Offset'], info['X']['Offset'])

        self.xmin = min(self.xmin, min(x))
        self.xmax = max(self.xmax, max(x))
        df = pd.DataFrame(y, columns=['S(Q)_%d' % info['index']], index=x)
        self.df_individuals = pd.concat([self.df_individuals, df], axis=1)
        return df

    def apply_cropping(self, x, y, xmin, xmax):
        y = y[np.logical_and(x >= xmin, x <= xmax)]
        x = x[np.logical_and(x >= xmin, x <= xmax)]
        return x, y

    def apply_scales_and_offset(self, x, y, yscale, yoffset, xoffset):
        y = self.scale(y, yscale)
        y = self.offset(y, yoffset)
        x = self.offset(x, xoffset)
        return x, y

    def merge_data(self):
        # Sum over single S(Q) columns into a merged S(Q)
        self.df_sq_master[self.sq_title] = self.df_individuals.iloc[:, :].mean(
            axis=1)
        return

    def offset(self, data, offset):
        data = data + offset
        return data

    def scale(self, data, scale):
        data = scale * data
        return data

    # -------------------------------------#
    # Transform Utilities

    def extend_axis_to_low_end(self,x,decimals=4):
        dx = x[1] - x[0]
        x = np.linspace(dx, x[-1], int(x[-1]/dx), endpoint=True)
        return np.around(x,decimals=decimals)

    def create_dr(self):
        self.dr = np.arange(self.Rdelta, self.Rmax + self.Rdelta, self.Rdelta)

    '''
    def bit_merged(self):
        self.bit( self.df_sq_master, self.sq_title, lorch=False)

    def bit(self, df, col_name, **kwargs):
        if self.dr is None:
            self.create_dr()

        q = df.index.values
        sofq = df[col_name].values

        kwargs['df'] = df
        r, gofr = self.transform(q, sofq, self.dr, **kwargs)

        self.df_gr_master[self.gr_title] = gofr
        self.df_gr_master = self.df_gr_master.set_index(r)
    '''

    def transform(self, xin, yin, xout, lorch=False, df=None, title=None):
        xmax = max(xin)
        xout = self.extend_axis_to_low_end(xout)

        factor = np.full_like(yin, 1.0)
        if lorch:
            PiOverXmax = np.pi / xmax
            factor = np.sin(PiOverXmax * xin) / (PiOverXmax * xin)

        x_by_y_minus_one = (yin- 1) * xin

        if title is not None:
            y_temp = (factor * x_by_y_minus_one) / xin + 1
            df_temp = pd.DataFrame(y_temp, columns=[title], index=xin)
            df = pd.concat([df, df_temp], axis=1)

        afactor = 2. / np.pi
        yout = np.zeros_like(xout)
        for i, x  in enumerate(xout):
            kernel = factor * x_by_y_minus_one * np.sin(xin * x)
            yout[i] = afactor * np.trapz(kernel, x=xin)

        # Correct for omitted small Q-region
        yout = self.qmin_correction(xin, yin, xout, yout, lorch)

        # Convert to G(r) -> g(r)
        FourPiRho = 4. * np.pi * self.density
        yout = yout / FourPiRho / xout + 1.

        return xout, yout 

    def qmin_correction(self, q, sofq, dr, gofr, lorch):
        qmin = min(q)
        qmax = max(q)
        sofq_qmin = sofq[0]
        PiOverQmax = np.pi / qmax

        correction = np.zeros_like(gofr)
        for i, r in enumerate(dr):
            v = qmin * r
            if lorch:
                vm = qmin * (r - PiOverQmax)
                vp = qmin * (r + PiOverQmax)
                term1 = (vm * np.sin(vm) + np.cos(vm) - 1.) / \
                    (r - PiOverQmax)**2.
                term2 = (vp * np.sin(vp) + np.cos(vp) - 1.) / \
                    (r + PiOverQmax)**2.
                F1 = (term1 - term2) / (2. * PiOverQmax)
                F2 = (np.sin(vm) / (r - PiOverQmax) - np.sin(vp) / \
                    (r + PiOverQmax) )/ (2. * PiOverQmax)
            else:
                F1 = (2. * v * np.sin(v) - (v * v - 2.) *
                      np.cos(v) - 2.) / r / r / r
                F2 = (np.sin(v) - v * np.cos(v)) / r / r

            correction[i] = (2 / np.pi) * (F1 * sofq_qmin / qmin - F2)
        gofr += correction
    
        self._omitted_correction = correction

        return gofr

    def fourier_filter(self):

        # Work figuring out Fourier Filter
        r  = self.df_gr_master[self.gr_title].index.values
        gr = self.df_gr_master[self.gr_title].values
        r, gr = self.apply_cropping(r, gr, 0.0, self.fourier_filter_cutoff)

        # Grab Q values
        q = self.df_sq_master[self.sq_title].index.values
        qmin = min(q)
        qmax = max(q)

        # Extend the low Q-range -> 0.0
        q_ft = self.extend_axis_to_low_end(q)

        # Calculate Fourier Correction
        q_ft, sq_ft = self.transform(r, gr+1., q_ft, lorch=False)
        sq_ft = ((sq_ft-1)*(self.density*self.density*(2.*np.pi)**3.))+1.
        self.df_sq_master = self.add_to_dataframe(q_ft,sq_ft,self.df_sq_master,self.ft_title)
        self.write_out_ft()

        if self.plot_flag:
            self.plot_sq(ylabel="FourierFilter(Q)", 
                         title="Fourier Transform of the filtered low-r region below cutoff")

        # Crop Fourier Correction to match the initial Q-range
        q_ft, sq_ft = self.apply_cropping(q_ft, sq_ft, qmin, qmax)
        
        # Apply Fourier Correction
        q = self.df_sq_master[self.sq_title].index.values
        sq = self.df_sq_master[self.sq_title].values
        q, sq = self.apply_cropping(q, sq, qmin, qmax)
        sq = (sq - sq_ft) + 1
        self.df_sq_master = self.add_to_dataframe(q, sq, self.df_sq_master, self.sq_ft_title)
        self.write_out_ft_sq()

        if self.plot_flag:
            self.plot_sq(title="Fourier Filtered S(Q)")

        # Transform back to g(r) with Fourier Filter Correction
        r  = self.df_gr_master[self.gr_title].index.values
        r, gr_ft = self.transform(q, sq, r, lorch=False)
        self.df_gr_master = self.add_to_dataframe(r, gr_ft, self.df_gr_master, self.gr_ft_title)
        self.write_out_ft_gr()

        if self.plot_flag:
            self.plot_gr(title="Fourier Filtered G(r)")

        return q, sq, r, gr_ft

    def apply_lorch(self,q,sq,r):
        r, gr_lorch = self.transform(q, sq, r, lorch=True)
        self.df_gr_master = self.add_to_dataframe(r, gr_lorch, 
                                                  self.df_gr_master, self.gr_lorch_title)
        self.write_out_lorched_gr()

        if self.plot_flag:
            self.plot_gr(title="Lorched G(r)")
            plt.show()

        return r, gr_lorch

    def get_lowR_mean_square(self):
        gofr = self.df_gr_master[self.gr_title].values
        return self.lowR_mean_square(self.dr, gofr)

    def lowR_mean_square(self, dr, gofr, limit=1.01):
        gofr = gofr[dr <= limit]
        gofr_sq = np.multiply(gofr, gofr)
        average = sum(gofr_sq)
        return np.sqrt(average)

    def add_keen_fq(self):
        fq_rmc = self.final_scale*(sq-1)
        self.df_sq_master = self.add_to_dataframe(q, fq_rmc, self.df_sq_master, self.fq_rmc_title)
        self.write_out_rmc_fq()

    def add_keen_gr(self):
        gr_rmc = self.final_scale*(gr_out-1)
        self.df_gr_master = self.add_to_dataframe(r, gr_rmc, self.df_gr_master, self.gr_rmc_title)
        self.write_out_rmc_gr()
 
    # -------------------------------------#
    # Plot Utilities

    def plot_sq(self,xlabel='Q', ylabel='S(Q)', title=''):
        stog.df_sq_master.plot()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def plot_gr(self,xlabel='r', ylabel='G(r)', title=''):
        stog.df_gr_master.plot()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()


    def plot_merged_sq(self):
        fig, (ax1, ax2) = plt.subplots(1, 2,sharey=True)
        self.df_individuals.plot(ax=ax1)
        self.df_sq_master.plot(ax=ax2)
        plt.xlabel("Q")
        plt.ylabel("S(Q)")
        ax1.set_title("Individual S(Q)")
        ax2.set_title("Merged S(Q)")
        plt.show()

    def plot_merged_gr(self):
        stog.df_gr_master.plot()
        plt.xlabel("r")
        plt.ylabel("G(r)")
        plt.title("Merged G(r)")
        plt.show()

    def plot_summary_sq(self):
        fig, (ax1, ax2) = plt.subplots(1, 2,sharey=True)
        df_sq = self.df_sq_master.ix[ :, self.df_sq_master.columns.difference([self.fq_rmc_title]) ]
        df_sq.plot(ax=ax1)
        df_fq = self.df_sq_master.ix[ :, [self.fq_rmc_title] ]
        df_fq.plot(ax=ax2)
        plt.xlabel("Q")
        ax1.set_ylabel("S(Q)")
        ax1.set_title("StoG S(Q) functions")
        ax2.set_ylabel("FK(Q)")
        ax2.set_title("Keen's F(Q)")
        plt.show()

    def plot_summary_gr(self):
        fig, (ax1, ax2) = plt.subplots(1, 2,sharey=True)
        df_gr = self.df_gr_master.ix[ :, self.df_gr_master.columns.difference([self.gr_rmc_title]) ]
        df_gr.plot(ax=ax1)
        df_gk = self.df_gr_master.ix[ :, [self.gr_rmc_title] ]
        df_gk.plot(ax=ax2)
        plt.xlabel("r")
        ax1.set_ylabel("S(Q)")
        ax1.set_title("StoG G(r) functions")
        ax2.set_ylabel("GK(r)")
        ax2.set_title("Keen's G(r)")
        plt.show()

    # -------------------------------------#
    # Output Utilities

    def add_to_dataframe(self,x,y,df,title):
        df_temp = pd.DataFrame(y, columns=[title], index=x)
        df = pd.concat([df, df_temp], axis=1)
        return df
        
    def write_out_df(self, df, cols, filename):
        if df.empty:
            print("Empty dataframe.")
            return
        with open(filename, 'w') as f:
            f.write("%d \n" % df.shape[0])
            f.write("# Comment line\n")
        with open(filename, 'a') as f:
            df.to_csv(f, sep='\t', columns=cols, header=False)

    def write_out_merged_sq(self, filename=None):
        if filename is None:
            filename = "%s.sq" % self.stem_name
        self.write_out_df(self.df_sq_master, [self.sq_title], filename)

    def write_out_merged_gr(self, filename=None):
        if filename is None:
            filename = "%s.gr" % self.stem_name
        self.write_out_df(self.df_gr_master, [self.gr_title], filename)

    def write_out_ft(self, filename=None):
        if filename is None:
            filename = "ft.dat"
        self.write_out_df(self.df_sq_master, [self.ft_title], filename)

    def write_out_ft_sq(self, filename=None):
        if filename is None:
            filename = "%s_ft.sq" % self.stem_name
        self.write_out_df(self.df_sq_master, [self.sq_ft_title], filename)

    def write_out_ft_gr(self, filename=None):
        if filename is None:
            filename = "%s_ft.gr" % self.stem_name
        self.write_out_df(self.df_gr_master, [self.gr_ft_title], filename)

    def write_out_ft_dr(self, filename=None):
        if filename is None:
            filename = "%s_ft.dr" % self.stem_name
        self.write_out_df(self.df_gr_master, [self.dr_ft_title], filename)

    def write_out_lorched_gr(self, filename=None):
        if filename is None:
            filename = "%s_ft_lorched.gr" % self.stem_name
        self.write_out_df(self.df_gr_master, [self.gr_lorch_title], filename)

    def write_out_rmc_fq(self, filename=None):
        if filename is None:
            filename = "%s_rmc.fq" % self.stem_name
        self.write_out_df(self.df_sq_master, [self.fq_rmc_title], filename)

    def write_out_rmc_gr(self, filename=None):
        if filename is None:
            filename = "%s_rmc.gr" % self.stem_name
        self.write_out_df(self.df_gr_master, [self.gr_rmc_title], filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None,
                        help="Read JSON input file for arguments")
    parser.add_argument("--density", type=float,
                        help="Number density (atoms/angstroms^3")
    parser.add_argument("-f", "--filename", nargs='*', action='append',
                        default=list(), dest='filenames',
                        help="Filename, qmin, qmax, yoffset, yscale, Q offset, function type."+
                             "Function Types are: %s" % json.dumps(ReciprocalSpaceChoices))
    parser.add_argument(
        "--stem-name",
        type=str,
        dest="stem_name",
        default="merged",
        help="Stem name for output files")
    parser.add_argument(
        "--Rmax",
        type=float,
        default=50.0,
        help="Maximum value in angstroms for real-space functions")
    parser.add_argument("--Rpoints", type=int, default=5000,
                        help="Number of points in R for real-space functions")
    parser.add_argument("--Rdelta", type=float, default=None,
                        help="Bin width to use in R for real-space functions")
    parser.add_argument("--fourier-filter-cutoff", type=float,
                        default=None, dest="fourier_filter_cutoff",
                        help="Bin width to use in R for real-space functions")
    parser.add_argument("--lorch-flag", action="store_true",
                        default=False, dest="lorch_flag",
                        help="Apply Lorch function")
    parser.add_argument("--final-scale", type=float, default=1.0, dest="final_scale",
                        help="The (sum c*bbar)^2 term needed for F(Q) and G(r) for RMC output")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Plots using matplotlib along the way")
    args = parser.parse_args()

    
    if args.json:
        print("loading config from '%s'" % args.json)
        with open(args.json, 'r') as f:
            kwargs = json.load(f)

    else:
        kwargs = parser_cli_args(args)

    # Merge S(Q) files
    stog = PyStoG(**kwargs)
    stog.read_all_data(skiprows=3)
    stog.merge_data()
    stog.write_out_merged_sq()

    # Initial S(Q) -> g(r) transform 
    q    = stog.df_sq_master[stog.sq_title].index.values
    sofq = stog.df_sq_master[stog.sq_title].values

    if kwargs["PlotFlag"]:
        stog.plot_merged_sq()

    stog.create_dr()

    r, gofr = stog.transform(q, sofq, stog.dr, lorch=False)
    stog.df_gr_master[stog.gr_title] = gofr
    stog.df_gr_master = stog.df_gr_master.set_index(r)
    stog.write_out_merged_gr()

    if kwargs["PlotFlag"]:
        stog.plot_merged_gr()

    #print stog.get_lowR_mean_square()

    # Set the S(Q) and g(r) if no Fourier Filter
    sq = stog.df_sq_master[stog.sq_title].values
    gr_out = gofr

    if "FourierFilter" in kwargs:
        q, sq, r, gr_out = stog.fourier_filter()

    '''
    # Write out D(r) as well
    dr_ft = (gr_out-1)*r
    stog.df_gr_master = stog.add_to_dataframe(r, dr_ft, stog.df_gr_master, stog.dr_ft_title)
    stog.write_out_ft_dr()
    '''

    # Apply Lorch
    if kwargs["LorchFlag"]:
        r, gr_out = stog.apply_lorch(q,sq,r)

    # Apply final scale number
    stog.add_keen_fq()
    stog.add_keen_gr()
   
    if kwargs["PlotFlag"]:
        stog.plot_summary_sq()
        stog.plot_summary_gr()
