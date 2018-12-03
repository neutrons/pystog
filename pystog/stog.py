"""
=============
StoG
=============

This module defines the StoG class
that tries to replicate the previous
stog program behavior in an organized fashion
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pystog.converter import Converter
from pystog.transformer import Transformer
from pystog.fourier_filter import FourierFilter


class StoG:
    """The StoG class is used to put together
    the Converter, Transformer, and FourierFilter
    class functionalities to reproduce the original
    stog program behavior. Yet, this pythonized-version
    uses pandas and numpy for data storage, organization, and
    manipulation and matplotlib for diagonostic visualization
    "under the hood".

    :examples:

    >>> import json
    >>> from pystog import StoG
    >>> with open("../data/examples/argon_pystog.json", 'r') as f:
    >>>     kwargs = json.load(f)
    >>> stog = StoG(**kwargs)
    >>> stog.read_all_data()
    >>> stog.merge_data()
    >>> stog.write_out_merged_data()
    """

    def __init__(self, **kwargs):
        # Reciprocal space attributes
        self.__sq_title = "S(Q) Merged"
        self.__qsq_minus_one_title = "Q[S(Q)-1] Merged"
        self.__ft_title = "FT term"
        self.__sq_ft_title = "S(Q) FT"
        self.__fq_rmc_title = "F(Q) RMC"

        self.__df_individuals = pd.DataFrame()
        self.__df_sq_individuals = pd.DataFrame()
        self.__df_sq_master = pd.DataFrame()

        # Real space attributes
        if "RealSpaceFunction" in kwargs:
            self.__real_space_function = str(kwargs["RealSpaceFunction"])
        else:
            self.__real_space_function = "g(r)"

        self.__df_gr_master = pd.DataFrame()
        self.__gr_ft_title = "%s FT" % self.real_space_function
        self.__dr_ft_title = "D(r) FT"
        self.__gr_lorch_title = "%s FT Lorched" % self.real_space_function
        self.__gr_title = "%s Merged" % self.real_space_function
        self.__gr_rmc_title = "G(r) RMC"

        # Plotting key-word arguments for visualization
        self._plotting_kwargs = {'figsize': (16, 8),
                                 'style': '-',
                                 'ms': 1,
                                 'lw': 1,
                                 }

        self.__files = kwargs["Files"]
        self.xmin = 100
        self.xmax = 0

        self.dr = None

        self.density = kwargs["NumberDensity"]
        self.stem_name = kwargs["Outputs"]["StemName"]
        self.Rmax = float(kwargs["Rmax"])

        self.low_q_correction = None

        if "FourierFilter" in kwargs:
            self.fourier_filter_cutoff = kwargs["FourierFilter"]["Cutoff"]
        if "Rdelta" in kwargs:
            self.Rdelta = kwargs["Rdelta"]
        elif "Rpoints" in kwargs:
            self.Rdelta = self.Rmax / kwargs["Rpoints"]
        else:
            raise Exception("ERROR: Need either Rpoints or Rdelta")
        if "LorchFlag" in kwargs:
            self.lorch_flag = kwargs["LorchFlag"]
        if "PlotFlag" in kwargs:
            self.plot_flag = kwargs["PlotFlag"]
        if "<b_coh>^2" in kwargs:
            self.bcoh_sqrd = kwargs["<b_coh>^2"]
        if "<b_tot^2>" in kwargs:
            self.btot_sqrd = kwargs["<b_tot^2>"]
        if 'OmittedXrangeCorrection' in kwargs:
            self.low_q_correction = kwargs['OmittedXrangeCorrection']

        if "Merging" in kwargs:
            self.merging = kwargs["Merging"]
        else:
            self.merging = {"Y": {"Offset": 0.0, "Scale": 1.0}}

        if "Transform" in self.merging:
            transform_opts = self.merging["Transform"]
            if "Qmin" in transform_opts:
                self.qmin = transform_opts["Qmin"]
            if "Qmax" in transform_opts:
                self.qmax = transform_opts["Qmax"]

        self.converter = Converter()
        self.transformer = Transformer()
        self.filter = FourierFilter()

        @property
        def plotting_kwargs(self):
            """The plot settings for visualization via matplotlib

            :getter: Returns the current arguments
            :setter: Sets the plotting kwargs
            :type: dict
            """
            return self._plot_kwargs

        @plotting_kwargs.setter
        def plotting_kwargs(self, kwargs):
            self._plotting_kwargs = kwargs

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

    def add_dataset(self, info, decimals=4, **kwargs):
        x = np.array(info['data']['x'])
        y = np.array(info['data']['y'])

        x, y = self.transformer.apply_cropping(
            x, y, info['Qmin'], info['Qmax'])
        x, y = self.apply_scales_and_offset(x, y,
                                            info['Y']['Scale'],
                                            info['Y']['Offset'],
                                            info['X']['Offset'])
        self.xmin = min(self.xmin, min(x))
        self.xmax = max(self.xmax, max(x))
        if hasattr(self, 'qmin'):
            if self.xmin < self.qmin:
                x, y = self.transformer.apply_cropping(
                    x, y, self.qmin, self.xmax)
        if hasattr(self, 'qmax'):
            if self.xmax > self.qmax:
                x, y = self.transformer.apply_cropping(
                    x, y, self.xmin, self.qmax)

        if info["ReciprocalFunction"] != "S(Q)":
            df = pd.DataFrame(
                y, columns=[
                    '%s_%d' %
                    (info['ReciprocalFunction'], info['index'])], index=x)
            self.df_individuals = pd.concat([self.df_individuals, df], axis=1)

        if info["ReciprocalFunction"] == "F(Q)":
            y = self.converter.F_to_S(x, y)
        elif info["ReciprocalFunction"] == "FK(Q)":
            y = self.converter.FK_to_S(x, y, **{'<b_coh>^2': self.bcoh_sqrd})
        elif info["ReciprocalFunction"] == "DCS(Q)":
            y = self.converter.DCS_to_S(
                x, y, **{'<b_coh>^2': self.bcoh_sqrd,
                         '<b_tot^2>': self.btot_sqrd})

        df = pd.DataFrame(y, columns=['S(Q)_%d' % info['index']], index=x)
        self.df_sq_individuals = pd.concat(
            [self.df_sq_individuals, df], axis=1)
        return df

    def apply_scales_and_offset(self, x, y, yscale, yoffset, xoffset):
        y = self.scale(y, yscale)
        y = self.offset(y, yoffset)
        x = self.offset(x, xoffset)
        return x, y

    def merge_data(self):
        # Sum over single S(Q) columns into a merged S(Q)
        single_sofqs = self.df_sq_individuals.iloc[:, :]
        self.df_sq_master[self.sq_title] = single_sofqs.mean(axis=1)

        x = self.df_sq_master[self.sq_title].index.values
        y = self.df_sq_master[self.sq_title].values

        x, y = self.apply_scales_and_offset(x, y,
                                            self.merging['Y']['Scale'],
                                            self.merging['Y']['Offset'],
                                            0.0)
        self.df_sq_master[self.sq_title] = y

        return

    def offset(self, data, offset):
        data = data + offset
        return data

    def scale(self, data, scale):
        data = scale * data
        return data

    # -------------------------------------#
    # Transform Utilities

    def create_dr(self):
        self.dr = np.arange(self.Rdelta, self.Rmax + self.Rdelta, self.Rdelta)

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
                F2 = (np.sin(vm) / (r - PiOverQmax) - np.sin(vp) /
                      (r + PiOverQmax)) / (2. * PiOverQmax)
            else:
                F1 = (2. * v * np.sin(v) - (v * v - 2.) *
                      np.cos(v) - 2.) / r / r / r
                F2 = (np.sin(v) - v * np.cos(v)) / r / r

            correction[i] = (2 / np.pi) * (F1 * sofq_qmin / qmin - F2)
        gofr += correction

        self._omitted_correction = correction

        return gofr

    def fourier_filter(self):
        kwargs = {'lorch': False,
                  'rho': self.density,
                  '<b_coh>^2': self.bcoh_sqrd,
                  'OmittedXrangeCorrection': self.low_q_correction
                  }
        cutoff = self.fourier_filter_cutoff

        # Get reciprocal and real space data
        r = self.df_gr_master[self.gr_title].index.values
        gr = self.df_gr_master[self.gr_title].values
        q = self.df_sq_master[self.sq_title].index.values
        sq = self.df_sq_master[self.sq_title].values

        # Fourier filter g(r)
        if self.real_space_function == "g(r)":
            q_ft, sq_ft, q, sq, r, gr = self.filter.g_using_S(
                r, gr, q, sq, cutoff, **kwargs)
        elif self.real_space_function == "G(r)":
            q_ft, sq_ft, q, sq, r, gr = self.filter.G_using_S(
                r, gr, q, sq, cutoff, **kwargs)
        elif self.real_space_function == "GK(r)":
            q_ft, sq_ft, q, sq, r, gr = self.filter.GK_using_S(
                r, gr, q, sq, cutoff, **kwargs)
        else:
            raise Exception(
                "ERROR: Unknown real space function %s" %
                self.real_space_function)

        # Add output to master dataframes and write files
        self.df_sq_master = self.add_to_dataframe(
            q_ft, sq_ft, self.df_sq_master, self.ft_title)
        self.write_out_ft()

        self.df_sq_master = self.add_to_dataframe(
            q, sq, self.df_sq_master, self.sq_ft_title)
        self.write_out_ft_sq()

        self.df_gr_master = self.add_to_dataframe(
            r, gr, self.df_gr_master, self.gr_ft_title)
        self.write_out_ft_gr()

        # Plot results
        if self.plot_flag:
            exclude_list = [self.qsq_minus_one_title, self.sq_ft_title]
            columns_diff = self.df_sq_master.columns.difference(exclude_list)
            df_sq = self.df_sq_master.ix[:, columns_diff]
            self.plot_sq(
                df_sq,
                ylabel="FourierFilter(Q)",
                title="Fourier Transform of the low-r region below cutoff")
            exclude_list = [self.qsq_minus_one_title]
            df_sq = self.df_sq_master.ix[:, columns_diff]
            self.plot_sq(df_sq, title="Fourier Filtered S(Q)")
            self.plot_gr(
                self.df_gr_master,
                title="Fourier Filtered %s" %
                self.real_space_function)

        return q, sq, r, gr

    def apply_lorch(self, q, sq, r):
        if self.real_space_function == "g(r)":
            r, gr_lorch = self.transformer.S_to_g(
                q, sq, r, **{'lorch': True, 'rho': self.density})
        elif self.real_space_function == "G(r)":
            r, gr_lorch = self.transformer.S_to_G(q, sq, r, **{'lorch': True})
        elif self.real_space_function == "GK(r)":
            r, gr_lorch = self.transformer.S_to_GK(
                q, sq, r, **{'lorch': True, 'rho': self.density})
        else:
            raise Exception(
                "ERROR: Unknown real space function %s" %
                self.real_space_function)

        self.df_gr_master = self.add_to_dataframe(
            r, gr_lorch, self.df_gr_master, self.gr_lorch_title)
        self.write_out_lorched_gr()

        if self.plot_flag:
            self.plot_gr(
                self.df_gr_master,
                title="Lorched %s" %
                self.real_space_function)

        return r, gr_lorch

    def get_lowR_mean_square(self):
        gofr = self.df_gr_master[self.gr_title].values
        return self.lowR_mean_square(self.dr, gofr)

    def lowR_mean_square(self, dr, gofr, limit=1.01):
        gofr = gofr[dr <= limit]
        gofr_sq = np.multiply(gofr, gofr)
        average = sum(gofr_sq)
        return np.sqrt(average)

    def add_keen_fq(self, q, sq):
        kwargs = {'rho': self.density, "<b_coh>^2": self.bcoh_sqrd}
        fq_rmc = self.converter.S_to_FK(q, sq, **kwargs)
        self.df_sq_master = self.add_to_dataframe(
            q, fq_rmc, self.df_sq_master, self.fq_rmc_title)
        self.write_out_rmc_fq()

    def add_keen_gr(self, r, gr):
        kwargs = {'rho': self.density, "<b_coh>^2": self.bcoh_sqrd}
        if self.real_space_function == "g(r)":
            gr_rmc = self.converter.g_to_GK(r, gr, **kwargs)
        elif self.real_space_function == "G(r)":
            gr_rmc = self.converter.G_to_GK(r, gr, **kwargs)
        elif self.real_space_function == "GK(r)":
            gr_rmc = gr
        else:
            raise Exception(
                "ERROR: Unknown real space function %s" %
                self.real_space_function)

        self.df_gr_master = self.add_to_dataframe(
            r, gr_rmc, self.df_gr_master, self.gr_rmc_title)
        self.write_out_rmc_gr()

    # -------------------------------------#
    # Plot Utilities

    def plot_sq(self, df, xlabel='Q', ylabel='S(Q)', title=''):
        df.plot(**self.plotting_kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def plot_gr(self, df, xlabel='r', ylabel='G(r)', title=''):
        df.plot(**self.plotting_kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def plot_merged_sq(self):
        plot_kwargs = self.plotting_kwargs.copy()
        plot_kwargs['style'] = 'o-'
        plot_kwargs['lw'] = 0.5

        fig, axes = plt.subplots(2, 2, sharex=True)
        plt.xlabel("Q")

        # Plot the inividual reciprocal functions
        if self.df_individuals.empty:
            self.df_sq_individuals.plot(ax=axes[0, 0], **plot_kwargs)
        else:
            self.df_individuals.plot(ax=axes[0, 0], **plot_kwargs)

        # Plot the inividual S(Q) functions
        self.df_sq_individuals.plot(ax=axes[0, 1], **plot_kwargs)
        axes[0, 1].set_ylabel("S(Q)")
        axes[0, 1].set_title("Individual S(Q)")

        # Plot the merged S(Q)
        df_sq = self.df_sq_master.ix[:, [self.sq_title]]
        df_sq.plot(ax=axes[1, 0], **plot_kwargs)
        axes[1, 0].set_title("Merged S(Q)")
        axes[1, 0].set_ylabel("S(Q)")

        # Plot the merged Q[S(Q)-1]
        df_fq = self.df_sq_master.ix[:, [self.qsq_minus_one_title]]
        df_fq.plot(ax=axes[1, 1], **plot_kwargs)
        axes[1, 1].set_title("Merged Q[S(Q)-1]")
        axes[1, 1].set_ylabel("Q[S(Q)-1]")

        plt.show()

    def plot_summary_sq(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        columns = self.df_sq_master.columns
        columns_diff = columns.difference([self.fq_rmc_title])
        df_sq = self.df_sq_master.ix[:, columns_diff]
        df_sq.plot(ax=ax1, **self.plotting_kwargs)
        df_fq = self.df_sq_master.ix[:, [self.fq_rmc_title]]
        df_fq.plot(ax=ax2, **self.plotting_kwargs)
        plt.xlabel("Q")
        ax1.set_ylabel("S(Q)")
        ax1.set_title("StoG S(Q) functions")
        ax2.set_ylabel("FK(Q)")
        ax2.set_title("Keen's F(Q)")
        plt.show()

    def plot_summary_gr(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        columns = self.df_gr_master.columns
        columns_diff = columns.difference([self.gr_rmc_title])
        df_gr = self.df_gr_master.ix[:, columns_diff]
        df_gr.plot(ax=ax1, **self.plotting_kwargs)
        df_gk = self.df_gr_master.ix[:, [self.gr_rmc_title]]
        df_gk.plot(ax=ax2, **self.plotting_kwargs)
        plt.xlabel("r")
        ax1.set_ylabel(self.real_space_function)
        ax1.set_title("StoG %s functions" % self.real_space_function)
        ax2.set_ylabel("GK(r)")
        ax2.set_title("Keen's G(r)")
        plt.show()

    # -------------------------------------#
    # Output Utilities

    def add_to_dataframe(self, x, y, df, title):
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
