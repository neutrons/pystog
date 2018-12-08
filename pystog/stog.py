"""
=============
StoG
=============

This module defines the StoG class
that tries to replicate the previous
stog program behavior in an organized fashion
with the ability to re-construct the workflow.
"""

import json
import numpy as np
import pandas as pd

from pystog.utils import create_domain, RealSpaceChoices, ReciprocalSpaceChoices
from pystog.converter import Converter
from pystog.transformer import Transformer
from pystog.fourier_filter import FourierFilter

# Required for non-display environment (i.e. Travis-CI)
import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


class StoG(object):
    """The StoG class is used to put together
    the Converter, Transformer, and FourierFilter
    class functionalities to reproduce the original
    **stog** Fortran program behavior. This class is meant to
    put together the functionality of the classes
    into higher-level calls to construct workflows
    for merging and processing multiple recprical space
    functions into a final output real space function.

    This pythonized-version of the original Fortran **stog**
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
        # General attributes
        self.__xdecimals = 2
        self.__ydecimals = 16
        self.__xmin = 100
        self.__xmax = 0
        self.__qmin = None
        self.__qmax = None
        self.__files = None
        self.__real_space_function = "g(r)"
        self.__rmin = 0.0
        self.__rmax = 50.0
        self.__rdelta = 0.01
        self.__update_dr()
        self.__density = 1.0
        self.__bcoh_sqrd = 1.0
        self.__btot_sqrd = 1.0
        self.__low_q_correction = False
        self.__lorch_flag = False
        self.__fourier_filter_cutoff = None
        self.__plot_flag = True
        self.__plotting_kwargs = {'figsize': (16, 8),
                                  'style': '-',
                                  'ms': 1,
                                  'lw': 1,
                                  }
        self.__merged_opts = {"Y": {"Offset": 0.0, "Scale": 1.0}}
        self.__stem_name = "out"

        # DataFrames for total scattering functions
        self.__df_individuals = pd.DataFrame()
        self.__df_sq_individuals = pd.DataFrame()
        self.__df_sq_master = pd.DataFrame()
        self.__df_gr_master = pd.DataFrame()

        # Attributes that do not (currently) change
        self.__sq_title = "S(Q) Merged"
        self.__qsq_minus_one_title = "Q[S(Q)-1] Merged"
        self._ft_title = "FT term"
        self.__sq_ft_title = "S(Q) FT"
        self.__fq_title = "F(Q) Merged"
        self.__dr_ft_title = "D(r) FT"
        self.__GKofR_title = "G(r) (Keen Version)"

        # Set real space title attributes
        self.__gr_title = "%s Merged" % self.__real_space_function
        self.__gr_ft_title = "%s FT" % self.__real_space_function
        self.__gr_lorch_title = "%s FT Lorched" % self.__real_space_function

        # Interior class attributes

        #: The **converter** attribute defines the Converter
        #: which is used to do all inner conversions necessary
        #: to go from the input reciprocal space functions,
        #: produce diagnostics for the selected real space functions,
        #: and finally output the desired real space function
        #: Must of type :class:`pystog.converter.Converter`
        self.converter = Converter()

        #: The **transformer** attribute defines the Transformer
        #: which is used to do all Fourier transforms necessary
        #: from reciprocal space to real space and vice versa.
        #: Must of type :class:`pystog.transformer. Transformer`
        self.transformer = Transformer()

        #: The **filter** attribute defines the FourierFilter
        #: which is used to do all Fourier filtering if the
        #: **fourier_filter_cutoff** attribute is supplied.
        #: Must of type :class:`pystog.fourier_filter.FourierFilter`
        self.filter = FourierFilter()

        # Use key-word arguments to set attributes
        self.__kwargs2attr(kwargs)

    def __kwargs2attr(self, kwargs):
        """Takes the key-word arguments supplied to the
        initialization of the class and maps them to
        attirbutes in the class. Commonly get the **kwargs**
        from the JSON input file.
        """
        if "Files" in kwargs:
            self.files = kwargs["Files"]
        if "RealSpaceFunction" in kwargs:
            self.real_space_function = str(kwargs["RealSpaceFunction"])
        if "Rmin" in kwargs:
            self.rmin = float(kwargs["Rmin"])
        if "Rmax" in kwargs:
            self.rmax = float(kwargs["Rmax"])
        if "Rdelta" in kwargs:
            self.rdelta = kwargs["Rdelta"]
        elif "Rpoints" in kwargs:
            self.rdelta = self.rmax / kwargs["Rpoints"]
        if "NumberDensity" in kwargs:
            self.density = kwargs["NumberDensity"]
        if 'OmittedXrangeCorrection' in kwargs:
            self.low_q_correction = kwargs['OmittedXrangeCorrection']
        if "LorchFlag" in kwargs:
            self.lorch_flag = kwargs["LorchFlag"]
        if "FourierFilter" in kwargs:
            if "Cutoff" in kwargs["FourierFilter"]:
                self.fourier_filter_cutoff = kwargs["FourierFilter"]["Cutoff"]
        if "PlotFlag" in kwargs:
            self.plot_flag = kwargs["PlotFlag"]
        if "<b_coh>^2" in kwargs:
            self.bcoh_sqrd = kwargs["<b_coh>^2"]
        if "<b_tot^2>" in kwargs:
            self.btot_sqrd = kwargs["<b_tot^2>"]
        if "Merging" in kwargs:
            self.merged_opts = kwargs["Merging"]
            if "Transform" in kwargs['Merging']:
                transform_opts = kwargs['Merging']["Transform"]
                if "Qmin" in transform_opts:
                    self.qmin = transform_opts["Qmin"]
                if "Qmax" in transform_opts:
                    self.qmax = transform_opts["Qmax"]
        if "Outputs" in kwargs:
            if "StemName" in kwargs["Outputs"]:
                self.stem_name = kwargs["Outputs"]["StemName"]

    # General attributes
    @property
    def xmin(self):
        """The minimum X value of all datasets to
        use for Fourier transforms (from recirocal space -> real space)

        :getter: Returns the current set value
        :setter: Set the xmin value for the Fourier transforms
        :type: float
        """
        return self.__xmin

    @xmin.setter
    def xmin(self, value):
        self.__xmin = value

    @property
    def xmax(self):
        """The maximum X value of all datasets to
        use for Fourier transforms (from recirocal space -> real space)

        :getter: Returns the current set value
        :setter: Set the xmax value for the Fourier transforms
        :type: float
        """
        return self.__xmax

    @xmax.setter
    def xmax(self, value):
        self.__xmax = value

    @property
    def qmin(self):
        """The :math:`Q_{min}` value to use for the Fourier
        transforms (from recirocal space -> real space). This
        overrides **xmin** attribute if **xmin** < **qmin**.

        :getter: Returns the current set value
        :setter: Set the :math:`Q_{min}` value for the Fourier transforms
        :type: float
        """
        return self.__qmin

    @qmin.setter
    def qmin(self, value):
        self.__qmin = value

    @property
    def qmax(self):
        """The :math:`Q_{max}` value to use for the Fourier
        transforms (from recirocal space -> real space). This
        overrides **xmax** attribute if **qmax** < **xmax**.

        :getter: Returns the current set value
        :setter: Set the qmax value for the Fourier transforms
        :type: float
        """
        return self.__qmax

    @qmax.setter
    def qmax(self, value):
        self.__qmax = value

    @property
    def files(self):
        """The files that contain the reciprocal space data
        to merge together.

        :getter: Current list of files to merge
        :setter: Set the list of files to merg
        :type: list
        """
        return self.__files

    @files.setter
    def files(self, file_list):
        self.__files = file_list

    def append_file(self, new_file):
        """Appends a file to the file list

        :param new_file: New file name to append
        :type new_file: str
        :return: File list with appended new_file
        :rtype: list
        """
        self.files = self.files + [new_file]
        return self.files

    def extend_file_list(self, new_files):
        """Extend the file list with a list of new files

        :param new_files: List of new files
        :type new_file: list
        :return: File list extended by new_files
        :rtype: list
        """
        self.files.extend(new_files)
        return self.files

    def __update_dr(self):
        """Uses **rdelta** and **rmax** attributes (:math:`\\Delta r` and
        :math:`R_{max}`, respectively) to construct **dr** attribute
        (:math:`r`-space vector) via its setter
        """
        self.dr = create_domain(self.rmin, self.rmax, self.rdelta)

    @property
    def rdelta(self):
        """The :math:`\\Delta r` for the :math:`r`-space vector

        :getter: Return :math:`\\Delta r` value
        :setter: Set the :math:`\\Delta r` value and update :math:`r`-space vector
                 via the **dr** attribute
        :type: value
        """
        return self.__rdelta

    @rdelta.setter
    def rdelta(self, value):
        self.__rdelta = value
        self.__update_dr()

    @property
    def rmin(self):
        """The :math:`R_{min}` valuefor the :math:`r`-space vector

        :getter: Return :math:`R_{min}` value
        :setter: Set the :math:`R_{min}` value and update :math:`r`-space vector
                 via the **dr** attribute
        :type: value
        """
        return self.__rmin

    @rmin.setter
    def rmin(self, value):
        self.__rmin = value
        self.__update_dr()

    @property
    def rmax(self):
        """The :math:`R_{max}` valuefor the :math:`r`-space vector

        :getter: Return :math:`R_{max}` value
        :setter: Set the :math:`R_{max}` value and update :math:`r`-space vector
                 via the **dr** attribute
        :type: value
        """
        return self.__rmax

    @rmax.setter
    def rmax(self, value):
        self.__rmax = value
        self.__update_dr()

    @property
    def dr(self):
        """The real space function X axis data, :math:`r`-space vector

        :getter: Return the :math:`r` vector
        :setter: Set the :math:`r` vector
        :type: numpy.array
        """
        return self.__dr

    @dr.setter
    def dr(self, r):
        self.__dr = r

    @property
    def density(self):
        """The number density used (atoms/:math:`\\AA^{3}`) used for the
        :math:`\\rho_{0}` term in the equations

        :getter: Return the density value
        :setter: Set the density value
        :type: float
        """
        return self.__density

    @density.setter
    def density(self, value):
        self.__density = value

    @property
    def bcoh_sqrd(self):
        """The average coherent scattering length, squared:
        :math:`\\langle b_{coh} \\rangle^2` =
        :math:`( \\sum_{i} c_{i} b_{coh,i} ) ( \\sum_{i} c_{i} b^*_{coh,i} )`
        where the subscript :math:`i` implies for atom type :math:`i`,
        :math:`c_{i}` is the concentration of :math:`i`,
        :math:`b_{coh,i}` is the coherent scattering length of :math:`i`,
        and :math:`b^*_{coh,i}` is the complex coherent scattering length of :math:`i`


        The real part of the :math:`b_{coh,i}` term can be found
        from the **Coh b** column of the NIST neutron scattering
        length and cross section table found here:
        https://www.ncnr.nist.gov/resources/n-lengths/list.html

        Units are in :math:`fm` in the table for the :math:`b_{coh,i}` term.
        Thus, :math:`\\langle b_{coh} \\rangle^2` has units of :math:`fm^2`
        (and what PyStoG expects). NOTE: 100 :math:`fm^2` == 1 :math:`barn`.

        :getter: Return the value of :math:`\\langle b_{coh} \\rangle^2`
        :setter: Set the value for :math:`\\langle b_{coh} \\rangle^2`
        :type: float
        """
        # TODO: Add an example with code to demostrate
        return self.__bcoh_sqrd

    @bcoh_sqrd.setter
    def bcoh_sqrd(self, value):
        self.__bcoh_sqrd = value

    @property
    def btot_sqrd(self):
        """The average coherent scattering length, squared:
        :math:`\\langle b_{tot}^2 \\rangle`
        = :math:`\\sum_{i} c_{i} b_{tot,i}^2`
        = :math:`\\frac{1}{4 \\pi} \\sum_i c_{i} \\sigma_{tot,i}`
        where the subscript :math:`i` implies for atom type :math:`i`,
        :math:`c_{i}` is the concentration of :math:`i` and
        :math:`\\sigma_{tot,i}` is the total cross-section of :math:`i`


        The real part of the :math:`b_{coh,i}` term can be found
        from the **Scatt xs** column of the NIST neutron scattering
        length and cross section table found here:
        https://www.ncnr.nist.gov/resources/n-lengths/list.html

        Units are in :math:`barn` (=100 :math:`fm^2`) in the table
        for the :math:`\\sigma_{tot,i}` term. Thus, you must multiply
        :math:`\\sum_{i} c_{i} b_{tot,i}^2` by 100 to go from :math:`barn`
        to :math:`fm^2` (what PyStoG expects).

        :getter: Return the value of :math:`\\sum_{i} c_{i} b_{tot,i}^2`
        :setter: Set the value for :math:`\\sum_{i} c_{i} b_{tot,i}^2`
        :type: float
        """
        # TODO: Add an example with code to demostrate
        return self.__btot_sqrd

    @btot_sqrd.setter
    def btot_sqrd(self, value):
        self.__btot_sqrd = value

    @property
    def stem_name(self):
        """A stem name to prefix for all output files. Replicates
        the **stog** Fortran program behavior.

        :getter: Return the currently set stem name
        :setter: Set the stem name for output files
        :type: str
        """
        return self.__stem_name

    @stem_name.setter
    def stem_name(self, name):
        self.__stem_name = name

    @property
    def low_q_correction(self):
        """This sets the option to perform a low-:math:`Q` correction
        for the omitted :math:`Q` range.

        See :class:`pystog.transformer.Transformer` **_low_x_correction**
        method for more information.

        :getter: Return bool of applying the low-:math:`Q` correction
        :setter: Set whether the correction is applied or not
        :type: bool
        """
        return self.__low_q_correction

    @low_q_correction.setter
    def low_q_correction(self, value):
        if isinstance(value, bool):
            self.__low_q_correction = value
        else:
            raise TypeError("Expected a bool, True or False")

    @property
    def lorch_flag(self):
        """This sets the option to perform the Lorch dampening correction
        for the :math:`Q` range. Generally, will help reduce Fourier "ripples",
        or AKA "Gibbs phenomenon", due to discontinuity at :math:`Q_{max}`
        if the reciprocal space function is not at the :math:`Q -> \\infty` limit.
        Yet, will also broaden real space function peaks, possibly dubiously.

        See :class:`pystog.transformer.Transformer` **fourier_transform** and
        **_low_x_correction** methods for where this is applied.

        :getter: Return bool of applying the Lorch dampening correction
        :setter: Set whether the correction is applied or not
        :type: bool
        """
        return self.__lorch_flag

    @lorch_flag.setter
    def lorch_flag(self, value):
        if isinstance(value, bool):
            self.__lorch_flag = value
        else:
            raise TypeError("Expected a bool, True or False")

    @property
    def fourier_filter_cutoff(self):
        """This sets the cutoff in :math:`r`-space for the Fourier
        filter. The minimum is automatically 0.0. Thus, from
        0.0 to **fourier_filter_cutoff** is reverse transfomed,
        subtracted in reciprocal space, and then the difference
        is back-transformed.

        See :class:`pystog.fourier_filter.FourierFilter` for more information.

        :getter: Return currently set cutoff value
        :setter: Set cutoff value
        :type: float
        """
        return self.__fourier_filter_cutoff

    @fourier_filter_cutoff.setter
    def fourier_filter_cutoff(self, value):
        self.__fourier_filter_cutoff = value

    @property
    def merged_opts(self):
        """This sets the options to perform after merging the
        reciprocal space functions together, such as an
        overall offset and scale.

        :getter: Return the options currently set
        :setter: Set the options for the merged pattern
        :type: dict
        """
        return self.__merged_opts

    @merged_opts.setter
    def merged_opts(self, options):
        self.__merged_opts = options

    # DataFrame attributes

    @property
    def df_individuals(self):
        """The DataFrame for the input reciprocal space functions
        loaded from **files** and with the loading processing from **add_dataset**
        class method.

        :getter: Returns the current individual, input
                 reciprocal space functions DataFrame
        :setter: Sets the DataFrame
        :type: pandas.DataFrame
        """
        return self.__df_individuals

    @df_individuals.setter
    def df_individuals(self, df):
        self.__df_individuals = df

    @property
    def df_sq_individuals(self):
        """The DataFrame for the :math:`S(Q)` generated from each input
        reciprocal space dataset in **df_individuals** class DataFrame.

        :getter: Returns the current individual :math:`S(Q)`
                 reciprocal space functions DataFrame
        :setter: Sets the DataFrame
        :type: pandas.DataFrame
        """
        return self.__df_sq_individuals

    @df_sq_individuals.setter
    def df_sq_individuals(self, df):
        self.__df_sq_individuals = df

    @property
    def df_sq_master(self):
        """The "master" DataFrame for the :math:`S(Q)` reciprocal
        space functions that are generated for each processing step.
        """
        return self.__df_sq_master

    @df_sq_master.setter
    def df_sq_master(self, df):
        self.__df_sq_master = df

    @property
    def df_gr_master(self):
        """The "master" DataFrame for the real space functions
        that are generated for each processing step.

        :getter: Returns the current "master" real space function DataFrame
        :setter: Sets the "master" real space function DataFrame
        :type: pandas.DataFrame
        """
        return self.__df_gr_master

    @df_gr_master.setter
    def df_gr_master(self, df):
        self.__df_gr_master = df

    # Visualization attributes

    @property
    def plot_flag(self):
        """This sets the option for matplotlib to display
        diagnostic plots or not.

        :getter: Return bool of flag
        :setter: Set whether the diagnostics are displayed or not
        :type: bool
        """
        return self.__plot_flag

    @plot_flag.setter
    def plot_flag(self, value):
        if isinstance(value, bool):
            self.__plot_flag = value
        else:
            raise TypeError("Expected a bool, True or False")

    @property
    def plotting_kwargs(self):
        """The plot settings for visualization via matplotlib

        :getter: Returns the current arguments
        :setter: Sets the plotting kwargs
        :type: dict
        """
        return self.__plotting_kwargs

    @plotting_kwargs.setter
    def plotting_kwargs(self, kwargs):
        self.__plotting_kwargs = kwargs

    @property
    def real_space_function(self):
        """The real space function to use throughoutt the processing

        :getter: Returns the currently select real space function
        :setter: Set the selected real space function and
                 updates other title attributes that rely on this
                 in their name.
        :type: str
        """
        return self.__real_space_function

    @real_space_function.setter
    def real_space_function(self, real_space_function):
        if real_space_function not in RealSpaceChoices:
            raise ValueError("real_space_function must be of %s" %
                             ','.join(RealSpaceChoices.keys()))
        self.__real_space_function = real_space_function
        self.__gr_ft_title = "%s FT" % real_space_function
        self.__gr_lorch_title = "%s FT Lorched" % real_space_function
        self.__gr_title = "%s Merged" % real_space_function

    @property
    def sq_title(self):
        """The title of the :math:`S(Q)` function directly after merging
        the reciprocal space functions without any further corrections.

        :getter: Returns the current title for this function
        :setter: Sets the title for this function
        :type: str
        """
        return self.__sq_title

    @sq_title.setter
    def sq_title(self, title):
        self.__sq_title = title

    @property
    def qsq_minus_one_title(self):
        """The title of the :math:`Q[S(Q)-1]` function
        directly after merging the reciprocal space
        functions without any further corrections.

        :getter: Returns the current title for this function
        :setter: Sets the title for this function
        :type: str
        """
        return self.__qsq_minus_one_title

    @qsq_minus_one_title.setter
    def qsq_minus_one_title(self, title):
        self.__qsq_minus_one_title = title

    @property
    def sq_ft_title(self):
        """The title of the :math:`S(Q)` function after
        merging and a fourier filter correction.

        :getter: Returns the current title for this function
        :setter: Sets the title for this function
        :type: str
        """
        return self.__sq_ft_title

    @sq_ft_title.setter
    def sq_ft_title(self, title):
        self.__sq_ft_title = title

    @property
    def fq_title(self):
        """The title of the :math:`F(Q)` function after
        merging and a fourier filter correction.

        :getter: Returns the current title for this function
        :setter: Sets the title for this function
        :type: str
        """
        return self.__fq_title

    @fq_title.setter
    def fq_title(self, title):
        self.__fq_title = title

    @property
    def gr_title(self):
        """The title of the real space function directly after merging
        the reciprocal space functions without any further corrections.

        :getter: Returns the current title for this function
        :setter: Sets the title for this function
        :type: str
        """
        return self.__gr_title

    @gr_title.setter
    def gr_title(self, title):
        self.__gr_title = title

    @property
    def gr_ft_title(self):
        """The title for the real space function after both
        merging and a fourier filter correction

        :getter: Returns the current title for this function
        :setter: Sets the title for this function
        :type: str
        """
        return self.__gr_ft_title

    @gr_ft_title.setter
    def gr_ft_title(self, title):
        self.__gr_ft_title = title

    @property
    def gr_lorch_title(self):
        """The title for the real space function with the lorch correction

        :getter: Returns the current title for this function
        :setter: Sets the title for this function
        :type: str
        """
        return self.__gr_lorch_title

    @gr_lorch_title.setter
    def gr_lorch_title(self, title):
        self.__gr_lorch_title = title

    @property
    def GKofR_title(self):
        """The title of the :math:`G_{Keen Version}(r)` with
        all corrections applied.

        :getter: Returns the current title for this function
        :setter: Sets the title for this function
        :type: str
        """
        return self.__GKofR_title

    @GKofR_title.setter
    def GKofR_title(self, title):
        self.__GKofR_title = title

# -------------------------------------#
# Reading and Merging Spectrum

    def read_all_data(self, **kwargs):
        """Reads all the data from the **files** attribute
        Uses the **read_dataset** method on each file.

        Will append all datasets as DataFrames to the
        **df_inviduals** attribute DataFrame
        and also convert to :math:`S(Q)` and add to the **df_sq_individuals**
        attribute DataFrame in **add_dataset** method via **read_dataset** method.
        """
        assert self.files is not None
        assert len(self.files) != 0

        for i, file_info in enumerate(self.files):
            file_info['index'] = i
            self.read_dataset(file_info, **kwargs)

    def read_dataset(
            self,
            info,
            xcol=0,
            ycol=1,
            sep=r"\s+",
            skiprows=2,
            **kwargs):
        """Reads an individual file and uses the **add_dataset**
        method to apply all dataset manipulations, such as
        scales, offsets, cropping, etc.

        Will append the DataFrame to the **df_inviduals** attribute DataFrame
        and also convert to :math:`S(Q)` and add to the **df_sq_individuals**
        attribute DataFrame in **add_dataset** method.

        :param info: Dict with information for dataset (filename, manipulations, etc.)
        :type info: dict
        :param xcol: The column in the data file that contains the X-axis
        :type xcol: int
        :param ycol: The column in the data file that contains the Y-axis
        :type ycol: int
        :param sep: Separator for the file used by pandas.read_csv
        :type sep: raw string
        :param skiprows: Number of rows to skip. Passed to pandas.read_csv
        :type skiprows: int
        """
        # TODO: Create a proper parser class so we can be
        # more accepting of file formats.
        data = pd.read_csv(info['Filename'],
                           sep=sep,
                           usecols=[xcol, ycol],
                           names=['x', 'y'],
                           skiprows=skiprows,
                           engine='python',
                           **kwargs)
        info['data'] = data
        self.add_dataset(info, index=info['index'], **kwargs)

    def add_dataset(
            self,
            info,
            index=0,
            yscale=1.,
            yoffset=0.,
            xoffset=0.,
            ydecimals=16,
            **kwargs):
        """Takes the info with the dataset and manipulations,
        such as scales, offsets, cropping, etc., and creates
        an invidual DataFrame.

        Will append the DataFrame to the **df_inviduals** attribute DataFrame
        and also convert to :math:`S(Q)` and add to the **df_sq_individuals**
        attribute DataFrame.

        :param info: Dict with information for dataset (filename, manipulations, etc.)
        :type info: dict
        :param index: Index of the added reciprocal space function dataset
        :type index: int
        :param yscale: Scale factor for the Y data (i.e. :math:`S(Q)`, :math:`F(Q)`, etc.)
        :type yscale: float
        :param yoffset: Offset factor for the Y data (i.e. :math:`S(Q)`, :math:`F(Q)`, etc.)
        :type yoffset: float
        :param xoffset: Offset factor for the X data (i.e. :math:`Q`)
        :type yoffset: float
        """
        # Extract data
        x = np.around(np.array(info['data']['x']), decimals=self.__xdecimals)
        y = np.around(np.array(info['data']['y']), decimals=self.__ydecimals)

        # Cropping
        xmin = min(x)
        xmax = max(x)
        if 'Qmin' in info:
            xmin = info['Qmin']
        if 'Qmax' in info:
            xmax = info['Qmax']
        x, y = self.transformer.apply_cropping(x, y, xmin, xmax)

        # Offset and scale
        adjusting = False
        if 'Y' in info:
            adjusting = True
            if 'Scale' in info['Y']:
                yscale = info['Y']['Scale']
            if 'Offset' in info['Y']:
                yoffset = info['Y']['Offset']

        if 'X' in info:
            adjusting = True
            if 'Offset' in info['X']:
                xoffset = info['X']['Offset']

        if adjusting:
            x, y = self._apply_scales_and_offset(
                x, y, yscale, yoffset, xoffset)

        # Save overal x-axis min and max
        self.xmin = min(self.xmin, xmin)
        self.xmax = max(self.xmax, xmax)

        # Use Qmin and Qmax to crop datasets
        if self.qmin is not None:
            if self.xmin < self.qmin:
                x, y = self.transformer.apply_cropping(
                    x, y, self.qmin, self.xmax)
        if self.qmax is not None:
            if self.xmax > self.qmax:
                x, y = self.transformer.apply_cropping(
                    x, y, self.xmin, self.qmax)

        # Default to S(Q) if function type not defined
        if "ReciprocalFunction" not in info:
            info["ReciprocalFunction"] = "S(Q)"

        if info["ReciprocalFunction"] not in ReciprocalSpaceChoices:
            error = "ReciprocalFunction was equal to %s.\n" % info["ReciprocalFunction"]
            error += "ReciprocalFunction must be one of the folloing %s" % json.dumps(
                ReciprocalSpaceChoices)
            raise ValueError(error)

        # Save reciprocal space function to the "invididuals" DataFrame
        df = pd.DataFrame(
            y, columns=[
                '%s_%d' %
                (info['ReciprocalFunction'], index)], index=x)
        self.df_individuals = pd.concat([self.df_individuals, df], axis=1)

        # Convert to S(Q) and save to the individual S(Q) DataFrame
        if info["ReciprocalFunction"] == "Q[S(Q)-1]":
            y = self.converter.F_to_S(x, y)
        elif info["ReciprocalFunction"] == "FK(Q)":
            y = self.converter.FK_to_S(x, y, **{'<b_coh>^2': self.bcoh_sqrd})
        elif info["ReciprocalFunction"] == "DCS(Q)":
            y = self.converter.DCS_to_S(x, y,
                                        **{'<b_coh>^2': self.bcoh_sqrd,
                                           '<b_tot^2>': self.btot_sqrd})

        df = pd.DataFrame(y, columns=['S(Q)_%d' % index], index=x)
        self.df_sq_individuals = pd.concat(
            [self.df_sq_individuals, df], axis=1)

    def _apply_scales_and_offset(
            self,
            x,
            y,
            yscale=1.0,
            yoffset=0.0,
            xoffset=0.0):
        """Applies scales to the Y-axis and offsets to both X and Y axes.

        :param x: X-axis data
        :type x: numpy.array or list
        :param y: Y-axis data
        :type y: numpy.array or list
        :param yscale: Y-axis scale factor
        :type yscale: float
        :param yoffset: Y-axis offset factor
        :type yoffset: float
        :param xoffset: X-axis offset factor
        :type xoffset: float
        :return: X and Y vectors after scales and offsets applied
        :rtype: numpy.array pair
        """
        y = self._scale(y, yscale)
        y = self._offset(y, yoffset)
        x = self._offset(x, xoffset)
        return x, y

    def _offset(self, data, offset):
        """Applies offset to data

        :param data: Input data
        :type data: numpy.array or list
        :param offset: Offset to apply to data
        :type offset: float
        :return: Data with offset applied
        :rtype: numpy.array
        """
        data = data + offset
        return data

    def _scale(self, data, scale):
        """Applies scale to data

        :param data: Input data
        :type data: numpy.array or list
        :param offset: Scale to apply to data
        :type offset: float
        :return: Data with scale applied
        :rtype: numpy.array
        """
        data = scale * data
        return data

    def merge_data(self):
        """Merges the reciprocal space data stored in the
        **df_individuals** class DataFrame into a single, merged
        recirocal space function. Stores the S(Q) result in
        **df_sq_master** class DataFrame.

        Also, converts this
        merged :math:`S(Q)` into :math:`Q[S(Q)-1]` via the
        **Converter** class and applies any modification
        specified in **merged_opts** dict attribute, specified
        by the **'Q[S(Q)-1]'** key of the dict. If there is modification,
        this modified :math:`Q[S(Q)-1]` will be converted to
        :math:`S(Q)` and replace the :math:`S(Q)` directly after merge.

        Example dict of **merged_opts** for scaling of
        :math:`S(Q)` by 2 and then offsetting :math:`Q[S(Q)-1]` by 5:

        .. highlight:: python
        .. code-block:: python

                {"Merging": { "Y": { "Offset": 0.0,
                                 "Scale": 2.0 },
                          "Q[S(Q)-1]": { "Y": "Offset": 5.0,
                                    "Scale": 1.0 }
                        }
        ...
        """
        # TODO: Refator to have "S(Q)" as key for S(Q) modifications

        # Sum over single S(Q) columns into a merged S(Q)
        single_sofqs = self.df_sq_individuals.iloc[:, :]
        self.df_sq_master[self.sq_title] = single_sofqs.mean(axis=1)

        q = self.df_sq_master[self.sq_title].index.values
        sq = self.df_sq_master[self.sq_title].values

        q, sq = self._apply_scales_and_offset(q, sq,
                                              self.merged_opts['Y']['Scale'],
                                              self.merged_opts['Y']['Offset'],
                                              0.0)
        self.df_sq_master[self.sq_title] = sq

        # Also, create merged Q[S(Q)-1] with modifications, if specified
        fofq = self.converter.S_to_F(q, sq)
        if "Q[S(Q)-1]" in self.merged_opts:
            fofq_opts = self.merged_opts["Q[S(Q)-1]"]
            if "Y" in fofq_opts:
                if "Scale" in fofq_opts["Y"]:
                    fofq *= fofq_opts["Y"]["Scale"]
                if "Offset" in fofq_opts["Y"]:
                    fofq += fofq_opts["Y"]["Offset"]
        self.df_sq_master[self.qsq_minus_one_title] = fofq

        # Convert this Q[S(Q)-1] back to S(Q) and overwrite the 1st one
        sq = self.converter.F_to_S(q, fofq)
        sq[np.isnan(sq)] = 0
        self.df_sq_master[self.sq_title] = sq

    # -------------------------------------#
    # Transform Utilities

    def transform_merged(self):
        """Performs the Fourier transform on the merged **df_sq_master**
        DataFrame to generate the desired real space function
        with this correction. The results for real space are:
        saved back to the **gr_master** DataFrame
        """
        # Create r-space vector if needed
        if self.dr is None or len(self.dr) == 0:
            self.__update_dr()

        # Get Q and S(Q)
        q = self.df_sq_master[self.sq_title].index.values
        sq = self.df_sq_master[self.sq_title].values

        # Perform the Fourier transform to selected real space function
        transform_kwargs = {'lorch': False,
                            'rho': self.density,
                            '<b_coh>^2': self.bcoh_sqrd
                            }
        if self.real_space_function == "g(r)":
            r, gofr = self.transformer.S_to_g(
                q, sq, self.dr, **transform_kwargs)
        elif self.real_space_function == "G(r)":
            r, gofr = self.transformer.S_to_G(
                q, sq, self.dr, **transform_kwargs)
        elif self.real_space_function == "GK(r)":
            r, gofr = self.transformer.S_to_GK(
                q, sq, self.dr, **transform_kwargs)

        self.df_gr_master[self.gr_title] = gofr
        self.df_gr_master = self.df_gr_master.set_index(r)

    def fourier_filter(self):
        """Performs the Fourier filter on the **df_sq_master**
        DataFrame to generate the desired real space function
        with this correction. The results from both reciprocal space and
        real space are:

        1. Saved back to the respective "master" DataFrames
        2. Saved to files via the **stem_name**
        3. (optional) Plotted for diagnostics
        4. Returned from function

        :return: Returns a tuple with :math:`r`, the selected real space function,
                 :math:`Q`, and :math:`S(Q)` functions
        :rtype: tuple of numpy.array
        """
        kwargs = {'lorch': False,
                  'rho': self.density,
                  '<b_coh>^2': self.bcoh_sqrd,
                  'OmittedXrangeCorrection': self.low_q_correction
                  }
        cutoff = self.fourier_filter_cutoff

        # Get reciprocal and real space data
        if self.gr_title not in self.df_gr_master.columns:
            print("WARNING: Fourier filtered before initial transform. Peforming now...")
            self.transform_merged()

        r = self.df_gr_master[self.gr_title].index.values
        gr = self.df_gr_master[self.gr_title].values
        q = self.df_sq_master[self.sq_title].index.values
        sq = self.df_sq_master[self.sq_title].values

        # Fourier filter g(r)
        # NOTE: Real space function setter will catch ValueError so
        # so no need for `else` to catch error
        if self.real_space_function == "g(r)":
            q_ft, sq_ft, q, sq, r, gr = self.filter.g_using_S(
                r, gr, q, sq, cutoff, **kwargs)
        elif self.real_space_function == "G(r)":
            q_ft, sq_ft, q, sq, r, gr = self.filter.G_using_S(
                r, gr, q, sq, cutoff, **kwargs)
        elif self.real_space_function == "GK(r)":
            q_ft, sq_ft, q, sq, r, gr = self.filter.GK_using_S(
                r, gr, q, sq, cutoff, **kwargs)

        # Round to avoid mismatch index in DataFrame and NaN for column values
        q = np.around(q, decimals=self.__xdecimals)
        sq = np.around(sq, decimals=self.__ydecimals)
        q_ft = np.around(q_ft, decimals=self.__xdecimals)
        sq_ft = np.around(sq_ft, decimals=self.__ydecimals)

        # Add output to master dataframes and write files
        self.df_sq_master = self.add_to_dataframe(
            q_ft, sq_ft, self.df_sq_master, self._ft_title)
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
            self.plot_sq(
                ylabel="FourierFilter(Q)",
                title="Fourier Transform of the low-r region below cutoff",
                exclude_list=exclude_list)
            exclude_list = [self.qsq_minus_one_title]
            self.plot_sq(
                title="Fourier Filtered S(Q)",
                exclude_list=exclude_list)
            self.plot_gr(
                title="Fourier Filtered %s" %
                self.real_space_function)

        return q, sq, r, gr

    def apply_lorch(self, q, sq, r):
        """Performs the Fourier transform using the Lorch
        dampening correction on the merged :math:`S(Q)` from
        the **df_sq_master** DataFrame to generate the
        desired real space function with
        this correction. The results from both reciprocal space and
        real space are:

        1. Saved back to the respective "master" DataFrames
        2. Saved to files via the **stem_name**
        3. (optional) Plotted for diagnostics
        4. Returned from function

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param sq: :math:`S(Q)` vector
        :type sq: numpy.array or list
        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :return: Returns a tuple with :math:`r` and selected real space function
        :rtype: tuple of numpy.array
        """
        if self.real_space_function == "g(r)":
            r, gr_lorch = self.transformer.S_to_g(
                q, sq, r, **{'lorch': True, 'rho': self.density})
        elif self.real_space_function == "G(r)":
            r, gr_lorch = self.transformer.S_to_G(q, sq, r, **{'lorch': True})
        elif self.real_space_function == "GK(r)":
            r, gr_lorch = self.transformer.S_to_GK(
                q, sq, r, **{'lorch': True, 'rho': self.density, '<b_coh>^2': self.bcoh_sqrd})

        self.df_gr_master = self.add_to_dataframe(
            r, gr_lorch, self.df_gr_master, self.gr_lorch_title)
        self.write_out_lorched_gr()

        if self.plot_flag:
            self.plot_gr(
                title="Lorched %s" %
                self.real_space_function)

        return r, gr_lorch

    def _get_lowR_mean_square(self):
        """Retuns the low-R mean square value for the real space function stored
        in the "master" real space function class DataFrame, **df_gr_master**.
        Used as a cost function for optimiziation of the :math:`Q_{max}` value
        by an iterative adjustment. Calls **_lowR_mean_square* method.
        **Currently not used in PyStoG workflow since was done manually.**

        :return: The calculated low-R mean-square value
        :rtype: float
        """
        # TODO: Automate the :math:`Q_{max}` adjustment in an iterative loop
        # using a minimizer.
        gr = self.df_gr_master[self.gr_title].values
        return self._lowR_mean_square(self.dr, gr)

    def _lowR_mean_square(self, r, gr, limit=1.01):
        """Calculates the low-R mean square value from a given real space function.
        Used as a cost function for optimiziation of the :math:`Q_{max}` value
        by an iterative adjustment.
        **Currently not used in PyStoG workflow since was done manually.**

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: real space function vector
        :type gr: numpy.array or list
        :param limit: The upper limit on :math:`r` to use for
                      the mean-square calculation
        :type limit: float
        :return: The calculated low-R mean-square value
        :rtype: float
        """
        # TODO: Automate the :math:`Q_{max}` adjustment in an iterative loop
        # using a minimizer.
        gr = gr[r <= limit]
        gr_sq = np.multiply(gr, gr)
        average = sum(gr_sq)
        return np.sqrt(average)

    def _add_keen_fq(self, q, sq):
        """Adds the Keen version of :math:`F(Q)` to the
        "master" recprical space DataFrame, **df_sq_master**, and
        writes it out to file using the **stem_name**.

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param sq: :math:`S(Q)` vector
        :type sq: numpy.array or list
        """
        kwargs = {'rho': self.density, "<b_coh>^2": self.bcoh_sqrd}
        fq = self.converter.S_to_FK(q, sq, **kwargs)
        self.df_sq_master = self.add_to_dataframe(
            q, fq, self.df_sq_master, self.fq_title)
        self.write_out_rmc_fq()

    def _add_keen_gr(self, r, gr):
        """Adds the Keen version of :math:`G(r)` to the
        "master" real space DataFrame, **df_gr_master**, and
        writes it out to file using the **stem_name**.

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: real space function vector
        :type gr: numpy.array or list
        """
        kwargs = {'rho': self.density, "<b_coh>^2": self.bcoh_sqrd}
        if self.real_space_function == "g(r)":
            GKofR = self.converter.g_to_GK(r, gr, **kwargs)
        elif self.real_space_function == "G(r)":
            GKofR = self.converter.G_to_GK(r, gr, **kwargs)
        elif self.real_space_function == "GK(r)":
            GKofR = gr

        self.df_gr_master = self.add_to_dataframe(
            r, GKofR, self.df_gr_master, self.GKofR_title)
        self.write_out_rmc_gr()

    # -------------------------------------#
    # Plot Utilities

    def _plot_df(self, df, xlabel, ylabel, title, exclude_list):
        """Utility function to help plot a DataFrame

        :param df: DataFrame to plot
        :type df: pandas.DataFrame
        :param xlabel: X-axis label
        :type xlabel: str
        :param ylabel: Y-axis label
        :type ylabel: str
        :param title: Title of plot
        :type title: str
        :param exclude_list: List of titles of columns in
                        DataFrame **df** to exclude from plot
        :type exclude_list: list of str
        """
        if exclude_list:
            columns_diff = df.columns.difference(exclude_list)
            columns_diff_ids = df.columns.get_indexer(columns_diff)
            df = df.iloc[:, columns_diff_ids]
        df.plot(**self.plotting_kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def plot_sq(self, xlabel='Q', ylabel='S(Q)', title='', exclude_list=None):
        """Helper function to plot the :math:`S(Q)` functions
        in the "master" DataFrame, **df_sq_master**.

        :param xlabel: X-axis label
        :type xlabel: str
        :param ylabel: Y-axis label
        :type ylabel: str
        :param title: Title of plot
        :type title: str
        :param exclude_list: List of titles of columns in
                        DataFrame to exclude from plot
        :type exclude_list: list of str
        """
        df_sq = self.df_sq_master
        self._plot_df(df_sq, xlabel, ylabel, title, exclude_list)

    def plot_merged_sq(self):
        """Helper function to multiplot the individual
        real space functions in the **df_individuals** DataFrame,
        these functions as individual :math:`S(Q)`, the merged
        :math:`S(Q)` from the individual functions, and
        :math:`Q[S(Q)-1]`.
        """

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
        df_sq = self.df_sq_master.loc[:, [self.sq_title]]
        df_sq.plot(ax=axes[1, 0], **plot_kwargs)
        axes[1, 0].set_title("Merged S(Q)")
        axes[1, 0].set_ylabel("S(Q)")

        # Plot the merged Q[S(Q)-1]
        df_fq = self.df_sq_master.loc[:, [self.qsq_minus_one_title]]
        df_fq.plot(ax=axes[1, 1], **plot_kwargs)
        axes[1, 1].set_title("Merged Q[S(Q)-1]")
        axes[1, 1].set_ylabel("Q[S(Q)-1]")

        plt.show()

    def plot_gr(self, xlabel='r', ylabel='G(r)', title='', exclude_list=None):
        """Helper function to plot the real space functions
        in the "master" DataFrame, **df_gr_master**.

        :param xlabel: X-axis label
        :type xlabel: str
        :param ylabel: Y-axis label
        :type ylabel: str
        :param title: Title of plot
        :type title: str
        :param exclude_list: List of titles of columns in
                        DataFrame to exclude from plot
        :type exclude_list: list of str
        """
        df_gr = self.df_gr_master
        self._plot_df(df_gr, xlabel, ylabel, title, exclude_list)

    def plot_summary_sq(self):
        """Helper function to multiplot the reciprocal space
        functions during processing and the :math:`F(Q)` function.
        """
        if self.fq_title not in self.df_sq_master.columns:
            q = self.df_sq_master[self.sq_title].index.values
            sq = self.df_sq_master[self.sq_title].values
            self._add_keen_fq(q, sq)

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        exclude_list = [self.fq_title]
        df = self.df_sq_master
        columns_diff = df.columns.difference(exclude_list)
        columns_diff_ids = df.columns.get_indexer(columns_diff)
        df_sq = self.df_sq_master.iloc[:, columns_diff_ids]
        df_sq.plot(ax=ax1, **self.plotting_kwargs)
        df_fq = self.df_sq_master.loc[:, [self.fq_title]]
        df_fq.plot(ax=ax2, **self.plotting_kwargs)
        plt.xlabel("Q")
        ax1.set_ylabel("S(Q)")
        ax1.set_title("StoG S(Q) functions")
        ax2.set_ylabel("FK(Q)")
        ax2.set_title("Keen's F(Q)")
        plt.show()

    def plot_summary_gr(self):
        """Helper function to multiplot the real space
        functions during processing and the :math:`G_{Keen Version}(Q)` function.
        """
        if self.GKofR_title not in self.df_gr_master.columns:
            r = self.df_gr_master[self.gr_title].index.values
            gr = self.df_gr_master[self.gr_title].values
            self._add_keen_gr(r, gr)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        df = self.df_gr_master
        columns_diff = df.columns.difference([self.GKofR_title])
        columns_diff_ids = df.columns.get_indexer(columns_diff)
        df_gr = df.iloc[:, columns_diff_ids]
        df_gr.plot(ax=ax1, **self.plotting_kwargs)
        df_gk = df.loc[:, [self.GKofR_title]]
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
        """Takes X,Y dataset and adds it to the given Datframe **df**,
        with the given **title**. Utility function for updating
        the class DataFrames.

        :param x: X-axis vector
        :type x: numpy.array or list
        :param y: Y-axis vector
        :type y: numpy.array or list
        :param df: DataFrame to append (**x**, **y**) pair to as a column
        :type df: pandas.DataFrame
        :param title: The title of the column in the DataFrame
        :type title: str
        :return: DataFrame with X,Y data appended with given title
        :rtype: pandas.DataFrame
        """
        df_temp = pd.DataFrame(y, columns=[title], index=x)
        if title in df.columns:
            df[title] = df_temp[title]
            return df
        df = pd.concat([df, df_temp], axis=1)
        return df

    def _write_out_df(self, df, cols, filename):
        """Helper function for writing out the DataFrame **df**
        and the given columns, **cols**, to the filename in
        the RMCProfile format.

        :param df: DataFrame to write from to filename
        :type df: pandas.DataFrame
        :param cols: Column title list for columns to write out
        :type cols: List of str
        :param filename: Filename to write to
        :type filename: str
        """
        if df.empty:
            raise ValueError("Empty dataframe. Cannot write out.")

        with open(filename, 'w') as f:
            f.write("%d \n" % df.shape[0])
            f.write("# Comment line\n")
        with open(filename, 'a') as f:
            df.to_csv(f, sep='\t', columns=cols, header=False)

    def write_out_merged_sq(self, filename=None):
        """Helper function for writing out the merged :math:`S(Q)`

        :param filename: Filename to write to
        :type filename: str
        """
        if filename is None:
            filename = "%s.sq" % self.stem_name
        self._write_out_df(self.df_sq_master, [self.sq_title], filename)

    def write_out_merged_gr(self, filename=None):
        """Helper function for writing out the merged real space function

        :param filename: Filename to write to
        :type filename: str
        """
        if filename is None:
            filename = "%s.gr" % self.stem_name
        self._write_out_df(self.df_gr_master, [self.gr_title], filename)

    def write_out_ft(self, filename=None):
        """Helper function for writing out the Fourier filter correction.

        :param filename: Filename to write to
        :type filename: str
        """
        if filename is None:
            filename = "ft.dat"
        self._write_out_df(self.df_sq_master, [self._ft_title], filename)

    def write_out_ft_sq(self, filename=None):
        """Helper function for writing out the Fourier filtered :math:`S(Q)`

        :param filename: Filename to write to
        :type filename: str
        """
        if filename is None:
            filename = "%s_ft.sq" % self.stem_name
        self._write_out_df(self.df_sq_master, [self.sq_ft_title], filename)

    def write_out_ft_gr(self, filename=None):
        """Helper function for writing out the Fourier filtered real space function

        :param filename: Filename to write to
        :type filename: str
        """
        if filename is None:
            filename = "%s_ft.gr" % self.stem_name
        self._write_out_df(self.df_gr_master, [self.gr_ft_title], filename)

    def write_out_lorched_gr(self, filename=None):
        """Helper function for writing out the Lorch dampened real space function

        :param filename: Filename to write to
        :type filename: str
        """
        if filename is None:
            filename = "%s_ft_lorched.gr" % self.stem_name
        self._write_out_df(self.df_gr_master, [self.gr_lorch_title], filename)

    def write_out_rmc_fq(self, filename=None):
        """Helper function for writing out the output :math:`F(Q)`

        :param filename: Filename to write to
        :type filename: str
        """
        if filename is None:
            filename = "%s_rmc.fq" % self.stem_name
        self._write_out_df(self.df_sq_master, [self.fq_title], filename)

    def write_out_rmc_gr(self, filename=None):
        """Helper function for writing out the output :math:`G_{Keen Version}(Q)`

        :param filename: Filename to write to
        :type filename: str
        """
        if filename is None:
            filename = "%s_rmc.gr" % self.stem_name
        self._write_out_df(self.df_gr_master, [self.GKofR_title], filename)
