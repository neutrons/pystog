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

from pystog.utils import create_domain, RealSpaceChoices, ReciprocalSpaceChoices
from pystog.converter import Converter
from pystog.transformer import Transformer
from pystog.fourier_filter import FourierFilter


class NoInputFilesException(Exception):
    """Exception when no files are given to process"""


class StoG(object):
    """
    The StoG class is used to put together
    the Converter, Transformer, and FourierFilter
    class functionalities to reproduce the original
    **stog** Fortran program behavior. This class is meant to
    put together the functionality of the classes
    into higher-level calls to construct workflows
    for merging and processing multiple recprical space
    functions into a final output real space function.

    This pythonized-version of the original Fortran **stog**
    uses numpy for data storage, organization, and
    manipulation "under the hood".

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
        self.__merged_opts = {"Y": {"Offset": 0.0, "Scale": 1.0}}
        self.__stem_name = "out"

        # Storage arrays for total scattering functions
        self.__reciprocal_individuals = np.empty([3, 0])
        self.__sq_individuals = np.empty([3, 0])
        self.__q_master = {}
        self.__sq_master = {}
        self.__r_master = {}
        self.__gr_master = {}

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
        """
        Takes the key-word arguments supplied to the
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
        """
        The minimum X value of all datasets to
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
        """
        The maximum X value of all datasets to
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
        """
        The :math:`Q_{min}` value to use for the Fourier
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
        """
        The :math:`Q_{max}` value to use for the Fourier
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
        """
        The files that contain the reciprocal space data
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
        """
        Appends a file to the file list

        :param new_file: New file name to append
        :type new_file: str
        :return: File list with appended new_file
        :rtype: list
        """
        self.files = self.files + [new_file]
        return self.files

    def extend_file_list(self, new_files):
        """
        Extend the file list with a list of new files

        :param new_files: List of new files
        :type new_file: list
        :return: File list extended by new_files
        :rtype: list
        """
        self.files.extend(new_files)
        return self.files

    def __update_dr(self):
        """
        Uses **rdelta** and **rmax** attributes (:math:`\\Delta r` and
        :math:`R_{max}`, respectively) to construct **dr** attribute
        (:math:`r`-space vector) via its setter
        """
        self.dr = create_domain(self.rmin, self.rmax, self.rdelta)

    @property
    def rdelta(self):
        """
        The :math:`\\Delta r` for the :math:`r`-space vector

        :getter: Return :math:`\\Delta r` value
        :setter: Set the :math:`\\Delta r` value
                 and update :math:`r`-space vector
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
        """
        The :math:`R_{min}` valuefor the :math:`r`-space vector

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
        """
        The :math:`R_{max}` valuefor the :math:`r`-space vector

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
        """
        The real space function X axis data, :math:`r`-space vector

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
        """
        The number density used (atoms/:math:`\\AA^{3}`) used for the
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
        """
        The average coherent scattering length, squared:
        :math:`\\langle b_{coh} \\rangle^2` =
        :math:`( \\sum_{i} c_{i} b_{coh,i} ) ( \\sum_{i} c_{i} b^*_{coh,i} )`
        where the subscript :math:`i` implies for atom type :math:`i`,
        :math:`c_{i}` is the concentration of :math:`i`,
        :math:`b_{coh,i}` is the coherent scattering length of :math:`i`,
        and :math:`b^*_{coh,i}`
        is the complex coherent scattering length of :math:`i`


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
        """
        The average coherent scattering length, squared:
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
        """
        A stem name to prefix for all output files. Replicates
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
        """
        This sets the option to perform a low-:math:`Q` correction
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
        """
        This sets the option to perform the Lorch dampening correction
        for the :math:`Q` range. Generally, will help reduce Fourier "ripples",
        or AKA "Gibbs phenomenon", due to discontinuity at :math:`Q_{max}`
        if the reciprocal space function is not at the
        :math:`Q -> \\infty` limit.
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
        """
        This sets the cutoff in :math:`r`-space for the Fourier
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
        """
        This sets the options to perform after merging the
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

    # Storage array attributes

    @property
    def reciprocal_individuals(self):
        """
        The storage array for the input reciprocal space functions
        loaded from **files** and with the loading processing
        from **add_dataset** class method.

        :getter: Returns the current individual, input reciprocal space
                 functions numpy array.
                 The dimenions is :math:`3 x N*M` where N is the number
                 of patterns stored and M is the length of the patterns.
        :setter: Sets the numpy array
        :type: numpy.ndarray
        """
        return self.__reciprocal_individuals

    @reciprocal_individuals.setter
    def reciprocal_individuals(self, individuals):
        self.__reciprocal_individuals = individuals

    @property
    def sq_individuals(self):
        """
        The storage array for the :math:`S(Q)` generated from each input
        reciprocal space dataset in **reciprocal_individuals** array.

        :getter: Returns the current individual :math:`S(Q)` reciprocal space
                 functions numpy array.
                 The dimenions is :math:`3 x N*M` where N is the number
                 of patterns stored and M is the length of the patterns.
        :setter: Sets the numpy array
        :type: numpy.ndarray
        """
        return self.__sq_individuals

    @sq_individuals.setter
    def sq_individuals(self, individuals):
        self.__sq_individuals = individuals

    @property
    def sq_master(self):
        """
        The "master" dictionary for the :math:`S(Q)` reciprocal
        space functions that are generated for each processing step.

        :getter: Returns the current "master" :math:`S(Q)` reciprocal space
                 functions dictionary generated up to the current step in
                 the workflow.
        :setter: Sets the "master" :math:`S(Q)` function dictionary
        :type: dict[str:numpy.ndarray]
        """
        return self.__sq_master

    @sq_master.setter
    def sq_master(self, sq):
        self.__sq_master = sq

    @property
    def gr_master(self):
        """
        The "master" dictionary for the real space functions
        that are generated for each processing step.

        :getter: Returns the current "master" real space
                 functions dictionary generated up to the current step in
                 the workflow.
        :setter: Sets the "master" real space function dictionary
        :type: dict[str:numpy.ndarray]
        """
        return self.__gr_master

    @gr_master.setter
    def gr_master(self, gr):
        self.__gr_master = gr

    @property
    def q_master(self):
        """
        The "master" dictionary for the domain :math:Q` of the reciprocal
        space functions that are generated for each processing step.

        :getter: Returns the current "master" :math:`Q` reciprocal space
                 functions dictionary generated up to the current step in
                 the workflow.
        :setter: Sets the "master" :math:`Q` dictionary
        :type: dict[str:numpy.ndarray]
        """
        return self.__q_master

    @q_master.setter
    def q_master(self, q):
        self.__q_master = q

    @property
    def r_master(self):
        """
        The "master" dictionary for the domain :math:r` of the real
        space functions that are generated for each processing step.

        :getter: Returns the current "master" :math:`r` real space
                 functions dictionary generated up to the current step in
                 the workflow.
        :setter: Sets the "master" :math:`r` dictionary
        :type: dict[str:numpy.ndarray]
        """
        return self.__r_master

    @r_master.setter
    def r_master(self, r):
        self.__r_master = r

    @property
    def real_space_function(self):
        """
        The real space function to use throughoutt the processing

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
        """
        The title of the :math:`S(Q)` function directly after merging
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
        """
        The title of the :math:`Q[S(Q)-1]` function
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
        """
        The title of the :math:`S(Q)` function after
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
        """
        The title of the :math:`F(Q)` function after
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
        """
        The title of the real space function directly after merging
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
        """
        The title for the real space function after both
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
        """
        The title for the real space function with the lorch correction

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
        """
        The title of the :math:`G_{Keen Version}(r)` with
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
        """
        Reads all the data from the **files** attribute
        Uses the **read_dataset** method on each file.

        Will append all datasets to the numpy storage array,
        **reciprocal_individuals**, and also convert to :math:`S(Q)` and add to
        the **sq_individuals** numpy storage array in **add_dataset** method
        via **read_dataset** method.
        """
        # Check that we have files to operate on
        if not self.files:
            raise NoInputFilesException("No input files given in arguments")

        # Read in all the data files
        for i, file_info in enumerate(self.files):
            self.read_dataset(file_info, **kwargs)

    def read_dataset(
            self,
            info,
            xcol=0,
            ycol=1,
            dycol=2,
            sep=r"\s+",
            skiprows=2,
            **kwargs):
        """
        Reads an individual file and uses the **add_dataset**
        method to apply all dataset manipulations, such as
        scales, offsets, cropping, etc.

        :param info: Dict with information for dataset
                     (filename, manipulations, etc.)
        :type info: dict
        :param xcol: Column in data file for X-axis
        :type xcol: int
        :param ycol: Column in data file for Y-axis
        :type ycol: int
        :param dycol: Column in data file for Y uncertainty
        :type dycol: int
        :param sep: Separator for the file used by numpy.loadtxt
        :type sep: raw string
        :param skiprows: Number of rows to skip. Passed to numpy.loadtxt
        :type skiprows: int
        """
        _data = np.loadtxt(
            info['Filename'],
            skiprows=skiprows,
            comments='#',
            unpack=True)
        if _data.shape[0] <= xcol or _data.shape[0] <= ycol:
            raise RuntimeError("Data format incompatible with input parameters")
        if _data.shape[0] <= dycol:
            array_seq = (_data[xcol], _data[ycol], np.zeros_like(_data[ycol]))
            data = np.stack(array_seq)
        else:
            data = np.stack((_data[xcol], _data[ycol], _data[dycol]))
        info['data'] = data
        self.add_dataset(info, **kwargs)

    def add_dataset(
            self,
            info,
            yscale=1.,
            yoffset=0.,
            xoffset=0.,
            ydecimals=16,
            **kwargs):
        """
        Takes the info with the dataset and manipulations,
        such as scales, offsets, cropping, etc., and creates
        an invidual numpy array for the pattern.

        :param info: Dict with information for dataset
                     (filename, manipulations, etc.)
        :type info: dict
        :param yscale: Scale factor for the Y data
                       (i.e. :math:`S(Q)`, :math:`F(Q)`, etc.)
        :type yscale: float
        :param yoffset: Offset factor for the Y data
                        (i.e. :math:`S(Q)`, :math:`F(Q)`, etc.)
        :type yoffset: float
        :param xoffset: Offset factor for the X data (i.e. :math:`Q`)
        :type yoffset: float
        """
        # Extract data
        x = np.around(np.array(info['data'][0]), decimals=self.__xdecimals)
        y = np.around(np.array(info['data'][1]), decimals=self.__ydecimals)
        if len(info['data']) == 3:
            dy = np.around(np.array(info['data'][2]), decimals=self.__ydecimals)
        else:
            dy = np.zeros_like(y)

        # Cropping
        xmin = min(x)
        xmax = max(x)
        if 'Qmin' in info:
            xmin = info['Qmin']
        if 'Qmax' in info:
            xmax = info['Qmax']
        x, y, dy = self.transformer.apply_cropping(x, y, xmin, xmax, dy=dy)

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
            x, y, dy = self.apply_scales_and_offset(
                x, y, dy=dy,
                yscale=yscale,
                yoffset=yoffset,
                xoffset=xoffset)

        # Save overal x-axis min and max
        self.xmin = min(self.xmin, xmin)
        self.xmax = max(self.xmax, xmax)

        # Use Qmin and Qmax to crop datasets
        if self.qmin is not None:
            if self.xmin < self.qmin:
                x, y, dy = self.transformer.apply_cropping(
                    x, y, self.qmin, self.xmax, dy=dy)
        if self.qmax is not None:
            if self.xmax > self.qmax:
                x, y, dy = self.transformer.apply_cropping(
                    x, y, self.xmin, self.qmax, dy=dy)

        # Default to S(Q) if function type not defined
        if "ReciprocalFunction" not in info:
            info["ReciprocalFunction"] = "S(Q)"

        if info["ReciprocalFunction"] not in ReciprocalSpaceChoices:
            error = "ReciprocalFunction was equal to {given}.\n"
            error += "ReciprocalFunction must be one of the folloing {options}"
            error = error.format(
                given=info["ReciprocalFunction"],
                options=json.dumps(ReciprocalSpaceChoices))
            raise ValueError(error)

        # Save reciprocal space function to the "invididuals" array
        array_seq = (self.reciprocal_individuals, np.stack((x, y, dy)))
        self.reciprocal_individuals = np.concatenate(array_seq, axis=1)

        # Convert to S(Q) and save to the individual S(Q) array
        if info["ReciprocalFunction"] == "Q[S(Q)-1]":
            y, dy = self.converter.F_to_S(x, y, dfq=dy)
        elif info["ReciprocalFunction"] == "FK(Q)":
            y, dy = self.converter.FK_to_S(
                x, y, dfq_keen=dy, **{'<b_coh>^2': self.bcoh_sqrd})
        elif info["ReciprocalFunction"] == "DCS(Q)":
            y, dy = self.converter.DCS_to_S(x, y, ddcs=dy,
                                            **{'<b_coh>^2': self.bcoh_sqrd,
                                               '<b_tot^2>': self.btot_sqrd})
        array_seq = (self.sq_individuals, np.stack((x, y, dy)))
        self.sq_individuals = np.concatenate(array_seq, axis=1)

    @staticmethod
    def apply_scales_and_offset(
            x,
            y,
            dy=None,
            yscale=1.0,
            yoffset=0.0,
            xoffset=0.0):
        """
        Applies scales to the Y-axis and offsets to both X and Y axes.

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
        y = y * yscale
        y = y + yoffset
        x = x + xoffset
        if dy is None:
            dy = np.zeros_like(y)
        dy = dy * yscale
        return x, y, dy

    def merge_data(self):
        """
        Merges the reciprocal space data stored in the
        **reciprocal_individuals** numpy array into a single, merged
        recirocal space function. Stores the S(Q) result in
        **sq_master** dictionary using **sq_title** (default: "S(Q) Merged").

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

                {
                    "Merging": {
                        "Y": {
                            "Offset": 0.0,
                            "Scale": 2.0
                        },
                        "Q[S(Q)-1]": {
                            "Y": "Offset": 5.0,
                            "Scale": 1.0
                        }
                    }
                }
        ...
        """
        data_merged = []
        _n_total = 0
        _n_sum = 0
        _n_err = 0
        _previous_x = None

        # At this point we have a concatenated array of data
        # We need to sort according to Q first
        # TODO: this is ugly but works for now.
        _x = self.sq_individuals[0]
        _y = self.sq_individuals[1]
        _dy = self.sq_individuals[2]
        zipped = zip(_x, _y, _dy)
        ordered = sorted(zipped, key=lambda a: a[0])
        _x, _y, _dy = zip(*ordered)
        self.sq_individuals = np.stack(
            (np.asarray(_x), np.asarray(_y), np.asarray(_dy)))

        # Go through the data and compute the average of points
        # with the same Q.
        for item in self.sq_individuals.T:
            if item[0] == _previous_x:
                _n_total += 1.
                _n_sum += item[1]
                _n_err += item[2]**2
            else:
                if _n_total > 0:
                    data_merged.append([
                        _previous_x,
                        _n_sum / _n_total,
                        np.sqrt(_n_err) / _n_total])
                _n_total = 1.
                _n_sum = item[1]
                _n_err = item[2]**2
            _previous_x = item[0]
        if _n_total > 0:
            data_merged.append([
                _previous_x,
                _n_sum / _n_total,
                np.sqrt(_n_err) / _n_total])

        data_merged = np.asarray(data_merged).T

        q = data_merged[0]
        sq = data_merged[1]
        dsq = data_merged[2]

        q, sq, dsq = self.apply_scales_and_offset(
            q, sq,
            yscale=self.merged_opts['Y']['Scale'],
            yoffset=self.merged_opts['Y']['Offset'],
            xoffset=0.0, dy=dsq)
        self.q_master[self.sq_title] = q
        self.sq_master[self.sq_title] = sq

        # Also, create merged Q[S(Q)-1] with modifications, if specified
        fofq, dfofq = self.converter.S_to_F(q, sq, dsq=dsq)
        if "Q[S(Q)-1]" in self.merged_opts:
            fofq_opts = self.merged_opts["Q[S(Q)-1]"]
            if "Y" in fofq_opts:
                if "Scale" in fofq_opts["Y"]:
                    fofq *= fofq_opts["Y"]["Scale"]
                if "Offset" in fofq_opts["Y"]:
                    fofq += fofq_opts["Y"]["Offset"]
        self.q_master[self.qsq_minus_one_title] = q
        self.sq_master[self.qsq_minus_one_title] = fofq

        # Convert this Q[S(Q)-1] back to S(Q) and overwrite the 1st one
        sq, dsq = self.converter.F_to_S(q, fofq, dfq=dfofq)
        sq[np.isnan(sq)] = 0
        self.sq_master[self.sq_title] = sq

    # -------------------------------------#
    # Transform Utilities

    def transform_merged(self):
        """
        Performs the Fourier transform on the merged **sq_master**
        pattern to generate the desired real space function
        with this correction. The results for real space are:
        the domain is saved to the **r_master** dictionary and
        the range is saved to the **gr_master** dictionary,
        with both using the **gr_title** for the key of the dictionaries.
        """
        # Create r-space vector if needed
        if self.dr is None or len(self.dr) == 0:
            self.__update_dr()

        # Get Q and S(Q)
        q = self.q_master[self.sq_title]
        sq = self.sq_master[self.sq_title]

        # Perform the Fourier transform to selected real space function
        transform_kwargs = {'lorch': False,
                            'rho': self.density,
                            '<b_coh>^2': self.bcoh_sqrd
                            }
        if self.real_space_function == "g(r)":
            r, gofr, dgofr = self.transformer.S_to_g(
                q, sq, self.dr, **transform_kwargs)
        elif self.real_space_function == "G(r)":
            r, gofr, dgofr = self.transformer.S_to_G(
                q, sq, self.dr, **transform_kwargs)
        elif self.real_space_function == "GK(r)":
            r, gofr, dgofr = self.transformer.S_to_GK(
                q, sq, self.dr, **transform_kwargs)

        self.gr_master[self.gr_title] = gofr
        self.r_master[self.gr_title] = r

    def fourier_filter(self):
        """
        Performs the Fourier filter on the **sq_master**
        pattern to generate the desired real space function
        with this correction. The results from both reciprocal space and
        real space are:

        1. Saved back to the respective "master" dictionaries
        2. Saved to files via the **stem_name**
        3. Returned from function

        :return: Returns a tuple with :math:`r`,
                 the selected real space function,
                 :math:`Q`,
                 and :math:`S(Q)` functions
        :rtype: tuple of numpy.array
        """
        kwargs = {'lorch': False,
                  'rho': self.density,
                  '<b_coh>^2': self.bcoh_sqrd,
                  'OmittedXrangeCorrection': self.low_q_correction
                  }
        cutoff = self.fourier_filter_cutoff

        # Get reciprocal and real space data
        if self.gr_title not in self.gr_master:
            msg = (
                "WARNING: Fourier filtered before initial transform. "
                "Peforming now...")
            print(msg)
            self.transform_merged()

        r = self.r_master[self.gr_title]
        gr = self.gr_master[self.gr_title]
        q = self.q_master[self.sq_title]
        sq = self.sq_master[self.sq_title]

        # Fourier filter g(r)
        # NOTE: Real space function setter will catch ValueError so
        # so no need for `else` to catch error
        if self.real_space_function == "g(r)":
            q_ft, sq_ft, q, sq, r, gr, _, _, _ = self.filter.g_using_S(
                r, gr, q, sq, cutoff, **kwargs)
        elif self.real_space_function == "G(r)":
            q_ft, sq_ft, q, sq, r, gr, _, _, _ = self.filter.G_using_S(
                r, gr, q, sq, cutoff, **kwargs)
        elif self.real_space_function == "GK(r)":
            q_ft, sq_ft, q, sq, r, gr, _, _, _ = self.filter.GK_using_S(
                r, gr, q, sq, cutoff, **kwargs)

        # Round to avoid mismatch index in domain and NaN
        q = np.around(q, decimals=self.__xdecimals)
        sq = np.around(sq, decimals=self.__ydecimals)
        q_ft = np.around(q_ft, decimals=self.__xdecimals)
        sq_ft = np.around(sq_ft, decimals=self.__ydecimals)

        # Add output to master dataframes and write files
        self.q_master[self._ft_title] = q_ft
        self.sq_master[self._ft_title] = sq_ft
        self.write_out_ft()

        self.q_master[self.sq_ft_title] = q
        self.sq_master[self.sq_ft_title] = sq
        self.write_out_ft_sq()

        self.r_master[self.gr_ft_title] = r
        self.gr_master[self.gr_ft_title] = gr
        self.write_out_ft_gr()

        return q, sq, r, gr

    def apply_lorch(self, q, sq, r):
        """
        Performs the Fourier transform using the Lorch
        dampening correction on the merged :math:`S(Q)` from
        the **sq_master** dictionary to generate the
        desired real space function with
        this correction. The results from both reciprocal space and
        real space are:

        1. Saved back to the respective "master" dictionaries
        2. Saved to files via the **stem_name**
        3. Returned from function

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
            r, gr_lorch, _ = self.transformer.S_to_g(
                q, sq, r, **{'lorch': True, 'rho': self.density})
        elif self.real_space_function == "G(r)":
            r, gr_lorch, _ = self.transformer.S_to_G(
                q, sq, r, **{'lorch': True})
        elif self.real_space_function == "GK(r)":
            r, gr_lorch, _ = self.transformer.S_to_GK(
                q, sq, r,
                **{
                    'lorch': True,
                    'rho': self.density,
                    '<b_coh>^2': self.bcoh_sqrd
                })

        self.gr_master[self.gr_lorch_title] = gr_lorch
        self.r_master[self.gr_lorch_title] = r
        self.write_out_lorched_gr()

        return r, gr_lorch

    def _get_lowR_mean_square(self):
        """
        Retuns the low-R mean square value for the real space function stored
        in the "master" real space function, **gr_master**.
        Used as a cost function for optimiziation of the :math:`Q_{max}` value
        by an iterative adjustment. Calls **_lowR_mean_square* method.
        **Currently not used in PyStoG workflow since was done manually.**

        :return: The calculated low-R mean-square value
        :rtype: float
        """
        # TODO: Automate the :math:`Q_{max}` adjustment in an iterative loop
        # using a minimizer.
        gr = self.gr_master[self.gr_title]
        return self._lowR_mean_square(self.dr, gr)

    def _lowR_mean_square(self, r, gr, limit=1.01):
        """
        Calculates the low-R mean square value from a given real space function.
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
        """
        Adds the Keen version of :math:`F(Q)` to the
        "master" recprical space storage array, **sq_master**, and
        writes it out to file using the **stem_name**.

        :param q: :math:`Q`-space vector
        :type q: numpy.array or list
        :param sq: :math:`S(Q)` vector
        :type sq: numpy.array or list
        """
        kwargs = {'rho': self.density, "<b_coh>^2": self.bcoh_sqrd}
        fq, dfq = self.converter.S_to_FK(q, sq, **kwargs)
        self.sq_master[self.fq_title] = fq
        self.q_master[self.fq_title] = q
        self.write_out_rmc_fq()

    def _add_keen_gr(self, r, gr):
        """
        Adds the Keen version of :math:`G(r)` to the
        "master" real space storage array, **gr_master**, and
        writes it out to file using the **stem_name**.

        :param r: :math:`r`-space vector
        :type r: numpy.array or list
        :param gr: real space function vector
        :type gr: numpy.array or list
        """
        kwargs = {'rho': self.density, "<b_coh>^2": self.bcoh_sqrd}
        if self.real_space_function == "g(r)":
            GKofR, dGKofR = self.converter.g_to_GK(r, gr, **kwargs)
        elif self.real_space_function == "G(r)":
            GKofR, dGKofR = self.converter.G_to_GK(r, gr, **kwargs)
        elif self.real_space_function == "GK(r)":
            GKofR = gr

        self.gr_master[self.GKofR_title] = GKofR
        self.r_master[self.GKofR_title] = r
        self.write_out_rmc_gr()

    # -------------------------------------#
    # Output Utilities
    def _write_out_to_file(self, x, y, filename, places=12):
        """
        Helper function for writing out X Y data
        to the filename in the RMCProfile format.

        :param x: X data to write out
        :type x: list
        :param y: Y data to write out
        :type y: list
        :param filename: Filename to write to
        :type filename: str
        """
        with open(filename, 'w') as f:
            f.write("%d \n" % len(x))
            f.write("# Comment line\n")
        with open(filename, 'a') as f:
            for i, j in zip(x, y):
                fmt = "{:.{places}f} {:.{places}f}\n"
                f.write(fmt.format(i, j, places=places))

    def write_out_merged_sq(self, filename=None):
        """
        Helper function for writing out the merged :math:`S(Q)`

        :param filename: Filename to write to
        :type filename: str
        """
        if filename is None:
            filename = "%s.sq" % self.stem_name
        x = self.q_master[self.sq_title]
        y = self.sq_master[self.sq_title]
        self._write_out_to_file(x, y, filename)

    def write_out_merged_gr(self, filename=None):
        """
        Helper function for writing out the merged real space function

        :param filename: Filename to write to
        :type filename: str
        """
        if filename is None:
            filename = "%s.gr" % self.stem_name
        x = self.r_master[self.gr_title]
        y = self.gr_master[self.gr_title]
        self._write_out_to_file(x, y, filename)

    def write_out_ft(self, filename=None):
        """
        Helper function for writing out the Fourier filter correction.

        :param filename: Filename to write to
        :type filename: str
        """
        if filename is None:
            filename = "ft.dat"
        x = self.q_master[self._ft_title]
        y = self.sq_master[self._ft_title]
        self._write_out_to_file(x, y, filename)

    def write_out_ft_sq(self, filename=None):
        """
        Helper function for writing out the Fourier filtered :math:`S(Q)`

        :param filename: Filename to write to
        :type filename: str
        """
        if filename is None:
            filename = "%s_ft.sq" % self.stem_name
        x = self.q_master[self.sq_ft_title]
        y = self.sq_master[self.sq_ft_title]
        self._write_out_to_file(x, y, filename)

    def write_out_ft_gr(self, filename=None):
        """
        Helper function for writing out the Fourier filtered real space function

        :param filename: Filename to write to
        :type filename: str
        """
        if filename is None:
            filename = "%s_ft.gr" % self.stem_name
        x = self.r_master[self.gr_ft_title]
        y = self.gr_master[self.gr_ft_title]
        self._write_out_to_file(x, y, filename)

    def write_out_lorched_gr(self, filename=None):
        """
        Helper function for writing out the Lorch dampened real space function

        :param filename: Filename to write to
        :type filename: str
        """
        if filename is None:
            filename = "%s_ft_lorched.gr" % self.stem_name
        x = self.r_master[self.gr_lorch_title]
        y = self.gr_master[self.gr_lorch_title]
        self._write_out_to_file(x, y, filename)

    def write_out_rmc_fq(self, filename=None):
        """
        Helper function for writing out the output :math:`F(Q)`

        :param filename: Filename to write to
        :type filename: str
        """
        if filename is None:
            filename = "%s_rmc.fq" % self.stem_name
        x = self.q_master[self.fq_title]
        y = self.sq_master[self.fq_title]
        self._write_out_to_file(x, y, filename)

    def write_out_rmc_gr(self, filename=None):
        """
        Helper function for writing out the output :math:`G_{Keen Version}(Q)`

        :param filename: Filename to write to
        :type filename: str
        """
        if filename is None:
            filename = "%s_rmc.gr" % self.stem_name
        x = self.r_master[self.GKofR_title]
        y = self.gr_master[self.GKofR_title]
        self._write_out_to_file(x, y, filename)
