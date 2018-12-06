import unittest
import numpy as np
import pandas as pd
from utils import \
    get_test_data_path, load_test_data, get_index_of_function, \
    REAL_HEADERS, RECIPROCAL_HEADERS
from materials import Argon
from pystog.stog import StoG

# Real Space Function


class TestStogBase(unittest.TestCase):
    rtol = 1e-5
    atol = 1e-8

    def initialize_material(self):
        # setup input data
        self.kwargs = self.material.kwargs

        # setup the tolerance
        self.first = self.material.reciprocal_space_first
        self.last = self.material.reciprocal_space_last

        data = load_test_data(self.material.reciprocal_space_filename)
        self.q = data[:, get_index_of_function("Q", RECIPROCAL_HEADERS)]
        self.sq = data[:, get_index_of_function("S(Q)", RECIPROCAL_HEADERS)]
        self.fq = data[:, get_index_of_function("F(Q)", RECIPROCAL_HEADERS)]
        self.fq_keen = data[:, get_index_of_function(
            "FK(Q)", RECIPROCAL_HEADERS)]
        self.dcs = data[:, get_index_of_function("DCS(Q)", RECIPROCAL_HEADERS)]

        # targets for 1st peaks
        self.sq_target = self.material.sq_target
        self.fq_target = self.material.fq_target
        self.fq_keen_target = self.material.fq_keen_target
        self.dcs_target = self.material.dcs_target

        # setup the first, last indices
        self.real_space_first = self.material.real_space_first
        self.real_space_last = self.material.real_space_last

        data = load_test_data(self.material.real_space_filename)
        self.r = data[:, get_index_of_function("r", REAL_HEADERS)]
        self.gofr = data[:, get_index_of_function("g(r)", REAL_HEADERS)]
        self.GofR = data[:, get_index_of_function("G(r)", REAL_HEADERS)]
        self.GKofR = data[:, get_index_of_function("GK(r)", REAL_HEADERS)]

        # targets for 1st peaks
        self.gofr_target = self.material.gofr_target
        self.GofR_target = self.material.GofR_target
        self.GKofR_target = self.material.GKofR_target

        # targets for 1st peaks
        self.gofr_ff_target = self.material.gofr_ff_target
        self.GofR_ff_target = self.material.GofR_ff_target
        self.GKofR_ff_target = self.material.GKofR_ff_target

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.material = Argon()
        self.initialize_material()

        self.real_xtarget = 3.525
        self.reciprocal_xtarget = 1.94
        self.fourier_filter_cutoff = 1.5

        self.kwargs_for_files = {
            'Files': [
                {'Filename': get_test_data_path(self.material.reciprocal_space_filename),
                 'ReciprocalFunction': 'S(Q)',
                 'Qmin': 0.02,
                 'Qmax': 15.0,
                 'Y': {'Offset': 0.0,
                       'Scale': 1.0},
                 'X': {'Offset': 0.0}
                 },
                {'Filename': get_test_data_path(self.material.reciprocal_space_filename),
                 'ReciprocalFunction': 'S(Q)',
                 'Qmin': 1.90,
                 'Qmax': 35.2,
                 'Y': {'Offset': 0.0,
                       'Scale': 1.0},
                 'X': {'Offset': 0.0}
                 }
            ]
        }

        self.kwargs_for_stog_input = {
            'NumberDensity': self.material.kwargs['rho'],
            '<b_coh>^2': self.material.kwargs['<b_coh>^2'],
            '<b_tot^2>': self.material.kwargs['<b_tot^2>'],
            'FourierFilter': {'Cutoff': self.fourier_filter_cutoff},
            'OmittedXrangeCorrection': False,
            'Rdelta': self.r[1] - self.r[0],
            'Rmin': min(self.r),
            'Rmax': max(self.r)
        }

    def tearDown(self):
        unittest.TestCase.tearDown(self)


class TestStogInit(TestStogBase):
    def setUp(self):
        super(TestStogInit, self).setUp()

    def test_stog_init(self):
        stog = StoG()
        self.assertEqual(stog.xmin, 100)
        self.assertEqual(stog.xmax, 0)
        self.assertEqual(stog.qmin, None)
        self.assertEqual(stog.qmax, None)
        self.assertEqual(stog.files, None)
        self.assertEqual(stog.sq_title, "S(Q) Merged")
        self.assertEqual(stog.qsq_minus_one_title, "Q[S(Q)-1] Merged")
        self.assertEqual(stog.sq_ft_title, "S(Q) FT")
        self.assertEqual(stog.real_space_function, "g(r)")
        self.assertEqual(stog.gr_title, "g(r) Merged")
        self.assertEqual(stog.gr_ft_title, "g(r) FT")
        self.assertEqual(stog.gr_lorch_title, "g(r) FT Lorched")
        self.assertEqual(stog.rmin, 0.0)
        self.assertEqual(stog.rmax, 50.0)
        self.assertEqual(stog.rdelta, 0.01)
        self.assertEqual(stog.density, 1.0)
        self.assertEqual(stog.bcoh_sqrd, 1.0)
        self.assertEqual(stog.btot_sqrd, 1.0)
        self.assertEqual(stog.low_q_correction, False)
        self.assertEqual(stog.lorch_flag, False)
        self.assertEqual(stog.fourier_filter_cutoff, None)
        self.assertEqual(stog.plot_flag, True)
        self.assertEqual(stog.plotting_kwargs, {'figsize': (16, 8),
                                                'style': '-',
                                                'ms': 1,
                                                'lw': 1,
                                                }
                         )
        self.assertEqual(
            stog.merged_opts, {
                "Y": {
                    "Offset": 0.0, "Scale": 1.0}})
        self.assertEqual(stog.stem_name, "out")
        self.assertTrue(stog.df_individuals.empty)
        self.assertTrue(stog.df_sq_master.empty)
        self.assertTrue(stog.df_sq_individuals.empty)
        self.assertTrue(stog.df_gr_master.empty)

    def test_stog_init_kwargs_files(self):
        stog = StoG(**{'Files': ['file1.txt', 'file2.txt']})
        self.assertEqual(stog.files, ['file1.txt', 'file2.txt'])

    def test_stog_init_kwargs_real_space_function(self):
        stog = StoG(**{'RealSpaceFunction': 'G(r)'})
        self.assertEqual(stog.real_space_function, 'G(r)')

    def test_stog_init_kwargs_rmin(self):
        stog = StoG(**{'Rmin': 2.0})
        self.assertEqual(stog.rmin, 2.0)

    def test_stog_init_kwargs_rmax(self):
        stog = StoG(**{'Rmax': 25.0})
        self.assertEqual(stog.rmax, 25.0)

    def test_stog_init_kwargs_rdelta(self):
        stog = StoG(**{'Rdelta': 0.5})
        self.assertEqual(stog.rdelta, 0.5)

    def test_stog_init_kwargs_rpoints(self):
        stog = StoG(**{'Rpoints': 250})
        self.assertEqual(stog.rdelta, 0.2)

    def test_stog_init_kwargs_density(self):
        stog = StoG(**{'NumberDensity': 2.0})
        self.assertEqual(stog.density, 2.0)

    def test_stog_init_kwargs_low_q_correction(self):
        stog = StoG(**{'OmittedXrangeCorrection': True})
        self.assertEqual(stog.low_q_correction, True)

    def test_stog_init_kwargs_lorch_flag(self):
        stog = StoG(**{'LorchFlag': True})
        self.assertEqual(stog.lorch_flag, True)

    def test_stog_init_kwargs_fourier_filter_cutoff(self):
        stog = StoG(**{'FourierFilter': {'Cutoff': 1.0}})
        self.assertEqual(stog.fourier_filter_cutoff, 1.0)

    def test_stog_init_kwargs_plot_flag(self):
        stog = StoG(**{'PlotFlag': False})
        self.assertEqual(stog.plot_flag, False)

    def test_stog_init_kwargs_bcoh_sqrd(self):
        stog = StoG(**{'<b_coh>^2': 0.5})
        self.assertEqual(stog.bcoh_sqrd, 0.5)

    def test_stog_init_kwargs_btot_sqrd(self):
        stog = StoG(**{'<b_tot^2>': 0.75})
        self.assertEqual(stog.btot_sqrd, 0.75)

    def test_stog_init_kwargs_qmin_only(self):
        stog = StoG(**{'Merging': {'Transform': {'Qmin': 0.5}}})
        self.assertEqual(stog.qmin, 0.5)
        self.assertEqual(stog.qmax, None)

    def test_stog_init_kwargs_qmax_only(self):
        stog = StoG(**{'Merging': {'Transform': {'Qmax': 30.0}}})
        self.assertEqual(stog.qmin, None)
        self.assertEqual(stog.qmax, 30.0)

    def test_stog_init_kwargs_qmin_and_qmax(self):
        stog = StoG(**{'Merging': {'Transform': {'Qmin': 0.5, 'Qmax': 30.0}}})
        self.assertEqual(stog.qmin, 0.5)
        self.assertEqual(stog.qmax, 30.0)

    def test_stog_init_kwargs_output_stem_name(self):
        stog = StoG(**{'Outputs': {'StemName': 'myName'}})
        self.assertEqual(stog.stem_name, 'myName')


class TestStogAttributes(TestStogBase):
    def setUp(self):
        super(TestStogAttributes, self).setUp()

    def test_stog_xmin_setter(self):
        stog = StoG()
        stog.xmin = 0.25
        self.assertEqual(stog.xmin, 0.25)

    def test_stog_xmax_setter(self):
        stog = StoG()
        stog.xmax = 10.0
        self.assertEqual(stog.xmax, 10.0)

    def test_stog_dr_getter(self):
        stog = StoG()
        self.assertAlmostEqual(stog.dr[0], 0.0)
        self.assertAlmostEqual(stog.dr[-1], 50.0)

    def test_stog_dr_setter_rmax(self):
        stog = StoG()
        stog.rmax = 25.0
        self.assertAlmostEqual(stog.dr[0], 0.0)
        self.assertAlmostEqual(stog.dr[-1], 25.0)

    def test_stog_dr_setter_rmin(self):
        stog = StoG()
        stog.rmin = 10.0
        self.assertAlmostEqual(stog.dr[0], 10.0)
        self.assertAlmostEqual(stog.dr[-1], 50.0)

    def test_stog_dr_setter_rdelta(self):
        stog = StoG()
        stog.rdelta = 0.5
        self.assertEqual(stog.dr[1] - stog.dr[0], 0.5)

    def test_stog_sq_title_function_setter(self):
        stog = StoG()
        stog.sq_title = "S(Q) dog"
        self.assertEqual(stog.sq_title, "S(Q) dog")

    def test_stog_qsq_minus_one_title_setter(self):
        stog = StoG()
        stog.qsq_minus_one_title = "Q[S(Q)-1] dog"
        self.assertEqual(stog.qsq_minus_one_title, "Q[S(Q)-1] dog")

    def test_stog_sq_ft_title_setter(self):
        stog = StoG()
        stog.sq_ft_title = "S(Q) FT dog"
        self.assertEqual(stog.sq_ft_title, "S(Q) FT dog")

    def test_stog_real_space_function_setter(self):
        stog = StoG()
        stog.real_space_function = "GK(r)"
        self.assertEqual(stog.real_space_function, "GK(r)")
        self.assertEqual(stog.gr_title, "GK(r) Merged")
        self.assertEqual(stog.real_space_function, "GK(r)")
        self.assertEqual(stog.gr_ft_title, "GK(r) FT")
        self.assertEqual(stog.gr_lorch_title, "GK(r) FT Lorched")

    def test_stog_plotting_kwargs_setter(self):
        stog = StoG()
        new_kwargs = {'figsize': (4, 4),
                      'style': 'o',
                      'ms': 2,
                      'lw': 2,
                      }
        stog.plotting_kwargs = new_kwargs
        self.assertEqual(stog.plotting_kwargs, new_kwargs)

    def test_stog_low_q_correction_exception(self):
        stog = StoG()
        with self.assertRaises(TypeError):
            stog.low_q_correction = 1.0

    def test_stog_lorch_flag_exception(self):
        stog = StoG()
        with self.assertRaises(TypeError):
            stog.lorch_flag = 1.0

    def test_stog_plot_flag_exception(self):
        stog = StoG()
        with self.assertRaises(TypeError):
            stog.plot_flag = 1.0

    def test_stog_real_space_function_exception(self):
        stog = StoG()
        with self.assertRaises(ValueError):
            stog.real_space_function = "Dog"


class TestStogDataFrameAttributes(TestStogBase):
    def setUp(self):
        super(TestStogDataFrameAttributes, self).setUp()
        self.df_target = pd.DataFrame(np.random.randn(10, 2),
                                      index=np.arange(10),
                                      columns=list('XY'))

    def test_stog_df_individuals_setter(self):
        stog = StoG()
        stog.df_individuals = self.df_target
        self.assertTrue(stog.df_individuals.equals(self.df_target))

    def test_stog_df_sq_individuals_setter(self):
        stog = StoG()
        stog.df_sq_individuals = self.df_target
        self.assertTrue(stog.df_sq_individuals.equals(self.df_target))

    def test_stog_df_sq_master_setter(self):
        stog = StoG()
        stog.df_sq_master = self.df_target
        self.assertTrue(stog.df_sq_master.equals(self.df_target))

    def test_stog_df_gr_master_setter(self):
        stog = StoG()
        stog.df_gr_master = self.df_target
        self.assertTrue(stog.df_gr_master.equals(self.df_target))


class TestStogGeneralMethods(TestStogBase):
    def setUp(self):
        super(TestStogGeneralMethods, self).setUp()

    def test_stog_append_file(self):
        stog = StoG(**{'Files': ['file1.txt', 'file2.txt']})
        stog.append_file('file3.txt')
        self.assertEqual(stog.files, ['file1.txt', 'file2.txt', 'file3.txt'])

    def test_stog_extend_file_list(self):
        stog = StoG(**{'Files': ['file1.txt', 'file2.txt']})
        stog.extend_file_list(['file3.txt', 'file4.txt'])
        self.assertEqual(
            stog.files, [
                'file1.txt', 'file2.txt', 'file3.txt', 'file4.txt'])


class TestStogDatasetSpecificMethods(TestStogBase):
    def setUp(self):
        super(TestStogDatasetSpecificMethods, self).setUp()

    def test_stog_add_dataset(self):
        # Number of decimal places for precision
        places = 5

        # Initialize with material info for Argon
        stog = StoG(**{'<b_coh>^2': self.kwargs['<b_coh>^2'],
                       '<b_tot^2>': self.kwargs['<b_tot^2>']})

        # Add the S(Q) data set and check values against targets
        index = 0
        info = {'data': pd.DataFrame({'x': self.q, 'y': self.sq}),
                'ReciprocalFunction': 'S(Q)'}
        stog.add_dataset(info, index=index)
        self.assertEqual(
            stog.df_individuals.iloc[self.first].name, self.reciprocal_xtarget)
        self.assertAlmostEqual(stog.df_individuals.iloc[self.first]['S(Q)_%d' % index],
                               self.sq_target[0],
                               places=places)
        self.assertAlmostEqual(stog.df_sq_individuals.iloc[self.first]['S(Q)_%d' % index],
                               self.sq_target[0],
                               places=places)

        # Add the Q[S(Q)-1] data set and check values for it and S(Q) against
        # targets
        index = 1
        info = {'data': pd.DataFrame({'x': self.q, 'y': self.fq}),
                'ReciprocalFunction': 'Q[S(Q)-1]'}
        stog.add_dataset(info, index=index)
        self.assertAlmostEqual(stog.df_individuals.iloc[self.first]['Q[S(Q)-1]_%d' % index],
                               self.fq_target[0],
                               places=places)
        self.assertAlmostEqual(stog.df_sq_individuals.iloc[self.first]['S(Q)_%d' % index],
                               self.sq_target[0],
                               places=places)

        # Add the FK(Q) data set and check values for it and S(Q) against
        # targets
        index = 2
        info = {'data': pd.DataFrame({'x': self.q, 'y': self.fq_keen}),
                'ReciprocalFunction': 'FK(Q)'}
        stog.add_dataset(info, index=index)
        self.assertAlmostEqual(stog.df_individuals.iloc[self.first]['FK(Q)_%d' % index],
                               self.fq_keen_target[0],
                               places=places)
        self.assertAlmostEqual(stog.df_sq_individuals.iloc[self.first]['S(Q)_%d' % index],
                               self.sq_target[0],
                               places=places)

        # Add the DCS(Q) data set and check values for it and S(Q) against
        # targets
        index = 3
        info = {'data': pd.DataFrame({'x': self.q, 'y': self.dcs}),
                'ReciprocalFunction': 'DCS(Q)'}
        stog.add_dataset(info, index=index)
        self.assertAlmostEqual(stog.df_individuals.iloc[self.first]['DCS(Q)_%d' % index],
                               self.dcs_target[0],
                               places=places)
        self.assertAlmostEqual(stog.df_sq_individuals.iloc[self.first]['S(Q)_%d' % index],
                               self.sq_target[0],
                               places=places)

    def test_stog_add_dataset_yscale(self):
        # Scale S(Q) and make sure it does not equal original target values
        stog = StoG()
        index = 0
        info = {'data': pd.DataFrame({'x': self.q, 'y': self.sq}),
                'ReciprocalFunction': 'S(Q)',
                'Y': {'Scale': 2.0}}
        stog.add_dataset(info, index=index)
        self.assertNotEqual(stog.df_individuals.iloc[self.first]['S(Q)_%d' % index],
                            self.sq_target[0])
        self.assertNotEqual(stog.df_sq_individuals.iloc[self.first]['S(Q)_%d' % index],
                            self.sq_target[0])

    def test_stog_add_dataset_yoffset(self):
        # Offset S(Q) and make sure it does not equal original target values
        stog = StoG()
        index = 0
        info = {'data': pd.DataFrame({'x': self.q, 'y': self.sq}),
                'ReciprocalFunction': 'S(Q)',
                'Y': {'Offset': 2.0}}
        stog.add_dataset(info, index=index)
        self.assertNotEqual(stog.df_individuals.iloc[self.first]['S(Q)_%d' % index],
                            self.sq_target[0])
        self.assertNotEqual(stog.df_sq_individuals.iloc[self.first]['S(Q)_%d' % index],
                            self.sq_target[0])

    def test_stog_add_dataset_xoffset(self):
        # Offset Q from 1.96 -> 2.14
        stog = StoG()
        index = 0
        info = {'data': pd.DataFrame({'x': self.q, 'y': self.sq}),
                'ReciprocalFunction': 'S(Q)',
                'X': {'Offset': 0.2}}
        stog.add_dataset(info, index=index)
        self.assertEqual(stog.df_individuals.iloc[self.first].name, 2.14)

    def test_stog_add_dataset_qmin_qmax_crop(self):
        # Check qmin and qmax apply cropping
        stog = StoG()
        index = 0
        info = {'data': pd.DataFrame({'x': self.q, 'y': self.sq}),
                'ReciprocalFunction': 'S(Q)'}
        stog.qmin = 1.5
        stog.qmax = 12.0
        stog.add_dataset(info, index=index)
        self.assertEqual(stog.df_individuals.iloc[0].name, stog.qmin)
        self.assertEqual(stog.df_individuals.iloc[-1].name, stog.qmax)

    def test_stog_add_dataset_default_reciprocal_space_function(self):
        # Checks the default reciprocal space function is S(Q) and the index is
        # set
        stog = StoG()
        index = 300
        info = {'data': pd.DataFrame({'x': self.q, 'y': self.sq})}
        stog.add_dataset(info, index=index)
        self.assertEqual(
            list(
                stog.df_individuals.columns.values), [
                'S(Q)_%d' %
                index])

    def test_stog_add_dataset_wrong_reciprocal_space_function_exception(self):
        # Check qmin and qmax apply cropping
        stog = StoG()
        index = 0
        info = {'data': pd.DataFrame({'x': self.q, 'y': self.sq}),
                'ReciprocalFunction': 'ABCDEFG(Q)'}
        with self.assertRaises(ValueError):
            stog.add_dataset(info, index=index)

    def test_stog_read_dataset(self):
        # Number of decimal places for precision
        places = 5

        # Load S(Q) for Argon from test data
        stog = StoG(**{'<b_coh>^2': self.kwargs['<b_coh>^2'],
                       '<b_tot^2>': self.kwargs['<b_tot^2>']})
        info = {
            'Filename': get_test_data_path(
                self.material.reciprocal_space_filename),
            'ReciprocalFunction': 'S(Q)',
            'Qmin': 0.02,
            'Qmax': 35.2,
            'Y': {
                'Offset': 0.0,
                'Scale': 1.0},
            'X': {
                'Offset': 0.0}}

        info['index'] = 0
        stog.read_dataset(info)

        # Check S(Q) data against targets
        self.assertEqual(
            stog.df_individuals.iloc[self.first].name, self.reciprocal_xtarget)
        self.assertAlmostEqual(stog.df_individuals.iloc[self.first]['S(Q)_%d' % info['index']],
                               self.sq_target[0],
                               places=places)
        self.assertAlmostEqual(stog.df_sq_individuals.iloc[self.first]['S(Q)_%d' % info['index']],
                               self.sq_target[0],
                               places=places)

    def test_stog_read_all_data_assertion(self):
        stog = StoG()
        with self.assertRaises(AssertionError):
            stog.read_all_data()

        stog.files = list()
        with self.assertRaises(AssertionError):
            stog.read_all_data()

    def test_stog_read_all_data_for_files_length(self):
        # Load S(Q) for Argon from test data
        stog = StoG()
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()

        # Check S(Q) data against targets
        self.assertEqual(len(stog.files), len(self.kwargs_for_files['Files']))

    def test_stog_read_all_data(self):
        # Number of decimal places for precision
        places = 5

        # Load S(Q) for Argon from test data
        stog = StoG()
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()

        # Check S(Q) data against targets
        self.assertEqual(
            stog.df_individuals.iloc[self.first].name, self.reciprocal_xtarget)
        for index in range(len(stog.files)):
            self.assertAlmostEqual(stog.df_individuals.iloc[self.first]['S(Q)_%d' % index],
                                   self.sq_target[0],
                                   places=places)
            self.assertAlmostEqual(stog.df_sq_individuals.iloc[self.first]['S(Q)_%d' % index],
                                   self.sq_target[0],
                                   places=places)

    def test_stog_merge_data(self):
        # Number of decimal places for precision
        places = 5

        # Load S(Q) for Argon from test data
        stog = StoG()
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()
        stog.merge_data()

        # Check S(Q) data against targets
        self.assertEqual(
            stog.df_sq_master.iloc[self.first].name, self.reciprocal_xtarget)
        self.assertAlmostEqual(stog.df_sq_master.iloc[self.first][stog.sq_title],
                               self.sq_target[0],
                               places=places)


class TestStogTransformSpecificMethods(TestStogDatasetSpecificMethods):
    def setUp(self):
        super(TestStogTransformSpecificMethods, self).setUp()

    def test_stog_transform_merged(self):
        # Number of decimal places for precision
        places = 2

        # Load S(Q) for Argon from test data
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.plot_flag = True
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()

        # Check g(r) data against targets
        self.assertAlmostEqual(stog.df_gr_master.iloc[self.real_space_first].name,
                               self.real_xtarget,
                               places=places)
        self.assertAlmostEqual(stog.df_gr_master.iloc[self.real_space_first][stog.gr_title],
                               self.gofr_target[0],
                               places=places)

    def test_stog_fourier_filter(self):
        # Number of decimal places for precision
        places = 1

        # Load S(Q) for Argon from test data
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.plot_flag = False
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()
        stog.fourier_filter()

        # Check g(r) data against targets
        self.assertAlmostEqual(stog.df_gr_master.iloc[self.real_space_first].name,
                               self.real_xtarget,
                               places=places)
        self.assertAlmostEqual(stog.df_gr_master.iloc[self.real_space_first][stog.gr_ft_title],
                               self.gofr_ff_target[0],
                               places=places)


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
