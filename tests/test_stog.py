import numpy as np
import os
import tempfile
import unittest

from tests.utils import \
    get_data_path, \
    get_index_of_function, \
    load_data
from tests.materials import Argon
from pystog.utils import \
    RealSpaceHeaders, \
    ReciprocalSpaceHeaders
from pystog.stog import NoInputFilesException, StoG


class TestStogBase(unittest.TestCase):
    rtol = 1.0
    atol = 1.0

    def initialize_material(self):
        # setup input data
        self.kwargs = self.material.kwargs

        # setup the tolerance
        self.first = self.material.reciprocal_space_first
        self.last = self.material.reciprocal_space_last

        data = load_data(self.material.reciprocal_space_filename)
        self.q = data[:, get_index_of_function("Q", ReciprocalSpaceHeaders)]
        self.sq = data[:, get_index_of_function(
            "S(Q)", ReciprocalSpaceHeaders)]
        self.fq = data[:, get_index_of_function(
            "Q[S(Q)-1]", ReciprocalSpaceHeaders)]
        self.fq_keen = data[:, get_index_of_function(
            "FK(Q)", ReciprocalSpaceHeaders)]
        self.dcs = data[:, get_index_of_function(
            "DCS(Q)", ReciprocalSpaceHeaders)]

        # targets for 1st peaks
        self.sq_target = self.material.sq_target
        self.dsq_target = 3.087955
        self.fq_target = self.material.fq_target
        self.fq_keen_target = self.material.fq_keen_target
        self.dcs_target = self.material.dcs_target

        # setup the first, last indices
        self.real_space_first = self.material.real_space_first
        self.real_space_last = self.material.real_space_last

        data = load_data(self.material.real_space_filename)
        self.r = data[:, get_index_of_function("r", RealSpaceHeaders)]
        self.gofr = data[:, get_index_of_function("g(r)", RealSpaceHeaders)]
        self.GofR = data[:, get_index_of_function("G(r)", RealSpaceHeaders)]
        self.GKofR = data[:, get_index_of_function("GK(r)", RealSpaceHeaders)]

        self.gofr_target = self.material.gofr_target
        self.GofR_target = self.material.GofR_target
        self.GKofR_target = self.material.GKofR_target

        self.gofr_ff_target = self.material.gofr_ff_target
        self.GofR_ff_target = self.material.GofR_ff_target
        self.GKofR_ff_target = self.material.GKofR_ff_target

        self.gofr_lorch_target = self.material.gofr_lorch_target
        self.GofR_lorch_target = self.material.GofR_lorch_target
        self.GKofR_lorch_target = self.material.GKofR_lorch_target

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.material = Argon()
        self.initialize_material()

        self.real_xtarget = 3.525
        self.reciprocal_xtarget = 1.94
        self.fourier_filter_cutoff = 1.5

        filename = get_data_path(self.material.reciprocal_space_filename)
        self.kwargs_for_files = {
            'Files': [
                {
                    'Filename': filename,
                    'ReciprocalFunction': 'S(Q)',
                    'Qmin': 0.02,
                    'Qmax': 15.0,
                    'Y': {
                        'Offset': 0.0,
                        'Scale': 1.0
                    },
                    'X': {
                        'Offset': 0.0
                    }
                },
                {
                    'Filename': filename,
                    'ReciprocalFunction': 'S(Q)',
                    'Qmin': 1.90,
                    'Qmax': 35.2,
                    'Y': {
                        'Offset': 0.0,
                        'Scale': 1.0},
                    'X': {
                        'Offset': 0.0}
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
        self.assertEqual(stog.fq_title, "F(Q) Merged")
        self.assertEqual(stog.real_space_function, "g(r)")
        self.assertEqual(stog.gr_title, "g(r) Merged")
        self.assertEqual(stog.gr_ft_title, "g(r) FT")
        self.assertEqual(stog.gr_lorch_title, "g(r) FT Lorched")
        self.assertEqual(stog.GKofR_title, "G(r) (Keen Version)")
        self.assertEqual(stog.rmin, 0.0)
        self.assertEqual(stog.rmax, 50.0)
        self.assertEqual(stog.rdelta, 0.01)
        self.assertEqual(stog.density, 1.0)
        self.assertEqual(stog.bcoh_sqrd, 1.0)
        self.assertEqual(stog.btot_sqrd, 1.0)
        self.assertEqual(stog.low_q_correction, False)
        self.assertEqual(stog.lorch_flag, False)
        self.assertEqual(stog.fourier_filter_cutoff, None)
        self.assertEqual(
            stog.merged_opts, {
                "Y": {
                    "Offset": 0.0, "Scale": 1.0}})
        self.assertEqual(stog.stem_name, "out")
        self.assertEqual(stog.reciprocal_individuals.size, 0)
        self.assertEqual(stog.q_master, {})
        self.assertEqual(stog.sq_master, {})
        self.assertEqual(stog.sq_individuals.size, 0)
        self.assertEqual(stog.r_master, {})
        self.assertEqual(stog.gr_master, {})

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

    def test_stog_gr_title_function_setter(self):
        stog = StoG()
        stog.gr_title = "G(r) dog"
        self.assertEqual(stog.gr_title, "G(r) dog")

    def test_stog_gr_ft_title_function_setter(self):
        stog = StoG()
        stog.gr_ft_title = "G(r) FT dog"
        self.assertEqual(stog.gr_ft_title, "G(r) FT dog")

    def test_stog_gr_lorch_title_function_setter(self):
        stog = StoG()
        stog.gr_lorch_title = "G(r) FT Lorch dog"
        self.assertEqual(stog.gr_lorch_title, "G(r) FT Lorch dog")

    def test_stog_GKofR_title_function_setter(self):
        stog = StoG()
        stog.GKofR_title = "GK(r) dog"
        self.assertEqual(stog.GKofR_title, "GK(r) dog")

    def test_stog_sq_title_function_setter(self):
        stog = StoG()
        stog.sq_title = "S(Q) dog"
        self.assertEqual(stog.sq_title, "S(Q) dog")

    def test_stog_fq_title_function_setter(self):
        stog = StoG()
        stog.fq_title = "F(Q) dog"
        self.assertEqual(stog.fq_title, "F(Q) dog")

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

    def test_stog_low_q_correction_exception(self):
        stog = StoG()
        with self.assertRaises(TypeError):
            stog.low_q_correction = 1.0

    def test_stog_lorch_flag_exception(self):
        stog = StoG()
        with self.assertRaises(TypeError):
            stog.lorch_flag = 1.0

    def test_stog_real_space_function_exception(self):
        stog = StoG()
        with self.assertRaises(ValueError):
            stog.real_space_function = "Dog"


class TestStogStorageArrays(TestStogBase):
    def setUp(self):
        super(TestStogStorageArrays, self).setUp()
        self.target = np.random.randn(3, 10)

    def test_stog_reciprocal_individuals_setter(self):
        stog = StoG()
        stog.reciprocal_individuals = self.target
        np.testing.assert_allclose(stog.reciprocal_individuals, self.target)

    def test_stog_sq_individuals_setter(self):
        stog = StoG()
        stog.sq_individuals = self.target
        np.testing.assert_allclose(stog.sq_individuals, self.target)

    def test_stog_q_master_setter(self):
        stog = StoG()
        stog.q_master = self.target
        np.testing.assert_allclose(stog.q_master, self.target)

    def test_stog_sq_master_setter(self):
        stog = StoG()
        stog.sq_master = self.target
        np.testing.assert_allclose(stog.sq_master, self.target)

    def test_stog_r_master_setter(self):
        stog = StoG()
        stog.r_master = self.target
        np.testing.assert_allclose(stog.r_master, self.target)

    def test_stog_gr_master_setter(self):
        stog = StoG()
        stog.gr_master = self.target
        np.testing.assert_allclose(stog.gr_master, self.target)


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

    def test_stog_apply_scales_and_offset(self):
        q, sq, dq = StoG.apply_scales_and_offset(self.q, self.sq)
        np.testing.assert_allclose(q, self.q)
        np.testing.assert_allclose(sq, self.sq)
        np.testing.assert_allclose(dq, np.zeros_like(sq))

    def test_stog_apply_scales_and_offset_with_dy(self):
        q, sq, dq = StoG.apply_scales_and_offset(self.q, self.sq, dy=self.sq)
        np.testing.assert_allclose(q, self.q)
        np.testing.assert_allclose(sq, self.sq)
        np.testing.assert_allclose(dq, self.sq)

    def test_stog_apply_scales_and_offset_with_yscale(self):
        q, sq, dq = StoG.apply_scales_and_offset(
            self.q, self.sq, dy=self.sq, yscale=2.0)
        np.testing.assert_allclose(q, self.q)
        np.testing.assert_allclose(sq, self.sq * 2.0)
        np.testing.assert_allclose(dq, self.sq * 2.0)

    def test_stog_apply_scales_and_offset_with_yoffset(self):
        q, sq, dq = StoG.apply_scales_and_offset(
            self.q, self.sq, dy=self.sq, yoffset=2.0)
        np.testing.assert_allclose(q, self.q)
        np.testing.assert_allclose(sq, self.sq + 2.0)
        np.testing.assert_allclose(dq, self.sq)

    def test_stog_apply_scales_and_offset_with_xoffset(self):
        q, sq, dq = StoG.apply_scales_and_offset(
            self.q, self.sq, dy=self.sq, xoffset=2.0)
        np.testing.assert_allclose(q, self.q + 2.0)
        np.testing.assert_allclose(sq, self.sq)
        np.testing.assert_allclose(dq, self.sq)

    def test_stog_add_dataset(self):
        # Number of decimal places for precision
        places = 5

        # Initialize with material info for Argon
        stog = StoG(**{'<b_coh>^2': self.kwargs['<b_coh>^2'],
                       '<b_tot^2>': self.kwargs['<b_tot^2>']})

        # Add the S(Q) data set and check values against targets
        info = {
            'data': [self.q, self.sq],
            'ReciprocalFunction': 'S(Q)'}
        stog.add_dataset(info)
        self.assertEqual(
            stog.reciprocal_individuals[0][self.first],
            self.reciprocal_xtarget)
        self.assertAlmostEqual(
            stog.reciprocal_individuals[1][self.first],
            self.sq_target[0],
            places=places)
        self.assertEqual(stog.reciprocal_individuals[2][self.first], 0.0)
        self.assertAlmostEqual(
            stog.sq_individuals[1][self.first],
            self.sq_target[0],
            places=places)
        self.assertEqual(stog.sq_individuals[2][self.first], 0.0)

        # Add the Q[S(Q)-1] data set and check values for it and S(Q) against
        # targets
        stride = stog.reciprocal_individuals.shape[1]
        info = {
            'data': [self.q, self.fq],
            'ReciprocalFunction': 'Q[S(Q)-1]'
        }
        stog.add_dataset(info)

        self.assertAlmostEqual(
            stog.reciprocal_individuals[1][self.first + stride],
            self.fq_target[0],
            places=places)
        self.assertEqual(
            stog.reciprocal_individuals[2][self.first + stride],
            0.0)
        self.assertAlmostEqual(
            stog.sq_individuals[1][self.first],
            self.sq_target[0],
            places=places)
        self.assertEqual(stog.sq_individuals[2][self.first], 0.0)

        # Add the FK(Q) data set and check values for it and S(Q) against
        # targets
        stride = stog.reciprocal_individuals.shape[1]
        info = {
            'data': [self.q, self.fq_keen],
            'ReciprocalFunction': 'FK(Q)'}
        stog.add_dataset(info)
        self.assertAlmostEqual(
            stog.reciprocal_individuals[1][self.first + stride],
            self.fq_keen_target[0],
            places=places)
        self.assertEqual(
            stog.reciprocal_individuals[2][self.first + stride],
            0.0)
        self.assertAlmostEqual(
            stog.sq_individuals[1][self.first],
            self.sq_target[0],
            places=places)
        self.assertEqual(stog.sq_individuals[2][self.first], 0.0)

        # Add the DCS(Q) data set and check values for it and S(Q) against
        # targets
        stride = stog.reciprocal_individuals.shape[1]
        info = {
            'data': [self.q, self.dcs],
            'ReciprocalFunction': 'DCS(Q)'}
        stog.add_dataset(info)
        self.assertAlmostEqual(
            stog.reciprocal_individuals[1][self.first + stride],
            self.dcs_target[0],
            places=places)
        self.assertEqual(
            stog.reciprocal_individuals[2][self.first + stride],
            0.0)
        self.assertAlmostEqual(
            stog.sq_individuals[1][self.first],
            self.sq_target[0],
            places=places)
        self.assertEqual(stog.sq_individuals[2][self.first], 0.0)

    def test_stog_add_dataset_yscale(self):
        # Scale S(Q) and make sure it does not equal original target values
        stog = StoG()
        info = {
            'data': [self.q, self.sq],
            'ReciprocalFunction': 'S(Q)',
            'Y': {'Scale': 2.0}}
        stog.add_dataset(info)
        self.assertNotEqual(
            stog.reciprocal_individuals[1][self.first],
            self.sq_target[0])
        self.assertEqual(stog.reciprocal_individuals[2][self.first], 0.0)
        self.assertNotEqual(
            stog.sq_individuals[1][self.first],
            self.sq_target[0])
        self.assertEqual(stog.sq_individuals[2][self.first], 0.0)

    def test_stog_add_dataset_yoffset(self):
        # Offset S(Q) and make sure it does not equal original target values
        stog = StoG()
        info = {
            'data': [self.q, self.sq],
            'ReciprocalFunction': 'S(Q)',
            'Y': {'Offset': 2.0}}
        stog.add_dataset(info)
        self.assertNotEqual(
            stog.reciprocal_individuals[1][self.first],
            self.sq_target[0])
        self.assertEqual(stog.reciprocal_individuals[2][self.first], 0.0)
        self.assertNotEqual(
            stog.sq_individuals[1][self.first],
            self.sq_target[0])
        self.assertEqual(stog.sq_individuals[2][self.first], 0.0)

    def test_stog_add_dataset_yscale_with_dy(self):
        # Scale S(Q) and make sure it does not equal original target values
        stog = StoG()
        info = {
            'data': [self.q, self.sq, self.sq],
            'ReciprocalFunction': 'S(Q)',
            'Y': {'Scale': 2.0}}
        stog.add_dataset(info)

        self.assertNotEqual(
            stog.reciprocal_individuals[1][self.first],
            self.sq_target[0])

        dsq_target = info['Y']['Scale'] * 2.59173
        self.assertEqual(
            stog.reciprocal_individuals[2][self.first],
            dsq_target)

        self.assertNotEqual(
            stog.sq_individuals[1][self.first],
            self.sq_target[0])

        self.assertEqual(stog.sq_individuals[2][self.first], dsq_target)

    def test_stog_add_dataset_yoffset_with_dy(self):
        # Offset S(Q) and make sure it does not equal original target values
        stog = StoG()
        info = {
            'data': [self.q, self.sq, self.sq],
            'ReciprocalFunction': 'S(Q)',
            'Y': {'Offset': 2.0}}
        stog.add_dataset(info)

        self.assertNotEqual(
            stog.reciprocal_individuals[1][self.first],
            self.sq_target[0])

        dsq_target = 2.59173
        self.assertEqual(
            stog.reciprocal_individuals[2][self.first],
            dsq_target)

        self.assertNotEqual(
            stog.sq_individuals[1][self.first],
            self.sq_target[0])

        self.assertEqual(stog.sq_individuals[2][self.first], dsq_target)

    def test_stog_add_dataset_xoffset(self):
        # Offset Q from 1.96 -> 2.14
        stog = StoG()
        info = {
            'data': [self.q, self.sq],
            'ReciprocalFunction': 'S(Q)',
            'X': {'Offset': 0.2}}
        stog.add_dataset(info)
        self.assertEqual(stog.reciprocal_individuals[0][self.first], 2.14)

    def test_stog_add_dataset_qmin_qmax_crop(self):
        # Check qmin and qmax apply cropping
        stog = StoG()
        info = {'data': [self.q, self.sq],
                'ReciprocalFunction': 'S(Q)'}
        stog.qmin = 1.5
        stog.qmax = 12.0
        stog.add_dataset(info)
        self.assertEqual(stog.reciprocal_individuals[0][0], stog.qmin)
        self.assertEqual(stog.reciprocal_individuals[0][-1], stog.qmax)

    def test_stog_add_dataset_default_reciprocal_space_function(self):
        # Checks the default reciprocal space function is S(Q)
        places = 5
        stog = StoG()
        info = {'data': [self.q, self.sq]}
        stog.add_dataset(info)
        self.assertEqual(
            stog.reciprocal_individuals[0][self.first],
            self.reciprocal_xtarget)
        self.assertEqual(stog.reciprocal_individuals[2][self.first], 0.0)
        self.assertAlmostEqual(
            stog.reciprocal_individuals[1][self.first],
            self.sq_target[0],
            places=places)
        self.assertEqual(stog.sq_individuals[2][self.first], 0.0)

    def test_stog_add_dataset_wrong_reciprocal_space_function_exception(self):
        # Check qmin and qmax apply cropping
        stog = StoG()
        info = {'data': [self.q, self.sq],
                'ReciprocalFunction': 'ABCDEFG(Q)'}
        with self.assertRaises(ValueError):
            stog.add_dataset(info)

    def test_stog_read_dataset(self):
        # Number of decimal places for precision
        places = 5

        # Load S(Q) for Argon from test data
        stog = StoG(**{'<b_coh>^2': self.kwargs['<b_coh>^2'],
                       '<b_tot^2>': self.kwargs['<b_tot^2>']})
        info = {
            'Filename': get_data_path(
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
            stog.reciprocal_individuals[0][self.first],
            self.reciprocal_xtarget)
        self.assertAlmostEqual(
            stog.reciprocal_individuals[1][self.first],
            self.sq_target[0],
            places=places)
        self.assertEqual(
            stog.reciprocal_individuals[2][self.first],
            self.dsq_target)
        self.assertAlmostEqual(
            stog.sq_individuals[1][self.first],
            self.sq_target[0],
            places=places)
        self.assertEqual(
            stog.sq_individuals[2][self.first],
            self.dsq_target)

    def test_stog_read_dataset_xcol_data_format_exception(self):
        stog = StoG()
        filename = get_data_path(self.material.reciprocal_space_filename)
        with self.assertRaises(RuntimeError):
            stog.read_dataset({'Filename': filename}, xcol=99)

    def test_stog_read_dataset_ycol_data_format_exception(self):
        stog = StoG()
        filename = get_data_path(self.material.reciprocal_space_filename)
        with self.assertRaises(RuntimeError):
            stog.read_dataset({'Filename': filename}, ycol=99)

    def test_stog_read_dataset_dycol_too_large(self):
        # Number of decimal places for precision
        places = 5

        # Load S(Q) for Argon from test data
        stog = StoG(**{'<b_coh>^2': self.kwargs['<b_coh>^2'],
                       '<b_tot^2>': self.kwargs['<b_tot^2>']})
        info = {
            'Filename': get_data_path(
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
        stog.read_dataset(info, dycol=99)

        # Check S(Q) data against targets
        self.assertEqual(
            stog.reciprocal_individuals[0][self.first],
            self.reciprocal_xtarget)
        self.assertAlmostEqual(
            stog.reciprocal_individuals[1][self.first],
            self.sq_target[0],
            places=places)
        self.assertEqual(
            stog.reciprocal_individuals[2][self.first],
            0.0)
        self.assertAlmostEqual(
            stog.sq_individuals[1][self.first],
            self.sq_target[0],
            places=places)

    def test_stog_read_all_data_assertion(self):
        stog = StoG()
        with self.assertRaises(NoInputFilesException):
            stog.read_all_data()

        stog.files = list()
        with self.assertRaises(NoInputFilesException):
            stog.read_all_data()

    def test_stog_read_all_data_for_files_length(self):
        # Load S(Q) for Argon from test data
        stog = StoG()
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()

        # Check S(Q) data against targets
        self.assertEqual(
            len(stog.files),
            len(self.kwargs_for_files['Files']))

    def test_stog_read_all_data(self):
        # Number of decimal places for precision
        places = 5

        # Load S(Q) for Argon from test data
        stog = StoG()
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()

        # Check S(Q) data against targets
        self.assertEqual(
            stog.reciprocal_individuals[0][self.first], self.reciprocal_xtarget)
        for index in range(len(stog.files)):
            self.assertAlmostEqual(
                stog.reciprocal_individuals[1][self.first],
                self.sq_target[0],
                places=places)
            self.assertAlmostEqual(
                stog.sq_individuals[1][self.first],
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
            stog.q_master[stog.sq_title][self.first],
            self.reciprocal_xtarget)
        self.assertAlmostEqual(
            stog.sq_master[stog.sq_title][self.first],
            self.sq_target[0],
            places=places)

    def test_stog_merge_data_qsq_opts_scale(self):
        # Number of decimal places for precision
        places = 5

        # Load S(Q) for Argon from test data
        stog = StoG()
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()
        qsq_opts = {'Y': {'Scale': 2.0}}

        # Test Q[S(Q)-1] scale
        stog.merged_opts['Q[S(Q)-1]'] = qsq_opts
        stog.merge_data()
        self.assertAlmostEqual(
            stog.sq_master[stog.qsq_minus_one_title][self.first],
            qsq_opts['Y']['Scale'] * self.fq_target[0],
            places=places)

    def test_stog_merge_data_qsq_opts_offset(self):
        # Number of decimal places for precision
        places = 5

        # Load S(Q) for Argon from test data
        stog = StoG()
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()
        qsq_opts = {'Y': {'Offset': 1.0}}

        # Test Q[S(Q)-1] scale
        stog.merged_opts['Q[S(Q)-1]'] = qsq_opts
        stog.merge_data()
        self.assertAlmostEqual(
            stog.sq_master[stog.qsq_minus_one_title][self.first],
            qsq_opts['Y']['Offset'] + self.fq_target[0],
            places=places)


class TestStogTransformSpecificMethods(TestStogDatasetSpecificMethods):
    def setUp(self):
        super(TestStogTransformSpecificMethods, self).setUp()
        self.lowR_target = 0.4720653

    def test_stog_transform_merged_default(self):
        # Number of decimal places for precision
        places = 2

        # Load S(Q) for Argon from test data
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()

        # Check g(r) data against targets
        self.assertAlmostEqual(
            stog.r_master[stog.gr_title][self.real_space_first],
            self.real_xtarget,
            places=places)
        self.assertAlmostEqual(
            stog.gr_master[stog.gr_title][self.real_space_first],
            self.gofr_target[0],
            places=places)

        # Test if no dr defined
        stog.dr = None
        stog.transform_merged()

        # Check g(r) data against targets
        self.assertAlmostEqual(
            stog.r_master[stog.gr_title][self.real_space_first],
            self.real_xtarget,
            places=places)
        self.assertAlmostEqual(
            stog.gr_master[stog.gr_title][self.real_space_first],
            self.gofr_target[0],
            places=places)

    def test_stog_transform_merged_GofR(self):
        # Number of decimal places for precision
        places = 2

        # Load S(Q) for Argon from test data
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.real_space_function = "G(r)"
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()

        # Check G(r) data against targets
        self.assertAlmostEqual(
            stog.r_master[stog.gr_title][self.real_space_first],
            self.real_xtarget,
            places=places)
        self.assertAlmostEqual(
            stog.gr_master[stog.gr_title][self.real_space_first],
            self.GofR_target[0],
            places=places)

    def test_stog_transform_merged_GKofR(self):
        # Number of decimal places for precision
        places = 2

        # Load S(Q) for Argon from test data
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.real_space_function = "GK(r)"
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()

        # Check GK(r) data against targets
        self.assertAlmostEqual(
            stog.r_master[stog.gr_title][self.real_space_first],
            self.real_xtarget,
            places=places)
        self.assertAlmostEqual(
            stog.gr_master[stog.gr_title][self.real_space_first],
            self.GKofR_target[0],
            places=places)

    def test_stog_transform_merged_for_nan_after_filter(self):
        # Load S(Q) for Argon from test data
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.real_space_function = "GK(r)"
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()

        self.assertFalse(
            np.isnan(stog.sq_master[stog.sq_title]).any())
        self.assertFalse(
            np.isnan(stog.gr_master[stog.gr_title]).any())

    def test_stog_fourier_filter(self):
        # Number of decimal places for precision
        places = 1

        # Load S(Q) for Argon from test data
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()
        stog.fourier_filter()

        # Check g(r) data against targets
        self.assertAlmostEqual(
            stog.r_master[stog.gr_ft_title][self.real_space_first],
            self.real_xtarget,
            places=places)
        self.assertAlmostEqual(
            stog.gr_master[stog.gr_ft_title][self.real_space_first],
            self.gofr_ff_target[0],
            places=places)

    def test_stog_fourier_filter_before_transform_merged_call(self):
        # Number of decimal places for precision
        places = 1

        # Load S(Q) for Argon from test data
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()
        stog.merge_data()
        stog.fourier_filter()

        # Check g(r) data against targets
        self.assertAlmostEqual(
            stog.r_master[stog.gr_ft_title][self.real_space_first],
            self.real_xtarget,
            places=places)
        self.assertAlmostEqual(
            stog.gr_master[stog.gr_ft_title][self.real_space_first],
            self.gofr_ff_target[0],
            places=places)

    def test_stog_fourier_filter_GofR(self):
        # Number of decimal places for precision
        places = 1

        # Load S(Q) for Argon from test data
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.real_space_function = "G(r)"
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()
        stog.fourier_filter()

        # Check g(r) data against targets
        self.assertAlmostEqual(
            stog.r_master[stog.gr_ft_title][self.real_space_first],
            self.real_xtarget,
            places=places)
        self.assertAlmostEqual(
            stog.gr_master[stog.gr_ft_title][self.real_space_first],
            self.GofR_ff_target[0],
            places=places)

    def test_stog_fourier_filter_GKofR(self):
        # Number of decimal places for precision
        places = 1

        # Load S(Q) for Argon from test data
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.real_space_function = "GK(r)"
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()
        stog.fourier_filter()

        # Check g(r) data against targets
        self.assertAlmostEqual(
            stog.r_master[stog.gr_ft_title][self.real_space_first],
            self.real_xtarget,
            places=places)
        self.assertAlmostEqual(
            stog.gr_master[stog.gr_ft_title][self.real_space_first],
            self.GKofR_ff_target[0],
            places=places)

    def test_stog_fourier_filter_for_nan_after_filter(self):
        # Load S(Q) for Argon from test data
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()
        stog.fourier_filter()

        self.assertFalse(
            np.isnan(stog.gr_master[stog.gr_title]).any())
        self.assertFalse(
            np.isnan(stog.sq_master[stog.sq_title]).any())
        self.assertFalse(
            np.isnan(stog.sq_master[stog._ft_title]).any())
        self.assertFalse(
            np.isnan(stog.sq_master[stog.sq_ft_title]).any())

    def test_stog_apply_lorch_default(self):
        # Number of decimal places for precision
        places = 5

        # Load S(Q) for Argon from test data
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()
        q, sq, r, gr = stog.fourier_filter()
        stog.apply_lorch(q, sq, r)

        self.assertAlmostEqual(
            stog.gr_master[stog.gr_lorch_title][self.real_space_first],
            self.gofr_lorch_target[0],
            places=places)

    def test_stog_apply_lorch_GofR(self):
        # Number of decimal places for precision
        places = 5

        # Load S(Q) for Argon from test data
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.real_space_function = "G(r)"
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()
        q, sq, r, gr = stog.fourier_filter()
        stog.apply_lorch(q, sq, r)

        self.assertAlmostEqual(
            stog.gr_master[stog.gr_lorch_title][self.real_space_first],
            self.GofR_lorch_target[0],
            places=places)

    def test_stog_apply_lorch_GKofR(self):
        # Number of decimal places for precision
        places = 5

        # Load S(Q) for Argon from test data
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.real_space_function = "GK(r)"
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()
        q, sq, r, gr = stog.fourier_filter()
        stog.apply_lorch(q, sq, r)

        self.assertAlmostEqual(
            stog.gr_master[stog.gr_lorch_title][self.real_space_first],
            self.GKofR_lorch_target[0], places=places)

    def test_stog_lowR_mean_square(self):
        # Load S(Q) for Argon from test data
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()
        gr = stog.gr_master[stog.gr_title]
        cost = stog._lowR_mean_square(stog.dr, gr)
        self.assertAlmostEqual(cost, self.lowR_target, places=7)

    def test_stog_get_lowR_mean_square(self):
        # Load S(Q) for Argon from test data
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()
        q, sq, r, gr = stog.fourier_filter()
        cost = stog._get_lowR_mean_square()
        self.assertAlmostEqual(cost, self.lowR_target, places=7)


class TestStogBookkeepingMethods(TestStogDatasetSpecificMethods):
    def setUp(self):
        super(TestStogBookkeepingMethods, self).setUp()
        stog = StoG(**self.kwargs_for_stog_input)
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()

        self.stog = stog

    def test_stog_add_keen_fq(self):
        stog = self.stog
        q = stog.sq_master[stog.sq_title]
        sq = stog.sq_master[stog.sq_title]
        stog._add_keen_fq(q, sq)
        self.assertTrue(stog.fq_title in stog.sq_master)

    def test_stog_add_keen_gr_default(self):
        stog = self.stog
        r = stog.gr_master[stog.gr_title]
        gr = stog.gr_master[stog.gr_title]
        stog._add_keen_gr(r, gr)
        self.assertTrue(stog.GKofR_title in stog.gr_master)

    def test_stog_add_keen_gr_GofR(self):
        stog = StoG(**self.kwargs_for_stog_input)
        stog.real_space_function = "G(r)"
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()
        r = stog.gr_master[stog.gr_title]
        gr = stog.gr_master[stog.gr_title]
        stog._add_keen_gr(r, gr)
        self.assertTrue(stog.GKofR_title in stog.gr_master)

    def test_stog_add_keen_gr_GKofR(self):
        stog = StoG(**self.kwargs_for_stog_input)
        stog.real_space_function = "GK(r)"
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()
        r = stog.gr_master[stog.gr_title]
        gr = stog.gr_master[stog.gr_title]
        stog._add_keen_gr(r, gr)
        self.assertTrue(stog.GKofR_title in stog.gr_master)


class TestStogOutputMethods(TestStogDatasetSpecificMethods):
    def setUp(self):
        super(TestStogOutputMethods, self).setUp()
        stog = StoG(**self.kwargs_for_stog_input)
        stog.stem_name = "dog"
        stog.files = self.kwargs_for_files['Files']
        stog.read_all_data()
        stog.merge_data()
        stog.transform_merged()

        self.stog = stog

    # Decorator to provide the data to run each write_out_<type> test
    def data_provider(self, x, y, filename):
        def write_out_decorator(write_out_func):
            def wrap_function(*args):
                # Using stem name
                write_out_func()

                data = {}
                data['x'], data['y'] = np.genfromtxt(
                    filename,
                    skip_header=2,
                    unpack=True)

                np.testing.assert_allclose(data['x'], x)
                np.testing.assert_allclose(
                    data['y'] - y,
                    np.zeros(
                        len(y)),
                    rtol=2.0,
                    atol=2.0,
                    equal_nan=True)

                os.remove(filename)

                # Using set filename
                tmp_filename = tempfile.mkstemp()[1]

                write_out_func(filename=tmp_filename)

                data['x'], data['y'] = np.genfromtxt(
                    tmp_filename,
                    skip_header=2,
                    unpack=True)

                np.testing.assert_allclose(data['x'], x)
                np.testing.assert_allclose(data['y'], y)
                os.remove(tmp_filename)

            return wrap_function
        return write_out_decorator

    # Tests
    def test_stog_write_df(self):
        outfile_path = tempfile.mkstemp()[1]
        x = self.stog.q_master[self.stog.sq_title]
        y = self.stog.sq_master[self.stog.sq_title]
        self.stog._write_out_to_file(x, y, outfile_path)
        data = {}
        data['x'], data['y'] = np.genfromtxt(
            outfile_path,
            skip_header=2,
            unpack=True)

        q = self.stog.q_master[self.stog.sq_title]
        sq = self.stog.sq_master[self.stog.sq_title]

        np.testing.assert_allclose(data['x'], q)
        np.testing.assert_allclose(data['y'], sq)
        os.remove(outfile_path)

    def test_write_out_merged_sq(self):
        # Have to decorate after the setUp() is called for the self.* args to
        # work
        @self.data_provider(
            self.stog.q_master[self.stog.sq_title],
            self.stog.sq_master[self.stog.sq_title],
            "dog.sq")
        def decorated_write_out_merged(*args, **kwargs):
            self.stog.write_out_merged_sq(*args, **kwargs)
        decorated_write_out_merged()

    def test_write_out_merged_gr(self):
        # Have to decorate after the setUp() is called for the self.* args to
        # work
        @self.data_provider(
            self.stog.r_master[self.stog.gr_title],
            self.stog.gr_master[self.stog.gr_title],
            "dog.gr")
        def decorated_write_out_merged(*args, **kwargs):
            self.stog.write_out_merged_gr(*args, **kwargs)
        decorated_write_out_merged()

    def test_write_out_ft_sq(self):
        self.stog.fourier_filter()
        # Have to decorate after the setUp() is called for the self.* args to
        # work

        @self.data_provider(
            self.stog.q_master[self.stog.sq_ft_title],
            self.stog.sq_master[self.stog.sq_ft_title],
            'dog_ft.sq')
        def decorated_write_out_merged(*args, **kwargs):
            self.stog.write_out_ft_sq(*args, **kwargs)
        decorated_write_out_merged()


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
