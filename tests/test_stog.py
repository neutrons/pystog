import unittest
import numpy
import pandas
from pystog.stog import StoG

# Real Space Function


class TestStogBase(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)

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
        self.assertEqual(stog.real_space_function, "g(r)")
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
        self.assertEqual(stog.merged_opts, {"Y": {"Offset": 0.0, "Scale": 1.0}})
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
        self.assertAlmostEqual(stog.dr[0], 0.01)
        self.assertAlmostEqual(stog.dr[-1], 50.0)

    def test_stog_dr_setter_rmax(self):
        stog = StoG()
        stog.rmax = 25.0
        self.assertAlmostEqual(stog.dr[0], 0.01)
        self.assertAlmostEqual(stog.dr[-1], 25.0)

    def test_stog_dr_setter_rdelta(self):
        stog = StoG()
        stog.rdelta = 0.5
        self.assertAlmostEqual(stog.dr[0], 0.5)
        self.assertAlmostEqual(stog.dr[-1], 50.0)

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


class TestStogDataFrames(TestStogBase):
    def setUp(self):
        super(TestStogDataFrames, self).setUp()
        self.df_target = pandas.DataFrame(numpy.random.randn(10, 2),
                                          index=numpy.arange(10),
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


class TestStogMethods(TestStogBase):
    def setUp(self):
        super(TestStogMethods, self).setUp()

    def test_stog_append_file(self):
        stog = StoG(**{'Files': ['file1.txt', 'file2.txt']})
        stog.append_file('file3.txt')
        self.assertEqual(stog.files, ['file1.txt', 'file2.txt', 'file3.txt'])

    def test_stog_extend_file_list(self):
        stog = StoG(**{'Files': ['file1.txt', 'file2.txt']})
        stog.extend_file_list(['file3.txt', 'file4.txt'])
        self.assertEqual(stog.files, ['file1.txt', 'file2.txt', 'file3.txt', 'file4.txt'])

    def test_stog_read_all_data_assertion(self):
        stog = StoG()
        with self.assertRaises(AssertionError):
            stog.read_all_data()

        stog.files = list()
        with self.assertRaises(AssertionError):
            stog.read_all_data()
