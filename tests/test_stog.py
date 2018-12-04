import unittest
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
