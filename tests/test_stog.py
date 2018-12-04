import unittest
from pystog.stog import StoG

# Real Space Function


class TestStogBase(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.stog = StoG()

    def tearDown(self):
        unittest.TestCase.tearDown(self)


class TestStogInit(TestStogBase):
    def setUp(self):
        super(TestStogInit, self).setUp()

    def test_stog_init(self):
        self.assertEqual(self.stog.xmin, 100)
        self.assertEqual(self.stog.xmax, 0)
        self.assertEqual(self.stog.qmin, None)
        self.assertEqual(self.stog.qmax, None)
        self.assertEqual(self.stog.files, None)
        self.assertEqual(self.stog.real_space_function, "g(r)")
        self.assertEqual(self.stog.Rmax, 50.0)
        self.assertEqual(self.stog.Rdelta, 0.01)
        self.assertEqual(self.stog.density, 1.0)
        self.assertEqual(self.stog.bcoh_sqrd, 1.0)
        self.assertEqual(self.stog.btot_sqrd, 1.0)
        self.assertEqual(self.stog.low_q_correction, False)
        self.assertEqual(self.stog.lorch_flag, False)
        self.assertEqual(self.stog.fourier_filter_cutoff, None)
        self.assertEqual(self.stog.plot_flag, True)
        self.assertEqual(self.stog.plotting_kwargs, {'figsize': (16, 8),
                                                     'style': '-',
                                                     'ms': 1,
                                                     'lw': 1,
                                                     }
                         )
        self.assertEqual(self.stog.merged_opts, {"Y": {"Offset": 0.0, "Scale": 1.0}})
        self.assertEqual(self.stog.stem_name, "out")
        self.assertTrue(self.stog.df_individuals.empty)
        self.assertTrue(self.stog.df_sq_master.empty)
        self.assertTrue(self.stog.df_sq_individuals.empty)
        self.assertTrue(self.stog.df_gr_master.empty)
