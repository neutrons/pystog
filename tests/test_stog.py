import unittest
from utils import \
    load_test_data, get_index_of_function, \
    REAL_HEADERS, RECIPROCAL_HEADERS
from materials import Nickel, Argon
from pystog.stog import StoG

# Real Space Function


class TestStogBase(unittest.TestCase):
    rtol = 0.2
    atol = 0.2

    def initialize_material(self):
        # setup input data
        self.kwargs = self.material.kwargs

        # setup the first, last indices
        self.real_space_first = self.material.real_space_first
        self.real_space_last = self.material.real_space_last

        data = load_test_data(self.material.real_space_filename)
        self.r = data[:, get_index_of_function("r", REAL_HEADERS)]
        self.gofr = data[:, get_index_of_function("g(r)", REAL_HEADERS)]
        self.GofR = data[:, get_index_of_function("G(r)", REAL_HEADERS)]
        self.GKofR = data[:, get_index_of_function("GK(r)", REAL_HEADERS)]

        # targets for 1st peaks
        self.gofr_ff_target = self.material.gofr_ff_target
        self.GofR_ff_target = self.material.GofR_ff_target
        self.GKofR_ff_target = self.material.GKofR_ff_target

        data = load_test_data(self.material.reciprocal_space_filename)
        self.q = data[:, get_index_of_function("Q", RECIPROCAL_HEADERS)]
        self.sq = data[:, get_index_of_function("S(Q)", RECIPROCAL_HEADERS)]
        self.fq = data[:, get_index_of_function("F(Q)", RECIPROCAL_HEADERS)]
        self.fq_keen = data[:, get_index_of_function(
            "FK(Q)", RECIPROCAL_HEADERS)]
        self.dcs = data[:, get_index_of_function("DCS(Q)", RECIPROCAL_HEADERS)]

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


class TestStogNickel(TestStogBase):
    def setUp(self):
        super(TestStogNickel, self).setUp()
        self.material = Nickel()
        self.initialize_material()


class TestStogArgon(TestStogBase):
    def setUp(self):
        super(TestStogArgon, self).setUp()
        self.material = Argon()
        self.initialize_material()
