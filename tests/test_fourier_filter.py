import unittest
import numpy
from utils import \
    load_test_data, get_index_of_function, \
    REAL_HEADERS, RECIPROCAL_HEADERS
from materials import Nickel, Argon
from pystog.fourier_filter import FourierFilter

# Real Space Function


class TestFourierFilterBase(unittest.TestCase):
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

        data = load_test_data(self.material.reciprocal_space_filename)
        self.q = data[:, get_index_of_function("Q", RECIPROCAL_HEADERS)]
        self.sq = data[:, get_index_of_function("S(Q)", RECIPROCAL_HEADERS)]
        self.fq = data[:, get_index_of_function("F(Q)", RECIPROCAL_HEADERS)]
        self.fq_keen = data[:, get_index_of_function(
            "FK(Q)", RECIPROCAL_HEADERS)]
        self.dcs = data[:, get_index_of_function("DCS(Q)", RECIPROCAL_HEADERS)]

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.ff = FourierFilter()
        self.cutoff = 1.5

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    # Real space

    # g(r) tests

    def g_using_F(self):
        q_ft, fq_ft, q, fq, r, gr = self.ff.g_using_F(
            self.r, self.gofr, self.q, self.fq, self.cutoff,  **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        for i in gr[first:last]:
            print("%f" % i)
        self.assertTrue(numpy.allclose(gr[first:last],
                                       self.gofr_ff_target,
                                       rtol=self.rtol, atol=self.atol))


class TestFourierFilterNickel(TestFourierFilterBase):
    def setUp(self):
        super(TestFourierFilterNickel, self).setUp()
        self.material = Nickel()
        self.initialize_material()

    def test_g_using_F(self):
        self.g_using_F()


class TestFourierFilterArgon(TestFourierFilterBase):
    def setUp(self):
        super(TestFourierFilterArgon, self).setUp()
        self.material = Argon()
        self.initialize_material()

    def test_g_using_F(self):
        self.g_using_F()
