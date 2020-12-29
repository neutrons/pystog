import unittest
from numpy.testing import assert_allclose
from tests.utils import \
    load_data, get_index_of_function
from tests.materials import Nickel, Argon
from pystog.utils import \
    RealSpaceHeaders, ReciprocalSpaceHeaders
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

        data = load_data(self.material.real_space_filename)
        self.r = data[:, get_index_of_function("r", RealSpaceHeaders)]
        self.gofr = data[:, get_index_of_function("g(r)", RealSpaceHeaders)]
        self.GofR = data[:, get_index_of_function("G(r)", RealSpaceHeaders)]
        self.GKofR = data[:, get_index_of_function("GK(r)", RealSpaceHeaders)]

        # targets for 1st peaks
        self.gofr_ff_target = self.material.gofr_ff_target
        self.GofR_ff_target = self.material.GofR_ff_target
        self.GKofR_ff_target = self.material.GKofR_ff_target

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

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.ff = FourierFilter()
        self.cutoff = 1.5

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    # Real space

    # g(r) tests

    def g_using_F(self):
        q_ft, fq_ft, q, fq, r, gr, _, _, _ = self.ff.g_using_F(
            self.r, self.gofr, self.q, self.fq, self.cutoff, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(gr[first:last],
                        self.gofr_ff_target,
                        rtol=self.rtol, atol=self.atol)

    def g_using_S(self):
        q_ft, sq_ft, q, sq, r, gr, _, _, _ = self.ff.g_using_S(
            self.r, self.gofr, self.q, self.sq, self.cutoff, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(gr[first:last],
                        self.gofr_ff_target,
                        rtol=self.rtol, atol=self.atol)

    def g_using_FK(self):
        q_ft, fq_ft, q, fq, r, gr, _, _, _ = self.ff.g_using_FK(
            self.r, self.gofr, self.q, self.fq_keen, self.cutoff, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(gr[first:last],
                        self.gofr_ff_target,
                        rtol=self.rtol, atol=self.atol)

    def g_using_DCS(self):
        q_ft, dcs_ft, q, dcs, r, gr, _, _, _ = self.ff.g_using_DCS(
            self.r, self.gofr, self.q, self.dcs, self.cutoff, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(gr[first:last],
                        self.gofr_ff_target,
                        rtol=self.rtol, atol=self.atol)

    # G(r) tests

    def G_using_F(self):
        q_ft, fq_ft, q, fq, r, gr, _, _, _ = self.ff.G_using_F(
            self.r, self.GofR, self.q, self.fq, self.cutoff, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(gr[first:last],
                        self.GofR_ff_target,
                        rtol=self.rtol, atol=self.atol)

    def G_using_S(self):
        q_ft, sq_ft, q, sq, r, gr, _, _, _ = self.ff.G_using_S(
            self.r, self.GofR, self.q, self.sq, self.cutoff, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(gr[first:last],
                        self.GofR_ff_target,
                        rtol=self.rtol, atol=self.atol)

    def G_using_FK(self):
        q_ft, fq_ft, q, fq, r, gr, _, _, _ = self.ff.G_using_FK(
            self.r, self.GofR, self.q, self.fq_keen, self.cutoff, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(gr[first:last],
                        self.GofR_ff_target,
                        rtol=self.rtol, atol=self.atol)

    def G_using_DCS(self):
        q_ft, dcs_ft, q, dcs, r, gr, _, _, _ = self.ff.G_using_DCS(
            self.r, self.GofR, self.q, self.dcs, self.cutoff, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(gr[first:last],
                        self.GofR_ff_target,
                        rtol=self.rtol, atol=self.atol)

    # G(r) tests

    def GK_using_F(self):
        q_ft, fq_ft, q, fq, r, gr, _, _, _ = self.ff.GK_using_F(
            self.r, self.GKofR, self.q, self.fq, self.cutoff, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(gr[first:last],
                        self.GKofR_ff_target,
                        rtol=self.rtol, atol=self.atol)

    def GK_using_S(self):
        q_ft, sq_ft, q, sq, r, gr, _, _, _ = self.ff.GK_using_S(
            self.r, self.GKofR, self.q, self.sq, self.cutoff, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(gr[first:last],
                        self.GKofR_ff_target,
                        rtol=self.rtol, atol=self.atol)

    def GK_using_FK(self):
        q_ft, fq_ft, q, fq, r, gr, _, _, _ = self.ff.GK_using_FK(
            self.r,
            self.GKofR,
            self.q,
            self.fq_keen,
            self.cutoff,
            **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(gr[first:last],
                        self.GKofR_ff_target,
                        rtol=self.rtol, atol=self.atol)

    def GK_using_DCS(self):
        q_ft, dcs_ft, q, dcs, r, gr, _, _, _ = self.ff.GK_using_DCS(
            self.r, self.GKofR, self.q, self.dcs, self.cutoff, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(gr[first:last],
                        self.GKofR_ff_target,
                        rtol=self.rtol, atol=self.atol)


class TestFourierFilterNickel(TestFourierFilterBase):
    def setUp(self):
        super(TestFourierFilterNickel, self).setUp()
        self.material = Nickel()
        self.initialize_material()

    def test_g_using_F(self):
        self.g_using_F()

    def test_g_using_S(self):
        self.g_using_S()

    def test_g_using_FK(self):
        self.g_using_FK()

    def test_g_using_DCS(self):
        self.g_using_DCS()

    def test_G_using_F(self):
        self.G_using_F()

    def test_G_using_S(self):
        self.G_using_S()

    def test_G_using_FK(self):
        self.G_using_FK()

    def test_G_using_DCS(self):
        self.G_using_DCS()

    def test_GK_using_F(self):
        self.GK_using_F()

    def test_GK_using_S(self):
        self.GK_using_S()

    def test_GK_using_FK(self):
        self.GK_using_FK()

    def test_GK_using_DCS(self):
        self.GK_using_DCS()


class TestFourierFilterArgon(TestFourierFilterBase):
    def setUp(self):
        super(TestFourierFilterArgon, self).setUp()
        self.material = Argon()
        self.initialize_material()

    def test_g_using_F(self):
        self.g_using_F()

    def test_g_using_S(self):
        self.g_using_S()

    def test_g_using_FK(self):
        self.g_using_FK()

    def test_g_using_DCS(self):
        self.g_using_DCS()

    def test_G_using_F(self):
        self.G_using_F()

    def test_G_using_S(self):
        self.G_using_S()

    def test_G_using_FK(self):
        self.G_using_FK()

    def test_G_using_DCS(self):
        self.G_using_DCS()

    def test_GK_using_F(self):
        self.GK_using_F()

    def test_GK_using_S(self):
        self.GK_using_S()

    def test_GK_using_FK(self):
        self.GK_using_FK()

    def test_GK_using_DCS(self):
        self.GK_using_DCS()


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
