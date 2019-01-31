import unittest
import numpy
from numpy.testing import assert_array_equal, assert_allclose
from tests.utils import \
    load_data, get_index_of_function
from tests.materials import Nickel, Argon
from pystog.utils import \
    RealSpaceHeaders, ReciprocalSpaceHeaders
from pystog.transformer import Transformer

# Real Space Function


class TestTransformerBase(unittest.TestCase):
    rtol = 1e-2
    atol = 1e-2

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
        self.gofr_target = self.material.gofr_target
        self.GofR_target = self.material.GofR_target
        self.GKofR_target = self.material.GKofR_target

        # setup the tolerance
        self.reciprocal_space_first = self.material.reciprocal_space_first
        self.reciprocal_space_last = self.material.reciprocal_space_last

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
        self.fq_target = self.material.fq_target
        self.fq_keen_target = self.material.fq_keen_target
        self.dcs_target = self.material.dcs_target

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.transformer = Transformer()

        # Parameters for fourier transform testing
        self._ft_first = 28
        self._ft_last = 35
        fs = 100  # sample rate
        f = 10  # the frequency of the signal
        self._ft_xin = numpy.linspace(0.0, 100., 1000)
        self._ft_yin = numpy.asarray(
            [numpy.sin(2 * numpy.pi * f * (i / fs)) for i in self._ft_xin])
        self._ft_xout = numpy.linspace(0.0, 2.0, 100)

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    # Utilities

    def test_apply_cropping(self):
        xin = numpy.linspace(0.5, 1.0, 11)
        yin = numpy.linspace(4.5, 5.0, 11)
        x, y, _ = self.transformer.apply_cropping(xin, yin, 0.6, 0.7)
        assert_array_equal(x, [0.6, 0.65, 0.7])
        assert_array_equal(y, [4.6, 4.65, 4.7])

    def test_fourier_transform(self):
        xout, yout, _ = self.transformer.fourier_transform(self._ft_xin,
                                                           self._ft_yin,
                                                           self._ft_xout)
        yout_target = [-0.14265772,
                       -10.8854444,
                       18.13582784,
                       49.72976782,
                       26.3590524,
                       -8.08540764,
                       -3.38810001]
        assert_allclose(yout[self._ft_first:self._ft_last],
                        yout_target,
                        rtol=self.rtol, atol=self.atol)

    def test_fourier_transform_with_lorch(self):
        kwargs = {"lorch": True}
        xout, yout, _ = self.transformer.fourier_transform(self._ft_xin,
                                                           self._ft_yin,
                                                           self._ft_xout,
                                                           **kwargs)
        yout_target = [-1.406162,
                       3.695632,
                       18.788041,
                       29.370677,
                       21.980533,
                       6.184271,
                       -1.234159]
        assert_allclose(yout[self._ft_first:self._ft_last],
                        yout_target,
                        rtol=self.rtol, atol=self.atol)

    def test_fourier_transform_with_low_x(self):
        kwargs = {"OmittedXrangeCorrection": True}
        xout, yout, _ = self.transformer.fourier_transform(self._ft_xin,
                                                           self._ft_yin,
                                                           self._ft_xout,
                                                           **kwargs)
        yout_target = [-0.142658,
                       -10.885444,
                       18.135828,
                       49.729768,
                       26.359052,
                       -8.085408,
                       -3.388100]
        assert_allclose(yout[self._ft_first:self._ft_last],
                        yout_target,
                        rtol=self.rtol, atol=self.atol)

    def test_low_x_correction(self):
        kwargs = {"lorch": False}
        xout, yout, _ = self.transformer.fourier_transform(self._ft_xin,
                                                           self._ft_yin,
                                                           self._ft_xout,
                                                           **kwargs)
        yout = self.transformer._low_x_correction(self._ft_xin,
                                                  self._ft_yin,
                                                  xout, yout,
                                                  **kwargs)
        yout_target = [-0.142658,
                       -10.885444,
                       18.135828,
                       49.729768,
                       26.359052,
                       -8.085408,
                       -3.388100]
        assert_allclose(yout[self._ft_first:self._ft_last],
                        yout_target,
                        rtol=self.rtol, atol=self.atol)

    def test_low_x_correction_with_lorch(self):
        kwargs = {"lorch": True}
        xout, yout, _ = self.transformer.fourier_transform(self._ft_xin,
                                                           self._ft_yin,
                                                           self._ft_xout,
                                                           **kwargs)
        yout = self.transformer._low_x_correction(self._ft_xin,
                                                  self._ft_yin,
                                                  xout, yout,
                                                  **kwargs)
        yout_target = [-1.406162,
                       3.695632,
                       18.788041,
                       29.370677,
                       21.980533,
                       6.184271,
                       -1.234159]
        assert_allclose(yout[self._ft_first:self._ft_last],
                        yout_target,
                        rtol=self.rtol, atol=self.atol)

    # Real space

    # g(r) tests

    def g_to_S(self):
        q, sq, _ = self.transformer.g_to_S(
            self.r, self.gofr, self.q, **self.kwargs)
        first, last = self.reciprocal_space_first, self.reciprocal_space_last
        assert_allclose(sq[first:last],
                        self.sq_target,
                        rtol=self.rtol, atol=self.atol)

    def g_to_F(self):
        q, fq, _ = self.transformer.g_to_F(
            self.r, self.gofr, self.q, **self.kwargs)
        first, last = self.reciprocal_space_first, self.reciprocal_space_last
        assert_allclose(fq[first:last],
                        self.fq_target,
                        rtol=self.rtol, atol=self.atol)

    def g_to_FK(self):
        q, fq_keen, _ = self.transformer.g_to_FK(
            self.r, self.gofr, self.q, **self.kwargs)
        first, last = self.reciprocal_space_first, self.reciprocal_space_last
        assert_allclose(fq_keen[first:last],
                        self.fq_keen_target,
                        rtol=self.rtol, atol=self.atol)

    def g_to_DCS(self):
        q, dcs, _ = self.transformer.g_to_DCS(
            self.r, self.gofr, self.q, **self.kwargs)
        first, last = self.reciprocal_space_first, self.reciprocal_space_last
        assert_allclose(dcs[first:last],
                        self.dcs_target,
                        rtol=self.rtol, atol=self.atol)

    # G(r) tests

    def G_to_S(self):
        q, sq, _ = self.transformer.G_to_S(
            self.r, self.GofR, self.q, **self.kwargs)
        first, last = self.reciprocal_space_first, self.reciprocal_space_last
        assert_allclose(sq[first:last],
                        self.sq_target,
                        rtol=self.rtol, atol=self.atol)

    def G_to_F(self):
        q, fq, _ = self.transformer.G_to_F(
            self.r, self.GofR, self.q, **self.kwargs)
        first, last = self.reciprocal_space_first, self.reciprocal_space_last
        assert_allclose(fq[first:last],
                        self.fq_target,
                        rtol=self.rtol, atol=self.atol)

    def G_to_FK(self):
        q, fq_keen, _ = self.transformer.G_to_FK(
            self.r, self.GofR, self.q, **self.kwargs)
        first, last = self.reciprocal_space_first, self.reciprocal_space_last
        assert_allclose(fq_keen[first:last],
                        self.fq_keen_target,
                        rtol=self.rtol, atol=self.atol)

    def G_to_DCS(self):
        q, dcs, _ = self.transformer.G_to_DCS(
            self.r, self.GofR, self.q, **self.kwargs)
        first, last = self.reciprocal_space_first, self.reciprocal_space_last
        assert_allclose(dcs[first:last],
                        self.dcs_target,
                        rtol=self.rtol, atol=self.atol)
    # GK(r) tests

    def GK_to_S(self):
        q, sq, _ = self.transformer.GK_to_S(
            self.r, self.GKofR, self.q, **self.kwargs)
        first, last = self.reciprocal_space_first, self.reciprocal_space_last
        assert_allclose(sq[first:last],
                        self.sq_target,
                        rtol=self.rtol, atol=self.atol)

    def GK_to_F(self):
        q, fq, _ = self.transformer.GK_to_F(
            self.r, self.GKofR, self.q, **self.kwargs)
        first, last = self.reciprocal_space_first, self.reciprocal_space_last
        assert_allclose(fq[first:last],
                        self.fq_target,
                        rtol=self.rtol, atol=self.atol)

    def GK_to_FK(self):
        q, fq_keen, _ = self.transformer.GK_to_FK(
            self.r, self.GKofR, self.q, **self.kwargs)
        first, last = self.reciprocal_space_first, self.reciprocal_space_last
        assert_allclose(fq_keen[first:last],
                        self.fq_keen_target,
                        rtol=self.rtol, atol=self.atol)

    def GK_to_DCS(self):
        q, dcs, _ = self.transformer.GK_to_DCS(
            self.r, self.GKofR, self.q, **self.kwargs)
        first, last = self.reciprocal_space_first, self.reciprocal_space_last
        assert_allclose(dcs[first:last],
                        self.dcs_target,
                        rtol=self.rtol, atol=self.atol)

    # Reciprocal space

    # S(Q) tests
    def S_to_g(self):
        r, gofr, _ = self.transformer.S_to_g(
            self.q, self.sq, self.r, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(gofr[first:last],
                        self.gofr_target,
                        rtol=self.rtol, atol=self.atol)

    def S_to_G(self):
        r, GofR, _ = self.transformer.S_to_G(
            self.q, self.sq, self.r, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(GofR[first:last],
                        self.GofR_target,
                        rtol=self.rtol, atol=self.atol)

    def S_to_GK(self):
        r, GKofR, _ = self.transformer.S_to_GK(
            self.q, self.sq, self.r, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(GKofR[first:last],
                        self.GKofR_target,
                        rtol=self.rtol, atol=self.atol)
    # Q[S(Q)-1] tests

    def F_to_g(self):
        r, gofr, _ = self.transformer.F_to_g(
            self.q, self.fq, self.r, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(gofr[first:last],
                        self.gofr_target,
                        rtol=self.rtol, atol=self.atol)

    def F_to_G(self):
        r, GofR, _ = self.transformer.F_to_G(
            self.q, self.fq, self.r, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(GofR[first:last],
                        self.GofR_target,
                        rtol=self.rtol, atol=self.atol)

    def F_to_GK(self):
        r, GKofR, _ = self.transformer.F_to_GK(
            self.q, self.fq, self.r, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(GKofR[first:last],
                        self.GKofR_target,
                        rtol=self.rtol, atol=self.atol)
    # FK(Q) tests

    def FK_to_g(self):
        r, gofr, _ = self.transformer.FK_to_g(
            self.q, self.fq_keen, self.r, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(gofr[first:last],
                        self.gofr_target,
                        rtol=self.rtol, atol=self.atol)

    def FK_to_G(self):
        r, GofR, _ = self.transformer.FK_to_G(
            self.q, self.fq_keen, self.r, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(GofR[first:last],
                        self.GofR_target,
                        rtol=self.rtol, atol=self.atol)

    def FK_to_GK(self):
        r, GKofR, _ = self.transformer.FK_to_GK(
            self.q, self.fq_keen, self.r, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(GKofR[first:last],
                        self.GKofR_target,
                        rtol=self.rtol, atol=self.atol)
    # DCS(Q) tests

    def DCS_to_g(self):
        r, gofr, _ = self.transformer.DCS_to_g(
            self.q, self.dcs, self.r, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(gofr[first:last],
                        self.gofr_target,
                        rtol=self.rtol, atol=self.atol)

    def DCS_to_G(self):
        r, GofR, _ = self.transformer.DCS_to_G(
            self.q, self.dcs, self.r, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(GofR[first:last],
                        self.GofR_target,
                        rtol=self.rtol, atol=self.atol)

    def DCS_to_GK(self):
        r, GKofR, _ = self.transformer.DCS_to_GK(
            self.q, self.dcs, self.r, **self.kwargs)
        first, last = self.real_space_first, self.real_space_last
        assert_allclose(GKofR[first:last],
                        self.GKofR_target,
                        rtol=self.rtol, atol=self.atol)


class TestTransformerNickel(TestTransformerBase):
    def setUp(self):
        super(TestTransformerNickel, self).setUp()
        self.material = Nickel()
        self.initialize_material()

    def test_g_to_S(self):
        self.g_to_S()

    def test_g_to_F(self):
        self.g_to_F()

    def test_g_to_FK(self):
        self.g_to_FK()

    def test_g_to_DCS(self):
        self.g_to_DCS()

    def test_G_to_S(self):
        self.G_to_S()

    def test_G_to_F(self):
        self.G_to_F()

    def test_G_to_FK(self):
        self.G_to_FK()

    def test_G_to_DCS(self):
        self.G_to_DCS()

    def test_GK_to_S(self):
        self.GK_to_S()

    def test_GK_to_F(self):
        self.GK_to_F()

    def test_GK_to_FK(self):
        self.GK_to_FK()

    def test_GK_to_DCS(self):
        self.GK_to_DCS()


class TestTransformerArgon(TestTransformerBase):
    def setUp(self):
        super(TestTransformerArgon, self).setUp()
        self.material = Argon()
        self.initialize_material()

    def test_g_to_S(self):
        self.g_to_S()

    def test_g_to_F(self):
        self.g_to_F()

    def test_g_to_FK(self):
        self.g_to_FK()

    def test_g_to_DCS(self):
        self.g_to_DCS()

    def test_G_to_S(self):
        self.G_to_S()

    def test_G_to_F(self):
        self.G_to_F()

    def test_G_to_FK(self):
        self.G_to_FK()

    def test_G_to_DCS(self):
        self.G_to_DCS()

    def test_GK_to_S(self):
        self.GK_to_S()

    def test_GK_to_F(self):
        self.GK_to_F()

    def test_GK_to_FK(self):
        self.GK_to_FK()

    def test_GK_to_DCS(self):
        self.GK_to_DCS()

    def test_S_to_g(self):
        self.S_to_g()

    def test_S_to_G(self):
        self.S_to_G()

    def test_S_to_GK(self):
        self.S_to_GK()

    def test_F_to_g(self):
        self.F_to_g()

    def test_F_to_G(self):
        self.F_to_G()

    def test_F_to_GK(self):
        self.F_to_GK()

    def test_FK_to_g(self):
        self.FK_to_g()

    def test_FK_to_G(self):
        self.FK_to_G()

    def test_FK_to_GK(self):
        self.FK_to_GK()

    def test_DCS_to_g(self):
        self.DCS_to_g()

    def test_DCS_to_G(self):
        self.DCS_to_G()

    def test_DCS_to_GK(self):
        self.DCS_to_GK()


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
