import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from tests.utils import \
    load_data, get_index_of_function
from tests.materials import Nickel, Argon
from pystog.utils import \
    RealSpaceHeaders, ReciprocalSpaceHeaders
from pystog.converter import Converter


class TestConverterUtilities(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.converter = Converter()

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    def test_safe_divide(self):
        assert_array_equal(
            self.converter._safe_divide(np.arange(10), np.arange(10)),
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1])


# Real Space Function


class TestConverterRealSpaceBase(unittest.TestCase):
    rtol = 1e-5
    atol = 1e-8

    def initialize_material(self):
        # setup input data
        self.kwargs = self.material.kwargs

        # setup the tolerance
        self.first = self.material.real_space_first
        self.last = self.material.real_space_last

        data = load_data(self.material.real_space_filename)
        self.r = data[:, get_index_of_function("r", RealSpaceHeaders)]
        self.gofr = data[:, get_index_of_function("g(r)", RealSpaceHeaders)]
        self.GofR = data[:, get_index_of_function("G(r)", RealSpaceHeaders)]
        self.GKofR = data[:, get_index_of_function("GK(r)", RealSpaceHeaders)]

        # targets for 1st peaks
        self.gofr_target = self.material.gofr_target
        self.GofR_target = self.material.GofR_target
        self.GKofR_target = self.material.GKofR_target

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.converter = Converter()

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    # g(r) tests
    def g_to_G(self):
        GofR, dGofR = self.converter.g_to_G(self.r, self.gofr,
                                            np.ones_like(self.r),
                                            **self.kwargs)

        FourPiRhoR = 4 * np.pi * self.kwargs['rho'] * self.r

        assert_allclose(
            GofR[self.first:self.last],
            self.GofR_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(dGofR, np.ones_like(self.r) * FourPiRhoR)

    def g_to_GK(self):
        GKofR, dGKofR = self.converter.g_to_GK(self.r, self.gofr,
                                               np.ones_like(self.r),
                                               **self.kwargs)
        assert_allclose(
            GKofR[self.first:self.last],
            self.GKofR_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(dGKofR,
                        np.ones_like(self.r) * self.kwargs['<b_coh>^2'])

    # G(r) tests
    def G_to_g(self):
        gofr, dgofr = self.converter.G_to_g(self.r, self.GofR,
                                            np.ones_like(self.r),
                                            **self.kwargs)

        FourPiRhoR = 4 * np.pi * self.kwargs['rho'] * self.r

        assert_allclose(
            gofr[self.first:self.last],
            self.gofr_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(dgofr, np.ones_like(self.r) / FourPiRhoR)

    def G_to_GK(self):
        GKofR, dGKofR = self.converter.G_to_GK(self.r, self.GofR,
                                               np.ones_like(self.r),
                                               **self.kwargs)
        FourPiRhoR = 4 * np.pi * self.kwargs['rho'] * self.r
        bcoh_sqrd = self.kwargs['<b_coh>^2']

        assert_allclose(
            GKofR[self.first:self.last],
            self.GKofR_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(dGKofR,
                        np.ones_like(self.r) * bcoh_sqrd / FourPiRhoR)

    # GK(r) tests
    def GK_to_g(self):
        gofr, dgofr = self.converter.GK_to_g(self.r, self.GKofR,
                                             np.ones_like(self.r),
                                             **self.kwargs)
        assert_allclose(
            gofr[self.first:self.last],
            self.gofr_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(dgofr, np.ones_like(self.r) / self.kwargs['<b_coh>^2'])

    def GK_to_G(self):
        GofR, dGofR = self.converter.GK_to_G(self.r, self.GKofR,
                                             np.ones_like(self.r),
                                             **self.kwargs)

        FourPiRhoR = 4 * np.pi * self.kwargs['rho'] * self.r
        bcoh_sqrd = self.kwargs['<b_coh>^2']

        assert_allclose(
            GofR[self.first:self.last],
            self.GofR_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(dGofR,
                        np.ones_like(self.r) / bcoh_sqrd * FourPiRhoR)


class TestConverterRealSpaceNickel(TestConverterRealSpaceBase):
    def setUp(self):
        super(TestConverterRealSpaceNickel, self).setUp()
        self.material = Nickel()
        self.initialize_material()

    def test_g_to_G(self):
        self.g_to_G()

    def test_g_to_GK(self):
        self.g_to_GK()

    def test_G_to_g(self):
        self.G_to_g()

    def test_G_to_GK(self):
        self.G_to_GK()

    def test_GK_to_g(self):
        self.GK_to_g()

    def test_GK_to_G(self):
        self.GK_to_G()


class TestConverterRealSpaceArgon(TestConverterRealSpaceBase):
    def setUp(self):
        super(TestConverterRealSpaceArgon, self).setUp()
        self.material = Argon()
        self.initialize_material()

    def test_g_to_G(self):
        self.g_to_G()

    def test_g_to_GK(self):
        self.g_to_GK()

    def test_G_to_g(self):
        self.G_to_g()

    def test_G_to_GK(self):
        self.G_to_GK()

    def test_GK_to_g(self):
        self.GK_to_g()

    def test_GK_to_G(self):
        self.GK_to_G()


# Reciprocal Space Function


class TestConverterReciprocalSpaceBase(unittest.TestCase):
    rtol = 1e-5
    atol = 1e-8

    def initialize_material(self):
        # setup input data
        self.kwargs = self.material.kwargs

        # setup the tolerance
        self.first = self.material.reciprocal_space_first
        self.last = self.material.reciprocal_space_last

        data = load_data(self.material.reciprocal_space_filename)
        self.q = data[:, get_index_of_function("Q", ReciprocalSpaceHeaders)]
        self.sq = data[:,
                       get_index_of_function("S(Q)", ReciprocalSpaceHeaders)]
        self.fq = data[:,
                       get_index_of_function("Q[S(Q)-1]",
                                             ReciprocalSpaceHeaders)]
        self.fq_keen = data[:,
                            get_index_of_function("FK(Q)",
                                                  ReciprocalSpaceHeaders)]
        self.dcs = data[:,
                        get_index_of_function("DCS(Q)",
                                              ReciprocalSpaceHeaders)]

        # targets for 1st peaks
        self.sq_target = self.material.sq_target
        self.fq_target = self.material.fq_target
        self.fq_keen_target = self.material.fq_keen_target
        self.dcs_target = self.material.dcs_target

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.converter = Converter()

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    # S(Q) tests
    def S_to_F(self):
        fq, dfq = self.converter.S_to_F(self.q, self.sq, np.ones_like(self.q),
                                        **self.kwargs)
        assert_allclose(
            fq[self.first:self.last],
            self.fq_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(dfq, np.ones_like(self.q) * self.q)

    def S_to_FK(self):
        fq_keen, dfq_keen = self.converter.S_to_FK(self.q, self.sq,
                                                   np.ones_like(self.q),
                                                   **self.kwargs)
        assert_allclose(
            fq_keen[self.first:self.last],
            self.fq_keen_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(dfq_keen,
                        np.ones_like(self.q) * self.kwargs['<b_coh>^2'])

    def S_to_DCS(self):
        dcs, ddcs = self.converter.S_to_DCS(self.q, self.sq,
                                            np.ones_like(self.q),
                                            **self.kwargs)
        assert_allclose(
            dcs[self.first:self.last],
            self.dcs_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(ddcs, np.ones_like(self.q) * self.kwargs['<b_coh>^2'])

    # Q[S(Q)-1] tests

    def F_to_S(self):
        sq, dsq = self.converter.F_to_S(self.q, self.fq, np.ones_like(self.q),
                                        **self.kwargs)
        assert_allclose(
            sq[self.first:self.last],
            self.sq_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(dsq, np.ones_like(self.q) / self.q)

    def F_to_S_with_no_dfq(self):
        sq, dsq = self.converter.F_to_S(self.q, self.fq, **self.kwargs)
        assert_allclose(
            sq[self.first:self.last],
            self.sq_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(dsq, np.zeros_like(dsq))

    def F_to_FK(self):
        fq_keen, dfq_keen = self.converter.F_to_FK(
            self.q, self.fq, np.ones_like(self.q), **self.kwargs)
        assert_allclose(
            fq_keen[self.first:self.last],
            self.fq_keen_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(
            dfq_keen,
            np.ones_like(self.q) * self.kwargs['<b_coh>^2'] / self.q)

    def F_to_FK_with_no_dfq(self):
        fq_keen, dfq_keen = self.converter.F_to_FK(
            self.q, self.fq, **self.kwargs)
        assert_allclose(
            fq_keen[self.first:self.last],
            self.fq_keen_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(dfq_keen, np.zeros_like(dfq_keen))

    def F_to_DCS(self):
        dcs, ddcs = self.converter.F_to_DCS(self.q, self.fq,
                                            np.ones_like(self.q),
                                            **self.kwargs)
        assert_allclose(
            dcs[self.first:self.last],
            self.dcs_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(
            ddcs,
            np.ones_like(self.q) * self.kwargs['<b_coh>^2'] / self.q)

    # FK(Q) tests

    def FK_to_S(self):
        sq, dsq = self.converter.FK_to_S(self.q, self.fq_keen,
                                         np.ones_like(self.q), **self.kwargs)
        assert_allclose(
            sq[self.first:self.last],
            self.sq_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(dsq, np.ones_like(self.q) / self.kwargs['<b_coh>^2'])

    def FK_to_F(self):
        fq, dfq = self.converter.FK_to_F(self.q, self.fq_keen,
                                         np.ones_like(self.q), **self.kwargs)
        assert_allclose(
            fq[self.first:self.last],
            self.fq_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(
            dfq,
            np.ones_like(self.q) / self.kwargs['<b_coh>^2'] * self.q)

    def FK_to_DCS(self):
        dcs, ddcs = self.converter.FK_to_DCS(self.q, self.fq_keen,
                                             np.ones_like(self.q),
                                             **self.kwargs)
        assert_allclose(
            dcs[self.first:self.last],
            self.dcs_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(ddcs, np.ones_like(self.q))

    # DCS(Q) tests

    def DCS_to_S(self):
        sq, dsq = self.converter.DCS_to_S(self.q, self.dcs, np.ones_like(
            self.q), **self.kwargs)
        assert_allclose(
            sq[self.first:self.last],
            self.sq_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(dsq, np.ones_like(self.q) / self.kwargs['<b_coh>^2'])

    def DCS_to_F(self):
        fq, dfq = self.converter.DCS_to_F(self.q, self.dcs, np.ones_like(
            self.q), **self.kwargs)
        assert_allclose(
            fq[self.first:self.last],
            self.fq_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(
            dfq,
            np.ones_like(self.q) / self.kwargs['<b_coh>^2'] * self.q)

    def DCS_to_FK(self):
        fq_keen, dfq_keen = self.converter.DCS_to_FK(self.q, self.dcs,
                                                     np.ones_like(self.q),
                                                     **self.kwargs)
        assert_allclose(
            fq_keen[self.first:self.last],
            self.fq_keen_target,
            rtol=self.rtol,
            atol=self.atol)
        assert_allclose(dfq_keen, np.ones_like(self.q))


class TestConverterReciprocalSpaceNickel(TestConverterReciprocalSpaceBase):
    def setUp(self):
        super(TestConverterReciprocalSpaceNickel, self).setUp()
        self.material = Nickel()
        self.initialize_material()

    def test_S_to_F(self):
        self.S_to_F()

    def test_S_to_FK(self):
        self.S_to_FK()

    def test_S_to_DCS(self):
        self.S_to_DCS()

    def test_F_to_S(self):
        self.F_to_S()

    def test_F_to_S_with_no_dfq(self):
        self.F_to_S_with_no_dfq()

    def test_F_to_FK(self):
        self.F_to_FK()

    def test_F_to_FK_with_no_dfq(self):
        self.F_to_FK_with_no_dfq()

    def test_F_to_DCS(self):
        self.F_to_DCS()

    def test_FK_to_S(self):
        self.FK_to_S()

    def test_FK_to_F(self):
        self.FK_to_F()

    def test_FK_to_DCS(self):
        self.FK_to_DCS()

    def test_DCS_to_S(self):
        self.DCS_to_S()

    def test_DCS_to_F(self):
        self.DCS_to_F()

    def test_DCS_to_FK(self):
        self.DCS_to_FK()


class TestConverterReciprocalSpaceArgon(TestConverterReciprocalSpaceBase):
    def setUp(self):
        super(TestConverterReciprocalSpaceArgon, self).setUp()
        self.material = Argon()
        self.initialize_material()

    def test_S_to_F(self):
        self.S_to_F()

    def test_S_to_FK(self):
        self.S_to_FK()

    def test_S_to_DCS(self):
        self.S_to_DCS()

    def test_F_to_S(self):
        self.F_to_S()

    def test_F_to_S_with_no_dfq(self):
        self.F_to_S_with_no_dfq()

    def test_F_to_FK(self):
        self.F_to_FK()

    def test_F_to_FK_with_no_dfq(self):
        self.F_to_FK_with_no_dfq()

    def test_F_to_DCS(self):
        self.F_to_DCS()

    def test_FK_to_S(self):
        self.FK_to_S()

    def test_FK_to_F(self):
        self.FK_to_F()

    def test_FK_to_DCS(self):
        self.FK_to_DCS()

    def test_DCS_to_S(self):
        self.DCS_to_S()

    def test_DCS_to_F(self):
        self.DCS_to_F()

    def test_DCS_to_FK(self):
        self.DCS_to_FK()


if __name__ == '__main__':
    unittest.main()  # pragma: no cover
