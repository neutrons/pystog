import unittest
import numpy
from tests.utils import \
    load_test_data, get_index_of_function, \
    REAL_HEADERS, RECIPROCAL_HEADERS,\
    nickel_kwargs, argon_kwargs
from pystog.converter import Converter

#-------------------------------------------------------------------------------#
# Real Space Function

class TestConverterRealSpaceBase(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.converter = Converter()

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    # g(r) tests
    def g_to_G(self):
        GofR = self.converter.g_to_G(self.r, self.gofr, **self.kwargs)
        self.assertTrue(numpy.allclose(GofR[self.first:self.last], 
                                       self.GofR_target, 
                                       rtol=self.rtol, atol=self.atol))

    def g_to_GK(self):
        GKofR = self.converter.g_to_GK(self.r, self.gofr, **self.kwargs)
        self.assertTrue(numpy.allclose(GKofR[self.first:self.last], 
                                       self.GKofR_target, 
                                       rtol=self.rtol, atol=self.atol))

    # G(r) tests
    def G_to_g(self):
        gofr  = self.converter.G_to_g(self.r, self.GofR, **self.kwargs)
        self.assertTrue(numpy.allclose(gofr[self.first:self.last], 
                                       self.gofr_target, 
                                       rtol=self.rtol, atol=self.atol))
    
    def G_to_GK(self):
        GKofR  = self.converter.G_to_GK(self.r, self.GofR, **self.kwargs)
        self.assertTrue(numpy.allclose(GKofR[self.first:self.last], 
                                       self.GKofR_target, 
                                       rtol=self.rtol, atol=self.atol))

    # GK(r) tests
    def GK_to_g(self):
        gofr  = self.converter.GK_to_g(self.r, self.GKofR, **self.kwargs)
        self.assertTrue(numpy.allclose(gofr[self.first:self.last], 
                                       self.gofr_target, 
                                       rtol=self.rtol, atol=self.atol))
    def GK_to_G(self):
        GofR  = self.converter.GK_to_G(self.r, self.GKofR, **self.kwargs)
        self.assertTrue(numpy.allclose(GofR[self.first:self.last], 
                                       self.GofR_target, 
                                       rtol=self.rtol, atol=self.atol))
        
        
class TestConverterRealSpaceNickel(TestConverterRealSpaceBase):
    def setUp(self):
        super(TestConverterRealSpaceNickel, self).setUp()

        # setup nickel g(r) input data
        self.kwargs = nickel_kwargs

        # setup the tolerance
        self.first = 45
        self.last = 53
        self.rtol = 1e-5 
        self.atol = 1e-8

        data = load_test_data("nickel.real_space.dat")
        self.r = data[:,get_index_of_function("r",REAL_HEADERS)]
        self.gofr  = data[:,get_index_of_function("g(r)",REAL_HEADERS)]
        self.GofR  = data[:,get_index_of_function("G(r)",REAL_HEADERS)]
        self.GKofR = data[:,get_index_of_function("GK(r)",REAL_HEADERS)]

        # targets for 1st peaks
        self.gofr_target    = [0.036372,
                               0.832999,
                               5.705700,
                              12.894100,
                              10.489400,
                               3.267010,
                               0.416700,
                               0.021275]
        self.GofR_target    = [-2.57284109375, 
                               -0.45547376985, 
                               13.1043856417,  
                               33.8055086359, 
                               27.5157162282,   
                                6.703650364,
                               -1.75833641369, 
                               -3.00652731657]
        self.GKofR_target    = [-102.2312733,
                                 -17.71713609,
                                 499.227713,
                                1261.845069,
                                1006.730446,
                                 240.5070909,  
                                 -61.882297,
                                -103.83293525]

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

        # setup argon g(r) input data
        self.kwargs = argon_kwargs

        # setup the tolerance
        self.first = 69
        self.last = 76
        self.rtol = 1e-5 
        self.atol = 1e-8

        data = load_test_data("argon.real_space.dat")
        self.r = data[:,get_index_of_function("r",REAL_HEADERS)]
        self.gofr  = data[:,get_index_of_function("g(r)",REAL_HEADERS)]
        self.GofR  = data[:,get_index_of_function("G(r)",REAL_HEADERS)]
        self.GKofR = data[:,get_index_of_function("GK(r)",REAL_HEADERS)]

        # targets for 1st peaks
        self.gofr_target  = [2.3774,
                             2.70072,
                             2.90777,
                             3.01835,
                             2.99808,
                             2.89997,
                             2.75178]
        self.GofR_target  = [1.304478,
                             1.633527,
                             1.858025,
                             1.992835,
                             1.999663,
                             1.926998,
                             1.800236]
        self.GKofR_target = [5.019246,
                             6.197424,
                             6.951914,
                             7.354867,
                             7.281004,
                             6.923491,
                             6.383486]

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
        
#-------------------------------------------------------------------------------#
# Reciprocal Space Function
 
class TestConverterReciprocalSpaceBase(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.converter = Converter()

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    # S(Q) tests
    def S_to_F(self):
        fq = self.converter.S_to_F(self.q, self.sq, **self.kwargs)
        self.assertTrue(numpy.allclose(fq[self.first:self.last], 
                                       self.fq_target, 
                                       rtol=self.rtol, atol=self.atol))

    def S_to_FK(self):
        fq_keen = self.converter.S_to_FK(self.q, self.sq, **self.kwargs)
        self.assertTrue(numpy.allclose(fq_keen[self.first:self.last], 
                                       self.fq_keen_target, 
                                       rtol=self.rtol, atol=self.atol))

    def S_to_DCS(self):
        dcs = self.converter.S_to_DCS(self.q, self.sq, **self.kwargs)
        self.assertTrue(numpy.allclose(dcs[self.first:self.last], 
                                       self.dcs_target, 
                                       rtol=self.rtol, atol=self.atol))
    # F(Q) tests
    def F_to_S(self):
        sq = self.converter.F_to_S(self.q, self.fq, **self.kwargs)
        self.assertTrue(numpy.allclose(sq[self.first:self.last], 
                                       self.sq_target, 
                                       rtol=self.rtol, atol=self.atol))

    def F_to_FK(self):
        fq_keen = self.converter.F_to_FK(self.q, self.fq, **self.kwargs)
        self.assertTrue(numpy.allclose(fq_keen[self.first:self.last], 
                                       self.fq_keen_target, 
                                       rtol=self.rtol, atol=self.atol))

    def F_to_DCS(self):
        dcs = self.converter.F_to_DCS(self.q, self.fq, **self.kwargs)
        self.assertTrue(numpy.allclose(dcs[self.first:self.last], 
                                       self.dcs_target, 
                                       rtol=self.rtol, atol=self.atol))
     # FK(Q) tests
    def FK_to_S(self):
        sq = self.converter.FK_to_S(self.q, self.fq_keen, **self.kwargs)
        self.assertTrue(numpy.allclose(sq[self.first:self.last], 
                                       self.sq_target, 
                                       rtol=self.rtol, atol=self.atol))

    def FK_to_F(self):
        fq = self.converter.FK_to_F(self.q, self.fq_keen, **self.kwargs)
        self.assertTrue(numpy.allclose(fq[self.first:self.last], 
                                       self.fq_target, 
                                       rtol=self.rtol, atol=self.atol))

    def FK_to_DCS(self):
        dcs = self.converter.FK_to_DCS(self.q, self.fq_keen, **self.kwargs)
        self.assertTrue(numpy.allclose(dcs[self.first:self.last], 
                                       self.dcs_target, 
                                       rtol=self.rtol, atol=self.atol))
    # DCS(Q) tests
    def DCS_to_S(self):
        sq = self.converter.DCS_to_S(self.q, self.dcs, **self.kwargs)
        self.assertTrue(numpy.allclose(sq[self.first:self.last], 
                                       self.sq_target, 
                                       rtol=self.rtol, atol=self.atol))

    def DCS_to_F(self):
        fq = self.converter.DCS_to_F(self.q, self.dcs, **self.kwargs)
        self.assertTrue(numpy.allclose(fq[self.first:self.last], 
                                       self.fq_target, 
                                       rtol=self.rtol, atol=self.atol))

    def DCS_to_FK(self):
        fq_keen = self.converter.DCS_to_FK(self.q, self.dcs, **self.kwargs)
        self.assertTrue(numpy.allclose(fq_keen[self.first:self.last], 
                                       self.fq_keen_target, 
                                       rtol=self.rtol, atol=self.atol))
       
class TestConverterReciprocalSpaceNickel(TestConverterReciprocalSpaceBase):
    def setUp(self):
        super(TestConverterReciprocalSpaceNickel, self).setUp()

        # setup nickel g(r) input data
        self.kwargs = nickel_kwargs

        # setup the tolerance
        self.first = 150
        self.last = 157
        self.rtol = 1e-5 
        self.atol = 1e-8

        data = load_test_data("nickel.reciprocal_space.dat")
        self.q       = data[:,get_index_of_function("Q",RECIPROCAL_HEADERS)]
        self.sq      = data[:,get_index_of_function("S(Q)",RECIPROCAL_HEADERS)]
        self.fq      = data[:,get_index_of_function("F(Q)",RECIPROCAL_HEADERS)]
        self.fq_keen = data[:,get_index_of_function("FK(Q)",RECIPROCAL_HEADERS)]
        self.dcs     = data[:,get_index_of_function("DCS(Q)",RECIPROCAL_HEADERS)]

        # targets for 1st peaks
        self.sq_target       = [7.07469,
                                8.704824,
                                9.847706,
                                10.384142,
                                10.265869,
                                9.519633,
                                8.240809]
        self.fq_target       = [18.345563,
                                23.422666,
                                27.07398,
                                28.903156,
                                28.724193,
                                26.581256,
                                22.73614]
        self.fq_keen_target  = [644.463844,
                                817.404819,
                                938.653124,
                                995.563576,
                                983.016011,
                                903.847917,
                                768.177419]
        self.dcs_target      = [791.683844,
                                964.624819,
                                1085.873124,
                                1142.783576,
                                1130.236011,
                                1051.067917,
                                915.397419]

    def test_S_to_F(self):
        self.S_to_F()
    def test_S_to_FK(self):
        self.S_to_FK()
    def test_S_to_DCS(self):
        self.S_to_DCS()
    def test_F_to_S(self):
        self.F_to_S()
    def test_F_to_FK(self):
        self.F_to_FK()
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

        # setup argon g(r) input data
        self.kwargs = argon_kwargs

        # setup the tolerance
        self.first = 96 
        self.last = 103
        self.rtol = 1e-5 
        self.atol = 1e-8

        data = load_test_data("argon.reciprocal_space.dat")
        self.q       = data[:,get_index_of_function("Q",RECIPROCAL_HEADERS)]
        self.sq      = data[:,get_index_of_function("S(Q)",RECIPROCAL_HEADERS)]
        self.fq      = data[:,get_index_of_function("F(Q)",RECIPROCAL_HEADERS)]
        self.fq_keen = data[:,get_index_of_function("FK(Q)",RECIPROCAL_HEADERS)]
        self.dcs     = data[:,get_index_of_function("DCS(Q)",RECIPROCAL_HEADERS)]

        # targets for 1st peaks
        self.sq_target       = [2.59173,
                                2.706695,
                                2.768409,
                                2.770228,
                                2.71334,
                                2.605211,
                                2.458852]
        self.fq_target       = [3.087955,
                                3.345121,
                                3.50145,
                                3.540457,
                                3.460946,
                                3.27463,
                                3.005236]
        self.fq_keen_target  = [5.800262,
                                6.219195,
                                6.444083,
                                6.450712,
                                6.24341,
                                5.849389,
                                5.316058]
        self.dcs_target      = [11.235262,
                                11.654195,
                                11.879083,
                                11.885712,
                                11.67841,
                                11.284389,
                                10.751058]

    def test_S_to_F(self):
        self.S_to_F()
    def test_S_to_FK(self):
        self.S_to_FK()
    def test_S_to_DCS(self):
        self.S_to_DCS()
    def test_F_to_S(self):
        self.F_to_S()
    def test_F_to_FK(self):
        self.F_to_FK()
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
