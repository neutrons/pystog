import unittest
import numpy
from tests.utils import \
    load_test_data, get_index_of_function, REAL_HEADERS, nickel_kwargs, argon_kwargs
from pystog.converter import Converter

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
        self.gofr_target    = [2.3774,
                               2.70072,
                               2.90777,
                               3.01835,
                               2.99808,
                               2.89997,
                               2.75178]
        self.GofR_target    = [1.304478,
                               1.633527,
                               1.858025,
                               1.992835,
                               1.999663,
                               1.926998,
                               1.800236]
        self.GKofR_target    = [5.019246,
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
        

 
       
