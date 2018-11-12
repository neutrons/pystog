import unittest
import numpy
from tests.utils import load_nickel_gofr
from pystog.converter import Converter

def plot(x,y1,y2):
    import matplotlib.pyplot as plt
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.show()

class TestConverter(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.converter = Converter()

        # setup nickel g(r) input data
        self.r, self.gr = load_nickel_gofr()
        self.kwargs = { "rho" : 0.0913841384754395,
                        "<b_coh>^2" : 106.09,
                        "<b_tot^2>" : 147.22}

        # setup the tolerance
        self.first = 45
        self.last = 53
        self.rtol = 1e-5 
        self.atol = 1e-8

        # targets for 1st peaks
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

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    # g(r) tests
    def test_g_to_G(self):
        GofR = self.converter.g_to_G(self.r, self.gr, **self.kwargs)
        self.assertTrue(numpy.allclose(GofR[self.first:self.last], 
                                       self.GofR_target, 
                                       rtol=self.rtol, atol=self.atol))

    def test_g_to_GK(self):
        GKofR = self.converter.g_to_GK(self.r, self.gr, **self.kwargs)
        self.assertTrue(numpy.allclose(GKofR[self.first:self.last], 
                                       self.GKofR_target, 
                                       rtol=self.rtol, atol=self.atol))

    # G(r) tests
    def test_G_to_g(self):
        GofR = self.converter.g_to_G(self.r, self.gr, **self.kwargs)
        gofr  = self.converter.G_to_g(self.r, GofR, **self.kwargs)
        self.assertTrue(numpy.allclose(gofr[self.first:self.last], 
                                       self.gr[self.first:self.last], 
                                       rtol=self.rtol, atol=self.atol))
    
    def test_G_to_GK(self):
        GofR = self.converter.g_to_G(self.r, self.gr, **self.kwargs)
        GKofR  = self.converter.G_to_GK(self.r, GofR, **self.kwargs)
        self.assertTrue(numpy.allclose(GKofR[self.first:self.last], 
                                       self.GKofR_target, 
                                       rtol=self.rtol, atol=self.atol))
    

    # GK(r) tests
    def test_GK_to_g(self):
        GKofR = self.converter.g_to_GK(self.r, self.gr, **self.kwargs)
        gofr  = self.converter.GK_to_g(self.r, GKofR, **self.kwargs)
        self.assertTrue(numpy.allclose(gofr[self.first:self.last], 
                                       self.gr[self.first:self.last], 
                                       rtol=self.rtol, atol=self.atol))
    def test_GK_to_G(self):
        GKofR = self.converter.g_to_GK(self.r, self.gr, **self.kwargs)
        GofR  = self.converter.GK_to_G(self.r, GKofR, **self.kwargs)
        self.assertTrue(numpy.allclose(GofR[self.first:self.last], 
                                       self.GofR_target, 
                                       rtol=self.rtol, atol=self.atol))
        
        
        
        
