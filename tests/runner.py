import sys
import unittest

# import test modules
import test_converter
import test_transformer
import test_fourier_filter
import test_stog

# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromModule(test_converter))
suite.addTests(loader.loadTestsFromModule(test_transformer))
suite.addTests(loader.loadTestsFromModule(test_fourier_filter))
suite.addTests(loader.loadTestsFromModule(test_stog))

# initialize a runner and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite).wasSuccessful()
sys.exit(not result)  # weird "opposite" logic
