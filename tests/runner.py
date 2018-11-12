import unittest

# import test modules
import converter
from utils import print_test_header

# initialize the test suite
loader = unittest.TestLoader()
suite  = unittest.TestSuite()

# add tests to the test suite
print_test_header("Converter")
suite.addTests(loader.loadTestsFromModule(converter))

# initialize a runner and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)
