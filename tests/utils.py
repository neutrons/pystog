from __future__ import print_function
import os
import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "../data/test_data")

def load_nickel_gofr():
    test_data_path = os.path.join(TEST_DATA_DIR, "nickel.rdf")
    i, r, gr, nr = np.loadtxt(test_data_path, skiprows=5, unpack=True)
    return r, gr

def print_test_header(test,dashes=50):
    dashes = "-"*dashes
    print("#%s#" % dashes)
    print("# %s" % test)
    print("#%s#" % dashes)


