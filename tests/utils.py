from __future__ import print_function
import os
import numpy as np

from pystog.converter import Converter
from pystog.transformer import Transformer

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "../data/test_data")
REAL_HEADERS = ["r", "g(r)", "G(r)", "GK(r)"]
RECIPROCAL_HEADERS = ["Q", "S(Q)", "F(Q)", "FK(Q)", "DCS(Q)"]

nickel_kwargs = { "rho" : 0.0913841384754395, "<b_coh>^2" : 106.09, "<b_tot^2>" : 147.22}
argon_kwargs  = { "rho" : 0.02138, "<b_coh>^2" : 3.644, "<b_tot^2>" : 5.435}

#------------------------------------------------
# General utility functions

def multiplied_template(l):
    return ('%f ' * len(l))[:-1] % tuple(l)

def print_test_header(test,dashes=50):
    dashes = "-"*dashes
    print("#%s#" % dashes)
    print("# %s" % test)
    print("#%s#" % dashes)

#------------------------------------------------
# Test data utility functions
def load_lammps_rdf(filename):
    test_file_path = os.path.join(TEST_DATA_DIR, filename)
    i, r, gr, nr = np.loadtxt(test_file_path,unpack=True, skiprows=5)
    return r, gr

def load_test_data(filename):
    test_file_path = os.path.join(TEST_DATA_DIR, filename)
    data = np.loadtxt(test_file_path, skiprows=2)
    return data 

def get_index_of_function(func_string, headers):
    for i, func_type in enumerate(headers):
        if func_type == func_string:
            return i
    return None
    

def write_functions_to_file(headers, data, output_filename):
    header_string = ' '.join(headers)
    outfile_path = os.path.join(TEST_DATA_DIR, output_filename)
    with open(outfile_path,"w") as f:
        f.write("%d\n" % len(data))
        f.write("# %s\n" % header_string)
        for row in data:
            f.write("%s\n" % multiplied_template(row))

#------------------------------------------------
# Real-space test data utility functions
def create_real_space_functions(gr_filename,**kwargs):
    r, gr = load_lammps_rdf(gr_filename)
    c = Converter()
    GofR = c.g_to_G(r,gr,**kwargs)
    GKofR = c.g_to_GK(r,gr,**kwargs)
    data    = [ r, gr, GofR, GKofR]
    assert len(REAL_HEADERS) == len(data)
    data = np.transpose(data) # puts the data where each column is a function 
    return REAL_HEADERS, data

def create_and_write_real_space_functions(gr_filename, output_filename, **kwargs):
    headers, data = create_real_space_functions(gr_filename, **kwargs)
    write_functions_to_file(headers, data, output_filename)
 
#------------------------------------------------
# Real-space test data utility functions
def create_reciprocal_space_functions(gr_filename,dq=0.02,qmin=0.00,qmax=35.0,**kwargs):
    r, gr = load_lammps_rdf(gr_filename)
    q = np.arange(qmin, qmax+dq, dq)
    t = Transformer()
    q, sq = t.g_to_S(r, gr, q, **kwargs)
    q, fq = t.g_to_F(r, gr, q, **kwargs)
    q, fq_keen = t.g_to_FK(r, gr, q, **kwargs)
    q, dcs = t.g_to_DCS(r, gr, q, **kwargs)
    data    = [ q, sq, fq, fq_keen, dcs]
    assert len(RECIPROCAL_HEADERS) == len(data)
    data = np.transpose(data) # puts the data where each column is a function 
    return RECIPROCAL_HEADERS, data

def create_and_write_reciprocal_space_functions(gr_filename, output_filename, **kwargs):
    headers, data = create_reciprocal_space_functions(gr_filename, **kwargs)
    write_functions_to_file(headers, data, output_filename)

#------------------------------------------------
# Real and Reciprocal test data utility function
def create_and_write_both_type_of_functions(gr_filename,real_out,reciprocal_out,**kwargs):
    headers, data = create_real_space_functions(gr_filename, **kwargs)
    write_functions_to_file(headers, data, real_out)
    headers, data = create_reciprocal_space_functions(gr_filename, **kwargs)
    write_functions_to_file(headers, data, reciprocal_out)
    



  


