import os
import numpy as np

from pystog.utils import RealSpaceHeaders, ReciprocalSpaceHeaders
from pystog.converter import Converter
from pystog.transformer import Transformer

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "test_data")
if not os.path.exists(TEST_DATA_DIR):
    raise RuntimeError('Failed to find "test_data" directory')
TUTORIAL_DATA_DIR = os.path.join(TEST_DIR, "../tutorials/data")

# ------------------------------------------------
# General utility functions


def multiplied_template(num):
    return ('%f ' * len(num))[:-1] % tuple(num)

# Test data utility functions


def load_lammps_rdf(filename):
    test_file_path = os.path.join(TEST_DATA_DIR, filename)
    i, r, gr, nr = np.loadtxt(test_file_path, unpack=True, skiprows=5)
    return r, gr


def get_data_path(filename):
    return os.path.join(TEST_DATA_DIR, filename)


def load_data(filename, skiprows=2):
    test_file_path = get_data_path(filename)
    data = np.loadtxt(test_file_path, skiprows=skiprows)
    return data


def get_index_of_function(func_string, headers):
    for i, func_type in enumerate(headers):
        if func_type == func_string:
            return i
    return None


def write_functions_to_file(headers, data, output_filename, datadir='.'):
    header_string = ' '.join(headers)
    outfile_path = os.path.join(datadir, output_filename)
    with open(outfile_path, "w") as f:
        f.write("%d\n" % len(data))
        f.write("# %s\n" % header_string)
        for row in data:
            f.write("%s\n" % multiplied_template(row))

# ------------------------------------------------
# Real-space test data utility functions


def create_real_space_functions(gr_filename, **kwargs):
    r, gr = load_lammps_rdf(gr_filename)
    c = Converter()
    GofR = c.g_to_G(r, gr, **kwargs)
    GKofR = c.g_to_GK(r, gr, **kwargs)
    data = [r, gr, GofR, GKofR]
    assert len(RealSpaceHeaders) == len(data)
    data = np.transpose(data)  # puts the data where each column is a function
    return RealSpaceHeaders, data


def create_and_write_real_space_functions(
        gr_filename, output_filename, **kwargs):
    headers, data = create_real_space_functions(gr_filename, **kwargs)
    write_functions_to_file(
        headers,
        data,
        output_filename,
        datadir=TEST_DATA_DIR)
    write_functions_to_file(
        headers,
        data,
        output_filename,
        datadir=TUTORIAL_DATA_DIR)

# ------------------------------------------------
# Real-space test data utility functions


def create_reciprocal_space_functions(
        gr_filename,
        dq=0.02,
        qmin=0.00,
        qmax=35.0,
        **kwargs):
    r, gr = load_lammps_rdf(gr_filename)
    q = np.arange(qmin, qmax + dq, dq)
    t = Transformer()
    q, sq = t.g_to_S(r, gr, q, **kwargs)
    q, fq = t.g_to_F(r, gr, q, **kwargs)
    q, fq_keen = t.g_to_FK(r, gr, q, **kwargs)
    q, dcs = t.g_to_DCS(r, gr, q, **kwargs)
    data = [q, sq, fq, fq_keen, dcs]
    assert len(ReciprocalSpaceHeaders) == len(data)
    data = np.transpose(data)  # puts the data where each column is a function
    return ReciprocalSpaceHeaders, data


def create_and_write_reciprocal_space_functions(
        gr_filename, output_filename, **kwargs):
    headers, data = create_reciprocal_space_functions(gr_filename, **kwargs)
    write_functions_to_file(
        headers,
        data,
        output_filename,
        datadir=TEST_DATA_DIR)
    write_functions_to_file(
        headers,
        data,
        output_filename,
        datadir=TUTORIAL_DATA_DIR)

# ------------------------------------------------
# Real and Reciprocal test data utility function


def create_and_write_both_type_of_functions(
        gr_filename, real_out, reciprocal_out, **kwargs):
    headers, data = create_real_space_functions(gr_filename, **kwargs)
    write_functions_to_file(headers, data, real_out, datadir=TEST_DATA_DIR)
    write_functions_to_file(headers, data, real_out, datadir=TUTORIAL_DATA_DIR)
    headers, data = create_reciprocal_space_functions(gr_filename, **kwargs)
    write_functions_to_file(
        headers,
        data,
        reciprocal_out,
        datadir=TEST_DATA_DIR)
    write_functions_to_file(
        headers,
        data,
        reciprocal_out,
        datadir=TUTORIAL_DATA_DIR)
