
import numpy as np
from tests.utils import create_and_write_both_type_of_functions

def create_nickel():
    kwargs = { "rho" : 0.0913841384754395, "<b_coh>^2" : 106.09, "<b_tot^2>" : 147.22}
    create_and_write_both_type_of_functions("nickel.gr",
                                            "nickel.real_space.dat",
                                            "nickel.reciprocal_space.dat",**kwargs)

def create_argon():
    kwargs = { "rho" : 0.02138, "<b_coh>^2" : 3.644, "<b_tot^2>" : 5.435}
    create_and_write_both_type_of_functions("argon.gr",
                                            "argon.real_space.dat",
                                            "argon.reciprocal_space.dat",**kwargs)


if __name__ == "__main__":
    create_nickel()
    create_argon()
