
import numpy as np
from tests.utils import create_and_write_both_type_of_functions, nickel_kwargs, argon_kwargs

def create_nickel():
    create_and_write_both_type_of_functions("nickel.gr",
                                            "nickel.real_space.dat",
                                            "nickel.reciprocal_space.dat",**nickel_kwargs)

def create_argon():
    create_and_write_both_type_of_functions("argon.gr",
                                            "argon.real_space.dat",
                                            "argon.reciprocal_space.dat",**argon_kwargs)


if __name__ == "__main__":
    create_nickel()
    create_argon()
