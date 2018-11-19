from tests.utils import create_and_write_both_type_of_functions
from tests.materials import Nickel, Argon


def create_nickel():
    ni = Nickel()
    create_and_write_both_type_of_functions(ni.lammps_gr_filename,
                                            ni.real_space_filename,
                                            ni.reciprocal_space_filename,
                                            **ni.kwargs)


def create_argon():
    ar = Argon()
    create_and_write_both_type_of_functions(ar.lammps_gr_filename,
                                            ar.real_space_filename,
                                            ar.reciprocal_space_filename,
                                            **ar.kwargs)


if __name__ == "__main__":
    create_nickel()
    create_argon()
