"""
==============
PyStoG CLI
==============

This module defines the PyStoG CLI function that performs the workflow of the
original FORTRAN-based StoG CLI
"""


import json

from pystog import StoG
from pystog.stog import NoInputFilesException
from pystog.io import get_cli_parser, parse_cli_args


def pystog_cli(kwargs=None):
    """
    Main entry point for PyStoG CLI tool

    :param kwargs: Dict with CLI arguments.
                   If None, parsed from command line via get_cli_parser
    :type kwargs: dict
    """
    parser = get_cli_parser()

    if not kwargs:
        args = parser.parse_args()
        if args.json:
            print("loading config from '%s'" % args.json)
            with open(args.json, 'r') as f:
                kwargs = json.load(f)

        else:
            kwargs = parse_cli_args(args)

    if not kwargs.get("Files"):
        parser.print_help()
        raise NoInputFilesException("No input files given in arguments")

    # Merge S(Q) files
    stog = StoG(**kwargs)
    stog.read_all_data(skiprows=3)
    stog.merge_data()
    stog.write_out_merged_sq()

    # Initial S(Q) -> g(r) transform
    stog.transform_merged()
    stog.write_out_merged_gr()

    # TODO: Add the lowR minimizer here
    # print stog.get_lowR_mean_square()

    # Set the S(Q) and g(r) if no Fourier Filter
    r = stog.r_master[stog.gr_title]
    q = stog.q_master[stog.sq_title]
    sq = stog.sq_master[stog.sq_title]
    gr_out = stog.gr_master[stog.gr_title]

    # Apply Fourier Filter
    if "FourierFilter" in kwargs:
        q, sq, r, gr_out = stog.fourier_filter()

    # Apply Lorch Modification
    if kwargs["LorchFlag"]:
        r, gr_out = stog.apply_lorch(q, sq, r)

    # Apply final scale number
    stog._add_keen_fq(q, sq)
    stog._add_keen_gr(r, gr_out)


if __name__ == "__main__":
    pystog_cli()
