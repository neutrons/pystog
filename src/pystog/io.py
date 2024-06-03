import json
import argparse
from pystog.utils import RealSpaceChoices, ReciprocalSpaceChoices


def get_cli_parser():
    """
    Create the argument parser for the CLI

    :return: Argument parser for PyStoG CLI
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Read JSON input file for arguments")

    parser.add_argument(
        "--density",
        type=float,
        help="Number density (atoms/angstroms^3")

    parser.add_argument(
        "-f",
        "--filename",
        nargs='*',
        action='append',
        default=list(),
        dest='filenames',
        help="Filename, qmin, qmax, yoffset, yscale, Qoffset, function type."
        "Function Types are: %s" %
        json.dumps(ReciprocalSpaceChoices))

    parser.add_argument(
        "--stem-name",
        type=str,
        dest="stem_name",
        default="merged",
        help="Stem name for output files")

    parser.add_argument(
        "--real-space-function",
        type=str,
        default="g(r)",
        dest="real_space_function",
        help="Real-space function typej. Choices are: %s" %
        json.dumps(RealSpaceChoices))

    parser.add_argument(
        "--Rmax",
        type=float,
        default=50.0,
        help="Maximum value in angstroms for real-space functions")

    parser.add_argument(
        "--Rpoints",
        type=int,
        default=5000,
        help="Number of points in R for real-space functions")

    parser.add_argument(
        "--Rdelta",
        type=float,
        default=None,
        help="Bin width to use in R for real-space functions")

    parser.add_argument(
        "--fourier-filter-cutoff",
        type=float,
        default=None,
        dest="fourier_filter_cutoff",
        help="Bin width to use in R for real-space functions")

    parser.add_argument(
        "--lorch-flag",
        action="store_true",
        default=False,
        dest="lorch_flag",
        help="Apply Lorch function")

    parser.add_argument(
        "--bcoh_sqrd",
        type=float,
        default=1.0,
        dest="bcoh_sqrd",
        help="The (sum c*bbar)^2 term needed for F(Q) and G(r) for RMC output")

    parser.add_argument(
        "--btot_sqrd",
        type=float,
        default=1.0,
        dest="btot_sqrd",
        help="The (sum c*b^2) term needed for DCS(Q) input")

    parser.add_argument(
        "--merging",
        nargs=2,
        type=float,
        default=[0.0, 1.0],
        help="Offset and Scale to apply to the merged S(Q)")

    parser.add_argument(
        "--low-q-correction",
        action='store_true',
        help="Apply low-Q correction during FT")

    return parser


def parse_cli_args(args):
    """
    Parse the CLI arguments and return a key-word dictionary with options to
    pass to StoG class

    :param args: Namespace returned from parsing the CLI arguments
    :type args: argparse.Namespace
    :return: Dictionary with options to pass to StoG class
    :rtype: dict
    """
    if not isinstance(args, argparse.Namespace):
        msg = "parse_cli_args takes a argparse.Namespace, "
        msg += "yet was given argument of type: {}"
        raise TypeError(msg.format(type(args)))

    # Get each file's info and story in dictionary
    files_info = list()
    for f in args.filenames:
        file_info = dict()
        file_info["Filename"] = f[0]
        file_info["Qmin"] = float(f[1])
        file_info["Qmax"] = float(f[2])
        file_info["Y"] = {"Offset": float(f[3]), "Scale": float(f[4])}
        file_info["X"] = {"Offset": float(f[5])}
        file_info["ReciprocalFunction"] = f[6]
        files_info.append(file_info)

    # Get general StoG options
    kwargs = {
        "Files": files_info,
        "NumberDensity": args.density,
        "Rmax": args.Rmax,
        "Rpoints": args.Rpoints,
        "FourierFilter": {
            "Cutoff": args.fourier_filter_cutoff},
        "LorchFlag": args.lorch_flag,
        "Outputs": {
            "StemName": args.stem_name},
        "Merging": {
            "Y": {
                "Offset": args.merging[0],
                "Scale": args.merging[1]}},
        "<b_coh>^2": args.bcoh_sqrd,
        "<b_tot^2>": args.btot_sqrd,
        "RealSpaceFunction": args.real_space_function,
        "OmittedXrangeCorrection": args.low_q_correction}
    if args.Rdelta:
        kwargs["Rdelta"] = args.Rdelta

    return kwargs
