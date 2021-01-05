import argparse
import pytest
from pystog.io import get_cli_parser, parse_cli_args


def test_get_cli_parser():
    parser = get_cli_parser()
    args = parser.parse_args([])
    assert not args.json
    assert not args.density
    assert not args.filenames
    assert args.stem_name == 'merged'
    assert args.real_space_function == 'g(r)'
    assert args.Rmax == 50.0
    assert args.Rpoints == 5000
    assert not args.Rdelta
    assert not args.fourier_filter_cutoff
    assert args.lorch_flag is False
    assert args.bcoh_sqrd == 1.0
    assert args.btot_sqrd == 1.0
    assert args.merging == [0.0, 1.0]
    assert args.low_q_correction is False


def test_parse_cli_args():
    args = argparse.Namespace()
    file0 = ["a.txt", 0.0, 30.0, 0.0, 1.0, 0.0, "S(Q)"]
    file1 = ["b.txt", 5.0, 20.0, 2.0, 3.0, 2.0, "F(Q)"]
    args.filenames = [
        file0,
        file1,
    ]
    args.density = 0.5
    args.Rmax = 100.0
    args.Rpoints = 10000
    args.fourier_filter_cutoff = 0.0
    args.lorch_flag = False
    args.stem_name = "deleteme"
    args.merging = [0.0, 1.0]
    args.bcoh_sqrd = 22.0
    args.btot_sqrd = 22.1
    args.real_space_function = "g(r)"
    args.low_q_correction = False
    args.Rdelta = 0.01
    kwargs = parse_cli_args(args)

    for i, f in enumerate([file0, file1]):
        assert kwargs["Files"][i]["Filename"] == f[0]
        assert kwargs["Files"][i]["Qmin"] == f[1]
        assert kwargs["Files"][i]["Qmax"] == f[2]
        assert kwargs["Files"][i]["Y"]["Offset"] == f[3]
        assert kwargs["Files"][i]["Y"]["Scale"] == f[4]
        assert kwargs["Files"][i]["X"]["Offset"] == f[5]
        assert kwargs["Files"][i]["ReciprocalFunction"] == f[6]
    assert kwargs["NumberDensity"] == args.density
    assert kwargs["Rmax"] == args.Rmax
    assert kwargs["Rpoints"] == args.Rpoints
    assert kwargs["FourierFilter"]["Cutoff"] == args.fourier_filter_cutoff
    assert kwargs["LorchFlag"] == args.lorch_flag
    assert kwargs["Outputs"]["StemName"] == args.stem_name
    assert kwargs["Merging"]["Y"]["Offset"] == args.merging[0]
    assert kwargs["Merging"]["Y"]["Scale"] == args.merging[1]
    assert kwargs["<b_coh>^2"] == args.bcoh_sqrd
    assert kwargs["<b_tot^2>"] == args.btot_sqrd
    assert kwargs["RealSpaceFunction"] == args.real_space_function
    assert kwargs["OmittedXrangeCorrection"] == args.low_q_correction
    assert kwargs["Rdelta"] == args.Rdelta


def test_parse_cli_args_exception():
    with pytest.raises(TypeError):
        parse_cli_args([])
