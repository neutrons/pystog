from pystog.io import get_cli_parser


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
