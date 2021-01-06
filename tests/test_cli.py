import pytest
from pystog.cli import pystog_cli
from pystog.stog import NoInputFilesException


def test_pystog_cli_no_files_exception():
    with pytest.raises(NoInputFilesException):
        pystog_cli({"cat": "meow"})
