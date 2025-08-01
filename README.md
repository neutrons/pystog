## Total Scattering Function Manipulator:

| Health                                                                                                                                                                                                                     | Release                                                                                                                                                                                                | Other                                                                                                                  |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| [![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fneutrons%2Fpystog%2Fbadge%3Fref%3Dmaster&style=plastic)](https://actions-badge.atrox.dev/neutrons/pystog/goto?ref=master) | [![PyPI version](https://badge.fury.io/py/pystog.svg)](https://badge.fury.io/py/pystog)                                                                                                                | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/neutrons/pystog/master?filepath=tutorials) |
| [![codecov](https://codecov.io/gh/neutrons/pystog/branch/master/graph/badge.svg)](https://codecov.io/gh/neutrons/pystog)                                                                                                   | [![Anaconda-Server Badge](https://anaconda.org/neutrons/pystog/badges/version.svg)](https://anaconda.org/neutrons/pystog)                                                                              | [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)        |
| [![Documentation Status](https://readthedocs.org/projects/pystog/badge/?version=latest)](https://pystog.readthedocs.io/en/latest/?badge=latest)                                                                            | [![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) |                                                                                                                        |

From total scattering functions, we have reciprocal-space structure factors and real-space pair distribution functions that are related via a Fourier transform.
PyStoG is a package that allows for:

1. Converting between the various functions used by different "communities" (ie researchers who study crystalline versus amorphous or glass materials). Conversions are for either real-space or reciprocal-space.
2. Perform the transform between the different available functions of choice
3. Fourier filter to remove spurious artificats in the data (ie aphysical, sub-angstrom low-r peaks in G(r) from experiments)

![alt text](https://raw.githubusercontent.com/neutrons/pystog/master/images/sofq_to_gofr.png)

The name **PyStoG** comes from the fact that this is a _Pythonized_ version of **StoG**, a ~30 year old Fortran program that is part of the [RMCProfile software suite](http://www.rmcprofile.org/Main_Page).
**StoG** means **"S(Q) to G(r)"** for the fact that it takes recirpocal-space S(Q) patterns from files and transforms them into a single G(r) pattern.
The original _StoG_ program has been developed, in reverse chronological order, by:

- Matthew Tucker and Martin Dove (~2009)
- Spencer Howells (~1989)
- Jack Carpenter (prior to 1989)

A current state of the **StoG** program is kept in the `fortran` directory of this package.

This project was initially just a "sandbox" for taking the capabilities of **StoG** and migrating them over to the [Mantid Framework](https://github.com/mantidproject/mantid).
Yet, with more and more use cases, **PyStoG** was further developed as the stand-alone project it is now.
Yet, migration to the Mantid Framework is still a goal since it feeds into the [ADDIE project](https://github.com/neutrons/addie)

## Installation

Installation is available via [`pip`](https://pip.pypa.io/en/stable/):

```bash
pip install pystog
```

And [conda](https://docs.conda.io/en/latest/):

```bash
conda install -c neutrons pystog
```

## Getting started

Once installed, you can access the packages classes that perform the function manipulation.

```python
import pystog
from pystog import Converter
from pystog import Transformer
from pystog import FourierFilter
from pystog import StoG
```

** WARNING: Testing of the CLI is still ongoing**

A CLI command is also included, which can be run with JSON input files. The script will be installed into the `bin` directory in your virtual environment directory.
For example:

- `.pixi/envs/default/bin/`
- `pystog/.venv/bin/`
- `.../miniconda/envs/pystog/bin/`

You can simply activate your virtual environment (`pixi shell`, `. .venv/bin/activate`, or `conda activate pystog`) and run `pystog-cli`:

```bash
pystog-cli --json <input json>
```

For a list of available options, run:

```bash
pystog-cli --help
```

An example JSON can be found [here](https://github.com/neutrons/pystog/blob/master/data/examples/argon_pystog.json)

## Documentation

The official documentation is hosted on readthedocs.org: [https://pystog.readthedocs.io/en/latest/](https://pystog.readthedocs.io/en/latest/)

Also, a useful example reference is the [PDFFourierTransform](http://docs.mantidproject.org/nightly/algorithms/PDFFourierTransform-v1.html) algorithm in the Mantid Framework that has similar yet limited capabilities.

Finally, tutorials in the form of Jupyter Notebooks can be launched via Binder by clicking the badge here [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/neutrons/pystog/master?filepath=tutorials) or at the top of the page.

## Development

Pystog uses [pixi](https://pixi.sh/latest) to manage packaging and dependencies.
To get started, [install pixi](https://pixi.sh/latest/installation), then install pystog by running:

```bash
cd pystog/
pixi install
```

Pixi will automatically create a virtual environment in `pystog/.pixi/`.
A number of convenience "tasks" are available, and can be viewed within the `pyproject.toml`, or by running:

```bash
pixi task list
```

### Testing

[pytest](https://docs.pytest.org/en/latest/) is used to write and run the test suite.

To run the tests, simply run:

```bash
pixi run test
```

Any additional flags or options you desire may be passed, for example:

```bash
pixi run test some.specific:test
# or
pixi run test -vv
```

### Formatting and Static analysis

[pre-commit](https://pre-commit.com/) is used for style enforcement and static analysis.
To install, after creating the environment run

```sh
pre-commit install
```

and it will run for every commit.

```sh
pre-commit run --all
```

will run it without committing.
