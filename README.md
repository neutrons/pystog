Total Scattering Function Manipulator:
-----------------------------------------------------------

| Health | Release | Other  |
|--------|---------|------------|
| [![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fneutrons%2Fpystog%2Fbadge%3Fref%3Dmaster&style=plastic)](https://actions-badge.atrox.dev/neutrons/pystog/goto?ref=master) | [![PyPI version](https://badge.fury.io/py/pystog.svg)](https://badge.fury.io/py/pystog) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/neutrons/pystog/master?filepath=tutorials) |
| [![codecov](https://codecov.io/gh/neutrons/pystog/branch/master/graph/badge.svg)](https://codecov.io/gh/neutrons/pystog) | [![Anaconda-Server Badge](https://anaconda.org/neutrons/pystog/badges/version.svg)](https://anaconda.org/neutrons/pystog)| [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)  |
|[![Documentation Status](https://readthedocs.org/projects/pystog/badge/?version=latest)](https://pystog.readthedocs.io/en/latest/?badge=latest) | [![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) |  |

From total scattering functions, we have reciprocal-space structure factors and real-space pair distribution functions that are related via a Fourier transform.
PyStoG is a package that allows for:
1. Converting between the various functions used by different "communities" (ie researchers who study crystalline versus amorphous or glass materials). Conversions are for either real-space or reciprocal-space.
2. Perform the transform between the different available functions of choice
3. Fourier filter to remove spurious artificats in the data (ie aphysical, sub-angstrom low-r peaks in G(r) from experiments)

![alt text](https://raw.githubusercontent.com/neutrons/pystog/master/images/sofq_to_gofr.png)


The name **PyStoG** comes from the fact that this is a _Pythonized_ version of **StoG**, a ~30 year old Fortran program that is part of the [RMCProfile software suite](http://www.rmcprofile.org/Main_Page).
**StoG** means **"S(Q) to G(r)"** for the fact that it takes recirpocal-space S(Q) patterns from files and transforms them into a single G(r) pattern.
The original *StoG* program has been developed, in reverse chronological order, by:

 * Matthew Tucker and Martin Dove (~2009)
 * Spencer Howells (~1989)
 * Jack Carpenter (prior to 1989)
 
 A current state of the **StoG** program is kept in the `fortran` directory of this package.

This project was initially just a "sandbox" for taking the capabilities of **StoG** and migrating them over to the [Mantid Framework](https://github.com/mantidproject/mantid).
Yet, with more and more use cases, **PyStoG** was further developed as the stand-alone project it is now.
Yet, migration to the Mantid Framework is still a goal since it feeds into the [ADDIE project](https://github.com/neutrons/addie)

## Installation

Installation is available via [`pip`](https://pip.pypa.io/en/stable/):
`pip install pystog`

And [conda](https://docs.conda.io/en/latest/):
`conda install -c neutrons pystog`

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

Also, there is a beta-version of a python script in the package that can be run on JSON input files and operates similarly to the original **StoG** program.
This is `python cli` and can be used as follows:

```bash
pystog_cli --json <input json>
```
An example JSON can be found [here](https://github.com/neutrons/pystog/blob/master/data/examples/argon_pystog.json)

### Documentation
The official documentation is hosted on readthedocs.org: [https://pystog.readthedocs.io/en/latest/](https://pystog.readthedocs.io/en/latest/)

Also, a useful example reference is the [PDFFourierTransform](http://docs.mantidproject.org/nightly/algorithms/PDFFourierTransform-v1.html) algorithm in the Mantid Framework that has similar yet limited capabilities.

Finally, tutorials in the form of Jupyter Notebooks can be launched via Binder by clicking the badge here [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/neutrons/pystog/master?filepath=tutorials) or at the top of the page.

## Development

The following are ways to setup the virtual environment, lint and test the package.

### Virtual environment setup

#### Using pipenv (recommended)
With [pipenv](https://pipenv.pypa.io/en/latest/) installed, run the following:
```bash
pipenv install --dev .
```

#### Using virtualenv
You can use [`virtualenv`](https://virtualenv.pypa.io/en/latest/) to setup an environment, here named `ENV`:

```bash
python -m virtualenv /path/to/ENV
source /path/to/ENV/bin/activate
pip install pystog
```

#### Using direnv
Also, [`direnv`](https://github.com/direnv/direnv) is a useful and recommended way to manage the virtual environment.
You can simply use an `.envrc` file which contains:

`layout python3`

Then, once inside the development directory, just install via `pip` as described above.

### Testing
[pytest](https://docs.pytest.org/en/latest/) is used to write the test suite.
[Tox](https://tox.readthedocs.io/en/latest/) is the general tool for running both linting and testing (single and multiple python version testing).
[pyenv](https://github.com/pyenv/pyenv) is the recommended tool for python version management.
From the parent directory of the module, run:

`pytest`

Using pipenv:
`pipenv run pytest`

Using tox for a single python version,
where `py<XY>` is one of [py36, py37, py38, py39]:
`tox -e py<XY>`

or with pipenv:
`pipenv run tox -e py<XY>`

Using tox for all stages of testing (all python versions and linting), just run:
`tox`
NOTE: You must have the version of python installed to run a test suite against that verion via tox.
Recommended way to install python versions is `pyenv`

### Linting
[Flake8](https://flake8.pycqa.org/en/latest/) is used for style guide enforcement (i.e. linting):

From the parent directory of the module, run:

`flake8 .`

Using pipenv and flake8:
`pipenv run flake8 .`

Using tox
`tox -e lint`

Using pipenv and tox:
`pipenv run tox -e lint`
