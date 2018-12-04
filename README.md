Total Scattering Function Manipulator:
-----------------------------------------------------------

| Dev | Other |
|-----|-------|
|[![Build Status](https://travis-ci.org/marshallmcdonnell/pystog.svg?branch=master)](https://travis-ci.org/marshallmcdonnell/pystog) | [![Documentation Status](https://readthedocs.org/projects/pystog/badge/?version=latest)](https://pystog.readthedocs.io/en/latest/?badge=latest) |
| [![codecov](https://codecov.io/gh/marshallmcdonnell/pystog/branch/master/graph/badge.svg)](https://codecov.io/gh/marshallmcdonnell/pystog) | [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) |
| [![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip) | |

From total scattering functions, we have reciprocal-space structure factors and real-space pair distribution functions that are related via a Fourier transform. PyStoG is a package that allows for:
1. Converting between the various functions used by different "communities" (ie researchers who study crystalline versus amorphous or glass materials). Conversions are for either real-space or reciprocal-space.
2. Perform the transform between the different available functions of choice
3. Fourier filter to remove spurious artificats in the data (ie aphysical, sub-angstrom low-r peaks in G(r) from experiments)

![alt text](https://raw.githubusercontent.com/marshallmcdonnell/mantid_total_scattering/master/images/sofq_to_gofr.png)


The name **PyStoG** comes from the fact that this is a _Pythonized_ version of **StoG**, a ~30 year old Fortran program that is part of the [RMCProfile software suite](http://www.rmcprofile.org/Main_Page). **StoG** means **"S(Q) to G(r)"** for the fact that it takes recirpocal-space S(Q) patterns from files and transforms them into a single G(r) pattern. The original *StoG* program has been developed, in reverse chronological order, by:

 * Matthew Tucker (~2009)
 * Spencer Howells (~1989)
 * Jack Carpenter (prior to 1989)
 
 A current state of the **StoG** program is kept in the `fortran` directory of this package.

This project was initially just a "sandbox" for taking the capabilities of **StoG** and migrating them over to the [Mantid Framework](https://github.com/mantidproject/mantid). Yet, with more and more use cases, **PyStoG** was further developed as the stand-alone project it is now. Yet, migration to the Mantid Framework is still a goal since it feeds into the [ADDIE project](https://github.com/neutrons/addie)

## Installation

Installation is available via `pip`. 

```bash 
pip install pystog
```

or for a local install
```bash
pip install pystog --user #locally installed in $HOME/.local
```
For a development environment, you can use [`virtualenv`](https://virtualenv.pypa.io/en/latest/) to setup an isolated environemnt, namely `ENV`:

```bash
python -m virtualenv /path/to/ENV
source /path/to/ENV/bin/activate
pip install pystog
```

Also, [`direnv`](https://github.com/direnv/direnv) is a useful and recommended way to manage the virtual environment. You can simply use an `.envrc` file which contains:

`layout python3`

Then, once inside the development directory, just install via `pip` as described above.

## Getting started

Once installed, you can access the packages classes that perform the function manipulation. 

```python
import pystog
from pystog import Converter
from pystog import Transformer
from pystog import FourierFilter
from pystog import StoG
```

Also, there is a beta-version of a python script in the package that can be run on JSON input files and operates similarly to the original **StoG** program, only with extra `matplotlib` visualization of the output. This is `python_cli` and can be used as follows:

```bash
pystog_cli --json <input json>
```
An example JSON can be found [here](https://github.com/marshallmcdonnell/pystog/blob/master/data/examples/argon_pystog.json)

### Documentation
This is a current work in progress but a useful example reference is the [PDFFourierTransform](http://docs.mantidproject.org/nightly/algorithms/PDFFourierTransform-v1.html) algorithm in the Mantid Framework that has similar yet limited capabilities.

## Running the tests
From the parent directory of the module, run:

```bash
python tests/runner.py
```
