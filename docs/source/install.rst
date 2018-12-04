============
Installation
============

Requirements
============

* Python (2.7, 3.5, 3.6 are tested)
* matplotlib_
* numpy_
* pandas_


.. _matplotlib: https://matplotlib.org/
.. _numpy: http://www.numpy.org/
.. _pandas: http://pandas.pydata.org/



Using PyPI
==========

Installation is available via `pip`.

.. code:: sh

    pip install pystog

or for a local install

.. code:: sh

    pip install pystog --user #locally installed in $HOME/.local

Using python setup.py
=======================

A setup.py is available to install but it is recommended to do 
this in a virtual environment. For system install, use `PyPi` instead.

.. code:: sh

   python setup.py install


Development
===========

For a development environment, you can use virtualenv_ to setup an isolated environemnt, namely `ENV`:

.. code:: sh

    python -m virtualenv /path/to/ENV
    source /path/to/ENV/bin/activate
    pip install pystog

Also, direnv_ is a useful and recommended way to manage the virtual environment. You can simply use an `.envrc` file which contains `layout python<version>` where `<version>` == 2 or 3 for the python version you would like (3 is recommended).

Then, once inside the development directory, just install via `pip` as described above.

.. _virtualenv: https://virtualenv.pypa.io/en/latest/
.. _direnv: https://github.com/direnv/direnv

Tests
=====

From the parent directory of the module, run:

.. code:: sh

    python tests/runner.py

