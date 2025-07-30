============
Installation
============

Requirements
============

* Python (3.10 is tested)
* numpy_

.. _numpy: http://www.numpy.org/

Installing
==========

Pystog releases can be installed via ``pip``.

.. code:: sh

    pip install pystog

As well as ``conda``.

.. code:: sh

    conda install -c neutrons pystog

Development
===========

`pixi <https://pixi.sh/latest>`_ is used to manage local installations and development environments:

.. code:: sh

    cd pystog
    pixi install

You can then activate the pixi environment or use it to run commands/tasks:

.. code:: sh

    pixi run <command or task>

    # or simply
    pixi shell

A number of convenience "tasks" are available, and can be viewed within the ``pyproject.toml``, or by running:

.. code:: sh

    pixi task list

Tests
=====

From the parent directory of the module, run:

.. code:: sh

    pixi run test
