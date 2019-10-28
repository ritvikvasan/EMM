.. highlight:: shell

============
Installation
============


Stable release
--------------

To install EMM, run these commands in your terminal:

* Create conda environment

.. code-block:: bash

    $ conda create --name cvae python=3.7

* Activate conda environment :

.. code-block:: bash

    $ conda activate cvae

* Install requirements in setup.py

.. code-block:: bash

    $ pip install -e .[all]

This is the preferred method to install EMM, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for EMM can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/ritvikvasan/emm

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/ritvikvasan/emm/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/ritvikvasan/emm
.. _tarball: https://github.com/ritvikvasan/emm/tarball/master
