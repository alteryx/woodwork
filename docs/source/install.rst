Install
*******

Woodwork is available for Python 3.6, 3.7, and 3.8. The recommended way to install woodwork is using ``pip``:
::

    python -m pip install woodwork


Install from Source
-------------------

To install Woodwork from source, clone the repository from `github
<https://github.com/FeatureLabs/woodwork>`_::

    git clone https://github.com/FeatureLabs/woodwork.git
    cd woodwork
    python setup.py install

or use ``pip`` locally if you want to install all dependencies as well::

    pip install .

You can view the list of all dependencies within the ``extras_require`` field
of ``setup.py``.


Development
-----------
Before making contributions to the codebase, please follow the guidelines `here <https://github.com/FeatureLabs/woodwork/blob/main/contributing.md>`_.

Virtualenv
~~~~~~~~~~
We recommend developing in a `virtualenv <https://virtualenvwrapper.readthedocs.io/en/latest/>`_::

    mkvirtualenv venv

Install development requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run::

    make installdeps

Test
~~~~

Run Woodwork tests::

    make test

Before committing make sure to run linting in order to pass CI::

    make lint

Some linting errors can be automatically fixed by running the command below::

    make lint-fix


Build Documentation
~~~~~~~~~~~~~~~~~~~
Build the docs with the commands below::

    cd docs/

    # small changes
    make html

    # rebuild from scatch
    make clean html
