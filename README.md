# SCAMPy_tests #

SCAMPy is a Single Column Atmospheric Model in Python.
The code is available at [SCAMPy](https://github.com/cmkaul/SCAMPy).

This is a (work in progress) suite of tests for SCAMPy.
Tests are done using [pytest](https://docs.pytest.org/en/latest/).
The workflow assumes that SCAMPy and SCAMPy_tests are available in the same folder
and that the SCAMPy code has been compiled.

# testing  #

To generate the automatic plots try:

```
$ py.test -s -v plots/

```

To run one case (for example Bomex) try:

```
$ py.test -s -v plots/test_plot_Bomex.py

```

Unit tests and functionality tests (TODO)
