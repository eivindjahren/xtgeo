# coding: utf-8
# Setup common stuff for pytests...
import os.environ

import pytest

from tests.fixtures import *


def pytest_runtest_setup(item):
    # called for running each test in 'a' directory
    print("\nSetting up test\n", item)


def assert_equal(this, that, txt=""):
    """Assert equal wrapper function."""
    assert this == that, txt


def assert_almostequal(this, that, tol, txt=""):
    """Assert almost equal wrapper function."""
    assert this == pytest.approx(that, abs=tol), txt


def pytest_addoption(parser):
    parser.addoption(
        "--testdatapath",
        help="path to xtgeo-testdata, defaults to ../xtgeo-testdata"
        "and is overriden by the XTG_TESTPATH environment variable."
        "Experimental feature, not all tests obey this option.",
        action="store",
        default="../xtgeo-testdata",
    )


@pytest.fixture()
def testpath(request):
    testdatapath = request.config.getoption("--testddatapath")
    environ_path = os.environ.get("XTG_TESTPATH", None)
    if environ_path:
        testdatapath = environ_path

    return testdatapath
