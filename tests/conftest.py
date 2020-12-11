# coding: utf-8
# Setup common stuff for pytests...
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
    parser.addoption("--testdatapath", action="store", default="../xtgeo-testdata")


@pytest.fixture()
def testpath(request):
    return request.config.getoption("--testdatapath")
