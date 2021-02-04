# coding: utf-8
from __future__ import absolute_import, division, print_function

import os
from os.path import basename, join

import numpy as np
import pytest

import tests.test_common.test_xtg as tsetup
import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit


@pytest.fixture
def dual_poro_grid(dual_poro_path):
    return xtgeo.grid3d.Grid(dual_poro_path + ".EGRID")


@pytest.fixture
def dual_poro_dual_perm_path(dual_poro_path):
    return dual_poro_path + "DK"


@pytest.fixture
def dual_poro_dual_perm_wg_path(grids_etc_path):
    # same as dual_poro_dual_perm but with water/gas
    # instead of oil/water
    return join(grids_etc_path, "TEST2_DPDK_WG")


class GridCase:
    def __init__(self, path, expected_dimensions, expected_perm):
        self.path = path
        self.expected_dimensions = expected_dimensions
        self.expected_perm = expected_perm

    @property
    def grid(self):
        return xtgeo.grid3d.Grid(self.path + ".EGRID")

    def get_property_from_init(self, name, **kwargs):
        return xtgeo.gridproperty_from_file(
            self.path + ".INIT", grid=self.grid, name=name, **kwargs
        )

    def get_property_from_restart(self, name, date, **kwargs):

        return xtgeo.gridproperty_from_file(
            self.path + ".UNRST", grid=self.grid, date=date, name=name, **kwargs
        )


@pytest.fixture
def dual_poro_case(dual_poro_path):
    return GridCase(dual_poro_path, (5, 3, 1), False)


@pytest.fixture
def dual_poro_dual_perm_case(dual_poro_dual_perm_path):
    return GridCase(dual_poro_dual_perm_path, (5, 3, 1), True)


@pytest.fixture
def dual_poro_dual_perm_wg_case(dual_poro_dual_perm_wg_path):
    return GridCase(dual_poro_dual_perm_wg_path, (5, 3, 1), True)


@pytest.fixture(
    params=[
        "dual_poro_case",
        "dual_poro_dual_perm_case",
        "dual_poro_dual_perm_wg_case",
    ]
)
def dual_cases(request):
    return request.getfixturevalue(request.param)


def test_dual_grid_dimensions(dual_cases):
    assert dual_cases.grid.dimensions == dual_cases.expected_dimensions


def test_dual_grid_dualporo_predicate(dual_cases):
    assert dual_cases.grid.dualporo is True


def test_dual_grid_predicates(dual_cases):
    assert dual_cases.grid.dualperm is dual_cases.expected_perm


def test_dual_case_grid_to_file(tmpdir, dual_cases):
    dual_cases.grid.to_file(join(tmpdir, basename(dual_cases.path)) + ".roff")


def test_dual_case_actnum_to_file(tmpdir, dual_cases):
    dual_cases.grid._dualactnum.to_file(
        join(tmpdir, basename(dual_cases.path) + "dualact.roff")
    )


def test_dual_grid_poro_property(tmpdir, dual_cases):
    poro = dual_cases.get_property_from_init("PORO")

    assert poro.values[0, 0, 0] == pytest.approx(0.1)
    assert poro.values[1, 1, 0] == pytest.approx(0.16)
    assert poro.values[4, 2, 0] == pytest.approx(0.24)

    assert poro.name == "POROM"

    poro.describe()


def test_dual_grid_fractured_poro_property(tmpdir, dual_cases):
    poro = dual_cases.get_property_from_init("PORO", fracture=True)

    assert poro.values[0, 0, 0] == pytest.approx(0.25)
    assert poro.values[4, 2, 0] == pytest.approx(0.39)

    assert poro.name == "POROF"

    poro.describe()


def test_dualperm_fractured_poro_values(dual_poro_dual_perm_case):
    poro = dual_poro_dual_perm_case.get_property_from_init(name="PORO", fracture=True)
    assert poro.values[3, 0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize("date", [20170121, 20170131])
def test_dual_grid_swat_property(tmpdir, dual_cases, date):
    swat = dual_cases.get_property_from_restart("SWAT", date=date)
    swat.describe()
    assert swat.name == f"SWATM_{date}"
    swat.to_file(join(tmpdir, basename(dual_cases.path) + str(date) + "swatm.roff"))


@pytest.mark.parametrize("date", [20170121, 20170131])
def test_dual_grid_fractured_swat_property(tmpdir, dual_cases, date):
    swat = dual_cases.get_property_from_restart("SWAT", date=date, fracture=True)
    swat.describe()
    assert swat.name == f"SWATF_{date}"
    swat.to_file(join(tmpdir, basename(dual_cases.path) + str(date) + "swatf.roff"))


def test_dual_case_swat_values(dual_poro_case):
    swat = dual_poro_case.get_property_from_restart("SWAT", date=20170121)
    assert swat.values[0, 0, 0] == pytest.approx(0.609244)


def test_dual_case_fractured_swat_values(dual_poro_case):
    swat = dual_poro_case.get_property_from_restart(
        "SWAT", date=20170121, fracture=True
    )
    assert swat.values[0, 0, 0] == pytest.approx(0.989687)


def test_dualperm_swat_property(dual_poro_dual_perm_case):
    swat = dual_poro_dual_perm_case.get_property_from_restart("SWAT", date=20170121)
    assert swat.values[3, 0, 0] == pytest.approx(0.5547487)


def test_dualperm_fractured_swat_property(dual_poro_dual_perm_case):
    swat = dual_poro_dual_perm_case.get_property_from_restart(
        "SWAT", date=20170121, fracture=True
    )
    assert swat.values[3, 0, 0] == pytest.approx(0.0)


def test_dualperm_wg_swat_property(dual_poro_dual_perm_wg_case):
    swat = dual_poro_dual_perm_wg_case.get_property_from_restart("SWAT", date=20170121)
    assert swat.values[3, 0, 0] == pytest.approx(0.933606)
    assert swat.values[0, 1, 0] == pytest.approx(0.0)
    assert swat.values[4, 2, 0] == pytest.approx(0.89304)


def test_dualperm_wg_fractured_swat_property(dual_poro_dual_perm_wg_case):
    swat = dual_poro_dual_perm_wg_case.get_property_from_restart(
        "SWAT", date=20170121, fracture=True
    )
    assert swat.values[3, 0, 0] == pytest.approx(0.0)
    assert swat.values[0, 1, 0] == pytest.approx(0.99818)
    assert swat.values[4, 2, 0] == pytest.approx(0.821589)


def test_dual_case_perm_property(tmpdir, dual_cases):
    perm = dual_cases.get_property_from_init("PERMX")

    assert perm.values[0, 0, 0] == pytest.approx(100.0)
    assert perm.values[3, 0, 0] == pytest.approx(100.0)
    assert perm.values[0, 1, 0] == pytest.approx(0.0)
    assert perm.values[4, 2, 0] == pytest.approx(100)

    perm.describe()
    assert perm.name == "PERMXM"
    perm.to_file(os.path.join(tmpdir, basename(dual_cases.path) + "permxm.roff"))


def test_dual_case_fractured_perm_property(tmpdir, dual_cases):
    perm = dual_cases.get_property_from_init("PERMX", fracture=True)

    assert perm.values[0, 0, 0] == pytest.approx(100.0)
    assert perm.values[0, 1, 0] == pytest.approx(100.0)
    assert perm.values[4, 2, 0] == pytest.approx(100)

    perm.describe()
    assert perm.name == "PERMXF"
    perm.to_file(os.path.join(tmpdir, basename(dual_cases.path) + "permxf.roff"))


def test_dualperm_perm_property(dual_poro_dual_perm_case):
    perm = dual_poro_dual_perm_case.get_property_from_init("PERMX", fracture=True)
    assert perm.values[3, 0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize("date", [20170121, 20170131])
def test_dual_cases_soil_property(tmpdir, dual_cases, date):
    soil = dual_cases.get_property_from_restart("SOIL", date=date)
    soil.describe()
    assert soil.name == f"SOILM_{date}"
    soil.to_file(
        os.path.join(tmpdir, basename(dual_cases.path) + str(date) + "soilxm.roff")
    )


@pytest.mark.parametrize("date", [20170121, 20170131])
def test_dual_cases_fractured_soil_property(tmpdir, dual_cases, date):
    soil = dual_cases.get_property_from_restart("SOIL", date=date, fracture=True)
    soil.describe()
    assert soil.name == f"SOILF_{date}"
    soil.to_file(
        os.path.join(tmpdir, basename(dual_cases.path) + str(date) + "soilxf.roff")
    )


def test_dualperm_soil_property(dual_poro_dual_perm_case):
    soil = dual_poro_dual_perm_case.get_property_from_restart("SOIL", date=20170121)
    assert soil.values[3, 0, 0] == pytest.approx(0.4452512)
    assert soil.values[0, 1, 0] == pytest.approx(0.0)
    assert np.ma.is_masked(soil.values[1, 2, 0])
    assert soil.values[3, 2, 0] == pytest.approx(0.0)
    assert soil.values[4, 2, 0] == pytest.approx(0.4127138)


def test_dualperm_fractured_soil_property(dual_poro_dual_perm_case):
    soil = dual_poro_dual_perm_case.get_property_from_restart(
        "SOIL", date=20170121, fracture=True
    )
    assert soil.values[3, 0, 0] == pytest.approx(0.0)
    assert soil.values[0, 1, 0] == pytest.approx(0.01174145)
    assert soil.values[3, 2, 0] == pytest.approx(0.11676442)


def test_dualpermwg_soil_property(dual_poro_dual_perm_wg_case):
    soil = dual_poro_dual_perm_wg_case.get_property_from_restart("SOIL", date=20170121)
    assert soil.values[3, 0, 0] == pytest.approx(0.0)
    assert soil.values[0, 1, 0] == pytest.approx(0.0)


def test_dualpermwg_fractured_soil_property(dual_poro_dual_perm_wg_case):
    soil = dual_poro_dual_perm_wg_case.get_property_from_restart(
        "SOIL", date=20170121, fracture=True
    )
    assert soil.values[3, 0, 0] == pytest.approx(0.0)
    assert soil.values[0, 1, 0] == pytest.approx(0.0)


@pytest.mark.parametrize("date", [20170121, 20170131])
def test_dual_cases_sgas_property(tmpdir, dual_cases, date):
    sgas = dual_cases.get_property_from_restart("SGAS", date=date)
    sgas.describe()
    assert sgas.name == f"SGASM_{date}"
    sgas.to_file(
        os.path.join(tmpdir, basename(dual_cases.path) + str(date) + "sgasxm.roff")
    )


@pytest.mark.parametrize("date", [20170121, 20170131])
def test_dual_cases_fractured_sgas_property(tmpdir, dual_cases, date):
    sgas = dual_cases.get_property_from_restart("SGAS", date=date, fracture=True)
    sgas.describe()
    assert sgas.name == f"SGASF_{date}"
    sgas.to_file(
        os.path.join(tmpdir, basename(dual_cases.path) + str(date) + "sgasxf.roff")
    )


def test_dualperm_sgas_property(dual_poro_dual_perm_case):
    sgas = dual_poro_dual_perm_case.get_property_from_restart("SGAS", date=20170121)
    assert sgas.values[3, 0, 0] == pytest.approx(0.0)
    assert sgas.values[0, 1, 0] == pytest.approx(0.0)


def test_dualperm_fractured_sgas_property(dual_poro_dual_perm_case):
    sgas = dual_poro_dual_perm_case.get_property_from_restart(
        "SGAS", date=20170121, fracture=True
    )
    assert sgas.values[3, 0, 0] == pytest.approx(0.0)
    assert sgas.values[0, 1, 0] == pytest.approx(0.0)


def test_dualperm_wg_sgas_property(dual_poro_dual_perm_wg_case):
    sgas = dual_poro_dual_perm_wg_case.get_property_from_restart("SGAS", date=20170121)
    assert sgas.values[3, 0, 0] == pytest.approx(0.0663941)
    assert sgas.values[0, 1, 0] == pytest.approx(0.0)
    assert sgas.values[4, 2, 0] == pytest.approx(0.1069594)


def test_dualperm_wg_fractured_sgas_property(dual_poro_dual_perm_wg_case):
    sgas = dual_poro_dual_perm_wg_case.get_property_from_restart(
        "SGAS", date=20170121, fracture=True
    )
    assert sgas.values[3, 0, 0] == pytest.approx(0.0)
    assert sgas.values[0, 1, 0] == pytest.approx(0.00181985)
    assert sgas.values[4, 2, 0] == pytest.approx(0.178411)
