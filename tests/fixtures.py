from os.path import join

import pytest

import xtgeo


@pytest.fixture
def reekpath(testpath):
    return join(testpath, "3dgrids", "reek")


@pytest.fixture
def reek3path(testpath):
    return join(testpath, "3dgrids", "reek3")


@pytest.fixture
def reek_egrid_file(reekpath):
    return xtgeo._XTGeoFile(join(reekpath, "REEK.EGRID"))


@pytest.fixture
def reek_init_file(reekpath):
    return xtgeo._XTGeoFile(join(reekpath, "REEK.INIT"))


@pytest.fixture
def reek_grid(reek_egrid_file):
    grid = xtgeo.Grid()
    return grid.from_file(reek_egrid_file._file, fformat="egrid")


@pytest.fixture
def eme1_path(testpath):
    return join(testpath, "3dgrids", "eme", "1")


@pytest.fixture
def emerald_grid_file(eme1_path):
    return join(eme1_path, "emerald_hetero_grid.roff")


@pytest.fixture
def emerald_grid(emerald_grid_file):
    grd = xtgeo.grid3d.Grid()
    return grd.from_file(emerald_grid_file, fformat="roff")


@pytest.fixture
def grids_etc_path(testpath):
    return join(testpath, "3dgrids", "etc")


@pytest.fixture
def banal6_grid_file(grids_etc_path):
    return join(grids_etc_path, "banal6.roff")


@pytest.fixture
def dual_poro_path(grids_etc_path):
    return join(grids_etc_path, "TEST_DP")


@pytest.fixture
def dual_poro_dual_perm_grid(dual_poro_path):
    return xtgeo.grid3d.Grid(dual_poro_path + "DK.EGRID")
