from os.path import join

import numpy as np
import pytest

import tests.test_common.test_xtg as tsetup
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import Grid, GridProperty
from xtgeo.surface import RegularSurface
from xtgeo.xyz import Polygons

# set default level
xtg = XTGeoDialog()

logger = xtg.basiclogger(__name__)


# ======================================================================================
# This tests a combination of methods, in order to produce maps of HC thickness
# ======================================================================================


@pytest.fixture
def reek_fault_polygons(testpath):
    zmap_file = join(testpath, "polygons", "reek", "1", "top_upper_reek_faultpoly.zmap")
    return Polygons(zmap_file, fformat="zmap")


@pytest.fixture
def reek_poro_property(reek_init_file, reek_grid):
    grid_property = GridProperty()
    return grid_property.from_file(
        reek_init_file._file, fformat="init", name="PORO", grid=reek_grid
    )


@tsetup.skipifroxar
def test_avg02(reek_grid, reek_poro_property, reek_fault_polygons, tmpdir):
    """Make average map from Reek Eclipse."""
    # get the dz and the coordinates
    dz = reek_grid.get_dz(mask=False)
    xc, yc, _zc = reek_grid.get_xyz(mask=False)

    # get actnum
    actnum = reek_grid.get_actnum()

    # convert from masked numpy to ordinary
    xcuse = np.copy(xc.values3d)
    ycuse = np.copy(yc.values3d)
    dzuse = np.copy(dz.values3d)
    pouse = np.copy(reek_poro_property.values3d)

    # dz must be zero for undef cells
    dzuse[actnum.values3d < 0.5] = 0.0
    pouse[actnum.values3d < 0.5] = 0.0

    # make a map... estimate from xc and yc
    zuse = np.ones((xcuse.shape))

    avgmap = RegularSurface(
        nx=200,
        ny=250,
        xinc=50,
        yinc=50,
        xori=457000,
        yori=5927000,
        values=np.zeros((200, 250)),
    )

    avgmap.avg_from_3dprop(
        xprop=xcuse,
        yprop=ycuse,
        zoneprop=zuse,
        zone_minmax=(1, 1),
        mprop=pouse,
        dzprop=dzuse,
        truncate_le=None,
    )

    # add the faults in plot
    fspec = {"faults": reek_fault_polygons}

    avgmap.quickplot(
        filename=join(tmpdir, "tmp_poro2.png"), xlabelrotation=30, faults=fspec
    )
    avgmap.to_file(join(tmpdir, "tmp.poro.gri"), fformat="irap_ascii")

    logger.info(avgmap.values.mean())
    assert avgmap.values.mean() == pytest.approx(0.1653, abs=0.01)


@tsetup.skipifroxar
def test_avg03(reek_grid, reek_poro_property, reek_fault_polygons, tmpdir):
    """Make average map from Reek Eclipse, speed up by zone_avgrd."""
    # get the dz and the coordinates
    dz = reek_grid.get_dz(mask=False)
    xc, yc, _zc = reek_grid.get_xyz(mask=False)

    # get actnum
    actnum = reek_grid.get_actnum()
    actnum = actnum.get_npvalues3d()

    # convert from masked numpy to ordinary
    xcuse = xc.get_npvalues3d()
    ycuse = yc.get_npvalues3d()
    dzuse = dz.get_npvalues3d(fill_value=0.0)
    pouse = reek_poro_property.get_npvalues3d(fill_value=0.0)

    # dz must be zero for undef cells
    dzuse[actnum < 0.5] = 0.0
    pouse[actnum < 0.5] = 0.0

    # make a map... estimate from xc and yc
    zuse = np.ones((xcuse.shape))

    avgmap = RegularSurface(
        nx=200,
        ny=250,
        xinc=50,
        yinc=50,
        xori=457000,
        yori=5927000,
        values=np.zeros((200, 250)),
    )

    avgmap.avg_from_3dprop(
        xprop=xcuse,
        yprop=ycuse,
        zoneprop=zuse,
        zone_minmax=(1, 1),
        mprop=pouse,
        dzprop=dzuse,
        truncate_le=None,
        zone_avg=True,
    )

    # add the faults in plot
    fspec = {"faults": reek_fault_polygons}

    avgmap.quickplot(
        filename=join(tmpdir, "tmp_poro3.png"), xlabelrotation=30, faults=fspec
    )
    avgmap.to_file(join(tmpdir, "tmp.poro3.gri"), fformat="irap_ascii")

    logger.info(avgmap.values.mean())
    assert avgmap.values.mean() == pytest.approx(0.1653, abs=0.01)
