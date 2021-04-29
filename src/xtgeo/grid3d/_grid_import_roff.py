# coding: utf-8
"""Private module, Grid Import private functions for ROFF format."""

import pathlib

import xtgeo

from ._roff_grid import RoffGrid

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_roff(self, gfile):
    filelike = gfile._file
    if isinstance(filelike, pathlib.Path):
        roff_grid = RoffGrid.from_file(str(gfile._file))
    else:
        roff_grid = RoffGrid.from_file(gfile._file)
    self._ncol = int(roff_grid.nx)
    self._nrow = int(roff_grid.ny)
    self._nlay = int(roff_grid.nz)
    self._actnumsv = roff_grid.xtgeo_actnum()
    self._coordsv = roff_grid.xtgeo_coord()
    self._zcornsv = roff_grid.xtgeo_zcorn()
    self._subgrids = roff_grid.xtgeo_subgrids()
    self._xtgformat = 2
