# coding: utf-8
"""Testing new xtg formats."""
import pathlib
import uuid
from os.path import join

import pytest

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit


def test_cube_export_import_many(tmpdir, testpath):
    """Test exporting etc to xtgregcube format."""
    cube1 = xtgeo.Cube(
        join(testpath, "cubes", "reek", "syntseis_20030101_seismic_depth_stack.segy")
    )

    nrange = 50

    fformat = "xtgregcube"

    fnames = []

    # timing of writer
    t1 = xtg.timer()
    for num in range(nrange):
        fname = uuid.uuid4().hex + "." + fformat

        fname = pathlib.Path(tmpdir) / fname
        fnames.append(fname)
        cube1.to_file(fname, fformat=fformat)

    logger.info("Timing export %s cubes with %s: %s", nrange, fformat, xtg.timer(t1))

    # timing of reader
    t1 = xtg.timer()
    for fname in fnames:
        cube2 = xtgeo.Cube()
        cube2.from_file(fname, fformat=fformat)

    logger.info("Timing import %s cubes with %s: %s", nrange, fformat, xtg.timer(t1))

    assert cube1.values.mean() == pytest.approx(cube2.values.mean())
