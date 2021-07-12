# coding: utf-8
"""Testing new xtg formats."""
import uuid
from os.path import join

import pytest

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

if not xtg.testsetup():
    raise SystemExit


@pytest.mark.benchmark()
def test_benchmark_cube_export(benchmark, testpath):
    cube1 = xtgeo.Cube(
        join(testpath, "cubes/reek/syntseis_20030101_seismic_depth_stack.segy")
    )

    fname = "syntseis_20030101_seismic_depth_stack.xtgrecube"

    @benchmark
    def write():
        cube1.to_file(fname, fformat="xtgregcube")


@pytest.mark.benchmark()
def test_benchmark_cube_import(benchmark, testpath):
    cube1 = xtgeo.Cube(
        join(testpath, "cubes/reek/syntseis_20030101_seismic_depth_stack.segy")
    )

    fname = "syntseis_20030101_seismic_depth_stack.xtgrecube"
    cube1.to_file(fname, fformat="xtgregcube")

    cube2 = xtgeo.Cube()

    @benchmark
    def read():
        cube2.from_file(fname, fformat="xtgregcube")

    assert cube1.values.mean() == pytest.approx(cube2.values.mean())
