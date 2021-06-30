import hypothesis.strategies as st
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

import xtgeo.grid3d._grdecl_grid as ggrid
from xtgeo.grid3d import Grid

from .grid_generator import indecies, xtgeo_grids

finites = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=32
)

units = st.sampled_from(ggrid.Units)
grid_relatives = st.sampled_from(ggrid.GridRelative)
orders = st.sampled_from(ggrid.Order)
orientations = st.sampled_from(ggrid.Orientation)
handedness = st.sampled_from(ggrid.Handedness)
coordinate_types = st.sampled_from(ggrid.CoordinateType)

map_axes = st.builds(
    ggrid.MapAxes,
    st.tuples(finites, finites),
    st.tuples(finites, finites),
    st.tuples(finites, finites),
)

gdorients = st.builds(ggrid.GdOrient, orders, orders, orders, orientations, handedness)


@st.composite
def gridunits(draw, relative=grid_relatives):
    return draw(st.builds(ggrid.GridUnit, units, relative))


@st.composite
def specgrids(draw, coordinates=coordinate_types):
    return draw(
        st.builds(ggrid.SpecGrid, indecies, indecies, indecies, st.just(1), coordinates)
    )


@st.composite
def grdecl_grids(
    draw,
    spec=specgrids(),
    mpaxs=map_axes,
    orient=gdorients,
    gunit=gridunits(),
    zcorn=lambda dims: arrays(
        shape=8 * dims[0] * dims[1] * dims[2],
        dtype=np.float32,
        elements=finites,
    ),
):
    specgrid = draw(spec)
    dims = specgrid.ndivix, specgrid.ndiviy, specgrid.ndiviz

    corner_size = (dims[0] + 1) * (dims[1] + 1) * 6
    coord = draw(
        arrays(
            shape=corner_size,
            dtype=np.float32,
            elements=finites,
        )
    )
    if draw(st.booleans()):
        actnum = draw(
            arrays(
                shape=dims[0] * dims[1] * dims[2],
                dtype=np.int32,
                elements=st.integers(min_value=0, max_value=3),
            )
        )
    else:
        actnum = None
    mapax = draw(mpaxs) if draw(st.booleans()) else None
    gdorient = draw(orient) if draw(st.booleans()) else None
    gridunit = draw(gunit) if draw(st.booleans()) else None

    return ggrid.GrdeclGrid(
        mapaxes=mapax,
        specgrid=specgrid,
        gridunit=gridunit,
        zcorn=draw(zcorn(dims)),
        actnum=actnum,
        coord=coord,
        gdorient=gdorient,
    )


@st.composite
def create_compatible_zcorn(draw, dims):
    nx, ny, nz = dims
    array = draw(
        arrays(
            shape=(2, nx, 2, ny, 2, nz),
            dtype=np.float32,
            elements=finites,
        )
    )
    array[:, :, :, :, 1, : nz - 1] = array[:, :, :, :, 0, 1:]
    for i in range(3, 6):
        array = np.swapaxes(array, i, 5 - i)
    return array.ravel()


xtgeo_compatible_grdecl_grids = grdecl_grids(
    spec=specgrids(
        coordinates=st.just(ggrid.CoordinateType.CARTESIAN),
    ),
    orient=st.just(ggrid.GdOrient()),
    gunit=gridunits(relative=st.just(ggrid.GridRelative.ORIGIN)),
    zcorn=create_compatible_zcorn,
)


@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(grdecl_grids())
def test_grdecl_grid_read_write(tmp_path, grgrid):
    tmp_file = tmp_path / "grid.grdecl"
    grgrid.to_file(tmp_file)
    assert ggrid.GrdeclGrid.from_file(tmp_file) == grgrid


@given(xtgeo_grids)
def test_to_from_xtgeogrid_format2(xtggrid):
    xtggrid._xtgformat2()
    grdecl_grid = ggrid.GrdeclGrid.from_xtgeo_grid(xtggrid)

    assert_allclose(grdecl_grid.xtgeo_actnum(), xtggrid._actnumsv, atol=0.02)
    assert_allclose(grdecl_grid.xtgeo_coord(), xtggrid._coordsv, atol=0.02)
    assert_allclose(grdecl_grid.xtgeo_zcorn(), xtggrid._zcornsv, atol=0.02)


@given(xtgeo_grids)
def test_to_from_xtgeogrid_format1(xtggrid):
    xtggrid._xtgformat1()
    grdecl_grid = ggrid.GrdeclGrid.from_xtgeo_grid(xtggrid)

    xtggrid._xtgformat2()
    assert_allclose(grdecl_grid.xtgeo_actnum(), xtggrid._actnumsv, atol=0.02)
    assert_allclose(grdecl_grid.xtgeo_coord(), xtggrid._coordsv, atol=0.02)
    nx, ny, nz = grdecl_grid.dimensions
    assert_allclose(grdecl_grid.xtgeo_zcorn(), xtggrid._zcornsv, atol=0.02)


@given(xtgeo_compatible_grdecl_grids)
def test_to_from_grdeclgrid(grdecl_grid):
    xtggrid = Grid()
    xtggrid._actnumsv = grdecl_grid.xtgeo_actnum()
    xtggrid._coordsv = grdecl_grid.xtgeo_coord()
    xtggrid._zcornsv = grdecl_grid.xtgeo_zcorn()
    nx, ny, nz = grdecl_grid.dimensions
    xtggrid._ncol = nx
    xtggrid._nrow = ny
    xtggrid._nlay = nz
    xtggrid._xtgformat = 2

    grdecl_grid2 = ggrid.GrdeclGrid.from_xtgeo_grid(xtggrid)
    assert_allclose(grdecl_grid2.xtgeo_actnum(), xtggrid._actnumsv, atol=0.02)
    assert_allclose(grdecl_grid2.xtgeo_coord(), xtggrid._coordsv, atol=0.02)
    assert_allclose(grdecl_grid2.xtgeo_zcorn(), xtggrid._zcornsv, atol=0.02)


@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(xtgeo_compatible_grdecl_grids)
def test_to_from_grdeclgrid_write(tmp_path, caplog, grdecl_grid):
    xtggrid = Grid()
    xtggrid._actnumsv = grdecl_grid.xtgeo_actnum()
    xtggrid._coordsv = grdecl_grid.xtgeo_coord()
    xtggrid._zcornsv = grdecl_grid.xtgeo_zcorn()
    nx, ny, nz = grdecl_grid.dimensions
    xtggrid._ncol = nx
    xtggrid._nrow = ny
    xtggrid._nlay = nz
    xtggrid._xtgformat = 2

    xtggrid.to_file(tmp_path / "xtggrid.grdecl", fformat="grdecl")
    grdecl_grid2 = ggrid.GrdeclGrid.from_file(tmp_path / "xtggrid.grdecl")

    assert_allclose(grdecl_grid.zcorn, grdecl_grid2.zcorn, atol=0.1)


@given(xtgeo_compatible_grdecl_grids)
def test_xtgeo_values_are_c_contiguous(grdecl_grid):
    assert grdecl_grid.xtgeo_coord().flags["C_CONTIGUOUS"]
    assert grdecl_grid.xtgeo_actnum().flags["C_CONTIGUOUS"]
    assert grdecl_grid.xtgeo_zcorn().flags["C_CONTIGUOUS"]


@given(grdecl_grids())
def test_eq_reflexivity(grdecl_grid):
    assert grdecl_grid == grdecl_grid


@given(grdecl_grids(), grdecl_grids())
def test_eq_symmetry(grdecl_grid1, grdecl_grid2):
    if grdecl_grid1 == grdecl_grid2:
        assert grdecl_grid2 == grdecl_grid1


@given(grdecl_grids(), grdecl_grids(), grdecl_grids())
def test_eq_transitivity(grdecl_grid1, grdecl_grid2, grdecl_grid3):
    if grdecl_grid1 == grdecl_grid2 and grdecl_grid2 == grdecl_grid3:
        assert grdecl_grid1 == grdecl_grid3
