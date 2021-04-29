import io

import hypothesis.strategies as st
import numpy as np
from hypothesis import HealthCheck, given
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from xtgeo.grid3d import Grid
from xtgeo.grid3d._roff_grid import RoffGrid

dimensions = st.tuples(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=10),
)

finites = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=32
)


@st.composite
def subgrids(draw, nz):
    if draw(st.booleans()):
        return None

    result = []
    res_sum = 0
    while res_sum < nz:
        sublayers = draw(st.integers(min_value=1, max_value=nz - res_sum))
        result.append(sublayers)
        res_sum += sublayers
    return np.array(result, dtype=np.int32)


@st.composite
def roff_grids(draw, dim=dimensions):
    dims = draw(dim)
    corner_size = (dims[0] + 1) * (dims[1] + 1) * 6
    corner_lines = draw(
        arrays(
            shape=corner_size,
            dtype=np.float32,
            elements=finites,
        )
    )
    num_nodes = (dims[0] + 1) * (dims[1] + 1) * (dims[2] + 1)
    split_enz = draw(
        # st.one_of(
        #    st.just(None),
        arrays(shape=num_nodes, dtype=np.int8, elements=st.sampled_from([1, 4])),
        # )
    ).tobytes()
    if split_enz is not None:
        numz = sum(split_enz)
    else:
        numz = num_nodes
    zvals = draw(arrays(shape=int(numz), dtype=np.float32, elements=finites))
    active = draw(
        arrays(
            shape=dims[0] * dims[1] * dims[2], dtype=np.bool_, elements=st.just(True)
        )
    )

    subs = draw(subgrids(dims[2]))

    rest = draw(st.tuples(*([finites] * 6)))
    return RoffGrid(*dims, subs, corner_lines, split_enz, zvals, active, *rest)


def create_xtgeo_grid(*args, **kwargs):
    grid = Grid()
    grid.create_box(*args, **kwargs)
    return grid


increments = st.floats(min_value=1.0, max_value=100.0)


xtgeo_grids = st.builds(
    create_xtgeo_grid,
    dimension=dimensions,
    origin=st.tuples(finites, finites, finites),
    increment=st.tuples(increments, increments, increments),
    rotation=st.floats(min_value=0.0, max_value=90),
)


@given(roff_grids())
def test_roff_grid_read_write(rgrid):
    buff = io.BytesIO()
    rgrid.to_file(buff)

    buff.seek(0)
    assert RoffGrid.from_file(buff) == rgrid


@given(xtgeo_grids)
def test_to_from_xtgeogrid_format2(xtggrid):
    xtggrid._xtgformat2()
    roff_grid = RoffGrid.from_xtgeo_grid(xtggrid)

    assert_allclose(roff_grid.xtgeo_actnum(), xtggrid._actnumsv, atol=0.02)
    assert_allclose(roff_grid.xtgeo_coord(), xtggrid._coordsv, atol=0.02)
    assert_allclose(roff_grid.xtgeo_zcorn(), xtggrid._zcornsv, atol=0.02)
    assert roff_grid.xtgeo_subgrids() == xtggrid._subgrids


@given(xtgeo_grids)
def test_to_from_xtgeogrid_format1(xtggrid):
    xtggrid._xtgformat1()
    roff_grid = RoffGrid.from_xtgeo_grid(xtggrid)

    xtggrid._xtgformat2()
    assert_allclose(roff_grid.xtgeo_actnum(), xtggrid._actnumsv, atol=0.02)
    assert_allclose(roff_grid.xtgeo_coord(), xtggrid._coordsv, atol=0.02)
    assert_allclose(roff_grid.xtgeo_zcorn(), xtggrid._zcornsv, atol=0.02)
    assert roff_grid.xtgeo_subgrids() == xtggrid._subgrids


@given(roff_grids())
def test_to_from_roffgrid(roff_grid):
    xtggrid = Grid()
    xtggrid._actnumsv = roff_grid.xtgeo_actnum()
    xtggrid._coordsv = roff_grid.xtgeo_coord()
    xtggrid._zcornsv = roff_grid.xtgeo_zcorn()
    xtggrid._subgrids = roff_grid.xtgeo_subgrids()
    xtggrid._ncol = roff_grid.nx
    xtggrid._nrow = roff_grid.ny
    xtggrid._nlay = roff_grid.nz

    roffgrid2 = RoffGrid.from_xtgeo_grid(xtggrid)
    assert roffgrid2.same_geometry(roff_grid)
    assert np.array_equal(roffgrid2.subgrids, roff_grid.subgrids)
