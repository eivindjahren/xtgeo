import io
import logging
import struct
from dataclasses import dataclass
from itertools import product
from typing import List, Optional

import hypothesis.strategies as st
import numpy as np
import roffio
from hypothesis import HealthCheck, given, note, settings
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo.grid3d import Grid


@dataclass
class RoffGrid:
    nx: int
    ny: int
    nz: int
    subgrids: Optional[np.ndarray]
    corner_lines: np.ndarray
    split_enz: Optional[np.ndarray]
    zvals: np.ndarray
    active: np.ndarray

    xoffset: float = 0.0
    yoffset: float = 0.0
    zoffset: float = 0.0
    xscale: float = 1.0
    yscale: float = 1.0
    zscale: float = -1.0

    def __repr__(self):
        return (
            f"RoffGrid(nx={self.nx}, ny={self.ny}, nz={self.nz},"
            f" subgrids={self.subgrids}, corner_lines={self.corner_lines},"
            f" split_enz={self.split_enz}, zvals={self.zvals},"
            f" active={self.active}, xoffset={self.xoffset}, yoffset={self.yoffset},"
            f" zoffset={self.zoffset}, xscale={self.xscale}, yscale={self.yscale},"
            f" zscale={self.zscale})"
        )

    def __str__(self):
        return (
            f"RoffGrid(nx={self.nx}, ny={self.ny}, nz={self.nz},"
            f" subgrids={self.subgrids}, corner_lines={self.corner_lines},"
            f" split_enz={self.split_enz}, zvals={self.zvals},"
            f" active={self.active}, xoffset={self.xoffset}, yoffset={self.yoffset},"
            f" zoffset={self.zoffset}, xscale={self.xscale}, yscale={self.yscale},"
            f" zscale={self.zscale})"
        )

    def __eq__(self, other):
        if not isinstance(other, RoffGrid):
            return False
        return (
            self.nx == other.nx
            and self.ny == other.ny
            and self.nz == other.nz
            and self.xoffset == other.xoffset
            and self.yoffset == other.yoffset
            and self.zoffset == other.zoffset
            and self.xscale == other.xscale
            and self.yscale == other.yscale
            and self.zscale == other.zscale
            and (
                (self.subgrids is None and other.subgrids is None)
                or (np.array_equal(self.subgrids, other.subgrids))
            )
            and (
                (self.split_enz is None and other.split_enz is None)
                or (np.array_equal(self.split_enz, other.split_enz))
            )
            and np.array_equal(self.zvals, other.zvals)
            and np.array_equal(self.corner_lines, other.corner_lines)
            and np.array_equal(self.active, other.active)
        )

    @property
    def num_nodes(self):
        return (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

    def _create_lookup(self):
        if not hasattr(self, "_lookup"):
            n = self.num_nodes
            self._lookup = np.zeros(n + 1, dtype=np.int32)
            for i in range(n):
                if self.split_enz is not None:
                    self._lookup[i + 1] = self.split_enz[i] + self._lookup[i]
                else:
                    self._lookup[i + 1] = 1 + self._lookup[i]

    def z_value(self, node):
        """
        To every point along a corner line there are up to
        eight cells attached. This returns the z-value for
        each of those 8 cells, in the following order:

        * below_sw
        * below_se
        * below_nw
        * below_ne
        * above_sw
        * above_se
        * above_nw
        * above_ne
        """
        i, j, k = node
        self._create_lookup()

        node_number = i * (self.ny + 1) * (self.nz + 1) + j * (self.nz + 1) + k
        pos = self._lookup[node_number]
        split = self._lookup[node_number + 1] - self._lookup[node_number]

        if split == 1:
            return np.array([self.zvals[pos]] * 8)
        elif split == 2:
            return np.array([self.zvals[pos]] * 4 + [self.zvals[pos + 1]] * 4)
        elif split == 4:
            return np.array(
                [
                    self.zvals[pos],
                    self.zvals[pos + 1],
                    self.zvals[pos + 2],
                    self.zvals[pos + 3],
                ]
                * 2
            )
        elif split == 8:
            return np.array(
                [
                    self.zvals[pos],
                    self.zvals[pos + 1],
                    self.zvals[pos + 2],
                    self.zvals[pos + 3],
                    self.zvals[pos + 4],
                    self.zvals[pos + 5],
                    self.zvals[pos + 6],
                    self.zvals[pos + 7],
                ]
            )
        else:
            raise ValueError("Only split types 1, 2, 4 and 8 are supported!")

    def local_to_utm(self, coordinates):
        (x, y, z) = coordinates
        x_utm = (x + self.xoffset) * self.xscale
        y_utm = (y + self.yoffset) * self.yscale
        tvd = (z + self.zoffset) * self.zscale
        return (x_utm, y_utm, tvd)

    def line_vertices(self, i, j):
        pos = 6 * (i * (self.ny + 1) + j)
        x_bot = self.corner_lines[pos]
        y_bot = self.corner_lines[pos + 1]
        z_bot = self.corner_lines[pos + 2]
        x_top = self.corner_lines[pos + 3]
        y_top = self.corner_lines[pos + 4]
        z_top = self.corner_lines[pos + 5]

        return ((x_bot, y_bot, z_bot), (x_top, y_top, z_top))

    def xtgeo_coord(self):
        offset = (self.xoffset, self.yoffset, self.zoffset)
        scale = (self.xscale, self.yscale, self.zscale)
        coordsv = self.corner_lines.reshape((self.nx + 1, self.ny + 1, 2, 3))
        coordsv = np.flip(coordsv, -2)
        coordsv = coordsv + offset
        coordsv *= scale
        return coordsv.reshape((self.nx + 1, self.ny + 1, 6))

    def xtgeo_actnum(self):
        actnum = self.active.reshape((self.nx, self.ny, self.nz))
        actnum = np.array(actnum, dtype=np.int32)
        actnum = np.flip(actnum, -1)
        return actnum

    def xtgeo_zcorn(self):
        zcornsv = np.zeros((self.nx + 1) * (self.ny + 1) * (self.nz + 1) * 4)
        split_enz = np.array(self.split_enz, dtype=np.int32)
        _cxtgeo.grd3d_roff2xtgeo_zcorn_np_arrs(
            int(self.nx),
            int(self.ny),
            int(self.nz),
            float(self.xoffset),
            float(self.yoffset),
            float(self.zoffset),
            float(self.xscale),
            float(self.yscale),
            float(self.zscale),
            split_enz,
            self.zvals,
            zcornsv,
        )
        return zcornsv.reshape((self.nx + 1, self.ny + 1, self.nz + 1, 4))

    def is_active(self, node):
        i, j, k = node
        return self.active[i * self.ny * self.nz + j * self.nz + k]

    def to_file(self, filelike):
        data = {
            "filedata": {"filetype": "grid"},
            "dimensions": {"nX": self.nx, "nY": self.ny, "nZ": self.nz},
            "translate": {
                "xoffset": np.float32(self.xoffset),
                "yoffset": np.float32(self.yoffset),
                "zoffset": np.float32(self.zoffset),
            },
            "scale": {
                "xscale": np.float32(self.xscale),
                "yscale": np.float32(self.yscale),
                "zscale": np.float32(self.zscale),
            },
            "cornerLines": {"data": self.corner_lines},
            "zvalues": {"data": self.zvals},
            "active": {"data": self.active},
        }
        if self.subgrids is not None:
            data["subgrids"] = {"nLayers": self.subgrids}
        if self.split_enz is not None:
            data["zvalues"]["splitEnz"] = self.split_enz
        roffio.write(filelike, data)

    @staticmethod
    def from_file(filelike):
        translate_kws = {
            "dimensions": {"nX": "nx", "nY": "ny", "nZ": "nz"},
            "translate": {
                "xoffset": "xoffset",
                "yoffset": "yoffset",
                "zoffset": "zoffset",
            },
            "scale": {
                "xscale": "xscale",
                "yscale": "yscale",
                "zscale": "zscale",
            },
            "cornerLines": {"data": "corner_lines"},
            "zvalues": {"splitEnz": "split_enz", "data": "zvals"},
            "active": {"data": "active"},
            "subgrids": {"nLayers": "subgrids"},
        }
        found = {
            tag_name: {key_name: None for key_name in tag_keys.keys()}
            for tag_name, tag_keys in translate_kws.items()
        }
        found["filedata"] = {"filetype": None}
        with roffio.lazy_read(filelike) as tag_generator:
            for tag, keys in tag_generator:
                if tag in found:
                    for name, k_value in keys:
                        if name in found[tag]:
                            if found[tag][name] is not None:
                                raise ValueError(
                                    f"Multiple tag, tagkey pair {tag}, {name}"
                                    " in {filelike}"
                                )
                            found[tag][name] = k_value

        filetype = found["filedata"]["filetype"]
        if filetype is None:
            raise ValueError(
                f"File {filelike} did not contain filetype key in filedata tag"
            )
        if filetype != "grid":
            raise ValueError(
                f"File {filelike} did not have filetype set to grid, found {filetype}"
            )

        if found["zvalues"]["splitEnz"] is not None:
            val = found["zvalues"]["splitEnz"]
            found["zvalues"]["splitEnz"] = np.ndarray(
                len(val),
                np.uint8,
                val,
            )

        return RoffGrid(
            **{
                translated: found[tag][key]
                for tag, tag_keys in translate_kws.items()
                for key, translated in tag_keys.items()
            }
        )


dimensions = st.tuples(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=10),
)

finites = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=32
)


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
    size = (dims[0] + 1) * (dims[1] + 1) * (dims[2] + 1)
    split_enz = draw(
        # st.one_of(
        #    st.just(None),
        arrays(shape=size, dtype=np.int8, elements=st.sampled_from([1, 4])),
        # )
    )
    if split_enz is not None:
        numz = sum(split_enz)
    else:
        numz = size
    zvals = draw(arrays(shape=int(numz), dtype=np.float32, elements=finites))
    active = draw(
        arrays(
            shape=dims[0] * dims[1] * dims[2], dtype=np.bool_, elements=st.just(True)
        )
    )

    rest = draw(st.tuples(*([finites] * 6)))
    return RoffGrid(*dims, None, corner_lines, split_enz, zvals, active, *rest)


@given(roff_grids())
def test_roff_grid_read_write(rgrid):
    buff = io.BytesIO()
    rgrid.to_file(buff)

    buff.seek(0)
    assert RoffGrid.from_file(buff) == rgrid


def interior_points(roff_grid):
    return product(
        range(1, roff_grid.nx - 1),
        range(1, roff_grid.ny - 1),
        range(1, roff_grid.nz - 1),
    )


def interior_layer_points(roff_grid):
    return product(range(1, roff_grid.nx - 1), range(1, roff_grid.ny - 1))


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(roff_grids())
def test_roff_grid_read_with_xtgeo(tmp_path, caplog, rgrid):
    caplog.set_level(logging.CRITICAL)
    path1 = tmp_path / "grid.roff"
    with path1.open("wb") as fil:
        rgrid.to_file(fil)

    xtggrid = Grid().from_file(path1)

    path2 = tmp_path / "grid2.roff"
    xtggrid.to_file(path2)
    read_grid = RoffGrid.from_file(str(path2))

    for line in interior_layer_points(rgrid):
        for v1, v2 in zip(read_grid.line_vertices(*line), rgrid.line_vertices(*line)):
            assert_allclose(
                read_grid.local_to_utm(v1), rgrid.local_to_utm(v2), atol=0.1
            )

    for node in interior_points(rgrid):
        assert_allclose(
            (read_grid.z_value(node) + read_grid.zoffset) * read_grid.zscale,
            (rgrid.z_value(node) + rgrid.zoffset) * rgrid.zscale,
            atol=0.2,
        )


def create_grid(*args, **kwargs):
    grid = Grid()
    grid.create_box(*args, **kwargs)
    return grid


indecies = st.integers(min_value=4, max_value=12)
coordinates = st.floats(min_value=-100.0, max_value=100.0)
increments = st.floats(min_value=1.0, max_value=100.0)
dimensions = st.tuples(indecies, indecies, indecies)


xtgeo_grids = st.builds(
    create_grid,
    dimension=dimensions,
    origin=st.tuples(coordinates, coordinates, coordinates),
    increment=st.tuples(increments, increments, increments),
    rotation=st.floats(min_value=0.0, max_value=90),
)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(xtgeo_grids)
def test_roff_grid_write_with_xtgeo(tmp_path, caplog, xtggrid):
    caplog.set_level(logging.CRITICAL)
    grid_path = tmp_path / "grid.roff"
    xtggrid.to_file(grid_path)
    read_grid = RoffGrid.from_file(str(grid_path))

    for line in interior_layer_points(read_grid):
        v1, v2 = read_grid.line_vertices(*line)
        utm_v1 = read_grid.local_to_utm(v1)
        utm_v2 = read_grid.local_to_utm(v2)
        assert_allclose(utm_v2 + utm_v1, xtggrid._coordsv[line[0]][line[1]], atol=0.001)

    for node in interior_points(read_grid):
        xtgnode = (node[0], node[1], read_grid.nz - node[2])
        xtgeo_zvalues = xtggrid._zcornsv[xtgnode]
        # xtgeo uses 4 z values per node, with no gap between layers
        # the roff format can have gap between layers so has 8 values
        # therefore we just double up the values here.
        xtgeo_zvalues = np.concatenate((xtgeo_zvalues, xtgeo_zvalues))
        assert_allclose(
            (read_grid.z_value(node) + read_grid.zoffset) * read_grid.zscale,
            xtgeo_zvalues,
            atol=0.2,
        )

    assert_allclose(read_grid.xtgeo_coord(), xtggrid._coordsv, atol=0.001)
    assert_allclose(read_grid.xtgeo_zcorn(), xtggrid._zcornsv, atol=0.001)
    assert np.array_equal(read_grid.xtgeo_actnum(), xtggrid._actnumsv)
    assert np.array_equal(read_grid.xtgeo_actnum(), xtggrid._actnumsv)
