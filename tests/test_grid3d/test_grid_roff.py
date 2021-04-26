import io
from dataclasses import dataclass
from typing import List, Optional

import hypothesis.strategies as st
import numpy as np
import roffio
from hypothesis import given, note
from hypothesis.extra.numpy import arrays


@dataclass
class RoffGrid:
    nx: int
    ny: int
    nz: int
    subgrids: Optional[List[int]]
    corner_lines: List[float]
    split_enz: Optional[List[int]]
    zvals: List[float]
    active: List[bool]

    xoffset: float = 0.0
    yoffset: float = 0.0
    zoffset: float = 0.0
    xscale: float = 1.0
    yscale: float = 1.0
    zscale: float = -1.0

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
            self._lookup = np.zeros(n + 1)
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
            return [self.zvals[pos]] * 8
        elif split == 2:
            return [self.zvals[pos]] * 4 + [self.zvals[pos + 1]] * 4
        elif split == 4:
            return [
                self.zvals[pos],
                self.zvals[pos + 1],
                self.zvals[pos + 2],
                self.zvals[pos + 3],
            ] * 2
        elif split == 8:
            return [
                self.zvals[pos],
                self.zvals[pos + 1],
                self.zvals[pos + 2],
                self.zvals[pos + 3],
                self.zvals[pos + 4],
                self.zvals[pos + 5],
                self.zvals[pos + 6],
                self.zvals[pos + 7],
            ]
        else:
            raise ValueError("Only split types 1, 2, 4 and 8 are supported!")

    def local_to_utm(self, coordinates):
        (x, y, z) = coordinates
        x_utm = (x + self.xoffset) * self.xscale
        y_utm = (y + self.yoffset) * self.yscale
        tvd = (z + self.zoffset) * self.zscale
        return (x_utm, y_utm, tvd)

    def is_active(self, node):
        i, j, k = node
        return self.active[i * self.ny * self.nz + j * self.nz + k]

    def to_file(self, filelike):
        data = {
            "filedata": {"filetype": "grid"},
            "dimensions": {"nX": self.nx, "nY": self.ny, "nZ": self.nz},
            "translate": {
                "xoffset": self.xoffset,
                "yoffset": self.yoffset,
                "zoffset": self.zoffset,
            },
            "scale": {
                "xscale": self.xscale,
                "yscale": self.yscale,
                "zscale": self.zscale,
            },
            "cornerLines": {"data": self.corner_lines},
            "zvalues": {"data": self.zvals},
            "active": {"data": self.active},
        }
        if self.subgrids is not None:
            data["subgrids"] = {"nLayers": self.subgrids}
        if self.split_enz is not None:
            data["zvalues"]["splitEnz"] = self.split_enz
        roffio.write(
            filelike,
            data,
        )

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
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=0, max_value=10),
)

finites = st.floats(allow_nan=False, allow_infinity=False, width=32)


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
        st.one_of(
            st.just(None),
            arrays(shape=size, dtype=np.int8, elements=st.sampled_from([1, 4])),
        )
    )
    if split_enz is not None:
        numz = sum(split_enz)
    else:
        numz = size
    zvals = draw(arrays(shape=int(numz), dtype=np.float32, elements=finites))
    active = draw(arrays(shape=size, dtype=np.bool_, elements=st.booleans()))

    rest = draw(st.tuples(*([finites] * 6)))
    return RoffGrid(*dims, None, corner_lines, split_enz, zvals, active, *rest)


@given(roff_grids())
def test_roff_grid_read_write(rgrid):
    buff = io.BytesIO()
    rgrid.to_file(buff)

    buff.seek(0)

    note(RoffGrid.from_file(buff))
    buff.seek(0)
    assert RoffGrid.from_file(buff) == rgrid
