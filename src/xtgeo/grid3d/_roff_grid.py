import warnings
from collections import OrderedDict
from dataclasses import dataclass
from itertools import product
from typing import Optional

import numpy as np
import roffio

import xtgeo.cxtgeo._cxtgeo as _cxtgeo


@dataclass
class RoffGrid:
    nx: int
    ny: int
    nz: int
    subgrids: Optional[np.ndarray]
    corner_lines: np.ndarray
    split_enz: Optional[bytes]
    zvals: np.ndarray
    active: np.ndarray

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

    def xtgeo_coord(self):
        offset = (self.xoffset, self.yoffset, self.zoffset)
        scale = (self.xscale, self.yscale, self.zscale)
        coordsv = self.corner_lines.reshape((self.nx + 1, self.ny + 1, 2, 3))
        coordsv = np.flip(coordsv, -2)
        coordsv = coordsv + offset
        coordsv *= scale
        return coordsv.reshape((self.nx + 1, self.ny + 1, 6)).astype(np.float64)

    def xtgeo_actnum(self):
        actnum = self.active.reshape((self.nx, self.ny, self.nz))
        actnum = np.flip(actnum, -1)
        return actnum.astype(np.int32)

    def xtgeo_zcorn(self):
        zcornsv = np.zeros(
            (self.nx + 1) * (self.ny + 1) * (self.nz + 1) * 4, dtype=np.float32
        )
        retval = _cxtgeo.grd3d_roff2xtgeo_splitenz(
            int(self.nz + 1),
            float(self.zoffset),
            float(self.zscale),
            self.split_enz,
            self.zvals,
            zcornsv,
        )
        if retval == 0:
            return zcornsv.reshape((self.nx + 1, self.ny + 1, self.nz + 1, 4))
        elif retval == -1:
            raise ValueError("Unsupported split type in split_enz")
        elif retval == -2:
            raise ValueError("Incorrect size of splitenz")
        elif retval == -3:
            raise ValueError("Incorrect size of zdata")
        elif retval == -4:
            raise ValueError(
                f"Incorrect size of zcorn, found {zcornsv.shape} should be multiple of {4 * self.nz}"
            )
        else:
            raise ValueError(f"Unknown error {retval} occurred")

    def xtgeo_subgrids(self):
        if self.subgrids is None:
            return None
        result = OrderedDict()
        next_ind = 0
        for i, current in enumerate(self.subgrids):
            result[f"subgrid_{i}"] = list(range(next_ind, current + next_ind))
            next_ind += current
        return result

    @staticmethod
    def from_xtgeo_subgrids(xtgeo_subgrids):
        if xtgeo_subgrids is None:
            return None
        subgrids = []
        for key, value in xtgeo_subgrids.items():
            if value != list(range(value[0], value[-1] + 1)):
                raise ValueError(
                    "Cannot convert non-consecutive subgrids to roff format."
                )
            subgrids.append(value[-1] + 1 - value[0])
        return np.array(subgrids, dtype=np.int32)

    @staticmethod
    def from_xtgeo_grid(xtgeo_grid):
        xtgeo_grid._xtgformat2()
        nx, ny, nz = xtgeo_grid.dimensions
        active = xtgeo_grid._actnumsv.reshape((nx, ny, nz))
        active = np.flip(active, -1).ravel().astype(np.bool_)
        corner_lines = xtgeo_grid._coordsv.reshape((nx + 1, ny + 1, 2, 3)) * np.array(
            [1, 1, -1]
        )
        corner_lines = np.flip(corner_lines, -2).ravel().astype(np.float32)
        zvals = xtgeo_grid._zcornsv.reshape((nx + 1, ny + 1, nz + 1, 4))
        zvals = np.flip(zvals, 2).ravel().view(np.float32) * -1
        split_enz = np.repeat(b"\x04", (nx + 1) * (ny + 1) * (nz + 1)).tobytes()
        subgrids = RoffGrid.from_xtgeo_subgrids(xtgeo_grid._subgrids)

        return RoffGrid(nx, ny, nz, subgrids, corner_lines, split_enz, zvals, active)

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

        return RoffGrid(
            **{
                translated: found[tag][key]
                for tag, tag_keys in translate_kws.items()
                for key, translated in tag_keys.items()
            }
        )

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

    def points(self):
        return product(
            range(1, self.nx - 1),
            range(1, self.ny - 1),
            range(1, self.nz - 1),
        )

    def layer_points(self):
        return product(range(1, self.nx - 1), range(1, self.ny - 1))

    def same_geometry(self, other):
        if not isinstance(other, RoffGrid):
            return False

        is_same = True
        for line in self.layer_points():
            for v1, v2 in zip(self.line_vertices(*line), other.line_vertices(*line)):
                is_same = is_same and np.allclose(
                    self.local_to_utm(v1), other.local_to_utm(v2), atol=0.1
                )

        for node in self.points():
            is_same = is_same and np.allclose(
                (other.z_value(node) + other.zoffset) * other.zscale,
                (self.z_value(node) + self.zoffset) * self.zscale,
                atol=0.2,
            )

        return is_same
