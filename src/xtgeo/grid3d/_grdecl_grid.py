from dataclasses import astuple, dataclass, fields
from enum import Enum, auto, unique
from typing import Optional, Tuple

import numpy as np

from ._grdecl_format import match_keyword, open_grdecl


@unique
class Units(Enum):
    METRES = auto()
    FEET = auto()
    CM = auto()

    def to_grdecl(self):
        return self.name

    @classmethod
    def from_grdecl(cls, unit_string):
        if match_keyword(unit_string, "METRES"):
            return cls.METRES
        if match_keyword(unit_string, "FEET"):
            return cls.FEET
        if match_keyword(unit_string, "CM"):
            return cls.CM


@unique
class GridRelative(Enum):
    MAP = auto()
    ORIGIN = auto()

    def to_grdecl(self):
        if self == GridRelative.MAP:
            return "MAP"
        else:
            return ""

    @classmethod
    def from_grdecl(cls, unit_string):
        if match_keyword(unit_string, "MAP"):
            return cls.MAP
        else:
            return cls.ORIGIN


@unique
class Order(Enum):
    INCREASING = auto()
    DECREASING = auto()

    def to_grdecl(self):
        return str(self.name)[0:3]

    @classmethod
    def from_grdecl(cls, order_string):
        if match_keyword(order_string, "INC"):
            return cls.INCREASING
        if match_keyword(order_string, "DEC"):
            return cls.DECREASING


@unique
class Orientation(Enum):
    UP = auto()
    DOWN = auto()

    def to_grdecl(self):
        return self.name

    @classmethod
    def from_grdecl(cls, orientation_string):
        if match_keyword(orientation_string, "UP"):
            return cls.UP
        if match_keyword(orientation_string, "DOWN"):
            return cls.DOWN


@unique
class Handedness(Enum):
    LEFT = auto()
    RIGHT = auto()

    def to_grdecl(self):
        return self.name

    @classmethod
    def from_grdecl(cls, orientation_string):
        if match_keyword(orientation_string, "LEFT"):
            return cls.LEFT
        if match_keyword(orientation_string, "RIGHT"):
            return cls.RIGHT


@unique
class CoordinateType(Enum):
    CARTESIAN = auto()
    CYLINDRICAL = auto()

    def to_grdecl(self):
        if self == CoordinateType.CARTESIAN:
            return "F"
        else:
            return "T"

    @classmethod
    def from_grdecl(cls, coord_string):
        if match_keyword(coord_string, "F"):
            return cls.CARTESIAN
        if match_keyword(coord_string, "T"):
            return cls.CYLINDRICAL


@dataclass
class GrdeclKeyword:
    def to_grdecl(self):
        return [value.to_grdecl() for value in astuple(self)]

    @classmethod
    def from_grdecl(cls, values):
        object_types = [f.type for f in fields(cls)]
        return cls(*[typ.from_grdecl(val) for val, typ in zip(values, object_types)])


@dataclass
class MapAxes(GrdeclKeyword):
    y_line: Tuple[float, float] = (0.0, 1.0)
    origin: Tuple[float, float] = (0.0, 0.0)
    x_line: Tuple[float, float] = (1.0, 0.0)

    def to_grdecl(self):
        return list(self.y_line) + list(self.origin) + list(self.x_line)

    @staticmethod
    def from_grdecl(values):
        if len(values) != 6:
            raise ValueError("MAPAXES must contain 6 values")
        return MapAxes(
            (values[0], values[1]),
            (values[2], values[3]),
            (values[4], values[5]),
        )


@dataclass
class GdOrient(GrdeclKeyword):
    i_order: Order = Order.INCREASING
    j_order: Order = Order.INCREASING
    k_order: Order = Order.INCREASING
    z_direction: Orientation = Orientation.DOWN
    handedness: Handedness = Handedness.RIGHT


@dataclass
class SpecGrid(GrdeclKeyword):
    ndivix: int = 1
    ndiviy: int = 1
    ndiviz: int = 1
    numres: int = 1
    coordinate_type: CoordinateType = CoordinateType.CARTESIAN

    def to_grdecl(self):
        return [
            self.ndivix,
            self.ndiviy,
            self.ndiviz,
            self.numres,
            self.coordinate_type.to_grdecl(),
        ]

    @classmethod
    def from_grdecl(cls, values):
        if len(values) != 5:
            raise ValueError("SPECGRID must contain 5 values")
        return cls(*values[0:-1], CoordinateType.from_grdecl(values[-1]))


@dataclass
class GridUnit(GrdeclKeyword):
    unit: Units = Units.METRES
    grid_relative: GridRelative = GridRelative.ORIGIN

    @classmethod
    def from_grdecl(cls, values):
        if len(values) == 1:
            return cls(Units.from_grdecl(values[0]))
        if len(values) == 2:
            return cls(
                Units.from_grdecl(values[0]),
                GridRelative.MAP
                if match_keyword(values[1], "MAP")
                else GridRelative.ORIGIN,
            )
        raise ValueError("GridUnit record must contain either 1 or 2 values")


@dataclass
class GrdeclGrid:
    coord: np.ndarray
    zcorn: np.ndarray
    specgrid: SpecGrid
    actnum: Optional[np.ndarray] = None
    mapaxes: Optional[MapAxes] = None
    mapunits: Optional[Units] = None
    gridunit: Optional[GridUnit] = None
    gdorient: Optional[GdOrient] = None

    def __eq__(self, other):
        if not isinstance(other, GrdeclGrid):
            return False
        return (
            self.specgrid == other.specgrid
            and self.mapaxes == other.mapaxes
            and self.mapunits == other.mapunits
            and self.gridunit == other.gridunit
            and self.gdorient == other.gdorient
            and (
                (self.actnum is None and other.actnum is None)
                or (np.array_equal(self.actnum, other.actnum))
            )
            and np.array_equal(self.coord, other.coord)
            and np.array_equal(self.zcorn, other.zcorn)
        )

    @staticmethod
    def from_file(filename):
        keyword_factories = {
            "COORD": lambda x: np.array(x, dtype=np.float32),
            "ZCORN": lambda x: np.array(x, dtype=np.float32),
            "ACTNUM": lambda x: np.array(x, dtype=np.int32),
            "MAPAXES": MapAxes.from_grdecl,
            "MAPUNITS": Units.from_grdecl,
            "GRIDUNIT": GridUnit.from_grdecl,
            "SPECGRID": SpecGrid.from_grdecl,
            "GDORIENT": GdOrient.from_grdecl,
        }
        results = {}
        with open_grdecl(
            filename,
            keyword_factories.keys(),
            max_len=8,
            ignore=["ECHO", "NOECHO"],
        ) as keyword_generator:
            for kw, values in keyword_generator:
                if len(results) == len(keyword_factories):
                    break
                if kw in results:
                    raise ValueError(f"Duplicate keyword {kw} in {filename}")
                factory = keyword_factories[kw]
                results[kw.lower()] = factory(values)
        return GrdeclGrid(**results)

    def to_file(self, filename):
        with open(filename, "w") as filestream:
            keywords = {
                "COORD": self.coord,
                "ZCORN": self.zcorn,
                "ACTNUM": self.actnum,
                "SPECGRID": self.specgrid.to_grdecl(),
                "MAPAXES": self.mapaxes.to_grdecl() if self.mapaxes else None,
                "MAPUNITS": [self.mapunits.name] if self.mapunits else None,
                "GRIDUNIT": self.gridunit.to_grdecl() if self.gridunit else None,
                "GDORIENT": self.gdorient.to_grdecl() if self.gdorient else None,
            }
            for kw, values in keywords.items():
                if values is None:
                    continue
                filestream.write(kw)
                if values is not None:
                    filestream.write("\n")
                    for value in values:
                        filestream.write(" ")
                        filestream.write(str(value))
                filestream.write("\n /\n")

    @property
    def dimensions(self):
        return (self.specgrid.ndivix, self.specgrid.ndiviy, self.specgrid.ndiviz)

    def _check_xtgeo_compatible(self):
        if self.gridunit and self.gridunit.grid_relative == GridRelative.MAP:
            raise NotImplementedError(
                "Xtgeo does not currently support"
                " translation of Map relative grdecl grids"
            )
        if (
            self.specgrid
            and self.specgrid.coordinate_type == CoordinateType.CYLINDRICAL
        ):
            raise NotImplementedError(
                "Xtgeo does not currently support cylindrical coordinate systems"
            )
        if self.gdorient and self.gdorient != GdOrient():
            raise NotImplementedError(
                "Xtgeo only supports default Grid orientation grdecl files"
            )

    def xtgeo_coord(self):
        self._check_xtgeo_compatible()
        nx, ny, nz = self.dimensions
        return self.coord.reshape((nx + 1, ny + 1, 6))

    def xtgeo_actnum(self):
        self._check_xtgeo_compatible()
        if self.actnum is None:
            return np.ones(shape=self.dimensions, dtype=np.int32)
        return self.actnum.reshape(self.dimensions)

    def xtgeo_zcorn(self):
        self._check_xtgeo_compatible()
        nx, ny, nz = self.dimensions
        zcorn = self.zcorn.reshape((nz, 2, 2, ny, 2, nx))
        for i in range(3, 6):
            zcorn = np.swapaxes(zcorn, i, 5 - i)

        if not np.allclose(
            zcorn[:, :, :, :, 0, : nz - 1],
            zcorn[:, :, :, :, 1, 1:],
        ):
            raise ValueError("xtgeo does not support grids with horizontal split")
        result = np.zeros((nx + 1, ny + 1, nz + 1, 4), dtype=np.float32)

        # xtgeo uses 4 z values per i,j,k to mean the 4 z values of
        # adjacent cells for the cornerline at position i,j,k assuming
        # no difference in z values between upper and lower cells. In
        # the order nw,ne,sw,se.

        # In grdecl, there are 8 zvalues per i,j,k meaning the z values
        # of each corner for the cell at i,j,k. In
        # the order "left" before "right" (2nd dimension), "near" before
        # "far" (4th dimension), "upper" before "bottom" (5th dimension)

        # Set the sw value of cornerline i+1,j+1 to
        # the far right corner of cell i,j to
        result[1:, 1:, 1:, 2] = zcorn[:, 1, :, 1, 0, :]
        result[1:, 1:, 0, 2] = zcorn[:, 1, :, 1, 1, 0]

        # Set the se value of cornerline i,j+1 to
        # the far left corner of cell i,j
        result[:nx, 1:, 1:, 3] = zcorn[:, 0, :, 1, 0, :]
        result[:nx, 1:, 0, 3] = zcorn[:, 0, :, 1, 1, 0]

        # Set the ne value of cornerline i,j to
        # the near left corner of cell i,j
        result[:nx, :ny, 1:, 1] = zcorn[:, 0, :, 0, 0, :]
        result[:nx, :ny, 0, 1] = zcorn[:, 0, :, 0, 1, 0]

        # Set the nw value of cornerline i+1,j to
        # the near right corner of cell i,j
        result[1:, :ny, 1:, 0] = zcorn[:, 1, :, 0, 0, :]
        result[1:, :ny, 0, 0] = zcorn[:, 1, :, 0, 1, 0]

        # For the remaining 6 faces, 4 lines, and 4 corners, they are all
        # special cases and have some insignificant values where we duplicate
        # the value of the adjacent cell.

        # sw->se face
        result[1:nx, 0, :, 2] = result[1:nx, 0, :, 0]
        result[1:nx, 0, :, 3] = result[1:nx, 0, :, 1]

        # vertical sw corner line
        result[0, 0, :, 0] = result[0, 0, :, 1]
        result[0, 0, :, 2] = result[0, 0, :, 1]
        result[0, 0, :, 3] = result[0, 0, :, 1]

        # se->ne face
        result[nx, 1:ny, :, 1] = result[nx, 1:ny, :, 0]
        result[nx, 1:ny, :, 3] = result[nx, 1:ny, :, 2]

        # vertical se corner line
        result[nx, 0, :, 1] = result[nx, 0, :, 0]
        result[nx, 0, :, 2] = result[nx, 0, :, 0]
        result[nx, 0, :, 3] = result[nx, 0, :, 0]

        # nw->ne face
        result[1:nx, ny, :, 0] = result[1:nx, ny, :, 2]
        result[1:nx, ny, :, 1] = result[1:nx, ny, :, 3]

        # vertical nw corner line
        result[0, ny, :, 0] = result[0, ny, :, 3]
        result[0, ny, :, 1] = result[0, ny, :, 3]
        result[0, ny, :, 2] = result[0, ny, :, 3]

        # sw->nw face
        result[0, 1:ny, :, 0] = result[0, 1:ny, :, 1]
        result[0, 1:ny, :, 2] = result[0, 1:ny, :, 3]

        # vertical ne corner line
        result[nx, ny, :, 0] = result[nx, ny, :, 2]
        result[nx, ny, :, 1] = result[nx, ny, :, 2]
        result[nx, ny, :, 3] = result[nx, ny, :, 2]

        return np.ascontiguousarray(result)

    @staticmethod
    def from_xtgeo_grid(xtgeo_grid):
        xtgeo_grid._xtgformat2()

        nx, ny, nz = xtgeo_grid.dimensions
        actnum = xtgeo_grid._actnumsv.ravel()
        if np.all(actnum == 1):
            actnum = None
        coord = xtgeo_grid._coordsv.ravel()
        zcorn = np.zeros((nx, 2, ny, 2, 2, nz))
        xtgeo_zcorn = xtgeo_grid._zcornsv.reshape((nx + 1, ny + 1, nz + 1, 4))

        # This is the reverse operation of that of xtgeo_zcorn,
        # see that function for description of operations.

        # Set the far right corner of cell i,j to
        # the sw value of cornerline i+1,j+1
        zcorn[:, 1, :, 1, 0, :nz] = xtgeo_zcorn[1:, 1:, 1:, 2]
        zcorn[:, 1, :, 1, 1, :] = xtgeo_zcorn[1:, 1:, :nz, 2]

        # Set the far left corner of cell i,j to
        # the se value of cornerline i,j+1
        zcorn[:, 0, :, 1, 0, :nz] = xtgeo_zcorn[:nx, 1:, 1:, 3]
        zcorn[:, 0, :, 1, 1, :] = xtgeo_zcorn[:nx, 1:, :nz, 3]

        # Set the near left corner of cell i,j to
        # the ne value of cornerline i,j
        zcorn[:, 0, :, 0, 0, :nz] = xtgeo_zcorn[:nx, :ny, 1:, 1]
        zcorn[:, 0, :, 0, 1, :] = xtgeo_zcorn[:nx, :ny, :nz, 1]

        # Set the near right corner of cell i,j to
        # the nw value of cornerline i+1,j
        zcorn[:, 1, :, 0, 0, :nz] = xtgeo_zcorn[1:, :ny, 1:, 0]
        zcorn[:, 1, :, 0, 1, :] = xtgeo_zcorn[1:, :ny, :nz, 0]

        for i in range(3, 6):
            zcorn = np.swapaxes(zcorn, i, 5 - i)
        zcorn = zcorn.ravel()

        return GrdeclGrid(
            coord=coord,
            zcorn=zcorn,
            actnum=actnum,
            specgrid=SpecGrid(nx, ny, nz),
        )
