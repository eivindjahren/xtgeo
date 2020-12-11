# -*- coding: utf-8 -*-
import hashlib
import os
import pathlib

import pytest

import tests.test_common.test_xtg as tsetup
import xtgeo
import xtgeo.common.sys as xsys

TRAVIS = False
if "TRAVISRUN" in os.environ:
    TRAVIS = True


@pytest.fixture
def topupperreek_surface(testpath):
    return xtgeo.RegularSurface(
        os.path.join(testpath, "surfaces", "reek", "1", "topupperreek.gri")
    )


def test_generic_hash():
    """Testing generic hashlib function."""
    ahash = xsys.generic_hash("ABCDEF")
    assert ahash == "8827a41122a5028b9808c7bf84b9fcf6"

    ahash = xsys.generic_hash("ABCDEF", hashmethod="sha256")
    assert ahash == "e9c0f8b575cbfcb42ab3b78ecc87efa3b011d9a5d10b09fa4e96f240bf6a82f5"

    ahash = xsys.generic_hash("ABCDEF", hashmethod="blake2b")
    assert ahash[0:12] == "0bb3eb1511cb"

    with pytest.raises(KeyError):
        ahash = xsys.generic_hash("ABCDEF", hashmethod="invalid")

    # pass a hashlib function
    ahash = xsys.generic_hash("ABCDEF", hashmethod=hashlib.sha224)
    assert ahash == "fd6639af1cc457b72148d78e90df45df4d344ca3b66fa44598148ce4"


def test_resolve_alias_md5(topupperreek_surface):
    """Testing resolving file alias function."""
    surf = topupperreek_surface
    md5hash = surf.generate_hash("md5")

    mname = xtgeo._XTGeoFile("whatever/$md5sum.gri", obj=surf)
    assert str(mname.file) == f"whatever/{md5hash}.gri"

    mname = xtgeo._XTGeoFile(pathlib.Path("whatever/$md5sum.gri"), obj=surf)
    assert str(mname.file) == f"whatever/{md5hash}.gri"


def test_resolve_alias_random(topupperreek_surface):
    surf = topupperreek_surface
    mname = xtgeo._XTGeoFile("whatever/$random.gri", obj=surf)
    assert len(str(mname.file)) == 45


def test_resolve_alias_fmu_v1(topupperreek_surface):
    surf = topupperreek_surface

    surf.metadata.opt.shortname = "topValysar"
    surf.metadata.opt.description = "Depth surface"

    mname = xtgeo._XTGeoFile(pathlib.Path("whatever/$fmu-v1.gri"), obj=surf)
    assert str(mname.file) == "whatever/topvalysar--depth_surface.gri"


@tsetup.skipifmac
@tsetup.skipifwindows
def test_xtgeocfile_reek_egrid(reek_egrid_file):
    assert isinstance(reek_egrid_file, xtgeo._XTGeoFile)
    assert isinstance(reek_egrid_file._file, pathlib.Path)

    assert reek_egrid_file._memstream is False
    assert reek_egrid_file._mode == "rb"
    assert reek_egrid_file._delete_after is False
    assert pathlib.Path(reek_egrid_file.name) == reek_egrid_file._file.resolve()
    assert reek_egrid_file.check_folder()

    assert "Swig" in str(reek_egrid_file.get_cfhandle())
    assert reek_egrid_file.cfclose() is True

    stem, suff = reek_egrid_file.splitext(lower=False)
    assert stem == "REEK"
    assert suff == "EGRID"


@tsetup.skipifmac
@tsetup.skipifwindows
def test_xtgeocfile_reek_folder(reek_egrid_file):
    reek_folder = xtgeo._XTGeoFile(reek_egrid_file._file.parent)

    assert reek_folder.exists()
    assert reek_folder.check_folder()


@tsetup.skipifmac
@tsetup.skipifwindows
def test_xtgeocfile_no_exists(testpath):
    nonexistant_file = xtgeo._XTGeoFile(
        os.path.join(testpath, "3dgrids", "reek", "NOSUCH.EGRID")
    )
    nonexistant_folder = xtgeo._XTGeoFile(
        os.path.join(testpath, "3dgrids", "noreek", "NOSUCH.EGRID")
    )

    assert pathlib.Path(nonexistant_file.name) == nonexistant_file._file.resolve()
    assert pathlib.Path(nonexistant_folder.name) == nonexistant_folder._file.resolve()

    assert not nonexistant_file.exists()
    assert not nonexistant_folder.exists()

    assert not nonexistant_file.check_file()
    assert not nonexistant_folder.check_folder()

    with pytest.raises(OSError):
        nonexistant_folder.check_folder(raiseerror=OSError)

    with pytest.raises(OSError):
        nonexistant_file.check_file(raiseerror=OSError)


@tsetup.skipifmac
@tsetup.skipifwindows
def test_xtgeocfile_fhandle(reek_egrid_file):
    """Test in particular C handle SWIG system"""

    chandle1 = reek_egrid_file.get_cfhandle()
    chandle2 = reek_egrid_file.get_cfhandle()
    assert reek_egrid_file._cfhandlecount == 2
    assert chandle1 == chandle2
    assert reek_egrid_file.cfclose() is False
    assert reek_egrid_file.cfclose() is True

    # try to close a cfhandle that does not exist
    with pytest.raises(RuntimeError):
        reek_egrid_file.cfclose()
