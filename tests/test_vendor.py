"""Engine download: checksum verification and extraction, without the network."""

from __future__ import annotations

import base64
import hashlib
import io
import tarfile

import pytest

from cifarium import vendor


def make_tarball(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def integrity_of(data: bytes) -> str:
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode("ascii")


def package_for(tarball: bytes, *, name: str = "demo", files: tuple[str, ...] = ("lib.js",)):
    return vendor.NpmPackage(name=name, version="1.0.0", integrity=integrity_of(tarball), files=files)


class FakeRegistry:
    """Stands in for urlopen: serves tarballs from memory and counts requests."""

    def __init__(self, tarballs: dict[str, bytes]):
        self.tarballs = tarballs
        self.calls: list[str] = []

    def __call__(self, url: str) -> io.BytesIO:
        self.calls.append(url)
        return io.BytesIO(self.tarballs[url])


def test_tarball_urls_match_the_registry():
    transformers, ort = vendor.ENGINE_PACKAGES
    assert transformers.tarball_url == (
        "https://registry.npmjs.org/@huggingface/transformers/-/transformers-4.3.2.tgz"
    )
    assert ort.tarball_url == (
        "https://registry.npmjs.org/onnxruntime-web/-/"
        "onnxruntime-web-1.31.0-dev.20260914-8d85527a0.tgz"
    )


def test_fetch_copies_only_the_listed_files(tmp_path):
    tarball = make_tarball(
        {
            "package/package.json": b"{}",
            "package/dist/lib.js": b"export const answer = 42;",
            "package/dist/other.js": b"not wanted",
        }
    )
    package = package_for(tarball)
    registry = FakeRegistry({package.tarball_url: tarball})

    vendor.fetch_package(package, tmp_path, open_url=registry)

    assert [p.name for p in tmp_path.iterdir()] == ["lib.js"]
    assert (tmp_path / "lib.js").read_bytes() == b"export const answer = 42;"


def test_fetch_rejects_a_tarball_with_the_wrong_checksum(tmp_path):
    tarball = make_tarball({"package/dist/lib.js": b"genuine"})
    package = package_for(tarball)
    tampered = make_tarball({"package/dist/lib.js": b"tampered"})
    registry = FakeRegistry({package.tarball_url: tampered})

    with pytest.raises(vendor.VendorError, match="Checksum mismatch"):
        vendor.fetch_package(package, tmp_path, open_url=registry)

    assert list(tmp_path.iterdir()) == []


def test_fetch_reports_a_file_missing_from_dist(tmp_path):
    tarball = make_tarball({"package/lib.js": b"wrong folder"})
    package = package_for(tarball)
    registry = FakeRegistry({package.tarball_url: tarball})

    with pytest.raises(vendor.VendorError, match="has no lib.js"):
        vendor.fetch_package(package, tmp_path, open_url=registry)


def test_fetch_never_writes_outside_the_destination(tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    tarball = make_tarball(
        {
            "package/dist/../../escaped.js": b"path traversal",
            "/absolute.js": b"absolute path",
            "package/dist/lib.js": b"ok",
        }
    )
    package = package_for(tarball)
    registry = FakeRegistry({package.tarball_url: tarball})

    vendor.fetch_package(package, dest, open_url=registry)

    assert [p.name for p in dest.iterdir()] == ["lib.js"]
    assert [p.name for p in tmp_path.iterdir()] == ["dest"]


def test_fetch_wraps_network_errors(tmp_path):
    package = package_for(make_tarball({"package/dist/lib.js": b"ok"}))

    def offline(url: str):
        raise OSError("network is unreachable")

    with pytest.raises(vendor.VendorError, match="Could not download"):
        vendor.fetch_package(package, tmp_path, open_url=offline)


def test_ensure_engine_downloads_once(tmp_path):
    first = make_tarball({"package/dist/lib.js": b"library"})
    second = make_tarball({"package/dist/run.mjs": b"loader", "package/dist/run.wasm": b"\0asm"})
    packages = (
        package_for(first, name="demo-lib"),
        package_for(second, name="demo-runtime", files=("run.mjs", "run.wasm")),
    )
    registry = FakeRegistry({packages[0].tarball_url: first, packages[1].tarball_url: second})

    folder = vendor.ensure_engine(tmp_path, packages=packages, open_url=registry)

    assert folder == tmp_path / "vendor" / vendor.TRANSFORMERS_VERSION
    assert sorted(p.name for p in folder.iterdir()) == ["lib.js", "run.mjs", "run.wasm"]
    assert len(registry.calls) == 2

    assert vendor.ensure_engine(tmp_path, packages=packages, open_url=registry) == folder
    assert len(registry.calls) == 2


def test_ensure_engine_leaves_nothing_behind_on_failure(tmp_path):
    good = make_tarball({"package/dist/lib.js": b"library"})
    packages = (package_for(good, name="demo-lib"),)
    registry = FakeRegistry({packages[0].tarball_url: make_tarball({"package/dist/lib.js": b"x"})})

    with pytest.raises(vendor.VendorError):
        vendor.ensure_engine(tmp_path, packages=packages, open_url=registry)

    assert list((tmp_path / "vendor").iterdir()) == []
