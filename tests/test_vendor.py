"""Engine download: checksum verification and extraction, without the network."""

from __future__ import annotations

import base64
import hashlib
import io
import re
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


def package_for(tarball: bytes, *, name: str = "demo", files: tuple[str, ...] = ("lib.js",), license=None):
    return vendor.NpmPackage(name=name, version="1.0.0", integrity=integrity_of(tarball), files=files,
                             license=license)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def test_the_engine_weight_lists_the_files_the_packages_provide():
    provided = {name for package in vendor.ENGINE_PACKAGES for name in package.files}

    assert set(vendor.ENGINE_BYTES) == provided
    assert vendor.engine_bytes() == sum(vendor.ENGINE_BYTES.values())


@pytest.mark.e2e
def test_the_engine_weight_matches_the_files_of_the_pinned_version():
    folder = vendor.ensure_engine()

    assert {name: (folder / name).stat().st_size for name in vendor.ENGINE_BYTES} == vendor.ENGINE_BYTES


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


TEXT = b"Apache License\nVersion 2.0, January 2004\n"


def test_fetch_stores_the_license_text_that_the_tarball_carries(tmp_path):
    tarball = make_tarball({"package/dist/lib.js": b"x", "package/LICENSE": TEXT, "package/README.md": b"r"})
    package = package_for(tarball, license=vendor.PackageLicense("LICENSE-demo.txt", sha256_of(TEXT)))

    vendor.fetch_package(package, tmp_path, open_url=FakeRegistry({package.tarball_url: tarball}))

    assert sorted(p.name for p in tmp_path.iterdir()) == ["LICENSE-demo.txt", "lib.js"]
    assert (tmp_path / "LICENSE-demo.txt").read_bytes() == TEXT


def test_fetch_takes_the_license_text_from_the_address_when_the_tarball_has_none(tmp_path):
    tarball = make_tarball({"package/dist/lib.js": b"x"})
    mit = b"MIT License\n"
    license = vendor.PackageLicense("LICENSE-demo.txt", sha256_of(mit), member=None,
                                    url="https://raw.example/demo/LICENSE")
    package = package_for(tarball, license=license)
    registry = FakeRegistry({package.tarball_url: tarball, license.url: mit})

    vendor.fetch_package(package, tmp_path, open_url=registry)

    assert (tmp_path / "LICENSE-demo.txt").read_bytes() == mit
    assert registry.calls == [package.tarball_url, license.url]


def test_a_license_text_that_is_not_the_pinned_one_is_refused(tmp_path):
    tarball = make_tarball({"package/dist/lib.js": b"x", "package/LICENSE": b"changed text"})
    package = package_for(tarball, license=vendor.PackageLicense("LICENSE-demo.txt", sha256_of(TEXT)))

    with pytest.raises(vendor.VendorError, match="Checksum mismatch for the license of demo"):
        vendor.fetch_package(package, tmp_path, open_url=FakeRegistry({package.tarball_url: tarball}))


def test_a_license_text_fetched_from_the_address_is_checked_too(tmp_path):
    tarball = make_tarball({"package/dist/lib.js": b"x"})
    license = vendor.PackageLicense("LICENSE-demo.txt", sha256_of(b"MIT License\n"), member=None,
                                    url="https://raw.example/demo/LICENSE")
    package = package_for(tarball, license=license)
    registry = FakeRegistry({package.tarball_url: tarball, license.url: b"something else"})

    with pytest.raises(vendor.VendorError, match="Checksum mismatch for the license of demo"):
        vendor.fetch_package(package, tmp_path, open_url=registry)


def test_a_tarball_without_the_license_member_is_an_error_that_names_it(tmp_path):
    tarball = make_tarball({"package/dist/lib.js": b"x"})
    package = package_for(tarball, license=vendor.PackageLicense("LICENSE-demo.txt", sha256_of(TEXT)))

    with pytest.raises(vendor.VendorError, match="demo 1.0.0 has no package/LICENSE"):
        vendor.fetch_package(package, tmp_path, open_url=FakeRegistry({package.tarball_url: tarball}))


def test_an_address_that_cannot_be_reached_is_a_vendor_error(tmp_path):
    tarball = make_tarball({"package/dist/lib.js": b"x"})
    license = vendor.PackageLicense("LICENSE-demo.txt", sha256_of(TEXT), member=None,
                                    url="https://raw.example/demo/LICENSE")
    package = package_for(tarball, license=license)

    def only_the_registry(url: str):
        if url == license.url:
            raise OSError("network is unreachable")
        return io.BytesIO(tarball)

    with pytest.raises(vendor.VendorError, match="Could not download https://raw.example/demo/LICENSE"):
        vendor.fetch_package(package, tmp_path, open_url=only_the_registry)


def test_an_engine_folder_from_before_the_licenses_is_downloaded_again(tmp_path):
    tarball = make_tarball({"package/dist/lib.js": b"library", "package/LICENSE": TEXT})
    package = package_for(tarball, name="demo-lib", license=vendor.PackageLicense("LICENSE-demo.txt", sha256_of(TEXT)))
    registry = FakeRegistry({package.tarball_url: tarball})
    folder = vendor.engine_dir(tmp_path)
    folder.mkdir(parents=True)
    (folder / "lib.js").write_bytes(b"library")  # what a cifarium without the licenses left behind

    vendor.ensure_engine(tmp_path, packages=(package,), open_url=registry)

    assert (folder / "LICENSE-demo.txt").read_bytes() == TEXT
    assert len(registry.calls) == 1
    vendor.ensure_engine(tmp_path, packages=(package,), open_url=registry)
    assert len(registry.calls) == 1


def test_the_pinned_packages_name_where_their_licenses_come_from_and_their_digests():
    transformers, ort = vendor.ENGINE_PACKAGES

    assert transformers.license.member == "package/LICENSE" and transformers.license.url is None
    assert transformers.license.stored_as == vendor.TRANSFORMERS_LICENSE_FILE
    # The onnxruntime-web tarball carries no LICENSE; the text is the repository's at the commit the
    # tarball names in package/__commit.txt, and that commit is the one of the pinned version.
    assert ort.license.member is None and ort.license.stored_as == vendor.ORT_LICENSE_FILE
    assert ort.license.url == ("https://raw.githubusercontent.com/microsoft/onnxruntime/"
                               f"{vendor.ORT_WEB_COMMIT}/LICENSE")
    assert vendor.ORT_WEB_COMMIT.startswith(vendor.ORT_WEB_VERSION.rsplit("-", 1)[-1])
    assert len(vendor.ORT_WEB_COMMIT) == 40
    for package in vendor.ENGINE_PACKAGES:
        assert re.fullmatch(r"[0-9a-f]{64}", package.license.sha256), package.name
        assert set(package.files).isdisjoint({package.license.stored_as})


@pytest.mark.e2e
def test_the_license_texts_of_the_pinned_engine_are_the_apache_and_the_mit_license():
    folder = vendor.ensure_engine()

    apache = (folder / vendor.TRANSFORMERS_LICENSE_FILE).read_text(encoding="utf-8")
    mit = (folder / vendor.ORT_LICENSE_FILE).read_text(encoding="utf-8")
    assert "Apache License" in apache and "Version 2.0, January 2004" in apache
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in apache
    assert mit.startswith("MIT License") and "Copyright (c) Microsoft Corporation" in mit
    assert "Permission is hereby granted, free of charge" in mit


def test_ensure_engine_leaves_nothing_behind_on_failure(tmp_path):
    good = make_tarball({"package/dist/lib.js": b"library"})
    packages = (package_for(good, name="demo-lib"),)
    registry = FakeRegistry({packages[0].tarball_url: make_tarball({"package/dist/lib.js": b"x"})})

    with pytest.raises(vendor.VendorError):
        vendor.ensure_engine(tmp_path, packages=packages, open_url=registry)

    assert list((tmp_path / "vendor").iterdir()) == []
