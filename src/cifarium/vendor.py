"""Engine files: Transformers.js and the onnxruntime-web build it was made for.

The browser needs three files (ARCHITECTURE.md, section 1.2). They are not
shipped inside the Python package. On first use the two npm tarballs are
downloaded from the registry, each one is checked against the sha512 digest
pinned below, and only the needed files are copied to
``<cache>/vendor/<Transformers.js version>/``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from . import paths

REGISTRY = "https://registry.npmjs.org"

TRANSFORMERS_VERSION = "4.3.2"
# The exact build named in the dependencies of @huggingface/transformers 4.3.2.
# The loader (.mjs) and the executor (.wasm) must come from the same build.
ORT_WEB_VERSION = "1.31.0-dev.20260914-8d85527a0"

# The only build of the library that runs without a bundler: the ".web" builds
# import "onnxruntime-web/webgpu" by a bare name the browser cannot resolve.
LIBRARY_FILE = "transformers.min.js"
# The executor the library asks for by default (WebAssembly with WebGPU support).
WASM_LOADER_FILE = "ort-wasm-simd-threaded.asyncify.mjs"
WASM_BINARY_FILE = "ort-wasm-simd-threaded.asyncify.wasm"

# Size in bytes of each engine file, which a visitor downloads on the first visit together with
# the model (ARCHITECTURE.md, section 1.2). The tarballs below are pinned by their sha512, so the
# sizes are fixed for these versions. Change them together with the versions; a test compares
# them with the files on disk.
ENGINE_BYTES = {
    LIBRARY_FILE: 587_917,
    WASM_LOADER_FILE: 53_057,
    WASM_BINARY_FILE: 26_861_777,
}


# The license texts that go into LICENSES.txt of a build (ARCHITECTURE.md, section 7), stored next to
# the engine files under these names.
TRANSFORMERS_LICENSE_FILE = "LICENSE-transformers-js.txt"
ORT_LICENSE_FILE = "LICENSE-onnxruntime-web.txt"
# The commit onnxruntime-web 1.31.0-dev.20260914-8d85527a0 was built from: its tarball names it in
# package/__commit.txt. The tarball carries no LICENSE (package.json says "MIT", the README sends
# the reader to GitHub), so the text is taken from the repository at this very commit.
ORT_WEB_COMMIT = "8d85527a010e294a26b274749f74294b2a32cec5"


class VendorError(RuntimeError):
    """Engine files could not be downloaded or failed verification."""


@dataclass(frozen=True)
class PackageLicense:
    """Where the text of a package's license is, and the digest it must have."""

    stored_as: str
    """File name in the engine folder."""
    sha256: str
    """Hex digest of the text as it is stored. The tarball is already pinned by its sha512; this
    one also covers a text fetched from elsewhere and shows in a diff when a version moves."""
    member: str | None = "package/LICENSE"
    """Path of the text inside the tarball, or None when the tarball has none."""
    url: str | None = None
    """Where to fetch the text when the tarball has none."""


@dataclass(frozen=True)
class NpmPackage:
    """A pinned npm package and the files cifarium takes from its ``dist/``."""

    name: str
    version: str
    #: ``dist.integrity`` from the registry: ``sha512-<base64 digest of the tarball>``.
    integrity: str
    files: tuple[str, ...]
    license: PackageLicense | None = None

    @property
    def tarball_url(self) -> str:
        basename = self.name.rsplit("/", 1)[-1]
        return f"{REGISTRY}/{self.name}/-/{basename}-{self.version}.tgz"

    @property
    def stored(self) -> tuple[str, ...]:
        """Every file this package leaves in the engine folder: its files and its license."""
        return self.files + ((self.license.stored_as,) if self.license else ())


ENGINE_PACKAGES: tuple[NpmPackage, ...] = (
    NpmPackage(
        name="@huggingface/transformers",
        version=TRANSFORMERS_VERSION,
        integrity=(
            "sha512-V3+dqf3iWlRYJ9I9GreM3zGQd0lAXnnoPeddmkOQYWAEE4l/"
            "+Fi6p7tx/qVUJlKQgLXGim2W76rTOzz+eC0i7g=="
        ),
        files=(LIBRARY_FILE,),
        license=PackageLicense(
            TRANSFORMERS_LICENSE_FILE,
            "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",  # Apache-2.0
        ),
    ),
    NpmPackage(
        name="onnxruntime-web",
        version=ORT_WEB_VERSION,
        integrity=(
            "sha512-Iy7rtadoBgxS/LLvDr3QW38DB1PNXRnr0GJMcL0TAt7c9qjgVQl83UlG"
            "CVyeAnK2InpmW8Uc3PL8XIuqtDeF6g=="
        ),
        files=(WASM_LOADER_FILE, WASM_BINARY_FILE),
        license=PackageLicense(
            ORT_LICENSE_FILE,
            "2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c",  # MIT
            member=None,
            url=f"https://raw.githubusercontent.com/microsoft/onnxruntime/{ORT_WEB_COMMIT}/LICENSE",
        ),
    ),
)


def _open_url(url: str) -> BinaryIO:
    return urllib.request.urlopen(url, timeout=60)


def engine_bytes() -> int:
    """Weight of the engine on a visitor's first visit, in bytes."""
    return sum(ENGINE_BYTES.values())


def engine_dir(cache: Path | None = None) -> Path:
    """Folder that holds the engine files of the pinned version."""
    return paths.vendor_dir(cache) / TRANSFORMERS_VERSION


def ensure_engine(
    cache: Path | None = None,
    *,
    packages: Sequence[NpmPackage] = ENGINE_PACKAGES,
    open_url: Callable[[str], BinaryIO] = _open_url,
    report: Callable[[str], None] | None = None,
) -> Path:
    """Return the folder with the engine files, downloading them on first use.

    The files are assembled in a staging folder and moved into place only after
    every tarball passed verification, so a finished folder is always complete.
    """
    target = engine_dir(cache)
    wanted = [name for package in packages for name in package.stored]
    if all((target / name).is_file() for name in wanted):
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=target.parent))
    try:
        for package in packages:
            if report:
                report(f"Downloading {package.name} {package.version}")
            fetch_package(package, staging, open_url=open_url)
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def fetch_package(
    package: NpmPackage,
    dest: Path,
    *,
    open_url: Callable[[str], BinaryIO] = _open_url,
) -> None:
    """Download one tarball, verify its sha512 and copy ``package.files`` and the license text
    into ``dest``.

    The archive is parsed only after its digest matched the pinned value; the license text is
    checked against its own pinned sha256.
    """
    algorithm, _, expected = package.integrity.partition("-")
    if algorithm != "sha512" or not expected:
        raise VendorError(f"Unsupported integrity value for {package.name}: {package.integrity!r}")

    digest = hashlib.sha512()
    with tempfile.TemporaryFile() as tarball:
        try:
            with open_url(package.tarball_url) as response:
                while chunk := response.read(1 << 16):
                    digest.update(chunk)
                    tarball.write(chunk)
        except OSError as error:
            raise VendorError(f"Could not download {package.tarball_url}: {error}") from error

        actual = base64.b64encode(digest.digest()).decode("ascii")
        if not hmac.compare_digest(actual, expected):
            raise VendorError(
                f"Checksum mismatch for {package.name} {package.version}: "
                f"expected sha512-{expected}, got sha512-{actual}"
            )

        tarball.seek(0)
        _copy_dist_files(tarball, package, dest)
        if package.license is not None:
            tarball.seek(0)
            _store_license(tarball, package, package.license, dest, open_url)


def _store_license(
    tarball: BinaryIO,
    package: NpmPackage,
    license: PackageLicense,
    dest: Path,
    open_url: Callable[[str], BinaryIO],
) -> None:
    """Write the license text of ``package`` into ``dest`` after checking its sha256."""
    text: bytes | None = None
    if license.member is not None:
        try:
            with tarfile.open(fileobj=tarball, mode="r:gz") as archive:
                member = archive.getmember(license.member)
                source = archive.extractfile(member) if member.isreg() else None
                text = source.read() if source is not None else None
        except KeyError:
            text = None
        except tarfile.TarError as error:
            raise VendorError(f"Could not read the {package.name} tarball: {error}") from error
        if text is None:
            raise VendorError(f"{package.name} {package.version} has no {license.member}")
    elif license.url is not None:
        try:
            with open_url(license.url) as response:
                text = response.read()
        except OSError as error:
            raise VendorError(f"Could not download {license.url}: {error}") from error
    if text is None:
        raise VendorError(f"{package.name} {package.version}: no source for its license text")
    actual = hashlib.sha256(text).hexdigest()
    if not hmac.compare_digest(actual, license.sha256):
        raise VendorError(
            f"Checksum mismatch for the license of {package.name} {package.version}: "
            f"expected sha256 {license.sha256}, got {actual}"
        )
    (dest / license.stored_as).write_bytes(text)


def _copy_dist_files(tarball: BinaryIO, package: NpmPackage, dest: Path) -> None:
    """Copy the listed files out of ``package/dist/``.

    Members are matched by exact name and written under names taken from
    ``package.files``, never from the archive, so nothing can land outside ``dest``.
    """
    wanted = {f"package/dist/{name}": name for name in package.files}
    found: set[str] = set()
    try:
        with tarfile.open(fileobj=tarball, mode="r:gz") as archive:
            for member in archive:
                name = wanted.get(member.name)
                if name is None or not member.isreg():
                    continue
                source = archive.extractfile(member)
                assert source is not None  # regular members always have data
                with open(dest / name, "wb") as output:
                    shutil.copyfileobj(source, output)
                found.add(name)
    except tarfile.TarError as error:
        raise VendorError(f"Could not read the {package.name} tarball: {error}") from error

    missing = sorted(set(package.files) - found)
    if missing:
        raise VendorError(
            f"{package.name} {package.version} has no {', '.join(missing)} in dist/"
        )
