"""Local server: isolation headers, explicit content types, no caching."""

from __future__ import annotations

import http.client
import mimetypes

import pytest

from cifarium.server import BenchServer, SiteServer

WASM = b"\0asm\1\0\0\0"


@pytest.fixture()
def site(tmp_path):
    bench = tmp_path / "bench"
    engine = tmp_path / "vendor" / "1.0"
    model = tmp_path / "models" / "org" / "model"
    for folder in (bench, engine, model / "onnx", model / ".cache"):
        folder.mkdir(parents=True)

    (bench / "index.html").write_text("<!doctype html><title>bench</title>", encoding="utf-8")
    (bench / "bench.js").write_text("export {};", encoding="utf-8")
    (engine / "loader.mjs").write_text("export default 1;", encoding="utf-8")
    (engine / "runtime.wasm").write_bytes(WASM)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "weights.bin").write_bytes(b"bin")
    (model / "onnx" / "model_quantized.onnx").write_bytes(b"onnx")
    (model / "onnx" / "model.onnx_data").write_bytes(b"data")
    (model / ".cache" / "meta").write_text("hidden", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside the served folders", encoding="utf-8")

    routes = {"bench": bench, "vendor": tmp_path / "vendor", "models": tmp_path / "models"}
    with BenchServer(routes) as server:
        yield server


def request(server: BenchServer, target: str, method: str = "GET"):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    try:
        connection.request(method, target)
        response = connection.getresponse()
        return response, response.read()
    finally:
        connection.close()


def test_listens_on_loopback_and_a_free_port(site, tmp_path):
    assert site.url == f"http://127.0.0.1:{site.port}"
    assert site.port > 0
    with BenchServer({"bench": tmp_path / "bench"}) as other:
        assert other.port != site.port


@pytest.mark.parametrize(
    ("target", "content_type"),
    [
        ("/bench/", "text/html; charset=utf-8"),
        ("/bench/index.html", "text/html; charset=utf-8"),
        ("/bench/bench.js", "text/javascript"),
        ("/vendor/1.0/loader.mjs", "text/javascript"),
        ("/vendor/1.0/runtime.wasm", "application/wasm"),
        ("/models/org/model/config.json", "application/json"),
        ("/models/org/model/weights.bin", "application/octet-stream"),
        ("/models/org/model/onnx/model_quantized.onnx", "application/octet-stream"),
        ("/models/org/model/onnx/model.onnx_data", "application/octet-stream"),
    ],
)
def test_content_types_are_explicit(site, target, content_type):
    response, _ = request(site, target)
    assert response.status == 200
    assert response.getheader("Content-Type") == content_type


def test_javascript_type_ignores_the_system_table(site, monkeypatch):
    # On Windows the registry can map .js to text/plain; the server must not care.
    monkeypatch.setitem(mimetypes.types_map, ".js", "text/plain")
    response, _ = request(site, "/bench/bench.js")
    assert response.getheader("Content-Type") == "text/javascript"


@pytest.mark.parametrize(
    ("method", "target", "status"),
    [
        ("GET", "/vendor/1.0/runtime.wasm", 200),
        ("HEAD", "/vendor/1.0/runtime.wasm", 200),
        ("GET", "/models/org/model/missing.json", 404),
        ("HEAD", "/models/org/model/missing.json", 404),
        ("GET", "/", 404),
        ("POST", "/bench/", 501),
    ],
)
def test_every_response_is_isolated_and_uncached(site, method, target, status):
    response, _ = request(site, target, method)
    assert response.status == status
    assert response.getheader("Cross-Origin-Opener-Policy") == "same-origin"
    assert response.getheader("Cross-Origin-Embedder-Policy") == "require-corp"
    assert response.getheader("Cache-Control") == "no-store"


def test_get_sends_the_whole_file(site):
    response, body = request(site, "/vendor/1.0/runtime.wasm")
    assert body == WASM
    assert response.getheader("Content-Length") == str(len(WASM))


def test_head_sends_headers_without_a_body(site):
    response, body = request(site, "/vendor/1.0/runtime.wasm", "HEAD")
    assert body == b""
    assert response.getheader("Content-Length") == str(len(WASM))
    assert response.getheader("Content-Type") == "application/wasm"


def test_query_string_is_ignored(site):
    response, body = request(site, "/vendor/1.0/runtime.wasm?v=2")
    assert response.status == 200
    assert body == WASM


@pytest.mark.parametrize(
    "target",
    [
        "/secret.txt",
        "/vendor/../secret.txt",
        "/vendor/%2e%2e/secret.txt",
        "/vendor/..%2fsecret.txt",
        "/vendor/..%5csecret.txt",
        "/vendor/1.0/../../secret.txt",
        "/vendor/1.0//runtime.wasm",
        "/models/org/model/.cache/meta",
        "/vendor/1.0",
        "/unknown/index.html",
    ],
)
def test_nothing_outside_the_served_folders_is_reachable(site, target):
    response, body = request(site, target)
    assert response.status == 404
    assert b"outside" not in body and b"hidden" not in body


def test_requests_are_recorded(site):
    request(site, "/bench/bench.js")
    request(site, "/models/org/model/missing.json")
    assert site.requests == [
        ("GET", "/bench/bench.js", 200),
        ("GET", "/models/org/model/missing.json", 404),
    ]


# --- SiteServer: a build folder at the root, with the headers of a hosting file ------------------

RULES = {
    "/": {"X-Rule": "root"},
    "/index.html": {"X-Rule": "index"},
    "/vendor/1.0/loader.mjs": {"X-Rule": "vendor", "Cache-Control": "public, max-age=31536000, immutable"},
}


@pytest.fixture()
def folder(tmp_path):
    root = tmp_path / "out"
    (root / "vendor" / "1.0").mkdir(parents=True)
    (root / "hosting").mkdir()
    (root / "index.html").write_text("<!doctype html><title>the folder's page</title>", encoding="utf-8")
    (root / "cifarium.json").write_text("{}", encoding="utf-8")
    (root / "vendor" / "1.0" / "loader.mjs").write_text("export default 1;", encoding="utf-8")
    (root / ".hidden").write_text("hidden", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside the folder", encoding="utf-8")
    return root


@pytest.fixture()
def hosted(folder):
    with SiteServer(folder, lambda path: RULES.get(path, {})) as server:
        yield server


def test_a_site_serves_the_folder_at_the_root_and_the_slash_is_the_page(hosted):
    page, body = request(hosted, "/")
    assert page.status == 200 and b"the folder's page" in body
    assert page.getheader("Content-Type") == "text/html; charset=utf-8"
    assert request(hosted, "/index.html")[1] == body
    assert request(hosted, "/cifarium.json")[0].getheader("Content-Type") == "application/json"
    assert request(hosted, "/vendor/1.0/loader.mjs")[0].getheader("Content-Type") == "text/javascript"


def test_a_site_sends_the_headers_the_rule_names_for_a_path_and_no_others(hosted):
    root = request(hosted, "/")[0]
    assert root.getheader("X-Rule") == "root"
    assert request(hosted, "/index.html")[0].getheader("X-Rule") == "index"
    vendor = request(hosted, "/vendor/1.0/loader.mjs")[0]
    assert vendor.getheader("Cache-Control") == "public, max-age=31536000, immutable"
    # A path no rule names gets none of the bench's headers: not isolation, not no-store.
    plain = request(hosted, "/cifarium.json")[0]
    for name in ("Cross-Origin-Opener-Policy", "Cross-Origin-Embedder-Policy", "Cache-Control", "X-Rule"):
        assert plain.getheader(name) is None, name


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_a_site_answers_a_missing_file_with_404_and_the_headers_of_its_path(folder, method):
    with SiteServer(folder, lambda path: {"X-Path": path}) as server:
        response, body = request(server, "/missing.js", method)

    assert response.status == 404
    assert response.getheader("X-Path") == "/missing.js"
    assert body == (b"Not found\n" if method == "GET" else b"")


def test_a_site_puts_another_file_at_the_paths_it_is_told_and_leaves_the_folder_alone(folder, tmp_path):
    page = tmp_path / "check.html"
    page.write_text("<!doctype html><title>the check page</title>", encoding="utf-8")
    before = sorted((path.relative_to(folder).as_posix(), path.read_bytes())
                    for path in folder.rglob("*") if path.is_file())

    with SiteServer(folder, lambda path: RULES.get(path, {}),
                    replace={"/": page, "/index.html": page}) as server:
        assert b"the check page" in request(server, "/")[1]
        assert b"the check page" in request(server, "/index.html")[1]
        assert request(server, "/")[0].getheader("X-Rule") == "root", "the rule of the path applies to the swap"
        assert request(server, "/cifarium.json")[0].status == 200

    after = sorted((path.relative_to(folder).as_posix(), path.read_bytes())
                   for path in folder.rglob("*") if path.is_file())
    assert after == before


def test_a_site_whose_swapped_file_is_missing_answers_404_not_the_page_of_the_folder(folder, tmp_path):
    with SiteServer(folder, lambda path: {}, replace={"/": tmp_path / "gone.html"}) as server:
        assert request(server, "/")[0].status == 404


@pytest.mark.parametrize("target", [
    "/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt", "/..%5csecret.txt",
    "/vendor/../../secret.txt", "/.hidden", "/vendor//1.0/loader.mjs", "/vendor/1.0", "/hosting",
])
def test_a_site_serves_nothing_outside_the_folder_nor_a_dotfile(hosted, target):
    response, body = request(hosted, target)
    assert response.status == 404
    assert b"outside" not in body and b"hidden" not in body


def test_a_site_records_its_requests(hosted):
    request(hosted, "/cifarium.json")
    request(hosted, "/nope.js")
    assert hosted.requests == [("GET", "/cifarium.json", 200), ("GET", "/nope.js", 404)]
