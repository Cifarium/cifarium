"""Local server: isolation headers, explicit content types, no caching."""

from __future__ import annotations

import http.client
import mimetypes

import pytest

from cifarium.server import BenchServer

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
