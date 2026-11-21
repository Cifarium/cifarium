"""The folder of a build on a site with a strict Content-Security-Policy, and on a site over plain http
that is not localhost, in the real Chrome (ARCHITECTURE.md, section 1.6).

A site with a strict policy runs its own script from a file: the example page of the folder has its
script inline, which `script-src 'self'` refuses, so the page here is the one a site would write, an
HTML file and `site.js` beside it that imports `cifarium.js`. The policy is sent on every file of the
folder, as a site sets it, and the page has it in a meta tag too: the antivirus of the author's laptop
rewrites the header of every HTML page served over plain http (it adds its own host and turns
`connect-src 'none'` into `'self'`), and a meta tag it leaves alone. The worker and the threads of
ONNX Runtime get the header only, which is what they obey.

The end-to-end run is not part of the default run: `pytest -m e2e`. It downloads nothing into the
cache of cifarium that tests/test_build_e2e.py does not; the folder with `host: hub` makes its page
download all-MiniLM from Hugging Face (about 24 MB) into a browser profile that is never kept. The
browser is a Chrome of this module's own, without the profile of the bench: the host that is not
localhost needs a flag of Chrome (`--host-resolver-rules`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pytest
from playwright.sync_api import Browser, sync_playwright
from test_build_e2e import ENGLISH

from cifarium import bundle, catalog, config, hosting
from cifarium.server import SiteServer

pytestmark = pytest.mark.e2e

TEXT, LABEL = ENGLISH[0]
NAMES = ["shipping", "returns", "technical"]
HOST = "cifarium.test"
"""A host that is not localhost: Chrome maps it to 127.0.0.1, so the page is over plain http on a
name that is not trustworthy, as a site without HTTPS is."""

HUB = "https://huggingface.co https://*.hf.co"
"""Where the page of a `host: hub` folder downloads from: the hub, and its CDN, whose host depends on
the place and the storage (us.aws.cdn.hf.co from the author's laptop)."""


def strict(connect: str = "") -> str:
    """The policy a strict site sends: nothing but its own origin, WebAssembly allowed, no `'unsafe-eval'`
    and no `'unsafe-inline'`."""
    return ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; "
            f"connect-src 'self'{' ' + connect if connect else ''}; object-src 'none'; base-uri 'none'")


SITE_JS = """// The script of a site, in a file of its own, as a strict policy wants it.
import { createCifarium } from './cifarium.js';

const out = {};
try {
  const model = await createCifarium('./cifarium.json');
  const info = await model.load();
  const answer = await model.run(new URLSearchParams(location.search).get('text'));
  out.label = answer.label;
  out.threads = info.threads;
  out.workerIsolated = info.crossOriginIsolated;
} catch (error) {
  out.code = error?.code ?? null;
  out.message = String(error?.message ?? error);
}
out.isolated = self.crossOriginIsolated;
out.secure = self.isSecureContext;
out.cacheApi = typeof caches !== 'undefined';
out.webgpu = 'gpu' in navigator;
out.cores = navigator.hardwareConcurrency || 1;
window.siteResult = out;
"""


@dataclass(frozen=True)
class Folders:
    bundle: Path
    hub: Path


@pytest.fixture(scope="module")
def folders(english_build, tmp_path_factory) -> Folders:
    """The folder of the English build, and the same model, labels and centres written with `host: hub`."""
    assert english_build.code == 0, english_build.err
    source = english_build.folder
    manifest = json.loads((source / bundle.MANIFEST).read_text(encoding="utf-8"))
    candidate = next(c for c in catalog.candidates("classify") if c.entry.repo == manifest["model"]["repo"]
                     and c.recipe.kind == manifest["recipe"])
    labels = [config.Label(item["name"], item.get("description")) for item in manifest["labels"]]
    assert [label.name for label in labels] == NAMES
    centres = np.frombuffer((source / manifest["data"]["centres"]["path"]).read_bytes(),
                            dtype="<f4").reshape(len(labels), -1)
    hub = tmp_path_factory.mktemp("csp-hub")
    bundle.write(hub, candidate, labels, centres, host="hub")
    return Folders(source, hub)


@pytest.fixture(scope="module")
def browser(folders) -> Iterator[Browser]:
    """The installed Chrome, started after the build has closed its own: Playwright's sync API runs one
    at a time in a thread."""
    with sync_playwright() as playwright:
        started = playwright.chromium.launch(channel="chrome", headless=True, args=[
            f"--host-resolver-rules=MAP {HOST} 127.0.0.1",
            # Chrome would try https:// first on a name it does not know; the server speaks http only.
            "--disable-features=HttpsUpgrades"])
        try:
            yield started
        finally:
            started.close()


def site_page(folder: Path, policy: str | None) -> dict[str, Path]:
    """The page of a site and its script, put at `/`, `/index.html` and `/site.js` of the folder's server."""
    meta = "" if policy is None else f'<meta http-equiv="Content-Security-Policy" content="{policy}">'
    page = folder / "page.html"
    page.write_text(f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">{meta}<title>a site</title>'
                    '</head><body><script type="module" src="./site.js"></script></body></html>\n', encoding="utf-8")
    script = folder / "site.js"
    script.write_text(SITE_JS, encoding="utf-8")
    return {"/": page, "/index.html": page, "/site.js": script}


def visit(browser: Browser, folder: Path, tmp_path: Path, *, policy: str | None, worker_policy: str | None = None,
          host: str = "127.0.0.1", visits: int = 1) -> list[dict[str, Any]]:
    """Serve ``folder`` behind the headers of its Netlify file plus ``policy`` on every file (``worker_policy``
    on every file but the page, when given), open the site's page ``visits`` times in one browser
    profile and say what the page found each time."""
    netlify = hosting.headers_of(folder, "netlify")
    others = policy if worker_policy is None else worker_policy

    def headers(path: str) -> dict[str, str]:
        own = dict(netlify(path))
        chosen = policy if path in ("/", "/index.html") else others
        if chosen is not None:
            own["Content-Security-Policy"] = chosen
        return own

    found = []
    context = browser.new_context()
    try:
        with SiteServer(folder, headers, replace=site_page(tmp_path, policy)) as server:
            for _ in range(visits):
                page = context.new_page()
                events: list[str] = []
                page.on("console", lambda message: events.append(f"console {message.type}: {message.text}"))
                page.goto(f"http://{host}:{server.port}/?text={quote(TEXT)}")
                page.wait_for_function("() => window.siteResult", timeout=120_000)
                found.append({**page.evaluate("() => window.siteResult"), "events": events[-6:]})
                page.close()
    finally:
        context.close()
    return found


def threads_of_isolated(cores: int) -> int:
    return min(4, math.ceil(cores / 2))


@pytest.mark.parametrize("which, policy", [("bundle", strict()), ("hub", strict(HUB))])
def test_under_a_strict_policy_on_every_file_the_page_names_the_label_in_threads(browser, folders, tmp_path,
                                                                                  which, policy):
    [seen] = visit(browser, getattr(folders, which), tmp_path, policy=policy)

    assert seen.get("label") == LABEL, seen
    assert seen["isolated"] is True and seen["workerIsolated"] is True
    assert seen["threads"] == threads_of_isolated(seen["cores"])


def test_without_wasm_unsafe_eval_for_the_worker_the_model_does_not_start(browser, folders, tmp_path):
    # The page itself compiles no WebAssembly: the worker does, and it obeys the header of its own file.
    # This is what proves the policy reaches the worker.
    [seen] = visit(browser, folders.bundle, tmp_path, policy=strict(),
                   worker_policy="default-src 'self'; script-src 'self'; worker-src 'self'; connect-src 'self'")

    assert seen.get("label") is None and seen["code"] == "load", seen
    assert "Content Security" in seen["message"] and "'unsafe-eval'" in seen["message"], seen["message"]


def test_the_page_needs_no_wasm_unsafe_eval_of_its_own(browser, folders, tmp_path):
    [seen] = visit(browser, folders.bundle, tmp_path, policy="default-src 'self'; script-src 'self'",
                   worker_policy=strict())

    assert seen.get("label") == LABEL, seen


@pytest.mark.parametrize("connect, why", [("", "only its own origin"),
                                          ("https://huggingface.co", "the hub without its CDN")])
def test_a_hub_folder_whose_worker_may_not_reach_the_cdn_of_hugging_face_cannot_download(browser, folders,
                                                                                         tmp_path, connect, why):
    [seen] = visit(browser, folders.hub, tmp_path, policy=strict(connect))

    assert seen.get("label") is None and seen["code"] == "network", (why, seen)


def test_over_plain_http_on_a_host_that_is_not_localhost_the_page_works_in_one_thread_without_a_store(
        browser, folders, tmp_path):
    # Not a secure context: the isolation headers are ignored, and there is no Cache API and no WebGPU.
    first, again = visit(browser, folders.bundle, tmp_path, policy=strict(), host=HOST, visits=2)

    for seen in (first, again):
        assert seen.get("label") == LABEL, seen
        assert (seen["secure"], seen["isolated"], seen["cacheApi"], seen["webgpu"]) == (False, False, False, False)
        assert seen["threads"] == 1
