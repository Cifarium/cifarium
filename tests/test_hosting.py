"""The readers of the three hosting files: what headers a host sends for a path. `check` serves a
folder with these headers, so a reader that gets a rule wrong makes the check lie about the site.
The files cifarium writes are read in tests/test_bundle.py; here the readers meet rules written by
hand, and the folder reading meets files that are missing or broken.
"""

from __future__ import annotations

import json

import pytest

from cifarium import hosting, paths

TEMPLATES = paths.web_dir() / "templates"


# --- netlify: _headers ---------------------------------------------------------------------------


NETLIFY = """\
# a comment, then a blank line

/
  X-One: a
  X-Two: b
/vendor/*
  X-One: under vendor
  Cache-Control: public, max-age=31536000, immutable
/*
  X-All: yes
"""


def test_netlify_matches_a_path_exactly_and_a_star_as_a_prefix():
    assert hosting.netlify(NETLIFY, "/") == {"X-One": "a", "X-Two": "b", "X-All": "yes"}
    assert hosting.netlify(NETLIFY, "/vendor/4.3.2/x.mjs")["Cache-Control"].startswith("public")
    assert hosting.netlify(NETLIFY, "/index.html") == {"X-All": "yes"}
    assert hosting.netlify(NETLIFY, "/vendor") == {"X-All": "yes"}  # the star wants the slash


def test_netlify_adds_the_headers_of_every_matching_rule_and_the_later_wins():
    headers = hosting.netlify("/*\n  X-One: first\n/a/*\n  X-One: second\n  X-Two: b\n", "/a/b")

    assert headers == {"X-One": "second", "X-Two": "b"}


def test_netlify_ignores_comments_and_blank_lines():
    assert hosting.netlify("# /\n\n/\n  # X-One: no\n  X-Two: b\n", "/") == {"X-Two": "b"}


# --- vercel: vercel.json -------------------------------------------------------------------------


def vercel_file(*rules: tuple[str, dict[str, str]]) -> str:
    return json.dumps({"headers": [
        {"source": source, "headers": [{"key": key, "value": value} for key, value in headers.items()]}
        for source, headers in rules]})


def test_vercel_takes_a_source_without_a_group_literally_and_one_with_a_group_as_a_pattern():
    text = vercel_file(("/", {"X-Root": "r"}), ("/vendor/(.*)", {"X-Vendor": "v"}), ("/a.b", {"X-Dot": "d"}))

    assert hosting.vercel(text, "/") == {"X-Root": "r"}
    assert hosting.vercel(text, "/vendor/x/y.mjs") == {"X-Vendor": "v"}
    assert hosting.vercel(text, "/vendor") == {}
    assert hosting.vercel(text, "/a.b") == {"X-Dot": "d"}
    assert hosting.vercel(text, "/aXb") == {}, "a source without a group is not a pattern"


def test_vercel_adds_the_headers_of_every_matching_rule():
    text = vercel_file(("/(.*)", {"X-All": "1", "X-One": "a"}), ("/x", {"X-One": "b"}))

    assert hosting.vercel(text, "/x") == {"X-All": "1", "X-One": "b"}


# --- nginx: nginx.conf ---------------------------------------------------------------------------


NGINX = r"""
server {
  # add_header X-Comment "no" always;
  location = /exact { add_header X-Kind "exact" always; }
  location /vendor/ { add_header X-Kind "prefix vendor" always; }
  location /vendor/deep/ { add_header X-Kind "prefix deep" always; }
  location ~ ^/vendor/.*\.map$ { add_header X-Kind "regular" always; }
  location / { add_header X-Kind "root" always; add_header X-Two "2" always; }
}
"""


@pytest.mark.parametrize("path, kind", [
    ("/exact", "exact"),
    ("/vendor/x.mjs", "prefix vendor"),
    ("/vendor/deep/x.mjs", "prefix deep"),
    ("/vendor/deep/x.map", "regular"),
    ("/anything", "root"),
])
def test_nginx_picks_the_exact_location_then_a_regular_one_then_the_longest_prefix(path, kind):
    assert hosting.nginx(NGINX, path)["X-Kind"] == kind


def test_nginx_gives_the_headers_of_that_one_location_and_no_others():
    assert hosting.nginx(NGINX, "/x") == {"X-Kind": "root", "X-Two": "2"}
    assert hosting.nginx(NGINX, "/exact") == {"X-Kind": "exact"}


def test_nginx_without_a_matching_location_sends_nothing():
    assert hosting.nginx("location = /a { add_header X-A \"1\" always; }", "/b") == {}


# --- the files of a folder -----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(hosting.FILES))
def test_the_files_cifarium_writes_isolate_the_page_and_the_worker(name):
    headers_of = hosting.headers_of(TEMPLATES, name)

    for path in ("/", "/index.html", "/cifarium-worker.js", "/vendor/4.3.2/x.mjs"):
        got = headers_of(path)
        assert {key: got[key] for key in hosting.ISOLATION} == hosting.ISOLATION, (name, path)
    assert not set(hosting.ISOLATION) & set(headers_of("/cifarium.json")), name


def test_the_default_is_the_file_netlify_and_cloudflare_pages_read():
    assert hosting.DEFAULT == "netlify" and hosting.FILES["netlify"] == "_headers"
    assert set(hosting.FILES.values()) == {"_headers", "vercel.json", "nginx.conf"}


def test_a_name_that_is_not_a_hosting_is_refused(tmp_path):
    with pytest.raises(hosting.HostingError, match="unknown hosting 'apache'.*netlify, nginx, vercel"):
        hosting.headers_of(tmp_path, "apache")


def test_a_file_that_is_not_in_the_folder_is_named(tmp_path):
    (tmp_path / "hosting").mkdir()

    with pytest.raises(hosting.HostingError, match=r"hosting/vercel\.json is not in"):
        hosting.headers_of(tmp_path, "vercel")


@pytest.mark.parametrize("name, text", [
    ("vercel", "{not json"),
    ("vercel", '{"no": "headers"}'),
    ("vercel", '{"headers": [{"source": "/(", "headers": []}]}'),
])
def test_a_file_the_reader_cannot_read_is_refused_before_a_page_depends_on_it(tmp_path, name, text):
    (tmp_path / "hosting").mkdir()
    (tmp_path / "hosting" / hosting.FILES[name]).write_text(text, encoding="utf-8")

    with pytest.raises(hosting.HostingError, match="cannot be read the way vercel reads it"):
        hosting.headers_of(tmp_path, name)


def test_a_file_that_is_not_text_is_refused(tmp_path):
    (tmp_path / "hosting").mkdir()
    (tmp_path / "hosting" / "_headers").write_bytes(b"\xff\xfe\x00bad")

    with pytest.raises(hosting.HostingError, match=r"hosting/_headers cannot be read"):
        hosting.headers_of(tmp_path)


def test_the_reader_of_a_folder_follows_the_file_it_read(tmp_path):
    (tmp_path / "hosting").mkdir()
    (tmp_path / "hosting" / "_headers").write_text("/\n  X-Mine: 1\n", encoding="utf-8")

    headers_of = hosting.headers_of(tmp_path)

    assert headers_of("/") == {"X-Mine": "1"} and headers_of("/other") == {}
