# The public pages and the edge's Caddyfile (scripts/site_check.py, design.md 6.23): no GPU, no Caddy, no network.
import shutil

import pytest

from scripts import site_check as S

PORT = 8127  # and 8128 (the stand-in Google)


@pytest.fixture(scope="module")
def edge():
    e = S.Edge(PORT, PORT + 1).start()
    yield PORT
    e.stop()


def test_pages_are_well_formed_and_their_links_resolve(edge):
    assert S.check_links(edge) == []


def test_no_other_origin_and_no_inline_code():
    assert S.check_sources() == []


def test_the_edge_routes_and_headers(edge):
    assert S.check_routes(edge) == []


def test_the_mock_flow_from_google_to_revoke(edge):
    assert S.check_flow(edge) == []


def test_the_fallback_numbers_are_6_20s():
    assert S.check_fallback_text() == []


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_the_scripts_under_node():
    assert S.check_node() == []


def test_the_caddyfile_block_is_read():
    conf = S.caddy_config()
    assert set(conf["matchers"]) >= {"api", "home", "pages", "slash", "assets"}
    assert conf["cache"]["pages"] == "no-cache" and conf["cache"]["assets"] == "no-cache"
    for name in ("home", "pages", "assets"):
        assert "sclmd_page" in conf["cache"]["_imports_" + name]
    assert "sclmd_page" not in conf["cache"]["_imports_api"]  # the API's answers keep the server's own headers
    assert conf["imports"] == ["sclmd_headers"]
