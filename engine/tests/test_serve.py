"""The live try-out server's guard rails (no simulations needed)."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from matchengine.serve import Usage, make_handler


def test_daily_cap_counts_and_persists(tmp_path):
    u = Usage(tmp_path / "usage.json", cap=10)
    assert u.remaining() == 10
    u.add(7)
    assert Usage(tmp_path / "usage.json", cap=10).remaining() == 3


def _server(tmp_path, cap):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(tmp_path / "cache"), Usage(tmp_path / "u.json", cap), "*"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_refuses_bad_requests_and_past_the_cap(tmp_path):
    srv, url = _server(tmp_path, cap=5)
    try:
        with urllib.request.urlopen(url + "/health") as r:
            assert json.loads(r.read())["matches_left_today"] == 5
        code, _ = _post(url + "/tryout", {"home": "RIV", "away": "KIN", "side": "home", "slot": 99, "player": "wes-20"})
        assert code == 400
        code, _ = _post(url + "/tryout", {"home": "RIV", "away": "KIN", "side": "home", "slot": 9, "player": "nobody"})
        assert code == 400
        # 12 matches each way needs 24 simulations: over a cap of 5, refused before any work starts
        code, body = _post(url + "/tryout", {"home": "RIV", "away": "KIN", "side": "home", "slot": 9, "player": "wes-20", "n": 12})
        assert code == 429 and "allowance" in body
    finally:
        srv.shutdown()
