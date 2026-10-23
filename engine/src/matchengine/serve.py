"""The try-out server: lets the 3D page simulate any player in any position, live.

``matchengine serve --cache data/demo/riv-kin-1119/tryouts`` and open the page with ``?api=http://localhost:8765``.

* ``POST /tryout`` with ``{"home", "away", "side", "slot", "player", "n"}`` streams one JSON line per finished
  match (the partial result, so the page's card fills in as the futures come in), then the final result.
* Only the match engine runs here: no AI models, so no model costs.
* Guard rails: one try-out at a time, at most ``MAX_N`` matches each way, and a **daily compute cap**
  (``MATCH_TRYOUT_DAILY_MATCHES``, default 240 matches, about two hours of one CPU). Past the cap the server
  refuses new try-outs until tomorrow (UTC), which keeps a hosted copy inside its free compute allowance.
  Results are cached, so repeating a try-out costs nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .formations import slots_for
from .league import get_club
from .tryout import RunCache, find_player, run_tryout

MAX_N = 24


def compact(r: dict) -> dict:
    """The fields the 3D page shows (same shape as the pre-simulated try-outs in the page)."""
    keep = ("futures", "win", "draw", "loss", "xwin", "xdraw", "xloss", "result_margin_95", "xpts", "xpts_margin_95", "xg_for",
            "xg_for_margin_95", "xg_against", "xg_against_margin_95", "minutes", "per90")

    def side(x: dict) -> dict:
        return {k: x.get(k) for k in keep}

    return {
        "candidate": {k: r["candidate"][k] for k in ("player_id", "name", "club", "role", "age")},
        "incumbent": {k: r["incumbent"][k] for k in ("player_id", "name", "club", "role", "age")},
        "position": r["position"],
        "with": side(r["with_candidate"]),
        "without": side(r["with_incumbent"]),
        "diff": r["difference"],
        "featured": None,
        "method": r.get("method"),
        "progress": r.get("progress"),
    }


class Usage:
    """Matches simulated today, persisted so a restart doesn't reset the cap."""

    def __init__(self, path: Path, cap: int):
        self.path, self.cap = path, cap
        self.lock = threading.Lock()

    def _load(self) -> dict:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            d = {}
        return d if d.get("day") == today else {"day": today, "matches": 0}

    def remaining(self) -> int:
        with self.lock:
            return max(0, self.cap - self._load()["matches"])

    def add(self, n: int) -> None:
        with self.lock:
            d = self._load()
            d["matches"] += n
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(d), encoding="utf-8")


def make_handler(cache_dir: str, usage: Usage, origin: str):
    busy = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # the response ends when the connection closes: simple streaming

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") in ("", "/health"):
                self._json(200, {"ok": True, "matches_left_today": usage.remaining(), "busy": busy.locked()})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/tryout":
                return self._json(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                home, away = str(req["home"]).upper(), str(req["away"]).upper()
                side, slot, player = req["side"], int(req["slot"]), str(req["player"])
                n = max(4, min(MAX_N, int(req.get("n") or 12)))
                if side not in ("home", "away"):
                    raise ValueError("side must be home or away")
                club = get_club(home if side == "home" else away)
                if not 0 <= slot < len(slots_for(club.formation)):
                    raise ValueError("no such position")
                find_player(player)
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                return self._json(400, {"error": f"bad request: {exc}"})
            cache = RunCache(cache_dir)
            cached = len(cache.load(home, away, side, slot, "base")) + len(cache.load(home, away, side, slot, player))
            need = max(0, 2 * n - cached)
            if need > usage.remaining():
                return self._json(429, {"error": "Today's simulation allowance is used up. Pre-simulated players still work; try again tomorrow."})
            if not busy.acquire(blocking=False):
                return self._json(429, {"error": "Another try-out is running. Try again in a minute or two."})
            try:
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                ran = [0]

                def progress(r: dict) -> None:
                    ran[0] = r["progress"]["done"]
                    self.wfile.write((json.dumps(compact(r)) + "\n").encode())
                    self.wfile.flush()

                try:
                    r = run_tryout(home, away, side, slot, player, n=n, cache_dir=cache_dir, on_progress=progress)
                    self.wfile.write((json.dumps(compact(r)) + "\n").encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the viewer left; finished matches are cached anyway
                except Exception as exc:  # noqa: BLE001 - report anything to the page
                    self.wfile.write((json.dumps({"error": str(exc)}) + "\n").encode())
                finally:
                    usage.add(ran[0])
            finally:
                busy.release()

        def log_message(self, fmt: str, *args) -> None:  # quieter logs
            print(f"[tryout] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="matchengine serve", description="Live player try-outs for the 3D page")
    ap.add_argument("--cache", required=True, help="try-out cache folder (e.g. data/demo/riv-kin-1119/tryouts)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    ap.add_argument("--origin", default=os.environ.get("MATCH_ALLOWED_ORIGIN", "*"), help="allowed page origin (CORS)")
    args = ap.parse_args(argv)
    cap = int(os.environ.get("MATCH_TRYOUT_DAILY_MATCHES", 240))
    usage = Usage(Path.home() / ".match-intelligence" / "tryout-usage.json", cap)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(args.cache, usage, args.origin))
    print(f"Try-out server on http://{args.host}:{args.port}  (daily cap {cap} matches, {usage.remaining()} left today)")
    print(f"Open the 3D page with ?api=http://{args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
