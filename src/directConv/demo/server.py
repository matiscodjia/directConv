"""Local web demo: `directconv-demo` (or `python -m directConv.demo`).

Standard library only. The server binds to localhost, and the browser can only pick a preset or
paste an http(s) URL: it can never make the server open an arbitrary local path.

Pages: `/` (home), `/lab` (the interactive comparison), `/why` (numpy vs frugal_ml), `/method`.
"""

import argparse
import json
import re
import socket
import ssl
import subprocess
import tempfile
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from directConv import _native, edge
from directConv.demo import runner

STATIC = Path(__file__).parent / "static"
PAGES = {"/": "home.html", "/lab": "lab.html", "/why": "why.html", "/method": "method.html"}
ASSET = re.compile(r"^[a-z0-9_.-]+\.(css|js|json|svg)$")
ASSET_TYPES = {"css": "text/css", "js": "text/javascript", "json": "application/json", "svg": "image/svg+xml"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}
IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
MEDIA_NAME = re.compile(r"^[0-9a-f]{32}\.mp4$")
MAX_BODY = 16_384
MAX_JOBS = 8  # finished jobs kept for polling


class Jobs:
    """Background runs, one at a time (timings must not compete), polled by the page."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._running = threading.Lock()

    def start(self, work) -> str | None:
        """Starts `work(progress)` in a thread, or returns None if a job is already running.
        `progress(step, text, pct)` updates what the page polls."""
        if not self._running.acquire(blocking=False):
            return None
        job_id = uuid.uuid4().hex
        job = {"state": "running", "step": "start", "stage": "Starting", "pct": None}
        with self._lock:
            self._jobs[job_id] = job
            for old in [k for k, v in self._jobs.items() if v["state"] != "running"][:-MAX_JOBS]:
                del self._jobs[old]

        def progress(step: str, text: str = "", pct: float | None = None):
            job.update(step=step, stage=text or step, pct=pct)

        def target():
            try:
                job["result"] = work(progress)
                job["state"] = "done"
            except Exception as exc:  # decoding, network, ...: report, keep serving
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["state"] = "error"
            finally:
                self._running.release()

        threading.Thread(target=target, daemon=True).start()
        return job_id

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


class Handler(BaseHTTPRequestHandler):
    presets: dict[str, str]
    media_dir: Path
    jobs = Jobs()
    lan = False  # --lan: also accept requests addressed to an IP address (a phone on the same Wi-Fi)

    server_version = "directConvDemo"

    def log_message(self, fmt, *args):
        pass  # keep the terminal for the URL and errors

    # -- helpers ----------------------------------------------------------
    def _host_ok(self) -> bool:
        # Refuses requests whose Host is not localhost (DNS rebinding). With --lan an IP literal is
        # fine too: rebinding attacks work through host names, never through a bare IP address.
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in LOCAL_HOSTS or (self.lan and bool(IPV4.match(host)))

    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload):
        self._send(status, json.dumps(payload).encode(), "application/json")

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        if not self._host_ok():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden host"})
        route = urlparse(self.path)
        path = route.path
        if path in PAGES:
            return self._send(200, (STATIC / PAGES[path]).read_bytes(), "text/html; charset=utf-8")
        if path.startswith("/static/"):
            return self._asset(path[len("/static/") :])
        if path == "/tradeoffs.json":
            return self._asset("tradeoffs.json")
        if path == "/api/config":
            return self._json(200, self._config())
        if path == "/api/last":
            return self._json(200, {"run": runner.last_result()})
        if path == "/api/preview":
            resolution = (parse_qs(route.query).get("resolution") or [""])[0]
            if resolution not in runner.LADDER:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "unknown resolution"})
            return self._json(200, {key: runner.cards_for(resolution, key) for key in edge.DEVICES})
        if path.startswith("/api/job/"):
            job = self.jobs.get(path[len("/api/job/") :])
            return self._json(200, job) if job else self._json(HTTPStatus.NOT_FOUND, {"error": "unknown job"})
        if path.startswith("/media/"):
            return self._media(path[len("/media/") :])
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    do_HEAD = do_GET

    def _asset(self, name: str):
        file = STATIC / name
        if not ASSET.match(name) or not file.is_file():
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        self._send(200, file.read_bytes(), ASSET_TYPES[name.rsplit(".", 1)[1]] + "; charset=utf-8")

    def do_POST(self):
        if not self._host_ok():
            return self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden host"})
        if self.path not in ("/api/run", "/api/play"):
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY:
                raise ValueError("bad request body")
            req = json.loads(self.rfile.read(length))
            media = str(self.media_dir)
            if self.path == "/api/run":
                source, frames, resolution = self._source(req), int(req["frames"]), str(req["resolution"])
                runner.validate_run(frames, resolution)  # fail fast, before starting a job
                work = lambda progress: runner.run(source, frames, resolution, media, progress)  # noqa: E731
            else:
                run_id, strategy, device = str(req["run_id"]), str(req["strategy"]), str(req["device"])
                camera_fps = float(req["camera_fps"])
                work = lambda progress: runner.play(run_id, strategy, device, camera_fps, media, progress)  # noqa: E731
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": f"{type(exc).__name__}: {exc}"})
        job_id = self.jobs.start(work)
        if job_id is None:
            return self._json(HTTPStatus.CONFLICT, {"error": "another job is in progress, wait for it to finish"})
        self._json(HTTPStatus.ACCEPTED, {"job": job_id})

    def _source(self, req: dict) -> str:
        if req.get("preset") is not None:
            if req["preset"] not in self.presets:
                raise ValueError("unknown preset")
            return self.presets[req["preset"]]
        url = str(req.get("url", "")).strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError("paste an http(s) video URL, or pick a preset")
        return url

    def _config(self) -> dict:
        return {
            "presets": list(self.presets),
            "length_steps": runner.LENGTH_STEPS,
            "default_frames": runner.DEFAULT_FRAMES,
            "max_camera_fps": runner.MAX_CAMERA_FPS,
            "ladder": [
                {
                    "name": name, "h": h, "w": w,
                    "direct_bytes": edge.direct_bytes(h, w),
                    "stream_bytes": edge.stream_bytes(w),
                    "max_frames": runner.max_frames(name),
                    "max_camera_fps": runner.max_camera_fps(h - 2, w - 2),
                }
                for name, h, w in edge.LADDER
            ],
            "devices": {
                key: {"name": d.name, "budget": d.budget, "note": d.note, "on_chip": d.clock_hz is not None, "ram": d.ram, "verified": d.verified}
                for key, d in edge.DEVICES.items()
            },
            "registered_shapes": len(_native.registered_shapes()),
        }  # fmt: skip

    def _media(self, name: str):
        if not MEDIA_NAME.match(name) or not (self.media_dir / name).is_file():
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        data = (self.media_dir / name).read_bytes()
        headers = {"Accept-Ranges": "bytes"}
        match = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range") or "")
        if match and (match[1] or match[2]):
            if match[1]:
                start = int(match[1])
                end = int(match[2]) if match[2] else len(data) - 1
            else:  # suffix range: the last N bytes
                start, end = max(0, len(data) - int(match[2])), len(data) - 1
            end = min(end, len(data) - 1)
            if start > end:
                return self._send(416, b"", "video/mp4", {"Content-Range": f"bytes */{len(data)}"})
            headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
            return self._send(206, data[start : end + 1], "video/mp4", headers)
        self._send(200, data, "video/mp4", headers)


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("10.255.255.255", 1))  # no packet is sent: this only picks the outgoing interface
        return probe.getsockname()[0]


def wrap_tls(server: ThreadingHTTPServer, ip: str, directory: str) -> None:
    """Serves `server` over HTTPS with a self-signed certificate for `ip`.

    Some browsers (Safari with "HTTPS-Only") refuse plain http:// links outright. The certificate is
    not trusted by anyone, so the browser warns once; it protects nothing beyond the phone's demands.
    """
    cert, key = f"{directory}/cert.pem", f"{directory}/key.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "7", "-subj", "/CN=directConv",
         "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost", "-keyout", key, "-out", cert],
        check=True, capture_output=True,
    )  # fmt: skip
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    # Handshake in the request thread, not in accept(): a client that stalls must not block the others.
    server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="frugal_ml (Rust) vs numpy, on a video, in the browser.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--video", help="also offer this local video file as a preset")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--lan", action="store_true", help="also listen on the local network, to open the demo from a phone")
    parser.add_argument("--https", action="store_true", help="serve HTTPS with a self-signed certificate (for browsers that refuse http://)")
    args = parser.parse_args()

    presets = dict(runner.PRESETS)
    if args.video:
        presets[f"Local file: {Path(args.video).name}"] = str(Path(args.video).resolve())

    with tempfile.TemporaryDirectory(prefix="directconv-demo-") as media_dir, tempfile.TemporaryDirectory(prefix="directconv-tls-") as tls_dir:
        Handler.presets = presets
        Handler.media_dir = Path(media_dir)
        Handler.lan = args.lan
        server = ThreadingHTTPServer(("0.0.0.0" if args.lan else "127.0.0.1", args.port), Handler)
        ip = lan_ip() if args.lan else "127.0.0.1"
        if args.https:
            wrap_tls(server, ip, tls_dir)
        scheme = "https" if args.https else "http"
        url = f"{scheme}://localhost:{args.port}"
        print(f"directConv demo on {url}  (Ctrl+C to stop)", flush=True)
        if args.lan:
            print(f"On your phone (same Wi-Fi): {scheme}://{ip}:{args.port}", flush=True)
            if args.https:
                print("The browser will warn about the certificate (self-signed): choose to visit the site anyway.", flush=True)
            print("Anyone on this network can use the demo while it runs, and make it download a video URL.", flush=True)
        if not args.no_browser:
            webbrowser.open(url + "/lab")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
