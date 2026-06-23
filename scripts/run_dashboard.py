# A_EYE 웹 대시보드 HTTP 서버와 SSE API를 실행합니다.
import argparse
import json
import math
import mimetypes
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dashboard.runtime import DashboardRuntime


ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT_DIR / "web_dashboard"
STATIC_DIR_RESOLVED = STATIC_DIR.resolve()
RUNTIME = DashboardRuntime(sensor_count=200, interval_sec=0.5)


class ClientError(ValueError):
    """Request errors that should be reported as 400 responses."""


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server_version = "A_EYE-Dashboard/0.1"

    def log_message(self, fmt, *args):
        sys.stdout.write("[dashboard] " + fmt % args + "\n")

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/state":
                self._send_json(RUNTIME.snapshot())
                return
            if parsed.path == "/api/sensor":
                params = parse_qs(parsed.query)
                sensor_id = self._parse_int(params.get("id", ["0"])[0], "id")
                self._send_json(RUNTIME.sensor_payload(sensor_id))
                return
            if parsed.path == "/api/events":
                self._send_events(parsed.query)
                return
            if parsed.path in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html")
                return
            candidate = self._resolve_static_path(parsed.path)
            if candidate.exists() and candidate.is_file():
                self._send_file(candidate)
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except ClientError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            body = self._read_json_body()
            if parsed.path == "/api/feedback":
                is_actual = self._parse_bool(body.get("is_actual_anomaly", True), "is_actual_anomaly")
                self._send_json(RUNTIME.apply_feedback(is_actual))
                return
            if parsed.path == "/api/threshold":
                if "threshold" not in body:
                    self._send_json({"error": "threshold is required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                threshold = self._parse_float(body["threshold"], "threshold")
                self._send_json(RUNTIME.update_threshold(threshold))
                return
            if parsed.path == "/api/control":
                action = body.get("action")
                if action == "start":
                    self._send_json(RUNTIME.set_running(True))
                    return
                if action == "stop":
                    self._send_json(RUNTIME.set_running(False))
                    return
                if action == "reset":
                    self._send_json(RUNTIME.reset())
                    return
                self._send_json({"error": "action must be start, stop, or reset"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except ClientError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ClientError("invalid content length") from exc
        if length <= 0:
            return {}
        data = self.rfile.read(length).decode("utf-8")
        try:
            return json.loads(data) if data else {}
        except json.JSONDecodeError as exc:
            raise ClientError("invalid JSON body") from exc

    def _send_json(self, payload, status=HTTPStatus.OK):
        encoded = json.dumps(payload, ensure_ascii=False, default=json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_file(self, path: Path):
        if not path.exists() or not path.is_file():
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        content = path.read_bytes()
        mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".js":
            mime_type = "text/javascript"
        content_type = mime_type + "; charset=utf-8" if mime_type.startswith("text/") else mime_type
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_events(self, query: str):
        params = parse_qs(query)
        poll_sec = self._parse_float(params.get("poll", ["0.5"])[0], "poll")
        poll_sec = max(0.1, min(5.0, poll_sec))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last_revision = -1
        while True:
            snapshot = RUNTIME.snapshot()
            current_revision = int(snapshot.get("revision", snapshot.get("step", 0)))
            if current_revision != last_revision:
                data = json.dumps(snapshot, ensure_ascii=False, default=json_default)
                message = f"event: state\ndata: {data}\n\n".encode("utf-8")
                try:
                    self.wfile.write(message)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    break
                last_revision = current_revision
            time.sleep(poll_sec)

    def _resolve_static_path(self, request_path: str) -> Path:
        decoded = unquote(request_path).replace("\\", "/").lstrip("/")
        parts = []
        for part in decoded.split("/"):
            if part in ("", "."):
                continue
            if part == ".." or ":" in part:
                raise ClientError("invalid static path")
            parts.append(part)
        if not parts:
            parts = ["index.html"]
        candidate = STATIC_DIR.joinpath(*parts).resolve()
        try:
            candidate.relative_to(STATIC_DIR_RESOLVED)
        except ValueError as exc:
            raise ClientError("invalid static path") from exc
        return candidate

    def _parse_int(self, value, field: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ClientError(f"{field} must be an integer") from exc

    def _parse_float(self, value, field: str) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ClientError(f"{field} must be a number") from exc
        if not math.isfinite(parsed):
            raise ClientError(f"{field} must be finite")
        return parsed

    def _parse_bool(self, value, field: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes", "y"):
                return True
            if lowered in ("false", "0", "no", "n"):
                return False
        raise ClientError(f"{field} must be a boolean")


def run(host: str, port: int):
    RUNTIME.start()
    server = ThreadingHTTPServer((host, port), DashboardRequestHandler)
    print(f"A_EYE dashboard running at http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        RUNTIME.shutdown()
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run A_EYE web dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run(args.host, args.port)
