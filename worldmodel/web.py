"""Play a game (or a trained world model) in a web browser.

A tiny stdlib HTTP server (no web framework) for headless boxes like Lightning.ai
or SSH where there is no display. Run it, open the forwarded port, and play.

    python -m worldmodel.web --game ball
    python -m worldmodel.web --model runs/ball_reg32/model.pt --compare
    python -m worldmodel.web --game gym:ALE/Breakout-v5 --resize 96 --port 8000

Design: the server runs the game loop on its own clock and streams frames as a
multipart PNG stream over a single long-lived connection (``GET /stream``). The browser
just shows ``<img src="/stream">``. Input is event-driven: the page POSTs to
``/input`` only when the pressed key changes. This keeps playback smooth even
over a high-latency tunnel, because no per-frame round-trip is on the path.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from worldmodel.core.contract import Box, Discrete
from worldmodel.play import build_env

_LOCK = threading.Lock()
_ENV = None
_INFO: dict = {}
_BOX_STATE: dict = {}
_STATE = {"intent": 0, "reset": False}


def _to_action(intent: int):
    space = _ENV.action_space
    if isinstance(space, Discrete):
        return min(max(int(intent), 0), space.n - 1)
    vec, step = _BOX_STATE["vec"], _BOX_STATE["step"]
    if intent in (1, 4):
        vec[0] = min(vec[0] + step[0], space.high[0])
    elif intent in (2, 3):
        vec[0] = max(vec[0] - step[0], space.low[0])
    else:
        vec[:] = vec + 0.5 * (0.5 * (space.low + space.high) - vec)
    return space.clip(vec)


def _encode(frame: np.ndarray) -> bytes:
    # PNG: lossless, crisp on the flat colors / hard edges of these games
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame), "RGB").save(buf, "PNG")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def setup(self):
        super().setup()
        try:  # kill the ~40ms Nagle/delayed-ack stall on small writes
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def _send(self, code, content_type, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = PAGE_TEMPLATE.replace("__INFO__", json.dumps(_INFO)).encode()
            self._send(200, "text/html; charset=utf-8", page)
        elif self.path.startswith("/stream"):
            self._stream()
        elif self.path.startswith("/frame"):
            with _LOCK:
                frame = _ENV._cur if _ENV._cur is not None else _ENV.reset()
                _ENV._cur = frame
            self._send(200, "image/png", _encode(frame))
        else:
            self._send(404, "text/plain", b"not found")

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        target = 1.0 / max(_INFO["fps"], 1.0)
        try:
            while True:
                t0 = time.time()
                with _LOCK:
                    if _STATE["reset"]:
                        frame = _ENV.reset()
                        _STATE["reset"] = False
                    else:
                        frame = _ENV.step(_to_action(_STATE["intent"]))[0]
                    _ENV._cur = frame
                png = _encode(frame)
                self.wfile.write(
                    b"--FRAME\r\nContent-Type: image/png\r\nContent-Length: "
                    + str(len(png)).encode()
                    + b"\r\n\r\n"
                    + png
                    + b"\r\n"
                )
                dt = time.time() - t0
                if dt < target:
                    time.sleep(target - dt)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client navigated away / reloaded

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode().strip() if n else ""
        if self.path == "/input":
            _STATE["intent"] = int(body or "0")
            self._send(200, "text/plain", b"ok")
        elif self.path == "/reset":
            _STATE["reset"] = True
            self._send(200, "text/plain", b"ok")
        else:
            self._send(404, "text/plain", b"not found")


PAGE_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>world model</title>
<style>
  body{background:#0d0d12;color:#cfd2dc;font-family:ui-monospace,Menlo,Consolas,monospace;
       display:flex;flex-direction:column;align-items:center;gap:14px;padding:24px}
  img{image-rendering:pixelated;border:1px solid #2a2c38;border-radius:8px;
      width:min(88vw,860px);height:auto;background:#000}
  .hud{font-size:13px;color:#8a8e9c;text-align:center;line-height:1.7}
  .keys{color:#cfd2dc}
  button{background:#1c1e28;color:#cfd2dc;border:1px solid #2a2c38;border-radius:6px;
         padding:6px 14px;font-family:inherit;cursor:pointer}
  b{color:#e8704a}
</style></head>
<body>
  <div class="hud" id="title"></div>
  <img id="screen" src="/stream" alt="loading stream...">
  <div class="hud">
    <span class="keys">W A S D</span> or <span class="keys">arrow keys</span> to move &nbsp;|&nbsp;
    <span class="keys">0-9</span> action &nbsp;|&nbsp; <span class="keys">R</span> reset
  </div>
  <!-- tabindex/-1 + mousedown-preventDefault so the button never takes keyboard
       focus; otherwise Space would activate it and reset instead of braking -->
  <button type="button" tabindex="-1" onmousedown="event.preventDefault()" onclick="doReset()">reset (R)</button>
<script>
const INFO = __INFO__;
document.getElementById('title').innerHTML =
  '<b>' + INFO.name + '</b> &nbsp; mode: ' + INFO.mode +
  (INFO.compare ? ' &nbsp;(left = real game, right = model dream)' : '') +
  ' &nbsp; streaming @ ' + INFO.fps + ' fps';

const pressed = new Set();
const DIR = {ArrowUp:1,w:1,ArrowDown:2,s:2,ArrowLeft:3,a:3,ArrowRight:4,d:4};
let lastIntent = -1;
function intent(){
  for (const k of pressed) if (k>='0' && k<='9') return parseInt(k);
  if (pressed.has(' ') && INFO.space_action) return INFO.space_action;
  for (const k of pressed) if (DIR[k]!==undefined) return DIR[k];
  return 0;
}
function sendIntent(){
  const i = intent();
  if (i === lastIntent) return;       // only POST when the action changes
  lastIntent = i;
  fetch('/input', {method:'POST', body:String(i)}).catch(()=>{});
}
addEventListener('keydown', e => {
  const k = e.key.length===1 ? e.key.toLowerCase() : e.key;
  if (k==='r'){ doReset(); e.preventDefault(); return; }
  pressed.add(k);
  if (DIR[k]!==undefined || k===' ' || (k>='0'&&k<='9')) e.preventDefault();
  sendIntent();
});
addEventListener('keyup', e => {
  const k = e.key.length===1 ? e.key.toLowerCase() : e.key;
  if (DIR[k]!==undefined || k===' ' || (k>='0'&&k<='9')) e.preventDefault();
  pressed.delete(k); sendIntent();
});
function doReset(){ fetch('/reset', {method:'POST'}).catch(()=>{}); }
</script>
</body></html>"""


def serve(args):
    global _ENV, _INFO
    env, ball_like, cfg, full = build_env(args)
    env._cur = None
    _ENV = env

    frame = env.reset()
    env._cur = frame
    h, w = frame.shape[:2]
    if isinstance(env.action_space, Box):
        sp = env.action_space
        _BOX_STATE["vec"] = (0.5 * (sp.low + sp.high)).astype(np.float32)
        _BOX_STATE["step"] = (0.34 * (sp.high - sp.low)).astype(np.float32)

    mode = "dream" if getattr(args, "model", None) else "real game"
    _INFO = {
        "name": getattr(env, "name", args.game),
        "mode": mode,
        "compare": bool(getattr(args, "compare", False)),
        "w": int(w),
        "h": int(h),
        "fps": float(args.fps),
        "space_action": int(getattr(env, "space_action", 0)),
    }

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {_INFO['name']} ({mode}) streaming @ {args.fps:g} fps")
    print(f"  local:  http://localhost:{args.port}")
    lightning_host = os.environ.get("LIGHTNING_CLOUDSPACE_HOST")
    if lightning_host:
        print(f"  public: https://{args.port}-{lightning_host}")
    else:
        print(f"  bound to {args.host}:{args.port} -- forward this port and open it")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()
        if hasattr(env, "close"):
            env.close()


def main():
    p = argparse.ArgumentParser(description="Play a game or world model in the browser")
    p.add_argument("--game", default="ball")
    p.add_argument("--model", default=None, help="checkpoint to play the world model instead")
    p.add_argument("--compare", action="store_true", help="with --model: real | dream side by side")
    p.add_argument("--sampler-steps", dest="sampler_steps", type=int, default=None)
    p.add_argument("--sampler", choices=["heun", "ddim"], default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resize", type=int, default=None, help="resize frames to NxN")
    p.add_argument("--resolution", type=int, default=None,
                   help="render the real game at this resolution (crisper; real-game mode only)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    serve(args)


if __name__ == "__main__":
    main()
