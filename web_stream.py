"""
실시간 웹 스트리밍 서버 (추가 설치 불필요 - 파이썬 표준 라이브러리 + OpenCV)
  http://<로봇IP>:<port>/        대시보드 (실시간 화면 + 상태 + 종료 버튼)
  /stream                        MJPEG 영상 스트림
  /status                        상태 JSON
  /stop  (POST)                  강제 종료 요청
"""
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>따개비 청소 로봇 관제</title>
<style>
 :root{--bg:#0f1418;--panel:#182027;--line:#2a343d;--text:#e6edf3;--muted:#8b98a5;--ok:#3fb950;--warn:#f0883e;--bad:#f85149}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"Malgun Gothic",sans-serif}
 header{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 20px;border-bottom:1px solid var(--line)}
 h1{font-size:18px;margin:0}
 .wrap{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:16px;padding:16px 20px;align-items:start}
 @media(max-width:900px){.wrap{grid-template-columns:1fr}}
 .video{background:#000;border:1px solid var(--line);border-radius:8px;overflow:hidden}
 .video img{display:block;width:100%;height:auto}
 .panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}
 .row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--line);font-size:14px}
 .row span:first-child{color:var(--muted)} .big{font-size:28px;font-weight:700;color:var(--warn)}
 #stop{background:var(--bad);color:#fff;border:0;border-radius:8px;padding:12px 20px;font-size:16px;font-weight:700;cursor:pointer}
 #stop:disabled{opacity:.5;cursor:default}
 #banner{display:none;margin:0 20px;padding:10px 14px;border-radius:8px;font-weight:700}
 #log{font-family:ui-monospace,Consolas,monospace;font-size:12px;line-height:1.5;max-height:260px;overflow:auto;margin-top:10px;color:var(--muted)}
 #log .obs{color:var(--warn);font-weight:700} #log .end{color:var(--ok);font-weight:700} #log .stop{color:var(--bad);font-weight:700}
</style></head><body>
<header><h1>따개비 청소 로봇 · 실시간 관제</h1><button id="stop">■ 강제 종료</button></header>
<div id="banner"></div>
<div class="wrap">
 <div class="video"><img src="/stream" alt="실시간 감지 화면"></div>
 <div class="panel">
  <div class="row"><span>따개비 구역</span><span class="big" id="zones">-</span></div>
  <div class="row"><span>상태</span><span id="state">-</span></div>
  <div class="row"><span>현재 동작</span><span id="action">-</span></div>
  <div class="row"><span>왕복 진행</span><span id="plan">-</span></div>
  <div class="row"><span>청소 통과(누적)</span><span id="passes">-</span></div>
  <div class="row"><span>장애물 회피(누적)</span><span id="obs">-</span></div>
  <div class="row"><span>연속 미검출</span><span id="empty">-</span></div>
  <div id="log"></div>
 </div>
</div>
<script>
const $=id=>document.getElementById(id);
const banner=(msg,color)=>{const b=$('banner');b.style.display='block';b.style.background=color;b.textContent=msg;};
$('stop').onclick=async()=>{$('stop').disabled=true;try{await fetch('/stop',{method:'POST'});}catch(e){}
  banner('강제 종료 요청을 보냈습니다. 하드웨어 안전 정지 중...','#f85149');};
async function poll(){
  try{
    const s=await (await fetch('/status',{cache:'no-store'})).json();
    $('zones').textContent=s.zones; $('state').textContent=s.state; $('action').textContent=s.action; $('plan').textContent=s.plan||'-';
    $('passes').textContent=s.passes; $('obs').textContent=s.obstacles;
    $('empty').textContent=s.empty_checks;
    $('log').innerHTML=s.events.map(e=>`<div class="${e.kind}">${e.t} ${e.msg}</div>`).reverse().join('');
    if(s.finished){banner(s.final_message,s.final_kind==='end'?'#238636':'#f85149');$('stop').disabled=true;return;}
  }catch(e){banner('로봇과의 연결이 끊겼습니다 (프로그램 종료됨)','#6e7681');$('stop').disabled=true;return;}
  setTimeout(poll,500);
}
poll();
</script></body></html>"""


class WebStreamServer:
    def __init__(self, host="0.0.0.0", port=8000, on_stop=None, jpeg_quality=75):
        self.host, self.port = host, port
        self.on_stop = on_stop or (lambda reason: None)
        self.jpeg_quality = jpeg_quality
        self._cond = threading.Condition()
        self._jpeg = None
        self._frame_id = 0
        self._running = False
        self.status = {"zones": "-", "state": "-", "action": "-", "passes": 0,
                       "obstacles": 0, "empty_checks": "-", "finished": False,
                       "final_message": "", "final_kind": ""}
        self.events = deque(maxlen=40)
        self._httpd = None
        self._thread = None

    # ---------- 데이터 갱신 ----------
    def push_frame(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._frame_id += 1
            self._cond.notify_all()

    def update_status(self, **kw):
        self.status.update(kw)

    def add_event(self, msg, kind=""):
        self.events.append({"t": time.strftime("%H:%M:%S"), "msg": msg, "kind": kind})

    # ---------- 서버 ----------
    def start(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # 콘솔 로그 억제
                pass

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                elif self.path.startswith("/status"):
                    data = dict(server.status, events=list(server.events))
                    self._send(200, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                               "application/json; charset=utf-8")
                elif self.path.startswith("/stream"):
                    self._stream()
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                if self.path.startswith("/stop"):
                    self._send(200, b'{"ok":true}', "application/json")
                    server.on_stop("웹 인터페이스 종료 버튼")
                else:
                    self._send(404, b"not found", "text/plain")

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = -1
                try:
                    while server._running:
                        with server._cond:
                            server._cond.wait_for(lambda: server._frame_id != last or not server._running, timeout=1.0)
                            if not server._running:
                                break
                            if server._frame_id == last or server._jpeg is None:
                                continue
                            jpeg, last = server._jpeg, server._frame_id
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass  # 브라우저 탭 닫힘

        ThreadingHTTPServer.daemon_threads = True
        ThreadingHTTPServer.allow_reuse_address = True
        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._running = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        print(f"[Web] 실시간 관제 화면: http://localhost:{self.port}  (같은 네트워크에서는 http://<로봇IP>:{self.port})")

    def stop(self):
        if not self._httpd:
            return
        self._running = False
        with self._cond:
            self._cond.notify_all()
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        print("[Web] 스트리밍 서버 종료")
