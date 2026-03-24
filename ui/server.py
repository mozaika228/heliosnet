from __future__ import annotations

import asyncio
import json
from pathlib import Path
from threading import Thread
from typing import Any

from core.service import BaseService

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, StreamingResponse
    import uvicorn
except Exception:
    FastAPI = None
    HTTPException = Exception
    HTMLResponse = None
    StreamingResponse = None
    uvicorn = None


class WebUIService(BaseService):
    def __init__(self, config, metrics, live_state):
        super().__init__("web_ui")
        self.config = config
        self.metrics = metrics
        self.live_state = live_state
        ui_cfg = getattr(config, "observability", {})
        self._enabled = bool(ui_cfg.get("web_ui", True))
        self._host = str(ui_cfg.get("web_ui_host", "0.0.0.0"))
        self._port = int(ui_cfg.get("web_ui_port", 8080))
        dist_cfg = getattr(config, "distributed", {})
        self._operator_cmd_path = Path(dist_cfg.get("operator_commands_path", "./data/operator_commands.jsonl"))
        self._operator_cmd_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = None
        self._started = False
        if self._enabled:
            self._start_server()

    def handle(self, item) -> None:
        self.push(item)

    def tick(self) -> None:
        return

    def _start_server(self) -> None:
        if not self._enabled or self._started:
            return
        if FastAPI is None or uvicorn is None:
            print("[web_ui] disabled: fastapi/uvicorn not installed", flush=True)
            return
        app = self._build_app()

        def _run() -> None:
            uvicorn.run(app, host=self._host, port=self._port, log_level="warning")

        self._thread = Thread(target=_run, daemon=True)
        self._thread.start()
        self._started = True
        print(f"[web_ui] listening on http://{self._host}:{self._port}", flush=True)

    def _build_app(self):
        app = FastAPI(title="HeliosNet Control")

        @app.get("/", response_class=HTMLResponse)
        def index():
            return _html()

        @app.get("/api/state")
        def state():
            return self.live_state.snapshot()

        @app.get("/api/events/stream")
        async def stream(seq: int = 0):
            async def gen():
                last = seq
                while True:
                    rows = self.live_state.events_since(last)
                    if rows:
                        for row in rows:
                            last = max(last, int(row.get("seq", 0)))
                            yield f"data: {json.dumps(row, ensure_ascii=True)}\\n\\n"
                    await asyncio.sleep(0.5)

            return StreamingResponse(gen(), media_type="text/event-stream")

        @app.post("/api/command")
        def command(cmd: dict[str, Any]):
            action = str(cmd.get("action", ""))
            if not action:
                raise HTTPException(status_code=400, detail="missing action")
            with self._operator_cmd_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(cmd, ensure_ascii=True) + "\\n")
            return {"ok": True, "queued": cmd}

        return app


def _html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width,initial-scale=1'/>
  <title>HeliosNet Command Grid</title>
  <style>
    :root { --bg:#0a1116; --card:#101e28; --ink:#d5e9f2; --accent:#11d3a7; --warn:#ffb84a; --danger:#ff5d73; --line:#2b4556; }
    *{box-sizing:border-box}
    body{margin:0;font-family:'Space Grotesk','Segoe UI',sans-serif;background:radial-gradient(1200px 700px at 80% -100px,#153143 0%,#0a1116 60%),var(--bg);color:var(--ink)}
    .grid{display:grid;grid-template-columns:1.4fr 1fr;gap:16px;padding:16px;min-height:100vh}
    .panel{background:linear-gradient(180deg,#11202a,#0c171f);border:1px solid var(--line);border-radius:14px;box-shadow:0 10px 30px rgba(0,0,0,.35)}
    .head{padding:12px 14px;border-bottom:1px solid var(--line);font-weight:700;letter-spacing:.04em}
    .body{padding:12px}
    #map{height:520px;position:relative;overflow:hidden;background:linear-gradient(180deg,#132834,#0c1a22)}
    .scan{position:absolute;inset:0;background:repeating-linear-gradient(0deg,transparent 0 18px,rgba(17,211,167,.04) 19px 20px);animation:drift 8s linear infinite}
    @keyframes drift{from{transform:translateY(0)}to{transform:translateY(20px)}}
    .dot{position:absolute;width:10px;height:10px;border-radius:999px;background:var(--accent);box-shadow:0 0 12px var(--accent)}
    .source{padding:10px;border:1px solid var(--line);border-radius:10px;margin:8px 0;background:#0f1d26}
    .kpi{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
    .pill{padding:8px;border-radius:10px;background:#0f1b23;border:1px solid var(--line)}
    .events{height:300px;overflow:auto;font-size:13px}
    .evt{padding:8px;border-bottom:1px solid #1f3340}
    .cmd{display:grid;grid-template-columns:1fr 1fr;gap:8px}
    button{background:#17303f;color:var(--ink);border:1px solid #2a4b5f;border-radius:10px;padding:10px 12px;cursor:pointer}
    button:hover{filter:brightness(1.1)}
    .warn{border-color:#8b5f18;background:#2f230f}
  </style>
</head>
<body>
  <div class='grid'>
    <div class='panel'>
      <div class='head'>DIGITAL TWIN</div>
      <div class='body'>
        <div id='map'><div class='scan'></div></div>
      </div>
    </div>
    <div>
      <div class='panel' style='margin-bottom:16px'>
        <div class='head'>MISSION STATS</div>
        <div class='body'>
          <div class='kpi'>
            <div class='pill'>Frames: <span id='frames'>0</span></div>
            <div class='pill'>Objects: <span id='objects'>0</span></div>
            <div class='pill'>Sources: <span id='sources'>0</span></div>
          </div>
          <div id='srcs'></div>
        </div>
      </div>
      <div class='panel' style='margin-bottom:16px'>
        <div class='head'>EVENT FEED</div>
        <div class='body events' id='events'></div>
      </div>
      <div class='panel'>
        <div class='head'>COMMAND LOOP</div>
        <div class='body cmd'>
          <button onclick='cmd({action:"set_battery",percent:80})'>Set Battery 80%</button>
          <button onclick='cmd({action:"set_battery",percent:20})' class='warn'>Set Battery 20%</button>
          <button onclick='cmd({action:"model_promote"})'>Promote Canary</button>
          <button onclick='cmd({action:"model_rollback"})' class='warn'>Rollback Model</button>
        </div>
      </div>
    </div>
  </div>
<script>
async function loadState(){
  const r=await fetch('/api/state'); const s=await r.json();
  document.getElementById('frames').textContent=s.stats.frames||0;
  document.getElementById('objects').textContent=s.stats.objects||0;
  const srcs=s.sources||{}; const keys=Object.keys(srcs);
  document.getElementById('sources').textContent=keys.length;
  const srcBox=document.getElementById('srcs'); srcBox.innerHTML='';
  const map=document.getElementById('map'); map.querySelectorAll('.dot').forEach(x=>x.remove());
  keys.forEach((k,i)=>{
    const row=srcs[k];
    const el=document.createElement('div'); el.className='source';
    el.textContent=`${k} tracks=${row.track_count} objects=${row.object_count}`;
    srcBox.appendChild(el);
    const d=document.createElement('div'); d.className='dot';
    d.style.left=(20+(i*70)%80)+'%'; d.style.top=(20+((i*43)%60))+'%';
    map.appendChild(d);
  });
}
function addEvent(e){
  const box=document.getElementById('events');
  const el=document.createElement('div'); el.className='evt';
  el.textContent=`${new Date((e.ts||0)*1000).toLocaleTimeString()} ${e.source_id||''} ${e.name||''}`;
  box.prepend(el); while(box.children.length>120) box.removeChild(box.lastChild);
}
function stream(){
  const es=new EventSource('/api/events/stream');
  es.onmessage=(m)=>{ try{ addEvent(JSON.parse(m.data)); }catch(_){} };
}
async function cmd(payload){
  await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
}
setInterval(loadState,1000); loadState(); stream();
</script>
</body>
</html>"""
