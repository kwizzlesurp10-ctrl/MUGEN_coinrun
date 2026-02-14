#!/usr/bin/env python3
"""Web UI for music generation with loading screen."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATUS_FILE = PROJECT_ROOT / "output" / "generation_status.json"
DEFAULT_CODEC = PROJECT_ROOT / "checkpoints" / "codec" / "codec_final.pt"
DEFAULT_MODEL = PROJECT_ROOT / "checkpoints" / "model" / "model_final.pt"
DEFAULT_PRESET = "tiny"


class GenerateRequest(BaseModel):
    prompt: str
    codec: str | None = None
    model: str | None = None
    preset: str = DEFAULT_PRESET


app = FastAPI(title="MusicGen UI")


def _read_status() -> dict:
    if not STATUS_FILE.exists():
        return {"status": "idle"}
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"status": "idle"}


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


def _checkpoints_ready() -> dict:
    codec_ok = DEFAULT_CODEC.exists()
    model_ok = DEFAULT_MODEL.exists()
    return {
        "ready": codec_ok and model_ok,
        "codec": codec_ok,
        "model": model_ok,
    }


@app.get("/api/checkpoints")
def get_checkpoints() -> JSONResponse:
    return JSONResponse(content=_checkpoints_ready())


@app.get("/api/status")
def get_status() -> JSONResponse:
    return JSONResponse(content=_read_status())


@app.post("/api/generate")
def start_generate(body: GenerateRequest) -> JSONResponse:
    status = _read_status()
    if status.get("status") == "running":
        return JSONResponse(
            content={"error": "Generation already in progress"},
            status_code=409,
        )
    codec_path = body.codec or str(DEFAULT_CODEC)
    model_path = body.model or str(DEFAULT_MODEL)
    preset = body.preset or DEFAULT_PRESET
    if not Path(codec_path).exists():
        return JSONResponse(
            content={"error": f"Codec not found: {codec_path}"},
            status_code=400,
        )
    if not Path(model_path).exists():
        return JSONResponse(
            content={"error": f"Model not found: {model_path}"},
            status_code=400,
        )
    output_path = PROJECT_ROOT / "output.wav"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    subprocess.Popen(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate.py"),
            "--prompt",
            body.prompt,
            "--codec",
            codec_path,
            "--model",
            model_path,
            "--output",
            str(output_path),
            "--preset",
            preset,
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return JSONResponse(content={"status": "started", "prompt": body.prompt})


@app.get("/output.wav")
def get_output_wav():
    from fastapi.responses import FileResponse

    path = PROJECT_ROOT / "output.wav"
    if not path.exists():
        return JSONResponse(content={"error": "Not found"}, status_code=404)
    return FileResponse(
        path,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MusicGen</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @keyframes pulse-glow {
      0%, 100% { opacity: 0.6; transform: scale(1); }
      50% { opacity: 1; transform: scale(1.02); }
    }
    @keyframes spin-slow {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
    .loading-spinner { animation: spin-slow 2s linear infinite; }
    .loading-card { animation: pulse-glow 1.5s ease-in-out infinite; }
  </style>
</head>
<body class="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col items-center justify-center p-6 font-sans">
  <main class="w-full max-w-lg space-y-6">
    <h1 class="text-2xl font-semibold text-center text-zinc-100">MusicGen</h1>
    <form id="form" class="space-y-4 p-4 rounded-xl bg-zinc-900/80 border border-zinc-800">
      <label class="block">
        <span class="text-sm text-zinc-400 block mb-1">Prompt</span>
        <input type="text" id="prompt" name="prompt" value="chill beats" placeholder="e.g. chill beats"
          class="w-full px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500">
      </label>
      <button type="submit" id="submitBtn"
        class="w-full py-2.5 rounded-lg bg-amber-500 text-zinc-950 font-medium hover:bg-amber-400 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:ring-offset-2 focus:ring-offset-zinc-950 disabled:opacity-50 disabled:pointer-events-none">
        Generate
      </button>
    </form>

    <div id="loading" class="hidden loading-card p-6 rounded-xl bg-zinc-900/90 border border-amber-500/30 text-center">
      <div class="loading-spinner inline-block w-12 h-12 rounded-full border-2 border-amber-500/50 border-t-amber-500 mb-3"></div>
      <p class="text-amber-200 font-medium">Generating…</p>
      <p id="loadingPrompt" class="text-sm text-zinc-400 mt-1"></p>
      <p id="loadingPct" class="text-sm text-amber-300/90 mt-2 font-mono"></p>
      <div class="mt-3 h-1.5 w-full rounded-full bg-zinc-800 overflow-hidden">
        <div id="loadingBar" class="h-full bg-amber-500 transition-[width] duration-300 ease-out" style="width: 0%"></div>
      </div>
    </div>

    <div id="result" class="hidden p-6 rounded-xl bg-zinc-900/80 border border-zinc-800">
      <p class="text-sm text-zinc-400 mb-2">Done</p>
      <audio id="audio" controls preload="auto" class="w-full rounded-lg"></audio>
      <p id="audioError" class="hidden mt-2 text-sm text-red-400"></p>
      <a id="download" href="/output.wav" download="output.wav" class="inline-block mt-3 text-sm text-amber-500 hover:text-amber-400">Download WAV</a>
    </div>

    <div id="error" class="hidden p-4 rounded-xl bg-red-950/50 border border-red-800 text-red-200 text-sm"></div>

    <div id="checkpointsWarning" class="hidden p-4 rounded-xl bg-amber-950/40 border border-amber-700/50 text-amber-200 text-sm">
      <p class="font-medium mb-1">Checkpoints missing</p>
      <p class="text-amber-200/90 text-xs font-mono break-all">make demo && make train-codec MANIFEST=data/demo/manifest.txt PRESET=tiny && make train-model MANIFEST=data/demo/manifest.txt PRESET=tiny</p>
    </div>
  </main>

  <script>
    const form = document.getElementById('form');
    const promptInput = document.getElementById('prompt');
    const submitBtn = document.getElementById('submitBtn');
    const loading = document.getElementById('loading');
    const loadingPrompt = document.getElementById('loadingPrompt');
    const loadingPct = document.getElementById('loadingPct');
    const loadingBar = document.getElementById('loadingBar');
    const result = document.getElementById('result');
    const audio = document.getElementById('audio');
    const audioError = document.getElementById('audioError');
    const download = document.getElementById('download');
    const errorEl = document.getElementById('error');
    const checkpointsWarning = document.getElementById('checkpointsWarning');

    function show(el) { el.classList.remove('hidden'); }
    function hide(el) { el.classList.add('hidden'); }

    function setProgress(pct) {
      const n = pct == null || pct === undefined ? 0 : Math.min(100, Math.max(0, Math.round(pct)));
      loadingBar.style.width = n + '%';
      loadingPct.textContent = n < 100 ? n + '%' : '';
    }

    audio.addEventListener('error', function() {
      audioError.textContent = 'Audio failed to load. Try downloading the WAV.';
      audioError.classList.remove('hidden');
    });
    audio.addEventListener('loadeddata', function() {
      audioError.classList.add('hidden');
    });

    async function pollStatus() {
      const r = await fetch('/api/status');
      return r.json();
    }

    async function waitForDone() {
      for (;;) {
        const s = await pollStatus();
        if (s.status === 'done') {
          setProgress(100);
          hide(loading);
          const url = '/output.wav?t=' + Date.now();
          audio.src = url;
          download.href = url;
          show(result);
          audio.load();
          audio.play().catch(function() {});
          return;
        }
        if (s.status === 'error') {
          hide(loading);
          errorEl.textContent = s.message || 'Generation failed';
          show(errorEl);
          return;
        }
        setProgress(s.progress);
        await new Promise(r => setTimeout(r, 1500));
      }
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      hide(errorEl);
      hide(result);
      const prompt = promptInput.value.trim();
      if (!prompt) return;
      submitBtn.disabled = true;
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });
      const data = await res.json();
      if (!res.ok) {
        errorEl.textContent = data.error || 'Failed to start';
        show(errorEl);
        submitBtn.disabled = false;
        return;
      }
      loadingPrompt.textContent = prompt;
      setProgress(0);
      show(loading);
      await waitForDone();
      submitBtn.disabled = false;
    });

    (async function init() {
      const cp = await fetch('/api/checkpoints').then(r => r.json());
      if (!cp.ready) show(checkpointsWarning);

      const s = await pollStatus();
      if (s.status === 'running') {
        loadingPrompt.textContent = s.prompt || '…';
        setProgress(s.progress);
        show(loading);
        submitBtn.disabled = true;
        await waitForDone();
        submitBtn.disabled = false;
      }
    })();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8765)


if __name__ == "__main__":
    main()
