"""redline-tiny server: Qwen-Image-2.1-Turbo edits and text-to-image on the Ryzen AI NPU, plus the web UI.

Text encoder (Qwen3-VL), DiT and VAE decoder run on the NPU; the VAE encoder runs on the iGPU (DirectML) in a helper
process, at the same time as the text encoder. One job at a time.

  GET  /                          the web UI (redline/ui/)
  GET  /api/status                {state: loading|ready|busy|error, message, load_s, job, limits}
  POST /api/tokens                {kind, prompt} -> {tokens, max, ok}
  POST /api/run                   {kind: edit|generate, prompt, image?: base64, seed?, steps?: 8|6, crop?} -> result
  GET  /api/history               {items: [result, ...]} newest first
  POST /api/open-folder           {id?} opens the outputs folder in Explorer
  POST /v1/images/edits           OpenAI-style: {prompt, image: base64, seed?, steps?, crop?} -> {data: [{b64_json}], ...}
  POST /v1/images/generations     OpenAI-style: {prompt, seed?, steps?}
  GET  /health

Memory: NPU buffers are pinned host RAM and the DiT alone peaks at ~12 GB, so only the DiT stays resident on the
device; the text encoder and VAE decoder keep their weights hot on the host and rebuild their device buffers per job.
"""

import argparse
import base64
import gc
import json
import msvcrt
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

import paths

HERE = Path(__file__).resolve().parent
UI = HERE / "ui"
TMP = paths.CACHE / "tmp"
REUSE = {8: (), 6: (2, 4)}           # DiT runs -> steps that reuse the previous velocity
MAX_TEXT = {"edit": 64, "generate": 512}   # text rows the prebuilt NPU programs cover (see check_prompt)
SYS = "Comprehend and analyze the provided prompt."
STEPS_TOTAL = 8


class NpuLock:
    """Machine-wide lock so two redline processes (or a benchmark) never share the NPU."""

    def __enter__(self):
        paths.LOCK.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(paths.LOCK, "a+"); self.f.seek(0)
        while True:
            try:
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1); return self
            except OSError:
                time.sleep(0.5)

    def __exit__(self, *exc):
        self.f.seek(0); msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1); self.f.close()


class Helper:
    """The iGPU VAE encoder + image prep + torch RNG, kept hot in a child process. Requests are async: send, wait()."""

    def __init__(self):
        self.p = subprocess.Popen([sys.executable, str(HERE / "igpu_vae.py"), "serve", "--fp16"], cwd=HERE, text=True,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        self.wait()

    def send(self, **req):
        self.p.stdin.write(json.dumps(req) + "\n"); self.p.stdin.flush()

    def wait(self):
        for line in self.p.stdout:
            i = line.find("@@")                      # native DirectML output can share the line
            if i >= 0:
                r = json.loads(line[i + 2:])
                if not r["ok"]:
                    raise RuntimeError("iGPU helper: " + r["err"])
                return r
        raise RuntimeError("the iGPU helper exited (see logs)")


def release_contexts():
    """Close every cached hardware context and its instruction buffers between stages: the stages together use more
    programs than the NPU's 16 contexts, and evicted contexts did not give their device memory back."""
    from aie.utils import DefaultNPURuntime
    import run_prebuilt
    import vlm_run
    DefaultNPURuntime.cleanup()
    vlm_run._handles.clear()
    run_prebuilt._loaded.clear()
    gc.collect()


def prune_cache(keep=6):
    """Prompt-length patched DiT programs pile up (~150 MB per length): keep the most recently used few."""
    d = paths.CACHE / "patched"
    if d.is_dir():
        for f in sorted(d.glob("*.elf"), key=lambda f: f.stat().st_atime, reverse=True)[keep * 2:]:
            try:
                f.unlink()
            except OSError:
                pass


class Job:
    """What the UI's progress bar shows. Fractions are rough wall-time shares of a warm job."""
    SHARE = {"edit": (0.20, 0.85), "generate": (0.10, 0.80)}   # end of text stage, end of DiT

    def __init__(self, kind):
        self.kind, self.t0, self.stage, self.step, self.fraction = kind, time.time(), "Starting", 0, 0.0

    def set(self, stage, fraction, step=0):
        self.stage, self.fraction, self.step = stage, fraction, step

    def dit_log(self, line):
        m = re.match(r"step (\d+)", line)
        if m:
            a, b = self.SHARE[self.kind]
            k = int(m.group(1))
            self.set("Denoising on the NPU, step", a + (b - a) * k / STEPS_TOTAL, min(k + 1, STEPS_TOTAL))

    def view(self):
        return {"kind": self.kind, "stage": self.stage, "step": self.step, "steps": STEPS_TOTAL if self.step else 0,
                "fraction": round(self.fraction, 3), "elapsed": round(time.time() - self.t0, 1)}


class Models:
    state, message, t_load, last, job = "loading", "Starting up…", 0.0, {}, None

    def load(self):
        try:
            t0 = time.perf_counter()
            self.message = "Starting the iGPU helper…"
            self.helper = Helper()
            from tokenizers import Tokenizer        # not transformers: torch in this process slows the NPU host code
            self.tok = Tokenizer.from_file(str(paths.MODEL / "processor" / "tokenizer.json"))
            self.message = "Loading the text encoder…"
            import vlm_run
            vlm_run.LM.STREAM = 2                    # LM weights in host RAM, streamed per layer (NPU memory cap)
            from vlm_encode import Encoder
            self.enc = Encoder()
            self.message = "Loading the image model (DiT) onto the NPU…"
            from forward import NPU, Weights, use_dit_elfs
            self.npu, self.w = NPU(), Weights()
            use_dit_elfs(self.npu)
            self.message = "Loading the VAE decoder…"
            from vae_decode import Dec
            self.dec = Dec(verbose=False)
            self.message = "Warming up the text encoder…"
            with NpuLock():
                self.enc.encode(str(paths.ASSETS / "warmup.png"), "warm-up")
                self.enc.lm = self.enc.vis = None
                release_contexts()
            self.t_load = time.perf_counter() - t0
            self.state, self.message = "ready", ""
            print(f"ready in {self.t_load:.1f} s", flush=True)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.state, self.message = "error", f"Could not load the models: {type(e).__name__}: {e}"
            if re.search(r"xrt|pyxrt|xclbin|hw_context|DLL load|device", f"{type(e).__name__} {e}", re.I):
                self.message += (" This usually means the NPU driver is missing or too old: you need 32.0.20101.3760 or"
                                 " newer, from AMD Software: Adrenalin Edition or your laptop maker (see the README's"
                                 " 'NPU driver' section). Run Install.cmd -CheckOnly to check.")

    # ---------- prompt length ----------
    def text_rows(self, kind, prompt):
        """Text rows the DiT sees, counted like vlm_preprocess: the template minus the system prefix (14 tokens);
        the edit's image placeholder is one token here and not a text row. Checked against the real preprocessing."""
        n = lambda s: len(self.tok.encode(s, add_special_tokens=False).ids)
        head = f"<|im_start|>system\n{SYS}<|im_end|>\n"
        if kind == "edit":
            return n(f"{head}<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{prompt or ' '}"
                     f"<|im_end|>\n<|im_start|>assistant\n") - n(head) - 1
        return n(f"{head}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n") - n(head)

    def check_prompt(self, kind, prompt):
        n = self.text_rows(kind, prompt)
        return {"tokens": n, "max": MAX_TEXT[kind], "ok": n <= MAX_TEXT[kind]}

    # ---------- jobs ----------
    def run(self, kind, prompt, seed, runs, image=None, crop="center", name=""):
        if not prompt.strip():
            raise ValueError("Please type a prompt.")
        c = self.check_prompt(kind, prompt)
        if not c["ok"]:
            raise ValueError(f"That prompt is {c['tokens']} tokens; {'edits' if kind == 'edit' else 'images'} can "
                             f"take up to {c['max']}. Please shorten it.")
        if runs not in REUSE:
            raise ValueError(f"steps must be one of {sorted(REUSE)}")
        day = paths.OUTPUTS / datetime.now().strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        TMP.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("%H%M%S") + ("_edit" if kind == "edit" else "_new")
        self.job = Job(kind)
        try:
            out, before, tim = (self._edit if kind == "edit" else self._generate)(prompt, seed, runs, day, stem,
                                                                                 image, crop)
        finally:
            self.job = None
            prune_cache()
        self.last = {k: round(v, 2) for k, v in tim.items()}
        rec = {"id": f"{day.name}/{stem}", "kind": kind, "prompt": prompt, "seed": seed, "steps": runs,
               "crop": crop if kind == "edit" else None, "source_name": name or None, "timings": self.last,
               "created": datetime.now().isoformat(timespec="seconds")}
        (day / f"{stem}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"[{rec['id']}] {kind}, seed {seed}, {runs} runs: " +
              ", ".join(f"{k} {v}" for k, v in self.last.items()), flush=True)
        return record_view(rec)

    def _generate(self, prompt, seed, runs, day, stem, image, crop):
        from forward import dit_denoise
        from vae_decode import png_write
        tim, t_all, job = {}, time.perf_counter(), self.job
        out, noise_path = day / f"{stem}.png", TMP / f"{stem}.noise.npy"
        self.helper.send(op="noise", seed=seed, dst=str(noise_path))   # torch RNG = same images as diffusers
        with NpuLock():
            job.set("Reading the prompt (text encoder, NPU)", 0.03)
            t = time.perf_counter()
            pe, mask = self.enc.encode(None, prompt)[:2]
            tim["text encoder (NPU)"] = time.perf_counter() - t
            self.enc.lm = None
            release_contexts()
            self.helper.wait()
            noise = np.load(noise_path)
            noise_path.unlink()
            job.set("Denoising on the NPU, step", job.SHARE["generate"][0], 1)
            t = time.perf_counter()
            lat = dit_denoise(self.npu, self.w, np.asarray(pe, np.float32).reshape(-1, 4096), np.asarray(mask).ravel(),
                              None, noise, log=job.dit_log, reuse=set(REUSE[runs]))
            self.npu.sync()
            release_contexts()
            tim[f"DiT ({runs} runs, NPU)"] = time.perf_counter() - t
            job.set("Decoding the image (VAE, NPU)", job.SHARE["generate"][1])
            t = time.perf_counter()
            self._decode(lat, out)
            tim["VAE decode (NPU)"] = time.perf_counter() - t
        tim["total"] = time.perf_counter() - t_all
        return out, None, tim

    def _edit(self, prompt, seed, runs, day, stem, image, crop):
        from forward import dit_denoise
        tim, t_all, job = {}, time.perf_counter(), self.job
        src, sq = TMP / f"{stem}.src", day / f"{stem}.before.png"
        lat_path, out = TMP / f"{stem}.ref.npy", day / f"{stem}.png"
        src.write_bytes(image)
        job.set("Preparing the photo (iGPU)", 0.01)
        t = time.perf_counter()
        self.helper.send(op="square", src=str(src), dst=str(sq), crop=crop); self.helper.wait()
        src.unlink()
        tim["prep (iGPU)"] = time.perf_counter() - t
        self.helper.send(op="encode", src=str(sq), dst=str(lat_path))   # iGPU encodes while the NPU reads the prompt
        with NpuLock():
            job.set("Reading the photo and prompt (text encoder, NPU)", 0.03)
            t = time.perf_counter()
            if self.enc.vis is None:
                from vlm_vision import Vision
                self.enc.vis = Vision()
            pe, mask = self.enc.encode(str(sq), prompt)[:2]
            tim["text encoder (NPU)"] = time.perf_counter() - t
            self.enc.lm = self.enc.vis = None        # one stage on the NPU at a time (pinned RAM)
            release_contexts()
            t = time.perf_counter()
            tim["VAE encode (iGPU, overlapped)"] = self.helper.wait()["t"]
            tim["waiting for the iGPU"] = time.perf_counter() - t
            ref = np.load(lat_path)
            lat_path.unlink()
            noise = np.random.default_rng(seed).standard_normal((4096, 64)).astype(np.float32)
            job.set("Denoising on the NPU, step", job.SHARE["edit"][0], 1)
            t = time.perf_counter()
            lat = dit_denoise(self.npu, self.w, np.asarray(pe, np.float32).reshape(-1, 4096), np.asarray(mask).ravel(),
                              ref, noise, log=job.dit_log, reuse=set(REUSE[runs]))
            self.npu.sync()
            release_contexts()                        # the step objects hold ~2-5 GB of NPU buffers
            tim[f"DiT ({runs} runs, NPU)"] = time.perf_counter() - t
            job.set("Decoding the image (VAE, NPU)", job.SHARE["edit"][1])
            t = time.perf_counter()
            self._decode(lat, out)
            tim["VAE decode (NPU)"] = time.perf_counter() - t
        tim["total"] = time.perf_counter() - t_all
        return out, sq, tim

    def _decode(self, lat, out):
        from vae_decode import png_write
        png_write(str(out), self.dec.decode(lat))
        self.dec.pool.clear(); self.dec.wt.clear()   # device buffers, rebuilt next decode (RAM budget)
        release_contexts()


def record_view(rec):
    day, stem = rec["id"].split("/")
    v = dict(rec, url=f"/outputs/{day}/{stem}.png")
    if rec["kind"] == "edit":
        v["before"] = f"/outputs/{day}/{stem}.before.png"
    return v


def history(limit=200):
    items = []
    if paths.OUTPUTS.is_dir():
        for f in sorted(paths.OUTPUTS.glob("*/*.json"), key=lambda f: f.name, reverse=True):
            if len(items) >= limit:
                break
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
                if (f.with_suffix(".png")).exists():
                    items.append(record_view(rec))
            except (OSError, ValueError, KeyError):
                continue
    items.sort(key=lambda r: r.get("created", ""), reverse=True)
    return items


M = Models()
WORK = threading.Lock()
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".png": "image/png", ".txt": "text/plain; charset=utf-8", ".svg": "image/svg+xml"}


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def file(self, root, rel):
        p = (root / rel).resolve()
        if root.resolve() not in p.parents or not p.is_file():
            return self.reply(404, {"error": "not found"})
        b = p.read_bytes()
        self.send_response(200); self.send_header("Content-Type", TYPES.get(p.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self.file(UI, "index.html")
        if path.startswith("/ui/"):
            return self.file(UI, path[4:])
        if path.startswith("/outputs/"):
            return self.file(paths.OUTPUTS, path[9:])
        if path == "/docs/QWEN-RESEARCH-LICENSE.txt":
            return self.file(paths.ROOT / "docs", "QWEN-RESEARCH-LICENSE.txt")
        if path == "/api/status":
            state = "busy" if M.state == "ready" and M.job else M.state
            return self.reply(200, {"state": state, "message": M.message, "load_s": round(M.t_load, 1),
                                    "job": M.job.view() if M.job else None, "limits": {"edit_tokens": MAX_TEXT["edit"]}})
        if path == "/api/history":
            return self.reply(200, {"items": history()})
        if path.rstrip("/") == "/health":
            return self.reply(200, {"ok": M.state == "ready", "state": M.state, "load_s": round(M.t_load, 1),
                                    "last": M.last})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        try:
            r = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            return self.reply(400, {"error": "bad JSON"})
        if path == "/api/open-folder":
            d = paths.OUTPUTS
            if r.get("id"):
                d = paths.OUTPUTS / r["id"].split("/")[0]
            d.mkdir(parents=True, exist_ok=True)
            os.startfile(str(d))
            return self.reply(200, {"ok": True})
        if M.state != "ready":
            return self.reply(503, {"error": M.message or "Still loading the models, give it a minute."})
        if path == "/api/tokens":
            return self.reply(200, M.check_prompt("edit" if r.get("kind") == "edit" else "generate",
                                                  str(r.get("prompt", ""))))
        openai = path in ("/v1/images/edits", "/v1/images/generations")
        if path != "/api/run" and not openai:
            return self.reply(404, {"error": "not found"})
        kind = ("edit" if path.endswith("edits") else "generate") if openai else r.get("kind")
        if kind not in ("edit", "generate"):
            return self.reply(400, {"error": "kind must be edit or generate"})
        if kind == "edit" and not r.get("image"):
            return self.reply(400, {"error": "Please add a photo to edit."})
        if not WORK.acquire(blocking=False):
            return self.reply(409, {"error": "Already working on an image; wait for it to finish."})
        try:
            seed = r.get("seed")
            seed = int(seed) if seed is not None and str(seed) != "" and int(seed) >= 0 else int(np.random.randint(0, 2 ** 31))
            res = M.run(kind, str(r.get("prompt", "")), seed, int(r.get("steps") or 8),
                        base64.b64decode(r["image"]) if kind == "edit" else None, r.get("crop") or "center",
                        str(r.get("name") or ""))
        except ValueError as e:
            return self.reply(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self.reply(500, {"error": f"{type(e).__name__}: {e}"})
        finally:
            WORK.release()
        if openai:
            f = paths.OUTPUTS / (res["id"] + ".png")
            return self.reply(200, {"data": [{"b64_json": base64.b64encode(f.read_bytes()).decode()}],
                                    "seed": res["seed"], "timings": res["timings"], "file": str(f)})
        self.reply(200, res)

    def log_message(self, *a):
        pass


class Tee:
    """Console output also goes to logs/server.log (REDLINE_LOG), so a bug report can include it."""

    def __init__(self, stream, f):
        self.stream, self.f = stream, f

    def write(self, s):
        self.stream.write(s); self.f.write(s); self.f.flush()

    def flush(self):
        self.stream.flush()


def main():
    if os.environ.get("REDLINE_LOG"):
        f = open(os.environ["REDLINE_LOG"], "a", encoding="utf-8")
        f.write(f"\n---- {datetime.now().isoformat(timespec='seconds')}\n")
        sys.stdout, sys.stderr = Tee(sys.stdout, f), Tee(sys.stderr, f)
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11290)
    ap.add_argument("--open", action="store_true", help="open the UI in the browser once the port is up")
    a = ap.parse_args()
    os.environ.setdefault("XRT_CONTEXT_CACHE_SIZE", "14")   # 16 hw contexts; the DiT keeps 2 outside the cache
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    url = f"http://127.0.0.1:{a.port}/"
    print(f"redline-tiny on {url} (loading the models, ~30 s)", flush=True)
    threading.Thread(target=M.load, daemon=True).start()
    if a.open:
        os.startfile(url)
    srv.serve_forever()


if __name__ == "__main__":
    main()
