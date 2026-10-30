"""Qwen-Image-2.1 edit text encoder (Qwen3-VL-8B) on the NPU: image + prompt -> prompt_embeds, image_pad_mask.

Host: preprocessing (helper subprocess: tokeniser + image resize/patchify, no model), the embedding gather,
mrope/RoPE tables, layout packing and a few row scatters. NPU: every layer of the vision tower (27 blocks, 4
mergers) and the language model (36 layers, deepstack after layers 0-2).
Usage (IRON env active): python vlm_encode.py IMAGE "PROMPT" [OUT.npz] [--check]
  --check compares with reference prompt_embeds in golden\\ (developer data, not shipped).
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from vlm_run import ART, LM, M, N_LAYERS, embed, mrope_cos_sin, mrope_positions
from vlm_vision import Vision
from paths import ROOT, need_build

GOLDEN_PY = sys.executable   # one venv in the release
HERE = Path(__file__).resolve().parent


def attn_artifact(nqt):
    name = f"vlm-attn-m{M}-t{nqt}-x"
    if not (ART / name / "final.xclbin").exists():  # one build per prompt-length bucket (64 tokens), ~8 s
        need_build(name)
        subprocess.run([sys.executable, str(HERE.parent / "build" / "npu_build.py"), "--name", name, "--design",
                        str(HERE.parent / "kernels" / "vlm" / "attn_causal_design.py"), "--family", "attncausal",
                        "--meta", f"M={M}", f"NQT={nqt}", "EXP2FIX=1", "--design-args", "--m", str(M), "--nqt",
                        str(nqt), "--exp2fix", "1"], check=True, env={**__import__("os").environ,
                                                                         "NPU_ART_ROOT": str(ART)})
    return name


class Encoder:
    def __init__(self, vision=True):
        """vision=False: text-to-image only (no vision tower weights or kernels)."""
        # the preprocessing worker's imports (~12 s) overlap the weight loading
        self.pre = subprocess.Popen([GOLDEN_PY, str(HERE / "vlm_preprocess.py"), "--serve"], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, text=True, bufsize=1)
        t = time.perf_counter()
        self.vis = Vision() if vision else None
        self.t_load_vision = time.perf_counter() - t
        self.lm = None
        self.lm_opts = {}   # LM attributes applied before each encode
        self.tmp = tempfile.TemporaryDirectory()
        assert self.pre.stdout.readline().strip() == "ready"

    def preprocess(self, image, prompt):
        import json
        f = Path(self.tmp.name) / "pre.npz"
        self.pre.stdin.write(json.dumps({"image": str(image) if image else "", "prompt": prompt, "out": str(f)}) + "\n")
        self.pre.stdin.flush()
        ans = self.pre.stdout.readline().strip()
        if ans != "ok":
            raise RuntimeError(f"preprocessing failed: {ans}")
        return dict(np.load(f))

    def encode(self, image, prompt):
        """image None = text-to-image (t2i template, no vision tower, no deepstack)."""
        tim = {}
        t0 = time.perf_counter()
        pre = self.preprocess(image, prompt)
        tim["preprocess (helper process: tokeniser + image)"] = time.perf_counter() - t0
        ids, img_id, drop = pre["input_ids"], int(pre["image_token_id"]), int(pre["drop_idx"])
        L = ids.size
        assert L <= M, f"prompt of {L} tokens > {M}"
        rows = np.flatnonzero(ids == img_id)

        t = time.perf_counter()
        if image:
            self.vis.t = {}
            img, ds = self.vis.forward(pre["pixel_values"])
            tim["vision tower (NPU, wall)"] = time.perf_counter() - t
            npu_vis = sum(self.vis.t.values())
        else:
            img, ds, npu_vis = np.zeros((0, 4096), np.float32), [], 0.0

        t = time.perf_counter()
        pos = mrope_positions(ids, img_id, tuple(pre["image_grid_thw"][0]) if image else (1, 64, 64))
        cos, sin = np.zeros((M, 64), np.float32), np.zeros((M, 64), np.float32)
        cos[:L], sin[:L] = mrope_cos_sin(pos)
        x = np.zeros((M, 4096), np.float32)
        x[:L] = embed(ids)
        x[rows] = img
        if self.lm is None:
            self.lm = LM(cos, sin, attn=attn_artifact(-(-L // 64)))
        else:  # weights stay resident; tables and the attention length bucket follow the prompt
            self.lm.set_tables(cos, sin)
            self.lm.attn = attn_artifact(-(-L // 64))
        for k, v in self.lm_opts.items():
            setattr(self.lm, k, v)
        self.lm.load_x(x)
        self.lm.set_deepstack(ds, rows)
        tim["LM host prep (embeddings, tables, deepstack scatter)"] = time.perf_counter() - t
        t = time.perf_counter()
        self.lm.t = {}
        for i in range(N_LAYERS):
            self.lm.layer(i)
        out = self.lm.x_out(N_LAYERS - 1)
        tim["language model (NPU, wall)"] = time.perf_counter() - t
        tim["total"] = time.perf_counter() - t0
        npu = {"vision": npu_vis, "lm": sum(self.lm.t.values())}
        return out[drop:L], (ids[drop:] == img_id), tim, npu, (self.vis.t if image else {}, self.lm.t)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    image, prompt = args[0], args[1]
    out_path = args[2] if len(args) > 2 else None
    t = time.perf_counter()
    enc = Encoder()
    print(f"vision weights resident in {enc.t_load_vision:.1f} s")
    pe, mask, tim, npu, (tv, tl) = enc.encode(image, prompt)
    print(f"(first encode incl. LM weight load) total {time.perf_counter() - t:.1f} s")
    pe, mask, tim, npu, (tv, tl) = enc.encode(image, prompt)   # warm: weights resident, xclbins loaded
    print("warm encode:")
    for k, v in tim.items():
        print(f"  {k:52s} {v * 1e3:8.0f} ms")
    print(f"  NPU time: vision {npu['vision']:.0f} ms (" + ", ".join(f"{k} {v:.0f}" for k, v in tv.items()) + ")")
    print(f"            LM {npu['lm']:.0f} ms (" + ", ".join(f"{k} {v:.0f}" for k, v in tl.items()) + ")")
    print(f"prompt_embeds {pe.shape}, image_pad_mask {int(mask.sum())} image rows")
    if out_path:
        np.savez(out_path, prompt_embeds=pe, image_pad_mask=mask)
    if "--check" in sys.argv:
        from pack_weights import SafeTensors
        gold = SafeTensors((ROOT / "golden" / "a_text.safetensors")).get("prompt_embeds")[0]
        txt = np.flatnonzero(~mask)
        cs = [float(np.corrcoef(pe[r], gold[r])[0, 1]) for r in txt]
        print(f"text rows ({txt.size}) vs golden: mean corr {np.mean(cs):.5f}, worst {min(cs):.5f} (row {txt[int(np.argmin(cs))]}), "
              f"all text rows {np.corrcoef(pe[txt].ravel(), gold[txt].ravel())[0, 1]:.5f}")
        print("  per row:", " ".join(f"{r}:{c:.3f}" for r, c in zip(txt, cs)))
        img_rows = np.flatnonzero(mask)
        print(f"image rows vs golden (unused by the DiT): corr {np.corrcoef(pe[img_rows].ravel(), gold[img_rows].ravel())[0, 1]:.5f}")


if __name__ == "__main__":
    main()
