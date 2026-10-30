"""Qwen-Image-2.1 VAE decode on the NPU: (4096, 64) latents -> 1024x1024 RGBA PNG.

Every layer runs on the NPU: convs (kernels/vae/vconv_design.py, 32 cores), channel RMS norm (+SiLU), residual adds,
2x upsample / DupUp shortcuts (kernels/vae/vglue_design.py). Activations stay on the device as octet-planar bf16
([H+2][C/8][W+2][8], zero border; buffers rounded up for vadd). Host glue: packing the latents, the mid
attention's K/V-as-weights repack and its softmax over 4096 keys, and the final clamp + PNG.
Artifacts are built on demand (build/npu_build.py, names vae2-*); weights are packed once into
data\\weights\\vae\\.
Usage (IRON env active): python vae_decode.py [--latents FILE.npy|golden] [--out PNG]
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.utils import DefaultNPURuntime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kernels" / "vae"))
from npu_layouts import V9  # noqa: E402
from pack_weights import SafeTensors  # noqa: E402
from aie.utils.npukernel import NPUKernel  # noqa: E402
from vae_ops import opb_pack, opb_unpack, pack_vconv, vae, vae_config  # noqa: E402
from vglue_design import add_len  # noqa: E402

from paths import NPU as ART, ROOT as _ROOT, WEIGHTS, need_build
WDIR = WEIGHTS / "vae"
GOLDEN = _ROOT / "golden"   # developer checks only


def conv_params(cin, cout):
    """(KC, NC, R) for vconv. Gives the decoder's original choices for its channel counts (artifact names don't
    encode them), and covers the encoder's 8 / 96 / 192 / 384 / 768 / 128 / 2304."""
    kc = 48 if cin % 48 == 0 else 32 if cin % 32 == 0 else 8
    if cout == 4096:
        return kc, 64, 4
    nc, r = next((nc, r) for nc, r in ((48, 4), (48, 3), (32, 4), (32, 3), (16, 4), (16, 3), (16, 1))
                 if cout % (nc * r) == 0)
    return kc, nc, r


def png_write(path, rgba):
    """Minimal PNG writer (8-bit RGBA, no PIL in the IRON env)."""
    h, w, c = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


class Dec:
    # Shape-generic programs (kernels/vae --gen 1). Each shape is its own artifact (vae2-*, for its insts.bin), but
    # every shape of one kernel config runs on ONE canonical xclbin (vae2x-<config>), so the runtime keeps one hardware
    # context per config: 20 programs for the decoder.

    def __init__(self, verbose=True, dry=False):
        self.dry = dry  # walk the decode and build artifacts only (no NPU runs, no lock needed)
        self.st = vae()
        self.cfg = vae_config()
        self.rt = DefaultNPURuntime
        self.t_npu, self.t_wall, self.n = defaultdict(float), defaultdict(float), defaultdict(int)
        self.t_load = 0.0
        self.pool = {}
        self.verbose = verbose
        self.wt = {}
        self.xclbin_of = {}               # artifact dir -> its config's canonical xclbin
        WDIR.mkdir(parents=True, exist_ok=True)

    # ---------- artifacts / weights ----------
    def art(self, name, family, meta, dargs):
        name, meta, dargs = "vae2-" + name[len("vae-"):], dict(meta, GEN=1), dargs + ["--gen", "1"]
        d = ART / name
        if not (d / "final.xclbin").exists():
            self._build(name, family, meta, dargs)
        if d not in self.xclbin_of:
            self.xclbin_of[d] = self._canonical(d, name, family, meta, dargs)
        return d

    def _build(self, name, family, meta, dargs, env=None):
        need_build(name)
        t = time.time()
        cmd = [sys.executable, str(ROOT / "build" / "npu_build.py"), "--name", name, "--design",
               str(ROOT / "kernels" / "vae" / ("vconv_design.py" if family == "vae-vconv" else "vglue_design.py")),
               "--family", family, "--meta"] + [f"{k}={v}" for k, v in meta.items()] + ["--design-args"] + dargs
        r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, NPU_ART_ROOT=str(ART), **(env or {})))
        if not (ART / name / "final.xclbin").exists():
            raise RuntimeError(f"build {name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        print(f"  built {name} ({time.time() - t:.0f} s)", flush=True)

    @staticmethod
    def gen_config(family, m):
        """What the core program depends on: shapes only change the insts.bin."""
        return {"vae-vconv": lambda: f"conv-p{m['P']}-k{m['KC']}-n{m['NC']}-t{m['TAPS']}-r{m['R']}-ds{m.get('DS', 0)}",
                "vae-vnorm": lambda: f"norm-c{m['C']}-s{m['SILU']}",
                "vae-vadd": lambda: "add",
                "vae-vup": lambda: f"up-{m['CIN']}-{m['COUT']}-m{m['MODE']}",
                "vae-vdown": lambda: f"down-c{m['C']}-m{m['MODE']}"}[family]()

    def _canonical(self, d, name, family, meta, dargs):
        """The config's canonical xclbin (vae2x-<config>/final.xclbin, the first shape built). Every shape's static
        configuration (final.prj/cdo_main: core ELFs, DMA/lock init, enables) must be byte-identical to it, or its
        insts.bin would run against a different hardware setup: checked here, not assumed."""
        cdir = ART / f"vae2x-{self.gen_config(family, meta)}"
        cdo = lambda x: {f.name: f.read_bytes() for f in sorted((x / "final.prj" / "cdo_main").glob("*.bin"))}
        if not cdo(d):   # an IRON compile-cache hit leaves no final.prj: rebuild this shape with an empty cache
            d.rename(ART / f"{name}.nocdo-{int(time.time())}")
            self._build(name, family, meta, dargs, env={"NPU_CACHE_HOME": str(ART / f"cache-gen-{int(time.time())}")})
        if not (cdir / "final.xclbin").exists():
            import shutil
            (cdir / "final.prj").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(d / "final.xclbin", cdir / "final.xclbin")
            shutil.copytree(d / "final.prj" / "cdo_main", cdir / "final.prj" / "cdo_main", dirs_exist_ok=True)
            (cdir / "meta.json").write_text(json.dumps({"config": cdir.name, "from": name}, indent=1))
        if cdo(d) != cdo(cdir):
            raise RuntimeError(f"{name}: static configuration differs from {cdir.name}; not shape-generic")
        return cdir / "final.xclbin"

    def weights(self, key, make):
        """Packed weight stream as an NPU tensor, cached in WDIR/<key>.bin."""
        if key not in self.wt:
            f = WDIR / f"{key}.bin"
            if not f.exists():   # write-then-rename: an interrupted first decode must not leave a truncated cache
                WDIR.mkdir(parents=True, exist_ok=True)
                tmp = f.with_suffix(".tmp")
                tmp.write_bytes(make().tobytes())
                os.replace(tmp, f)
            self.wt[key] = iron.tensor(np.fromfile(f, np.uint8).view(V9), dtype=V9, device="npu")
        return self.wt[key]

    def run(self, op, art, args):
        if self.dry:
            return
        t = time.perf_counter()
        # through the runtime's own LRU every time: Strix has 16 hardware contexts and the decode uses ~45 xclbins,
        # so contexts get evicted and re-created (a handle kept across an eviction is stale)
        h = self.rt.load(NPUKernel(self.xclbin_of.get(art, art / "final.xclbin"), art / "insts.bin"))
        tl = time.perf_counter()
        r = self.rt.run(h, args)
        self.t_load += tl - t
        self.t_wall[op] += time.perf_counter() - t
        self.t_npu[op] += r.npu_time / 1e9
        self.n[op] += 1

    def buf(self, H, W, C):
        """A device buffer for an OPB tensor, from the pool when one is free (borders are always zero: every op
        writes only interiors, and vadd writes 0 + 0 there)."""
        L = add_len(H, W, C)
        free = self.pool.setdefault(L, [])
        return free.pop() if free else iron.zeros((L,), dtype=bfloat16, device="npu")

    def give(self, *bufs):
        for b in bufs:
            if b is not None and not self.dry:
                self.pool.setdefault(b.numel() if hasattr(b, "numel") else b.size, []).append(b)

    # ---------- ops ----------
    def conv(self, lname, x, H, W, cin, cout_real, taps, w=None, b=None, cin_tot=0, cin0=0, wkey=None, op=None,
             ds=0):
        """ds = 1: the encoder's stride-2 downsample conv (H, W are the input size; output H/2 x W/2)."""
        cout = 16 if cout_real == 4 else cout_real
        kc, nc, r = conv_params(cin, cout)
        name = f"vae-c{H}-{cin}-{cout}-t{taps}" + (f"-ct{cin_tot}o{cin0}" if cin_tot else "") + ("-ds" if ds else "")
        meta = dict(H=H, W=W, CIN=cin, COUT=cout, TAPS=taps, P=64, KC=kc, NC=nc, R=r, DS=ds)
        dargs = ["--h", str(H), "--w", str(W), "--cin", str(cin), "--cout", str(cout), "--taps", str(taps),
                 "--p", "64", "--kc", str(kc), "--nc", str(nc), "--r", str(r)]
        if cin_tot:
            dargs += ["--cin-tot", str(cin_tot), "--cin0", str(cin0)]
        if ds:
            dargs += ["--ds", "1"]
        a = self.art(name, "vae-vconv", meta, dargs)
        if self.dry:
            return x
        if w is None:
            wkey = f"{lname}_k{kc}_n{nc}_c{cout}"
            wt = self.weights(wkey, lambda: pack_vconv(self.st.get(lname + ".weight"), self._b(lname), kc, nc, cout))
        elif wkey:
            wt = self.weights(f"{wkey}_k{kc}_n{nc}_c{cout}", lambda: pack_vconv(w, b, kc, nc, cout))
        else:
            wt = iron.tensor(pack_vconv(w, b, kc, nc, cout).view(V9), dtype=V9, device="npu")
        y = self.buf(H // 2, W // 2, cout) if ds else self.buf(H, W, cout)
        self.run(op or f"conv{taps}{'/2' if ds else ''} {cin}->{cout_real}@{H}", a, [x, wt, y])
        return y

    def _b(self, lname):
        return self.st.get(lname + ".bias") if lname + ".bias" in self.st.header else None

    def norm(self, x, H, W, C, gname, silu=1):
        name = f"vae-n{H}-{C}-s{silu}"
        a = self.art(name, "vae-vnorm", dict(H=H, W=W, C=C, SILU=silu),
                     ["--op", "vnorm", "--h", str(H), "--w", str(W), "--c", str(C), "--silu", str(silu)])
        if self.dry:
            return x
        key = f"{gname}_g"
        if key not in self.wt:
            g = self.st.get(gname).reshape(-1) * np.sqrt(C)
            self.wt[key] = iron.tensor(g.astype(bfloat16), dtype=bfloat16, device="npu")
        y = self.buf(H, W, C)
        self.run(f"norm{'+silu' if silu else ''} {C}@{H}", a, [x, self.wt[key], y])
        return y

    def add(self, x, y, H, W, C):
        L = add_len(H, W, C)
        a = self.art(f"vae-a{H}-{C}", "vae-vadd", dict(H=H, W=W, C=C, L=L),
                     ["--op", "vadd", "--h", str(H), "--w", str(W), "--c", str(C)])
        if self.dry:
            return x
        o = self.buf(H, W, C)
        self.run(f"add {C}@{H}", a, [x, y, o])
        return o

    def up(self, x, H, W, cin, cout, mode):
        a = self.art(f"vae-u{H}-{cin}-{cout}-m{mode}", "vae-vup", dict(H=H, W=W, CIN=cin, COUT=cout, MODE=mode),
                     ["--op", "vup", "--h", str(H), "--w", str(W), "--c", str(cin), "--cout", str(cout),
                      "--mode", str(mode)])
        if self.dry:
            return x
        y = self.buf(2 * H, 2 * W, cout)
        self.run(f"up m{mode} {cin}->{cout}@{2 * H}", a, [x, y])
        return y

    def down(self, x, H, W, C, mode):
        """2x2 average pool (AvgDown shortcut): mode 0 C -> C, mode 1 C -> 2C with zeros on even channels."""
        a = self.art(f"vae-d{H}-{C}-m{mode}", "vae-vdown", dict(H=H, W=W, C=C, MODE=mode),
                     ["--op", "vdown", "--h", str(H), "--w", str(W), "--c", str(C), "--mode", str(mode)])
        if self.dry:
            return x
        y = self.buf(H // 2, W // 2, 2 * C if mode else C)
        self.run(f"avgdown m{mode} {C}@{H}", a, [x, y])
        return y

    # ---------- blocks ----------
    def resblock(self, x, H, W, cin, cout, p):
        h = self.conv(p + ".conv_shortcut", x, H, W, cin, cout, 1) if cin != cout else x
        t1 = self.norm(x, H, W, cin, p + ".norm1.gamma")
        t2 = self.conv(p + ".conv1", t1, H, W, cin, cout, 3)
        self.give(t1)
        t3 = self.norm(t2, H, W, cout, p + ".norm2.gamma")
        self.give(t2)
        t4 = self.conv(p + ".conv2", t3, H, W, cout, cout, 3)
        self.give(t3)
        out = self.add(t4, h, H, W, cout)
        self.give(t4, h if h is not x else None)
        return out

    def attention(self, x, H=64, W=64, C=1152, p="decoder.mid_block.attentions.0"):
        t = self.norm(x, H, W, C, p + ".norm.gamma", silu=0)
        wq = self.st.get(p + ".to_qkv.weight").copy()
        bq = self.st.get(p + ".to_qkv.bias").copy()
        wq[:C] /= np.sqrt(C)  # SDPA's 1/sqrt(d), folded into q
        bq[:C] /= np.sqrt(C)
        qkv = self.conv(p + ".to_qkv", t, H, W, C, 3 * C, 1, w=wq, b=bq, wkey=p + "_qkv_qscaled", op="attn qkv")
        self.give(t)
        if self.dry:
            self.conv(None, qkv, H, W, C, H * W, 1, w=0, cin_tot=3 * C, cin0=0)
            self.conv(None, qkv, H, W, H * W, C, 1, w=0)
            self.conv(p + ".proj", qkv, H, W, C, C, 1)
            return self.add(x, x, H, W, C)
        t0 = time.perf_counter()
        a = opb_unpack(qkv.numpy()[:(H + 2) * 3 * C * (W + 2)], H, W, 3 * C).reshape(H * W, 3 * C)
        k, v = a[:, C:2 * C], a[:, 2 * C:]
        self.t_wall["host attn repack"] += time.perf_counter() - t0
        s = self.conv(None, qkv, H, W, C, H * W, 1, w=k[:, :, None, None], b=None, cin_tot=3 * C, cin0=0,
                      op="attn q.kT")
        t0 = time.perf_counter()
        sc = opb_unpack(s.numpy()[:(H + 2) * H * W * (W + 2)], H, W, H * W).reshape(H * W, H * W)
        sc = np.exp(sc - sc.max(1, keepdims=True))
        sc /= sc.sum(1, keepdims=True)
        pk = opb_pack(sc.reshape(H, W, H * W))
        pt = iron.tensor(np.concatenate([pk, np.zeros(add_len(H, W, H * W) - pk.size, bfloat16)]), dtype=bfloat16,
                         device="npu")
        self.t_wall["host softmax"] += time.perf_counter() - t0
        o = self.conv(None, pt, H, W, H * W, C, 1, w=np.ascontiguousarray(v.T)[:, :, None, None], b=None, op="attn p.v")
        self.give(qkv, s)
        o2 = self.conv(p + ".proj", o, H, W, C, C, 1, op="attn proj")
        self.give(o)
        out = self.add(o2, x, H, W, C)
        self.give(o2)
        return out

    def decode(self, latents):
        cfg = self.cfg
        if self.dry:
            x = None
            x = self.conv(None, x, 64, 64, 64, 64, 1, w=0)
            x = self.conv("decoder.conv_in", x, 64, 64, 64, 1152, 3)
            return self._body(x)
        z = np.asarray(latents, np.float32).reshape(64, 64, 64)  # (4096 tokens, 64) is already NHWC
        x = iron.tensor(np.concatenate([opb_pack(z), np.zeros(add_len(64, 64, 64) - opb_pack(z).size, bfloat16)]),
                        dtype=bfloat16, device="npu")
        # post_quant_conv with z * std + mean folded in
        w = self.st.get("post_quant_conv.weight")
        std, mean = np.array(cfg["latents_std"], np.float32), np.array(cfg["latents_mean"], np.float32)
        wf = w * std[None, :, None, None]
        bf_ = self.st.get("post_quant_conv.bias") + w[:, :, 0, 0] @ mean
        x0 = self.conv(None, x, 64, 64, 64, 64, 1, w=wf, b=bf_, wkey="post_quant_conv_denorm", op="post_quant_conv")
        x = self.conv("decoder.conv_in", x0, 64, 64, 64, 1152, 3)
        self.give(x0)
        return self._body(x)

    def _body(self, x):
        m = "decoder.mid_block"
        for f in (lambda v: self.resblock(v, 64, 64, 1152, 1152, m + ".resnets.0"), self.attention,
                  lambda v: self.resblock(v, 64, 64, 1152, 1152, m + ".resnets.1")):
            y = f(x)
            self.give(x)
            x = y
        H, C = 64, 1152
        for i, (cout, mode) in enumerate(((1152, 0), (1152, 0), (576, 1), (288, 2), (144, None))):
            pfx = f"decoder.up_blocks.{i}"
            xc, cin = x, C
            for j in range(3):
                y = self.resblock(x, H, H, C, cout, f"{pfx}.resnets.{j}")
                if x is not xc:
                    self.give(x)
                x, C = y, cout
            if mode is None:
                self.give(xc)
                break
            u = self.up(x, H, H, C, C, 0)
            self.give(x)
            x2 = self.conv(f"{pfx}.upsampler.resample.1", u, 2 * H, 2 * H, C, C, 3)
            self.give(u)
            s = self.up(xc, H, H, cin, cout, mode)
            self.give(xc)
            H *= 2
            x = self.add(x2, s, H, H, C)
            self.give(x2, s)
        n = self.norm(x, H, H, C, "decoder.norm_out.gamma")
        self.give(x)
        x = self.conv("decoder.conv_out", n, H, H, C, 4, 3)
        self.give(n)
        if self.dry:
            return None
        t0 = time.perf_counter()
        img = opb_unpack(x.numpy()[:(H + 2) * 16 * (H + 2)], H, H, 16)[:, :, :4]
        rgba = np.round(np.clip(img / 2 + 0.5, 0, 1) * 255).astype(np.uint8)
        self.give(x)
        self.t_wall["host output"] += time.perf_counter() - t0
        return rgba

    def report(self):
        rows = sorted(self.t_wall, key=lambda o: -self.t_wall[o])
        tn, tw = sum(self.t_npu.values()), sum(self.t_wall.values())
        for op in rows:
            print(f"  {op:28s} {self.n.get(op, 0):3d} calls  npu {self.t_npu.get(op, 0) * 1e3:8.1f} ms  "
                  f"wall {self.t_wall[op] * 1e3:8.1f} ms")
        print(f"  {'total':28s}            npu {tn * 1e3:8.1f} ms  wall {tw * 1e3:8.1f} ms  "
              f"(of which load/context switch {self.t_load * 1e3:.0f} ms)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents", default="golden", help="'golden' (latents_step7) or a .npy of shape (4096, 64)")
    p.add_argument("--out", default="npu_vae_golden.png")
    p.add_argument("--runs", type=int, default=1, help="decode this many times (the first builds/packs/loads)")
    p.add_argument("--build-only", action="store_true", help="build every artifact the decode needs (no NPU)")
    args = p.parse_args()
    if args.build_only:
        Dec(dry=True).decode(None)
        print("all artifacts built")
        return
    if args.latents == "golden":
        lat = SafeTensors(GOLDEN / "b_latents.safetensors").get("latents_step7")[0]
    else:
        lat = np.load(args.latents).reshape(4096, 64)
    dec = Dec()
    for r in range(args.runs):
        dec.t_npu.clear(), dec.t_wall.clear(), dec.n.clear()
        dec.t_load = 0.0
        t = time.perf_counter()
        rgba = dec.decode(lat)
        print(f"run {r}: decode {time.perf_counter() - t:.2f} s wall")
    dec.report()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    png_write(args.out, rgba)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
