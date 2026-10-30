"""The Qwen-Image-2.1 VAE on the iGPU (ONNX Runtime + DirectML).
  python igpu_vae.py export             -> data\\onnx\\vae_{encoder,decoder}.onnx (fp32, 1024x1024 RGBA)
  python igpu_vae.py fp16               -> vae_{encoder,decoder}_fp16.onnx (weights/compute fp16, fp32 I/O)
  python igpu_vae.py roundtrip IN.png OUT.png [--fp16]   encode (latent mode) -> decode on the iGPU, timed
  python igpu_vae.py bench IN.png [--fp16] [--fp16-decode] warm timings + round-trip PSNR, as 'METRIC name value' lines
  python igpu_vae.py encloop N IN.png [--fp16]           encode N times (load for the iGPU+NPU overlap test)
  python igpu_vae.py latents IN.png OUT.npy [--fp16]     normalised (4096, 64) latents for vae_decode.py --latents
  python igpu_vae.py serve [--fp16]     resident helper for server.py: JSON requests on stdin, '@@'-prefixed JSON replies
Weights are only read. Exporting traces the graph once on zero input (no image is run on the CPU).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from paths import MODEL, ONNX as OUT
MODEL = str(MODEL)
FP16 = "--fp16" in sys.argv                  # fp16 encoder; the fp16 decoder also needs --fp16-decode:
FP16_DEC = "--fp16-decode" in sys.argv       # it hangs the 890M (DirectML 887A0006, Windows GPU timeout)


def export():
    import torch
    from diffusers import AutoencoderKLQwenImage21
    OUT.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32).eval().requires_grad_(False)

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.vae = vae

        def forward(self, x):                      # (1, 4, 1, 1024, 1024) RGBA in [-1, 1], one frame
            return self.vae.encode(x).latent_dist.mode()

    class Dec(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.vae = vae

        def forward(self, z):
            return self.vae.decode(z, return_dict=False)[0]

    # nearest-exact has no legacy-exporter symbolic; at an integer 2x scale it picks the same pixels as nearest
    F = torch.nn.functional
    interp = F.interpolate
    def nearest(input, size=None, scale_factor=None, mode="nearest", *a, **k):
        return interp(input, size, scale_factor, "nearest" if mode == "nearest-exact" else mode, *a, **k)
    F.interpolate = nearest
    x = torch.zeros(1, 4, 1, 1024, 1024)
    with torch.no_grad():
        z = Enc()(x)
        print("latent shape", tuple(z.shape))
        torch.onnx.export(Enc(), (x,), str(OUT / "vae_encoder.onnx"), opset_version=17, dynamo=False, input_names=["x"], output_names=["z"])
        torch.onnx.export(Dec(), (z,), str(OUT / "vae_decoder.onnx"), opset_version=17, dynamo=False, input_names=["z"], output_names=["y"])
    print("exported to", OUT)


def fp16():
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    for n in ("vae_encoder", "vae_decoder"):
        m = convert_float_to_float16(onnx.load(str(OUT / f"{n}.onnx")), keep_io_types=True)
        onnx.save(m, str(OUT / f"{n}_fp16.onnx"))
        print("wrote", OUT / f"{n}_fp16.onnx")


def session(kind):
    import onnxruntime as ort
    return ort.InferenceSession(str(OUT / f"vae_{kind}{'_fp16' if (FP16 and kind == 'encoder') or FP16_DEC else ''}.onnx"), providers=["DmlExecutionProvider"])


def read(src):
    from PIL import Image
    x = np.asarray(Image.open(src).convert("RGBA").resize((1024, 1024)), np.float32) / 127.5 - 1
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None, :, None])


def to_rgb(y):
    return ((np.clip(y[0, :, 0].transpose(1, 2, 0), -1, 1) + 1) * 127.5).round().astype(np.uint8)[..., :3]


def timed(sess, name, feed, runs):
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); out = sess.run(None, {name: feed})[0]; ts.append(time.perf_counter() - t)
    return out, ts


def encode(src, runs=2):
    enc = session("encoder")
    z, ts = timed(enc, "x", read(src), runs)       # first run includes DirectML warm-up
    del enc
    return z, ts


def decode(z, runs=2):
    dec = session("decoder")                       # the encoder session is freed first: both at once ran out of memory
    return timed(dec, "z", z, runs)


def roundtrip(src, dst):
    from PIL import Image
    z, te = encode(src)
    np.save(str(Path(dst).with_suffix(".latents.npy")), z)  # (1, 64, 1, 64, 64), raw VAE space
    y, td = decode(z)
    print(f"iGPU encode {' / '.join(f'{t:.2f}' for t in te)} s, decode {' / '.join(f'{t:.2f}' for t in td)} s")
    Image.fromarray(to_rgb(y)).save(dst)
    print("wrote", dst)


def normalised(z):
    cfg = json.load(open(rf"{MODEL}\vae\config.json"))
    m, s = np.array(cfg["latents_mean"], np.float32), np.array(cfg["latents_std"], np.float32)
    return (z[0, :, 0].transpose(1, 2, 0).reshape(4096, 64) - m) / s


def bench(src):
    from PIL import Image
    z, te = encode(src, runs=3)
    y, td = decode(z, runs=3)
    x = np.asarray(Image.open(src).convert("RGB").resize((1024, 1024)), np.float64)
    psnr = 10 * np.log10(255 ** 2 / ((to_rgb(y).astype(np.float64) - x) ** 2).mean())
    print(f"METRIC encode_warm_s {min(te[1:]):.3f}")
    print(f"METRIC decode_warm_s {min(td[1:]):.3f}")
    print(f"METRIC roundtrip_psnr_db {psnr:.2f}")


def encloop(n, src):
    enc, x = session("encoder"), read(src)
    timed(enc, "x", x, 1)                           # warm-up, not counted
    t0 = time.perf_counter()
    _, ts = timed(enc, "x", x, n)
    print(f"METRIC loop_encode_mean_s {np.mean(ts):.3f}")
    print(f"METRIC loop_wall_s {time.perf_counter() - t0:.2f}")


def square(src, dst, crop="center"):
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB"); e = min(im.size)
    l = {"left": 0, "center": (im.size[0] - e) // 2, "right": im.size[0] - e}[crop]
    t = (im.size[1] - e) // 2
    im.crop((l, t, l + e, t + e)).resize((1024, 1024), Image.LANCZOS).save(dst)


def serve():
    """Keeps the encoder session hot. Ops: square {src, dst, crop}, encode {src, dst}, noise {seed, dst} (the
    diffusers pipeline's initial latents for a seed, torch RNG only; see generate.torch_noise). stdout may carry
    native log noise, so replies are single lines starting with '@@'."""
    import gc
    import onnxruntime  # noqa: F401  (imports stay hot; the session is per encode, see below)
    print("@@" + json.dumps({"ok": True, "ready": True}), flush=True)
    for line in sys.stdin:
        try:
            r, t = json.loads(line), time.perf_counter()
            if r["op"] == "square":
                square(r["src"], r["dst"], r.get("crop", "center"))
            elif r["op"] == "noise":
                import torch
                from diffusers.utils.torch_utils import randn_tensor
                z = randn_tensor((1, 1, 64, 64, 64), generator=torch.Generator("cpu").manual_seed(int(r["seed"])),
                                 device=torch.device("cpu"), dtype=torch.bfloat16)
                np.save(r["dst"], z.view(1, 64, 4096).transpose(1, 2)[0].float().numpy())
            elif r["op"] == "encode":
                # a session per encode, freed straight after: its ~2-3 GB of iGPU memory comes out of the same
                # device-memory budget the NPU's DiT needs; runs alongside the text encoder anyway
                enc = session("encoder")
                np.save(r["dst"], normalised(enc.run(None, {"x": read(r["src"])})[0]))
                del enc
                gc.collect()
            print("@@" + json.dumps({"ok": True, "t": round(time.perf_counter() - t, 3)}), flush=True)
        except Exception as e:
            print("@@" + json.dumps({"ok": False, "err": f"{type(e).__name__}: {e}"}), flush=True)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    {"export": lambda: export(), "fp16": lambda: fp16(), "roundtrip": lambda: roundtrip(a[1], a[2]),
     "bench": lambda: bench(a[1]), "encloop": lambda: encloop(int(a[1]), a[2]),
     "serve": lambda: serve(),
     "latents": lambda: np.save(a[2], normalised(encode(a[1])[0]))}[a[0]]()
