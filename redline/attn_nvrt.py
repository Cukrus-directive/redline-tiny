"""Runtime valid-key count for attention (taken from VigenFlow, whose manifests list `valid_keys_word_offsets`).

Artifacts built with `attn_design.py --nvrt ID` write an RTP word 0x7A4E<ID><nv> into every core at the start of
the runtime sequence (nv = valid keys in the last 64-key tile, 1..64); the kernel masks keys >= nv of that tile. The
host patches nv in the instruction stream (insts.bin) or in a merged ELF, so one artifact serves every prompt length
instead of one build per length (ensure_attn).

  patch(data, marker_id, nv)          bytes of insts.bin / an ELF -> (patched bytes, words replaced)
  nv_last(nkv, nk, nt=64)             valid keys of the last tile for nkv valid keys out of nk padded

Test (IRON env active):  python attn_nvrt.py ART_NVRT --ref NKV=ART_REF [NKV=ART_REF ...]
runs the NVRT artifact patched to each NKV and the per-length artifact on the same random inputs; outputs must be
byte-identical.
"""

import argparse
import json
from pathlib import Path

import numpy as np

MARK = 0x7A4E


def nv_last(nkv, nk, nt=64):
    nv = nkv - (nk - nt)
    assert 1 <= nv <= nt, f"{nkv} valid keys are not in the last tile of {nk}"
    return nv


def patch(data, marker_id, nv, expect=None):
    """Replace the low byte of every little-endian word 0x7A4E<marker_id><xx> in data with nv."""
    key = bytes([marker_id, MARK & 0xFF, MARK >> 8])   # bytes.find: a regex over a 100 MB ELF took seconds
    b = bytearray(data)
    hits, i = [], b.find(key, 1)
    while i >= 0:
        hits.append(i - 1)
        i = b.find(key, i + 1)
    if expect is not None:
        assert len(hits) == expect, f"marker {marker_id}: {len(hits)} words, expected {expect}"
    for h in hits:
        b[h] = nv
    return bytes(b), len(hits)


def main():
    import aie.iron as iron
    from aie.utils import DefaultNPURuntime
    from aie.utils.npukernel import NPUKernel
    from npu_layouts import V9, to_bfp16
    from paths import NPU as ART
    p = argparse.ArgumentParser()
    p.add_argument("art")
    p.add_argument("--ref", nargs="+", required=True, help="NKV=ARTIFACT (compile-time key count)")
    args = p.parse_args()
    a = ART / args.art
    m = json.loads((a / "meta.json").read_text())
    nk, H, W, PT = m["NK"], m["H"], m["W"], m["PT"]
    tile = 64 * 128 // 8
    rng = np.random.default_rng(0)
    rnd = lambda nblk, s=1.0: iron.tensor(to_bfp16((rng.standard_normal(nblk * 8) * s).astype(np.float32)).view(V9),
                                          dtype=V9, device="npu")
    # Q pre-scaled like qkv_post (log2e / sqrt(128)) so the softmax is peaky enough to see a wrong mask
    bufs = [rnd(H * W * 32 * tile, 0.3), rnd(H * 64 * 2 * tile), rnd(H * PT * 2 * tile)]
    out = lambda: iron.tensor(np.zeros(W * 32 * 64 * 32 * 128 // 8, dtype=V9), dtype=V9, device="npu")
    insts = (a / "insts.bin").read_bytes()
    rt = DefaultNPURuntime
    ok = True
    for r in args.ref:
        nkv, ref = r.split("=")
        nkv = int(nkv)
        o_ref, o_rt = out(), out()
        rt.run(rt.load(NPUKernel(ART / ref / "final.xclbin", ART / ref / "insts.bin")), bufs + [o_ref])
        pi, n = patch(insts, m["NVRT"], nv_last(nkv, nk), expect=32)
        f = a / f"insts_nkv{nkv}.bin"
        f.write_bytes(pi)
        res = rt.run(rt.load(NPUKernel(a / "final.xclbin", f)), bufs + [o_rt])
        same = np.array_equal(o_ref.numpy().view(np.uint8), o_rt.numpy().view(np.uint8))
        ok &= same
        print(f"nkv {nkv}: patched {n} words, {res.npu_time / 1e6:.2f} ms, "
              f"{'byte-identical' if same else 'DIFFERENT'} vs {ref}", flush=True)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
