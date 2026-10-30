"""End-of-install check: load every model and make one picture, on the NPU, the way the app does.
Also packs the VAE decoder's weights (done on the first decode). Saves the image to outputs\\ and prints the timings.
Usage: python selftest.py
"""

import os
import sys
import time

os.environ.setdefault("XRT_CONTEXT_CACHE_SIZE", "14")
import server  # noqa: E402

PROMPT = ("A cinematic, high-resolution photo of a short slicked-white haired young woman in red raincoat, black long "
          "sleeve and wide jeans standing in the doorway of a subway train, surrounded by a crowd of passengers. "
          "Surrounding them, the rest of the subway passengers appear motion-blurred, captured mid-movement, creating "
          "streaks and soft shapes that suggest rush and crowd density rather than individual identities. The "
          "fluorescent station light casts a cool, metallic tone. People around her look tired and absorbed in their "
          "own thoughts. A newspaper lies on the ground near the yellow line, enhancing the urban realism and slightly "
          "surreal, fashion-editorial atmosphere.")


def main():
    t = time.time()
    print("Loading models onto the NPU", flush=True)
    server.M.load()
    if server.M.state != "ready":
        sys.exit(f"ERROR: {server.M.message}")
    print(f"   Loaded: {time.time() - t:.0f} s\nGenerating test image (text-to-image, 1024x1024, 8 steps, seed 0)", flush=True)
    r = server.M.run("generate", PROMPT, 0, 8)
    f = server.paths.OUTPUTS / (r["id"] + ".png")
    print(f"   Output: {f}\n   Total: {r['timings']['total']:.0f} s (text encoder {r['timings']['text encoder (NPU)']:.1f}, "
          f"DiT {r['timings']['DiT (8 runs, NPU)']:.1f}, VAE decode {r['timings']['VAE decode (NPU)']:.1f})", flush=True)
    server.M.helper.p.kill()
    print("Self-test: PASS", flush=True)


if __name__ == "__main__":
    main()
