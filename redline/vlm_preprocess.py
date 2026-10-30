"""Text-encoder preprocessing for vlm_encode.py, run in a helper process (transformers + PIL; torch stays out
of the NPU process). Tokenisation and image preprocessing only, no model weights or forward passes:
the pipeline's resize (VaeImageProcessor, ~1024x1024 area), RGBA composited over white, the edit template, and
Qwen3VLProcessor -> input_ids, pixel_values, image_grid_thw. An empty image path = text-to-image (t2i template,
input_ids only).
Usage: python vlm_preprocess.py IMAGE "PROMPT" OUT.npz
       ... vlm_preprocess.py --serve   (persistent: one JSON request {"image", "prompt", "out"} per stdin line,
                                        answers "ok" or "error: ..." per line; pays the imports once)
"""

import sys

import numpy as np
from PIL import Image

from paths import MODEL
MODEL = str(MODEL)
SYS = "Comprehend and analyze the provided prompt."


_PROC = {}


def main(image_path, prompt, out):
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.qwenimage21 import pipeline_qwenimage21 as mod
    from transformers import AutoProcessor
    if "p" not in _PROC:
        _PROC["p"] = AutoProcessor.from_pretrained(MODEL + r"\processor")
    proc = _PROC["p"]
    sys_tokens = proc.apply_chat_template([{"role": "system", "content": [{"type": "text", "text": SYS}]}],
                                          tokenize=True, return_dict=False)
    if not image_path:   # text-to-image: the pipeline's t2i template, no vision prefix
        text = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n")
        mi = proc(text=[text], padding=True, padding_side="left", return_tensors="np")
        np.savez(out, input_ids=np.asarray(mi["input_ids"][0]), drop_idx=len(sys_tokens[0]),
                 image_token_id=proc.tokenizer.encode("<|image_pad|>")[0])
        return
    img = Image.open(image_path)
    img = img.convert("RGBA") if img.mode != "RGBA" else img
    w, h, _ = mod.calculate_dimensions(1024 * 1024, img.size[0] / img.size[1])
    img = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64).resize(img, width=w, height=h)
    if img.mode == "RGBA":
        white = Image.new("RGB", img.size, (255, 255, 255))
        white.paste(img, mask=img.getchannel("A"))
        img = white
    text = (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{prompt or ' '}<|im_end|>\n"
            f"<|im_start|>assistant\n")
    mi = proc(text=[text], images=[img], padding=True, padding_side="left", return_tensors="np")
    np.savez(out, input_ids=np.asarray(mi["input_ids"][0]), pixel_values=np.asarray(mi["pixel_values"], np.float32),
             image_grid_thw=np.asarray(mi["image_grid_thw"]), drop_idx=len(sys_tokens[0]),
             image_token_id=proc.tokenizer.encode("<|image_pad|>")[0], size=np.array([w, h]))


def serve():
    import json
    import diffusers.pipelines.qwenimage21.pipeline_qwenimage21  # noqa: F401  (warm the imports)
    from transformers import AutoProcessor
    _PROC["p"] = AutoProcessor.from_pretrained(MODEL + r"\processor")
    print("ready", flush=True)
    for line in sys.stdin:
        try:
            r = json.loads(line)
            main(r["image"], r["prompt"], r["out"])
            print("ok", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"error: {e!r}", flush=True)


if __name__ == "__main__":
    serve() if sys.argv[1:] == ["--serve"] else main(*sys.argv[1:4])
