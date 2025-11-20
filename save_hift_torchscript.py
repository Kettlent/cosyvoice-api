# save_hift_torchscript.py
import torch
import sys, os
from cosyvoice.cli.cosyvoice import CosyVoice2
sys.path.append('third_party/Matcha-TTS')
# --- Change this to your model directory ---
MODEL_DIR = 'pretrained_models/CosyVoice2-0.5B'

def main():
    print("=== Loading CosyVoice2 (no jit/trt/vllm) ===")
    cosy = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, load_vllm=False, fp16=False)

    hift = cosy.model.hift
    hift.eval()

    print("=== HiFT loaded ===")
    print("HiFT class:", hift.__class__)

    # ---- Build deterministic test mel feature ----
    # required size (1, 80, N)
    T = 512   # any long-enough mel length
    mel = torch.randn(1, 80, T, dtype=torch.float32)
    empty_cache = torch.zeros(1, 1, 0)

    print("Tracing with mel.shape =", mel.shape)

    # ------------ test one forward pass --------------
    print("Running hift.inference(...) once to check correctness...")
    try:
        out, src = hift.inference(speech_feat=mel, cache_source=empty_cache)
        print("Forward OK. out.shape:", out.shape, "dtype:", out.dtype)
    except Exception as e:
        print("ERROR: HiFT forward pass failed:")
        print(e)
        sys.exit(1)

    # ------------ Wrap for tracing -------------------
    class HiftWrapper(torch.nn.Module):
        def __init__(self, h):
            super().__init__()
            self.h = h

        def forward(self, x):
            empty = torch.zeros(1, 1, 0)
            y, _ = self.h.inference(speech_feat=x, cache_source=empty)
            return y

    wrap = HiftWrapper(hift).eval()

    print("Beginning TorchScript tracing… this may take a few seconds…")

    try:
        traced = torch.jit.trace(wrap, mel)
    except Exception as e:
        print("TorchScript TRACE failed:")
        print("Error:", e)
        sys.exit(1)

    # Save file
    out_path = "hift_traced.pt"
    traced.save(out_path)
    print("=== SUCCESS ===")
    print("Saved TorchScript HiFT to:", out_path)
    print("File exists?", os.path.exists(out_path))

if __name__ == "__main__":
    main()
