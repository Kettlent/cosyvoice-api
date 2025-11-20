# inspect_and_forward.py  (updated)
import torch
import numpy as np
import sys
import os
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

def show(obj, name):
    print(f"--- {name} ---")
    print("type:", type(obj))
    if hasattr(obj, '__dict__'):
        keys = list(obj.__dict__.keys())
        print("keys:", keys)
    print()

def main():
    model_dir = 'pretrained_models/CosyVoice2-0.5B'  # adjust if different
    print("Loading CosyVoice2 wrapper (no jit/trt/vllm, fp16=False) ...")
    cosy = CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    print("Top-level keys:", list(cosy.__dict__.keys()))
    show(cosy.frontend, "frontend")
    show(cosy.model, "model (CosyVoice2Model)")

    llm = cosy.model.llm
    flow = cosy.model.flow
    hift = cosy.model.hift
    print("LLM type:", type(llm))
    print("Flow type:", type(flow))
    print("HiFT type:", type(hift))
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # ---------- Build dummy inputs for flow.inference ----------
    token_len2 = 10   # non-prompt token length
    token_len1 = 5    # prompt token length

    try:
        vocab_size = flow.input_embedding.num_embeddings
    except Exception:
        vocab_size = 4096
    print("Using vocab_size:", vocab_size)

    token = torch.randint(low=0, high=vocab_size, size=(1, token_len2), dtype=torch.long)
    prompt_token = torch.randint(low=0, high=vocab_size, size=(1, token_len1), dtype=torch.long)
    token_len = torch.tensor([token_len2], dtype=torch.int32)
    prompt_token_len = torch.tensor([token_len1], dtype=torch.int32)

    try:
        spk_in = flow.spk_embed_affine_layer.in_features
    except Exception:
        spk_in = 192
    embedding = torch.randn(1, spk_in).float()

    input_frame_rate = getattr(flow, "input_frame_rate", 50)
    mel_len2 = int(token_len2 / input_frame_rate * 22050 / 256)
    mel_len1 = int(token_len1 / input_frame_rate * 22050 / 256)
    if mel_len2 <= 0:
        mel_len2 = 17
    if mel_len1 <= 0:
        mel_len1 = 10
    print("Computed mel_len1 (prompt)=", mel_len1, "mel_len2 (tokens)=", mel_len2)

    feat_dim = getattr(flow, "output_size", 80)
    prompt_feat = torch.randn(1, mel_len1, feat_dim).float()
    prompt_feat_len = torch.tensor([mel_len1], dtype=torch.int32)

    # Move to device
    token = token.to(device)
    prompt_token = prompt_token.to(device)
    token_len = token_len.to(device)
    prompt_token_len = prompt_token_len.to(device)
    prompt_feat = prompt_feat.to(device)
    prompt_feat_len = prompt_feat_len.to(device)
    embedding = embedding.to(device)

    print("Calling flow.inference(...) with signature for CausalMaskedDiffWithXvec (streaming, finalize)...")
    flow.eval()
    with torch.no_grad():
        # NOTE: CausalMaskedDiffWithXvec.inference(token, token_len, prompt_token, prompt_token_len,
        #        prompt_feat, prompt_feat_len, embedding, streaming, finalize)
        try:
            feat, flow_cache = flow.inference(
                token=token,
                token_len=token_len,
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_feat,
                prompt_feat_len=prompt_feat_len,
                embedding=embedding,
                streaming=False,
                finalize=True
            )
        except TypeError:
            # fallback for other flow variants (older signature)
            feat, flow_cache = flow.inference(
                token,
                token_len,
                prompt_token,
                prompt_token_len,
                prompt_feat,
                prompt_feat_len,
                embedding,
                None
            )

    print("flow.inference returned feat.shape:", feat.shape, "dtype:", feat.dtype)
    print("feat min/max:", float(feat.min()), float(feat.max()))

    # ---------- Build dummy mel for HiFT vocoder ----------
    # hift.inference expects speech_feat as (1, mel_len, feat_dim) (per code) — try that
    mel2 = feat  # usually (1, mel_len, feat_dim)
    cache_source = torch.zeros(1, 1, 0, device=device)  # empty as in code

    print("Calling hift.inference(speech_feat=mel2, cache_source=cache_source) ...")
    hift.eval()
    with torch.no_grad():
        try:
            speech_out, source_out = hift.inference(speech_feat=mel2, cache_source=cache_source)
            print("hift.inference returned speech_out.shape:", speech_out.shape, "dtype:", speech_out.dtype)
        except Exception as e:
            print("hift.inference failed with exception:", e)
            # try alternative layout (1, feat_dim, T)
            try:
                mel_alt = mel2.transpose(1, 2)
                speech_out, source_out = hift.inference(speech_feat=mel_alt, cache_source=cache_source)
                print("hift.inference returned speech_out.shape with alt layout:", speech_out.shape, "dtype:", speech_out.dtype)
            except Exception as e2:
                print("hift.inference also failed with alt layout:", e2)
                print("Dumping hift module for manual inspection:")
                print(hift)

    print("Done. Paste the printed shapes/any errors here and I'll craft the TorchScript + CoreML steps for HiFT (and then flow).")

if __name__ == '__main__':
    main()
