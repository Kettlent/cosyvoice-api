# save_hift_no_istft_torchscript.py
import torch
import sys
import numpy as np
import types
from typing import Any
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2

MODEL_DIR = 'pretrained_models/CosyVoice2-0.5B'  # adjust if needed

# ----- Utilities: normalize numpy ints to Python ints for TorchScript -----
def _to_py_int_if_np(x: Any):
    if isinstance(x, (np.integer,)):
        return int(x)
    return x

def _normalize_attr_value(v: Any):
    # Recursively convert numpy ints in lists/tuples/dicts to python ints
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (list, tuple)):
        converted = [_normalize_attr_value(i) for i in v]
        return tuple(converted) if isinstance(v, tuple) else converted
    if isinstance(v, dict):
        return {k: _normalize_attr_value(val) for k, val in v.items()}
    return v

def normalize_module_attrs(module: torch.nn.Module):
    changed = []
    # First, fix common convolution attributes if present
    for name, mod in module.named_modules():
        # typical conv attrs: kernel_size, stride, padding, dilation, output_padding
        for attr in ("kernel_size", "stride", "padding", "dilation", "output_padding"):
            if hasattr(mod, attr):
                val = getattr(mod, attr)
                new_val = _normalize_attr_value(val)
                if new_val != val:
                    try:
                        setattr(mod, attr, new_val)
                        changed.append((name, attr, val, new_val))
                    except Exception:
                        # Some attributes are readonly; skip if can't set
                        pass
        # Also sanitize module.__dict__ simple attributes that might contain numpy ints
        for k, v in list(mod.__dict__.items()):
            # skip params, buffers, submodules, tensors
            if k.startswith('_') or isinstance(v, (torch.Tensor, torch.nn.Module)):
                continue
            if isinstance(v, (np.integer,)):
                try:
                    setattr(mod, k, int(v))
                    changed.append((name, k, v, int(v)))
                except Exception:
                    pass
            elif isinstance(v, (list, tuple, dict)):
                new_v = _normalize_attr_value(v)
                if new_v != v:
                    try:
                        setattr(mod, k, new_v)
                        changed.append((name, k, v, new_v))
                    except Exception:
                        pass
    return changed

# Helper: deterministic patcher (used only for tracing fallback; scripting doesn't need it)
class _DeterministicRandomPatcher:
    def __init__(self):
        self.orig_randn = torch.randn
        self.orig_rand = torch.rand
        import torch.distributions as d
        self.orig_uniform = d.uniform.Uniform.sample if hasattr(d.uniform.Uniform, 'sample') else None

    def _zero_like_randn(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            size = tuple(args[0])
        else:
            size = tuple(args)
        dtype = kwargs.get('dtype', torch.float32)
        device = kwargs.get('device', torch.device('cpu'))
        return torch.zeros(size, dtype=dtype, device=device)

    def _zero_like_rand(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            size = tuple(args[0])
        else:
            size = tuple(args)
        dtype = kwargs.get('dtype', torch.float32)
        device = kwargs.get('device', torch.device('cpu'))
        return torch.zeros(size, dtype=dtype, device=device)

    def _uniform_sample_fixed(self, self_obj, *args, **kwargs):
        # Uniform.sample(self, sample_shape=...)
        if len(args) >= 1 and isinstance(args[0], (tuple, list)):
            size = tuple(args[0])
        else:
            size = ()
        device = getattr(self_obj, 'low', torch.tensor(0)).device if hasattr(self_obj, 'low') else torch.device('cpu')
        dtype = getattr(self_obj, 'low', torch.tensor(0)).dtype if hasattr(self_obj, 'low') else torch.float32
        return torch.zeros(size, dtype=dtype, device=device)

    def apply(self):
        torch.randn = self._zero_like_randn
        torch.rand = self._zero_like_rand
        import torch.distributions as d
        if hasattr(d, 'uniform') and hasattr(d.uniform.Uniform, 'sample'):
            d.uniform.Uniform.sample = self._uniform_sample_fixed

    def restore(self):
        torch.randn = self.orig_randn
        torch.rand = self.orig_rand
        import torch.distributions as d
        if hasattr(d, 'uniform') and self.orig_uniform is not None:
            d.uniform.Uniform.sample = self.orig_uniform

# ----- Wrapper module (no istft) -----
class HiFT_NoISTFT(torch.nn.Module):
    def __init__(self, hift_module):
        super().__init__()
        self.hift = hift_module.eval()

        # Copy submodules (no changes)
        self.f0_predictor = self.hift.f0_predictor
        self.f0_upsamp = self.hift.f0_upsamp
        self.m_source = self.hift.m_source

        self.conv_pre = self.hift.conv_pre
        self.ups = self.hift.ups
        self.source_downs = self.hift.source_downs
        self.source_resblocks = self.hift.source_resblocks
        self.resblocks = self.hift.resblocks
        self.conv_post = self.hift.conv_post

        # Make plain Python types for sizing info
        self.num_upsamples = int(self.hift.num_upsamples)
        self.num_kernels = int(self.hift.num_kernels)
        # istft_params may be a mapping with numpy ints — normalize immediately
        self.istft_params = {k: int(v) if isinstance(v, (np.integer,)) else v for k, v in dict(self.hift.istft_params).items()}
        self.lrelu_slope = float(self.hift.lrelu_slope)
        self.reflection_pad = self.hift.reflection_pad
        self.stft_window = self.hift.stft_window

    def forward(self, speech_feat: torch.Tensor, cache_source: torch.Tensor):
        # We purposely do NOT alter behavior at runtime. Deterministic monkeypatch is only needed for trace fallback.
        patcher = None
        if torch.jit.is_tracing():
            # patch randomness to deterministic zeros for trace validation (trace fallback only)
            patcher = _DeterministicRandomPatcher()
            patcher.apply()
            torch.manual_seed(0)
            np.random.seed(0)
            try:
                torch.cuda.manual_seed_all(0)
            except Exception:
                pass

        # === inference(): mel -> f0 -> source ===
        f0 = self.f0_predictor(speech_feat)
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, N, T]
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)  # now [B, 1, T] (source)

        # apply cache_source conservatively
        if cache_source is not None and cache_source.numel() > 0:
            L = int(cache_source.shape[2])
            if L > 0:
                s[:, :, :L] = cache_source

        # === EXACT decode(): STFT FIRST ===
        s_stft_real, s_stft_imag = self.hift._stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        # === conv_pre ===
        x = self.conv_pre(speech_feat)

        # === SINGLE upsample loop ===
        for i in range(self.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # fusion
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            # resblocks
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j](x)
                xs = rb if xs is None else xs + rb

            x = xs / self.num_kernels

        # === post conv ===
        x = torch.nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)
        x = self.conv_post(x)

        # === final magnitude & phase ===
        n_fft = int(self.istft_params["n_fft"])
        magnitude = torch.exp(x[:, : n_fft // 2 + 1, :])
        phase = torch.sin(x[:, n_fft // 2 + 1 :, :])

        if patcher is not None:
            patcher.restore()

        return magnitude, phase, s

# ----- main: load, normalize, script -----
def main():
    print("Loading CosyVoice2 and HiFT (no istft wrapper)...")
    cosy = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    hift = cosy.model.hift

    print("Normalizing numpy-based attributes to Python ints in HiFT...")
    changes = normalize_module_attrs(hift)
    if changes:
        print("Applied attribute normalizations (sample):")
        for c in changes[:20]:
            print("  module:", c[0], "attr:", c[1], "old:", c[2], "new:", c[3])
    else:
        print("No attribute normalization needed.")

    wrapper = HiFT_NoISTFT(hift).eval().to('cpu')

    # Representative input — match previous tracing (you used 512 earlier)
    mel_len = 512
    inp = torch.randn(1, 80, mel_len, dtype=torch.float32)
    cache = torch.zeros(1, 1, 0, dtype=torch.float32)

    print("Calling wrapper once for sanity (python run)...")
    with torch.no_grad():
        mag, pha, s = wrapper(inp, cache)
    print("mag.shape", mag.shape, "pha.shape", pha.shape, "s.shape", s.shape)

    print("Attempting to script wrapper to TorchScript (no istft inside)...")
    try:
        scripted = torch.jit.script(wrapper)
        scripted.save("hift_no_istft_scripted.pt")
        print("Saved hift_no_istft_scripted.pt")
    except Exception as e:
        print("Scripting failed with exception:")
        import traceback; traceback.print_exc()
        print("Falling back to tracing attempt (strict=False) for debugging...")
        try:
            # For tracing fallback we'll keep deterministic monkeypatch in forward during tracing,
            # already implemented above.
            traced = torch.jit.trace(wrapper, (inp, cache), strict=False)
            traced.save("hift_no_istft_traced_debug.pt")
            print("Saved hift_no_istft_traced_debug.pt (trace fallback)")
        except Exception:
            print("Trace fallback also failed — printing stack:")
            import traceback; traceback.print_exc()
            raise

if __name__ == "__main__":
    main()
