# convert_hift_to_coreml.py
import coremltools as ct
import torch
import numpy as np

# Load your exact traced model
model = torch.jit.load("hift_traced.pt", map_location="cpu")
model.eval()

# IMPORTANT: match EXACT input size used during tracing
# You traced with mel.shape = (1,80,512)
example_input = np.random.randn(1, 80, 512).astype(np.float32)

print("Converting TorchScript → CoreML (this can take a few minutes)...")

mlmodel = ct.convert(
    model,
    inputs=[ct.TensorType(shape=example_input.shape)],
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.CPU_ONLY  # or CPU_AND_NE / ALL
)

mlmodel.save("HiFT_vocoder.mlmodel")
print("=== CoreML conversion complete ===")
print("Saved as HiFT_vocoder.mlmodel")
