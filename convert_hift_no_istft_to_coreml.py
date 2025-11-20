# convert_hift_no_istft_to_coreml.py
import coremltools as ct
import torch

print("Loading traced HiFT model...")
ts = torch.jit.load("hift_no_istft_traced_debug.pt", map_location="cpu")

print("Converting TorchScript to CoreML...")
mlmodel = ct.convert(
    ts,
    convert_to="mlprogram",
    inputs=[
        ct.TensorType(name="speech_feat", shape=(1, 80, 512)),
        ct.TensorType(name="cache_source", shape=(1, 1, 0)),
    ],
    minimum_deployment_target=ct.target.iOS17,
)

mlmodel.save("HiFT_NoISTFT.mlpackage")
print("Saved HiFT_NoISTFT.mlpackage")
