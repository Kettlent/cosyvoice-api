import soundfile as sf

path = "./asset/conan.wav"

try:
    data, sr = sf.read(path)
    print("✅ SUCCESS")
    print("Sample rate:", sr)
    print("Shape:", data.shape)
    print("Dtype:", data.dtype)
except Exception as e:
    print("❌ FAILED")
    print(type(e).__name__, e)