# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from fastapi.responses import Response
import io
import wave
import tempfile
import torchaudio

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2 , CosyVoice3
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio
        
def chunk_text(text: str, max_words: int = 200) -> list[str]:
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)

    return chunks


def collect_pcm(model_output) -> bytes:
    pcm_chunks = []

    for i in model_output:
        pcm = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        pcm_chunks.append(pcm)

    return b"".join(pcm_chunks)

@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))



@app.post("/tts_zero_shot")
async def tts_zero_shot(
    tts_text: str = Form(...),
    prompt_text: str = Form(...),
    prompt_wav: UploadFile = File(...)
):
    # Load prompt audio (16kHz required)
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)

    # Run zero-shot inference (non-streaming)
    model_output = list(
        cosyvoice.inference_zero_shot(
            tts_text,
            prompt_text,
            prompt_speech_16k,
            stream=False
        )
    )

    # Combine all chunks of generated audio
    audio = np.concatenate([chunk["tts_speech"] for chunk in model_output], axis=-1)

    # Convert float [-1,1] → PCM int16
    pcm16 = (audio * (2 ** 15)).astype(np.int16)

    # Write WAV in memory
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)           # mono
        wav_file.setsampwidth(2)           # int16 → 2 bytes
        wav_file.setframerate(16000)       # 16kHz
        wav_file.writeframes(pcm16.tobytes())

    buffer.seek(0)

    return Response(
        content=buffer.read(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="tts.wav"'
        }
    )

# @app.post("/inference_cross_lingual")
# async def inference_cross_lingual(
#     tts_text: str = Form(...),
#     prompt_wav: UploadFile = File(...)
# ):
#     """
#     Cross-lingual TTS using a reference voice.
#     Accepts ANY audio format and normalizes to PCM WAV.
#     """

#     # 1. Create temp WAV path
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#         prompt_wav_path = f.name

#     try:
#         # 2. Decode uploaded audio (mp3/m4a/wav/etc.)
#         waveform, sr = torchaudio.load(prompt_wav.file)

#         # 3. Convert to mono
#         if waveform.dim() > 1 and waveform.shape[0] > 1:
#             waveform = waveform.mean(dim=0, keepdim=True)

#         # 4. Resample to 24k (CosyVoice default)
#         if sr != 24000:
#             waveform = torchaudio.functional.resample(
#                 waveform, sr, 24000
#             )

#         # 5. Save REAL PCM WAV
#         torchaudio.save(
#             prompt_wav_path,
#             waveform,
#             24000,
#             encoding="PCM_S",
#             bits_per_sample=16
#         )

#         # 6. Run CosyVoice inference
#         model_output = cosyvoice.inference_cross_lingual(
#             tts_text,
#             prompt_wav_path
#         )

#         # 7. Stream result
#         return StreamingResponse(
#             generate_data(model_output),
#             media_type="audio/wav"
#         )

#     finally:
#         # 8. Cleanup temp file
#         if os.path.exists(prompt_wav_path):
#             os.remove(prompt_wav_path)

@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    filename = prompt_wav.filename or "zero_shot_prompt.wav"
    asset_wav_path = os.path.join('/workspace/cosyvoice-api/asset/', filename)

    with open(asset_wav_path, "wb") as f:
        f.write(await prompt_wav.read())
    model_output = cosyvoice.inference_cross_lingual(tts_text, asset_wav_path)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))



@app.post("/inference_instruct2")
async def inference_instruct2(
    tts_text: str = Form(...),
    instruct_text: str = Form(...),
    prompt_wav: UploadFile = File(...)
):
    filename = prompt_wav.filename or "zero_shot_prompt.wav"
    asset_wav_path = os.path.join("/workspace/cosyvoice-api/asset", filename)

    # ✅ READ ONCE — BEFORE streaming
    prompt_bytes = await prompt_wav.read()

    async def pcm_stream_generator():
            # 1️⃣ Save prompt wav from cached bytes
            with open(asset_wav_path, "wb") as f:
                f.write(prompt_bytes)

            # 2️⃣ Split text
            chunks = chunk_text(tts_text, max_words=200)

            # 3️⃣ Inference + stream
            for chunk in chunks:
                model_output = cosyvoice.inference_instruct2(
                    chunk,
                    instruct_text,
                    asset_wav_path
                )

                for pcm_chunk in generate_data(model_output):
                    yield pcm_chunk

    
    return StreamingResponse(
        pcm_stream_generator()
    )

# @app.get("/inference_instruct2")
# @app.post("/inference_instruct2")
# async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
#     # prompt_speech_16k = load_wav(prompt_wav.file, 48000)
#     # speech, sample_rate = torchaudio.load(prompt_wav, backend='soundfile')
#     # speech = speech.mean(dim=0, keepdim=True)
#     filename = prompt_wav.filename or "zero_shot_prompt.wav"
#     asset_wav_path = os.path.join('/workspace/cosyvoice-api/asset/', filename)

#     with open(asset_wav_path, "wb") as f:
#         f.write(await prompt_wav.read())

#     chunks = chunk_text(tts_text, max_words=200)

#     pcm_all = []

#             # 3️⃣ Inference loop
#     for chunk in chunks:
#             model_output = cosyvoice.inference_instruct2(
#                 chunk,
#                 instruct_text,
#                 asset_wav_path
#             )

#             pcm = collect_pcm(model_output)
#             pcm_all.append(pcm)

#     final_pcm = b"".join(pcm_all)
#     # -----------------------------------
#     # 2️⃣ Pass FILE PATH to CosyVoice
#     # -----------------------------------
#     # model_output = cosyvoice.inference_instruct2(
#     #     tts_text,
#     #     instruct_text,
#     #     asset_wav_path   # ✅ EXACTLY like official example
#     # )
#    # model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, speech)
#     return StreamingResponse(iter([final_pcm]))


def generate_wav_audio(model_output, sample_rate=22050):
    audio_np = []

    for i in model_output:
        audio_np.append(i['tts_speech'].numpy())

    audio = np.concatenate(audio_np)
    audio_int16 = (audio * (2 ** 15)).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer.read()

@app.get("/inference_instruct3")
@app.post("/inference_instruct3")
async def inference_instruct3(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    # prompt_speech_16k = load_wav(prompt_wav.file, 48000)
    # speech, sample_rate = torchaudio.load(prompt_wav, backend='soundfile')
    # speech = speech.mean(dim=0, keepdim=True)
    filename = prompt_wav.filename or "zero_shot_prompt.wav"
    asset_wav_path = os.path.join('/workspace/cosyvoice-api/asset/', filename)

    with open(asset_wav_path, "wb") as f:
        f.write(await prompt_wav.read())

    # -----------------------------------
    # 2️⃣ Pass FILE PATH to CosyVoice
    # -----------------------------------
    model_output = cosyvoice.inference_instruct2(
        tts_text,
        instruct_text,
        asset_wav_path   # ✅ EXACTLY like official example
    )
    wav_bytes = generate_wav_audio(model_output)
   # model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, speech)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=tts.wav"
        }
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
        except Exception:
            try:
                cosyvoice = CosyVoice3(args.model_dir)
            except Exception:
               raise TypeError('no valid model_type!')
    uvicorn.run(app, host="0.0.0.0", port=args.port)
