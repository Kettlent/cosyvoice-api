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

@app.post("/inference_cross_lingual")
async def inference_cross_lingual(
    tts_text: str = Form(...),
    prompt_wav: UploadFile = File(...)
):
    # 1. Save uploaded audio to a temp WAV file
    suffix = os.path.splitext(prompt_wav.filename or "")[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await prompt_wav.read())
        prompt_wav_path = f.name

    try:
        # 2. Pass FILE PATH (not tensor!) to CosyVoice
        model_output = cosyvoice.inference_cross_lingual(
            tts_text,
            prompt_wav_path
        )

        # 3. Stream output (same as demo)
        return StreamingResponse(
            generate_data(model_output),
            media_type="audio/wav"
        )

    finally:
        # 4. Cleanup temp file
        if os.path.exists(prompt_wav_path):
            os.remove(prompt_wav_path)


# @app.get("/inference_cross_lingual")
# @app.post("/inference_cross_lingual")
# async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
#     prompt_speech_16k = load_wav(prompt_wav.file, 16000)
#     model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
#     return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


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
