# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
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
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

# inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
# instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
#                  '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
#                  '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮',
#                  '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'}


inference_mode_list = [
    'Pretrained Voice',
    '3s Instant Voice Cloning',
    'Cross-Language Voice Cloning',
    'Natural Language Control'
]

instruct_dict = {
    'Pretrained Voice':
        '1. Select a pretrained voice\n'
        '2. Click the Generate Audio button',

    '3s Instant Voice Cloning':
        '1. Select a prompt audio file or record a prompt audio (must be within 30 seconds). '
        'If both are provided, the prompt audio file will be used first\n'
        '2. Enter the prompt text\n'
        '3. Click the Generate Audio button',

    'Cross-Language Voice Cloning':
        '1. Select a prompt audio file or record a prompt audio (must be within 30 seconds). '
        'If both are provided, the prompt audio file will be used first\n'
        '2. Click the Generate Audio button',

    'Natural Language Control':
        '1. Select a pretrained voice\n'
        '2. Enter the instruction text\n'
        '3. Click the Generate Audio button'
}
stream_mode_list = [("No", False), ("Yes", True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['自然语言控制']:
        if instruct_text == '':
            gr.Warning('您正在使用自然语言控制模式, 请输入instruct文本')
            yield (cosyvoice.sample_rate, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略')
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['跨语种复刻']:
        if instruct_text != '':
            gr.Info('您正在使用跨语种复刻模式, instruct文本会被忽略')
        if prompt_wav is None:
            gr.Warning('您正在使用跨语种复刻模式, 请提供prompt音频')
            yield (cosyvoice.sample_rate, default_data)
        gr.Info('您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            yield (cosyvoice.sample_rate, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (cosyvoice.sample_rate, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['预训练音色']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')
        if sft_dropdown == '':
            gr.Warning('没有可用的预训练音色！')
            yield (cosyvoice.sample_rate, default_data)
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s极速复刻']:
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！')

    if mode_checkbox_group == '预训练音色':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '3s极速复刻':
        logging.info('get zero_shot inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '跨语种复刻':
        logging.info('get cross_lingual inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    预训练模型 [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### Please enter the text to be synthesized, select an inference mode, ""and follow the instructions step by step")

        tts_text = gr.Textbox(label="Input Text for Synthesis",lines=1,value="I am a new generative speech model launched by the Tongyi Lab speech team, delivering comfortable and natural text-to-speech synthesis.")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label="Select Inference Mode", value=inference_mode_list[0])
            instruction_text = gr.Text(label="Steps", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label="Select Pretrained Voice", value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label="Streaming Inference", value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="Speed Control (Non-streaming only)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="🎲")
                seed = gr.Number(value=0, label="Random Inference Seed")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources="upload", type="filepath", label="Select prompt audio file (sample rate ≥ 16kHz)")
            prompt_wav_record = gr.Audio(sources="microphone", type="filepath", label="Record prompt audio")
        prompt_text = gr.Textbox(label="Enter Prompt Text", lines=1, placeholder="Enter prompt text (must match the prompt audio content; automatic recognition not supported yet)...", value="")
        instruct_text = gr.Textbox(label="Enter Instruction Text", lines=1, placeholder="Enter instruction text.", value="")

        generate_button = gr.Button("Generate Audio")
        audio_output = gr.Audio(label="Synthesized Audio", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()