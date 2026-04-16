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
from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

# Set in __main__ before uvicorn serves (server-side path to reference wav)
prompt_wav_path = ''


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
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form()):
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav_path)
    return StreamingResponse(generate_data(model_output), media_type='application/octet-stream')


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form()):
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_wav_path)
    return StreamingResponse(generate_data(model_output), media_type='application/octet-stream')


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_wav_path)
    return StreamingResponse(generate_data(model_output), media_type='application/octet-stream')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    default_prompt = os.path.normpath(os.path.join(ROOT_DIR, '..', '..', '..', 'asset', 'zero_shot_prompt.wav'))
    parser.add_argument('--prompt_wav',
                        type=str,
                        default=default_prompt,
                        help='server-side wav path for zero_shot / cross_lingual / instruct2')
    args = parser.parse_args()
    prompt_wav_path = os.path.abspath(args.prompt_wav)
    if not os.path.isfile(prompt_wav_path):
        raise FileNotFoundError('prompt_wav not found: {}'.format(prompt_wav_path))
    cosyvoice = AutoModel(model_dir=args.model_dir, load_vllm=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
