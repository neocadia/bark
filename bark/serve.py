import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from IPython.display import Audio
import nltk  # we'll use this to split into sentences
import numpy as np

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE

### HTTP API Specific

import argparse
import json
import logging

from fastapi import fastapi
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn

from typing import Generator, Optional, List, Dict, Any

import time

import shortuuid
from pydantic import BaseModel, Field

### End HTTP API Specific

# 
# HTTP API Serve
# 

class AudioGenerationRequest(BaseModel):
    # TODO: support streaming, stop with a list of text etc.
    text: str
    history_prompt: Optional[Union[Dict, str]] = None
    text_temp: Optional[float] = 0.7
    waveform_temp: Optional[float] = 0.7
    silent: Optional[bool] = False
    output_full: Optional[bool] = False

logger = logging.getLogger(__name__)

# app_settings = AppSettings()
app = fastapi.FastAPI()
headers = {"User-Agent": "Bark API Server"}

def generate_audio_arrays(sentences, history_prompt, temp, min_eos_p):
    for sentence in sentences:
        semantic_tokens = generate_text_semantic(
            sentence,
            history_prompt,
            temp,
            min_eos_p,  # this controls how likely the generation is to end
        )

        audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER,)
        yield audio_array

@app.post("/v1/tts/generate_audio")
async def create_audio_generation(request: AudioGenerationRequest):
    """Creates an audio generation for a text prompt"""

    fullPrompt = request.text.replace("\n", " ").strip()
    nltk.download('punkt')
    sentences = nltk.sent_tokenize(fullPrompt)

    GEN_TEMP = 0.6
    SPEAKER = "v2/en_speaker_6"
    silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence
    
    return StreamingResponse(json.dump([generate_audio_arrays(sentences, SPEAKER, GEN_TEMP, 0.05).tolist(), silence.copy().tolist()], media_type='text/event-stream'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bark Restful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--allow-credentials", action="store_true", help="allow credentials")
    parser.add_argument("--allowed-origins", type=json.loads, default=["*"], help="allowed origins")
    parser.add_argument("--allowed-methods", type=json.loads, default=["*"], help="allowed methods")
    parser.add_argument("--allowed-headers", type=json.loads, default=["*"], help="allowed headers")

    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.debug(f"==== args ====\n{args}")

    uvicorn.run(app, host=args.host, port=args.port, reload=True)
