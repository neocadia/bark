import nltk  # we'll use this to split into sentences
import numpy as np
import argparse
import json
import logging
from typing import Generator, Optional, Union, List, Dict, Any
import time
import shortuuid
from collections import Counter

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn

from pydantic import BaseModel, Field

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE

# 
# HTTP API Serve (requires Python 3.10+)
#

logger = logging.getLogger(__name__)  # Initialize a logger object with the current module name
# app_settings = AppSettings()
app = FastAPI() # Create a FastAPI app instance
headers = {"User-Agent": "Bark API Server"} # Define custom headers for HTTP requests

debugMode = True
app.add_middleware(
       CORSMiddleware,
       allow_origins=['*'],
       allow_methods=['*'],
       allow_headers=['*'],
    )

if __name__ == "__main__": #checks if the script is being run as the main program (as opposed to being imported as a module into another script). If it is being run as the main program, the code that follows will be executed.
    parser = argparse.ArgumentParser( #creates an instance of the ArgumentParser class from the argparse module. This object is used to parse command-line arguments passed to the script
        description="Bark Restful API server."
    )
    # Define command-line arguments for the server
    parser.add_argument("--host", type=str, default="0.0.0.0", help="hostname")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--allow-credentials", default=False, action=argparse.BooleanOptionalAction, help="allow credentials")
    parser.add_argument("--debug", default=True, action=argparse.BooleanOptionalAction, help="enable debug mode")
    parser.add_argument("--allowed-origins", type=json.loads, default=["*"], help="allowed origins")
    parser.add_argument("--allowed-methods", type=json.loads, default=["*"], help="allowed methods")
    parser.add_argument("--allowed-headers", type=json.loads, default=["*"], help="allowed headers")

    args = parser.parse_args()
    # Add CORS middleware to the FastAPI app with the specified command-line arguments
    app.add_middleware(
       CORSMiddleware,
       allow_origins=args.allowed_origins,
       allow_credentials=args.allow_credentials,
       allow_methods=args.allowed_methods,
       allow_headers=args.allowed_headers,
    )

    debugMode = args.debug

    print("==== Bootstrapping ====")
    print('Preloading models...')
    preload_models()
    print('Models preloaded.')
    print('Loading nltk...')
    nltk.download('punkt') # Download the Punkt tokenizer for splitting the text into sentences
    print('Loaded nltk.')
    print("==== Bootstrapping Complete ====")

    print(f"==== args ====\n{args}") # Log the parsed command-line arguments
    
    # Run the Uvicorn ASGI server with the specified host and port
    uvicorn.run("serve:app", host=args.host, port=args.port, reload=False)

class AudioGenerationRequest(BaseModel):  # Define a class for audio generation requests using Pydantic BaseModel
    text: str  # Define the text field as a string
    history_prompt: Union[Union[Dict, str], None] = None  # Define the history_prompt field as a string, dict, or None
    text_temp: Union[float, None] = 0.7  # Define the text_temp field as a float or None, with a default value of 0.7
    waveform_temp: Union[float, None] = 0.7  # Define the waveform_temp field as a float or None, with a default value of 0.7
    gen_temp: Union[float, None] = 0.7 # Semantic temp.
    output_full: Union[bool, None] = False  # Define the output_full field as a bool or None, with a default value of False,
    min_eos_p: Union[float, None] = 0.05 # This controls how likely the generation is to end.
    silence: Union[float, None] = 0.0 # How much silence to include at the end of each sentence. Default 1/4 second.
    max_gen_duration_s: Union[float, None] = None # FIXME: Make this dynamic based on the estimated length of time for a sentence given from the tokenizer.
    top_k: None = None
    top_p: None = None
    allow_early_stop: bool = True

# Define a generator function that yields audio arrays and silences
def generate_audio_arrays(sentences, request):
    if (debugMode):
        print('##### Job Started #####')

    silence = np.zeros(int(request.silence * SAMPLE_RATE))  # quarter second of silence
    silence_copied = silence.copy()

    total_start_time = time.time()
    for index, sentence in enumerate(sentences): # Iterate through each sentence
        start_time = time.time()
        semantic_tokens = generate_text_semantic(
             sentence,
             history_prompt=request.history_prompt,
             temp=request.gen_temp,
             min_eos_p=request.min_eos_p,
             max_gen_duration_s=request.max_gen_duration_s,
             allow_early_stop=request.allow_early_stop,
             top_k=request.top_k,
             top_p=request.top_p,
             silent=not debugMode
         )
        # Generate the audio array for the current sentence
        audio_array = semantic_to_waveform(
            semantic_tokens,
            request.history_prompt,
            request.waveform_temp
        ) # generate the audio by converting the text into semantic tokens first, then generating a waveform, resulting in more natural-sounding audio, as the semantic tokens can provide additional context for the generated audio
        # audio_array = generate_audio(sentence, history_prompt) # less computationally expensive, does not use NLTK
        # json_string = son.dumps(concatenated_array.tolist()) # Convert the concatenated audio array to a JSON string
        # yield json_string # Yield the JSON string to the calling function (to be used for streaming the audio)

        concatenated_array = np.concatenate([audio_array, silence_copied]) # Concatenate the audio array with the silence buffer
        transformed_data = concatenated_array.tobytes() # Convert the concatenated audio array to binary data

        yield transformed_data
        end_time = time.time()
        if (debugMode):
            print(f'{index}) Sentence: {sentence}\n{index}) Generated in {round(end_time - start_time, 2)} seconds.')
    if (debugMode):
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        word_count = Counter(request.text.lower().split()).total()
        sentences_per_second = round((len(sentences) / total_elapsed_time), 2)
        words_per_second = round((word_count / total_elapsed_time), 2)
        characters_per_second = round((len(request.text) / total_elapsed_time), 2)
        print('##### Job Complete #####')
        print(f'Generated {len(sentences)} sentences ({sentences_per_second}/s)')
        print(f'Generated {word_count} words ({words_per_second}/s)')
        print(f'Generated {len(request.text)} characters ({characters_per_second}/s)')
        print(f'Job completed in {total_elapsed_time} seconds.')
        print('##### End Job Info #####')
        
@app.post("/v1/tts/generate_audio")
async def create_audio_generation(request: AudioGenerationRequest, response: Response):
    """Creates an audio generation for a text prompt"""
    fullPrompt = request.text.replace("\n", " ").strip() # Replace newline characters with spaces and strip whitespace from the text
    sentences = nltk.sent_tokenize(fullPrompt) # Split the text into sentences
    # Set CORS headers
    # response.headers["Access-Control-Allow-Origin"] = "*"
    # response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    # response.headers[
    #     "Access-Control-Allow-Headers"
    # ] = "Content-Type, Authorization"

    return StreamingResponse(generate_audio_arrays(sentences, request), media_type="application/octet-stream")
