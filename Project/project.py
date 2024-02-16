from IPython.display import display, Image, Audio
import google.generativeai as genai
from openai import OpenAI
from pathlib import Path
import requests
import base64
import wave
import time
import json
import cv2
import os

# Approximation: 1 second = 30 frames
# Approximation: 1 frame = 760 tokens
# gpt-4-vision-preview limit: 10k tokens
def video_to_base64(video_path):
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".png", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(len(base64Frames), "frames read.")
    return base64Frames

def generate_script(base64Frames):
    # Load config file to get credentials
    with open("config.json", 'r') as f:
        config = json.load(f)
        credentials = config['params']

    # Set up the OpenAI client
    client = OpenAI(
        organization=credentials['openai_organization'],
        api_key=credentials['openai_api_key'],
    )

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "These are frames of a video. Create a voiceover script by a narrator as if she was explaining the concept in a cool way. Note: Only include the narration & do not include timestamp. Strictly Limit it to 50 words",
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::125]),
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 500,
    }

    result = client.chat.completions.create(**params)
    print(result.choices[0].message.content)
    return result.choices[0].message.content

def generate_script_gemini(base64Frames):
    # Load config file to get credentials
    with open("config.json", 'r') as f:
        config = json.load(f)
        credentials = config['params']

    # Set up the gemini client
    genai.configure(api_key=credentials['gemini_api_key'])

    # Set up the model
    generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    model = genai.GenerativeModel(model_name="gemini-pro-vision",
                                generation_config=generation_config,
                                safety_settings=safety_settings)

    # Google GenerativeAI Content Generation
    image_parts = [
        {
            "mime_type": "image/png",  # Change mime_type if using a different format
            "data": base64_frame
        } for base64_frame in base64Frames[0::125]
    ]

    prompt_parts = [
        {"text": "These are frames of a video. Create a voiceover script by a narrator as if she was explaining the concept in a cool way. Note: Only include the narration & do not include timestamp."},
    ] + image_parts

    response = model.generate_content(prompt_parts)
    print(response.text)

    return response.text
    
def generate_audio(script):
    with open("config.json", 'r') as f:
        config = json.load(f)
        credentials = config['params']
        
    url = "https://v2.api.audio/speech/tts/sync"

    payload = {
        "sampling_rate": "24000",
        "bitrate": "192",
        "speed": 1.12,
        "text": script,
        # "voice": "Bronson"
        "voice": "Joanna"
        # "effect": "chewie"
    }
    headers = {
        "Accept": "audio/wav",
        "content-type": "application/json",
        "x-api-key": credentials['audiostack_api_key']
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        file_path = "static/audios/output.wav"

        with open(file_path, 'wb') as file:
            file.write(response.content)

        print(f"Audio saved to {file_path}")
        return file_path

    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

if __name__ == "__main__":
    video_path = "static/videos/football_mini.mp4"
    base64Frames = video_to_base64(video_path)
    script = generate_script_gemini(base64Frames)
    audio_path = generate_audio(script)
    display(Audio(audio_path))