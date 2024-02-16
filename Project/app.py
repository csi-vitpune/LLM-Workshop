from flask import Flask, render_template, request
import google.generativeai as genai
from openai import OpenAI
from project import *
import json

app = Flask(__name__)

@app.route('/')
def index():
    return "hello"

@app.route('/upload')
def upload():
    return render_template('generate.html')

@app.route('/generateSummary', methods=['POST'])
def generateSummary():
    video = request.files['video']
    video.save('static/videos/input.mp4')
    video_path = "static/videos/input.mp4"
    base64Frames = video_to_base64(video_path)
    script = generate_script_gemini(base64Frames)
    audio_path = generate_audio(script)
    return render_template('preview.html', audio="output.wav")

@app.route('/hello')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()