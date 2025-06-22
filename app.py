from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import os
from utils import extract_mfcc
import numpy as np
import pyaudio
import wave
import threading
import re

app = Flask(__name__)
CORS(app)

models = np.load("models_hmmlearn_0603.npy", allow_pickle=True)

word_list = ['数字', '语音', '语言', '处理', '中国', '忠告', '北京', '背景', '上海', '商行',
            'Speech', 'Speaker', 'Signal', 'Sequence', 'Processing', 'Print', 'Project', 'File', 'Open', 'Close']

def recognize(audio_file):
    fea = extract_mfcc(audio_file)
    scores = []
    for m in range(20):
        model = models[m]
        score, _ = model.decode(fea)
        scores.append(score)
    # print(scores)
    det_lab = np.argmax(scores)
    return word_list[det_lab]

record_path = None
record_idx = 0

def get_max_record_index(folder='uploads'):
    global record_idx
    pattern = re.compile(r'^000-(\d+)\.wav$')
    indices = []
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            indices.append(int(match.group(1)))
    if not indices:
        record_idx = -1
    else:
        record_idx = max(indices)

def recording():
    global record_path, record_idx
    record_idx += 1
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 8192
    RECORD_SECONDS = 1.5
    output_file = 'uploads/000-'+ str(record_idx) +'.wav'
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording:")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Recorded.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    record_path = output_file
    # print("record path: ", record_path)

@app.route('/record', methods=['POST'])
def start_recording():
    # print("Starting recording by python...")
    threading.Thread(target=recording).start()
    return jsonify({"status": "recording started, record_idx is " + str(record_idx)})

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize_audio():
    return recognize(record_path)

if __name__ == '__main__':
    get_max_record_index()
    app.run(debug=True)