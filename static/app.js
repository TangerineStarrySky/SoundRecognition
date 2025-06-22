// webkitURL is deprecated but kept for compatibility
URL = window.URL || window.webkitURL;

// Audio context setup
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext = new AudioContext();

// DOM elements
var recordButton = document.getElementById("recordButton");
var canvas = document.getElementById("analyser");
var canvasWidth = canvas.width;
var canvasHeight = canvas.height;
var analyserContext = canvas.getContext('2d');
var result = document.getElementById('result');
var recordingsList = document.getElementById('recordingsList');

// Audio nodes and recording state
var gumStream;
var rec;
var input;
var analyserNode;
var isRecording = false;
var recIndex = 0;

// Event binding
recordButton.addEventListener("click", startRecording);

function convertToMono(input) {
    var splitter = audioContext.createChannelSplitter(2);
    var merger = audioContext.createChannelMerger(2);
    input.connect(splitter);
    splitter.connect(merger, 0, 0);
    splitter.connect(merger, 0, 1);
    return merger;
}

function updateAnalysers() {
    if (!analyserContext || !isRecording) return;

    var SPACING = 3;
    var BAR_WIDTH = 1;
    var numBars = Math.round(canvasWidth / SPACING);
    var freqByteData = new Uint8Array(analyserNode.frequencyBinCount);

    analyserNode.getByteFrequencyData(freqByteData);

    analyserContext.clearRect(0, 0, canvasWidth, canvasHeight);
    analyserContext.lineCap = 'round';
    var multiplier = analyserNode.frequencyBinCount / numBars;

    for (var i = 0; i < numBars; ++i) {
        var magnitude = 0;
        var offset = Math.floor(i * multiplier);
        for (var j = 0; j < multiplier; j++)
            magnitude += freqByteData[offset + j];
        magnitude = magnitude / multiplier;
        analyserContext.fillStyle = "hsl(" + Math.round((i * 360) / numBars) + ", 100%, 50%)";
        analyserContext.fillRect(i * SPACING, canvasHeight, BAR_WIDTH, -magnitude);
    }

    window.requestAnimationFrame(updateAnalysers);
}

function startRecording() {
    console.log("recordButton clicked");
    recIndex++;

    fetch('http://localhost:5000/record', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            console.log(data.status);
            setTimeout(() => {
                stopRecording();
            }, 1600);
        })
        .catch(err => console.error("录音失败", err));

    isRecording = true;
    recordButton.disabled = true;

    navigator.mediaDevices.getUserMedia({ audio: true, video: false }).then(stream => {
        audioContext = new AudioContext();
        gumStream = stream;

        input = audioContext.createMediaStreamSource(stream);
        var inputPoint = audioContext.createGain();
        var audioInput = convertToMono(input);

        audioInput.connect(inputPoint);

        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 2048;
        inputPoint.connect(analyserNode);

        var zeroGain = audioContext.createGain();
        zeroGain.gain.value = 0.0;
        inputPoint.connect(zeroGain);
        zeroGain.connect(audioContext.destination);

        updateAnalysers();

        rec = new Recorder(input, { numChannels: 1 });
        rec.record();
        console.log("Recording started");
    }).catch(error => {
        recordButton.disabled = false;
        console.error("getUserMedia() error: ", error);
    });
}

function stopRecording() {
    isRecording = false;
    recordButton.disabled = false;
    result.innerHTML = "识别结果: ";
    rec.stop();
    gumStream.getAudioTracks()[0].stop();
    rec.exportWAV(recognition);
}

function recognition(blob) {
    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('li');

    recordButton.disabled = true;
    var filename = 'audio' + recIndex + '.wav';
    au.controls = true;
    au.src = url;
    li.appendChild(au);

    var upload = document.createElement('recobutton');
    var refresh = document.createElement('refreshbutton');
    upload.href = refresh.href = "#";
    upload.innerHTML = "识别";
    refresh.innerHTML = "重置";

    upload.addEventListener("click", function () {
        var xhr = new XMLHttpRequest();
        xhr.onload = function () {
            if (this.readyState === 4 && this.status === 200) {
                result.innerHTML = "识别结果: " + this.responseText;
            } else {
                result.innerHTML = "请求失败，状态码: " + this.status;
            }
        };
        var fd = new FormData();
        fd.append("file", blob, filename);
        xhr.open("POST", "/recognize", true);
        xhr.send(fd);
    });

    refresh.addEventListener("click", function () {
        recordButton.disabled = false;
        result.innerHTML = "";
        li.remove();
        var waveformCtx = canvas.getContext('2d');
        waveformCtx.clearRect(0, 0, canvas.width, canvas.height);
    });

    var fileReader = new FileReader();
    fileReader.onload = function (e) {
        audioContext.decodeAudioData(e.target.result).then(drawCompleteWaveform);
    };
    fileReader.readAsArrayBuffer(blob);

    li.appendChild(upload);
    li.appendChild(refresh);
    recordingsList.appendChild(li);
}

function drawCompleteWaveform(audioBuffer) {
    var waveformCtx = canvas.getContext('2d');
    waveformCtx.clearRect(0, 0, canvas.width, canvas.height);
    waveformCtx.strokeStyle = 'blue';
    waveformCtx.lineWidth = 2;
    waveformCtx.beginPath();

    var channelData = audioBuffer.getChannelData(0);
    var step = Math.ceil(channelData.length / canvas.width);
    var amp = canvas.height / 2;

    for (var i = 0; i < canvas.width; i++) {
        var min = 1.0;
        var max = -1.0;
        for (var j = 0; j < step; j++) {
            var datum = channelData[(i * step) + j];
            if (datum < min) min = datum;
            if (datum > max) max = datum;
        }
        waveformCtx.moveTo(i, (1 + min) * amp);
        waveformCtx.lineTo(i, (1 + max) * amp);
    }
    waveformCtx.stroke();
}
