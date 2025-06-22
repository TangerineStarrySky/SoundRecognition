import os
import wave
import numpy as np
import matplotlib.pyplot as plt
from playsound import playsound

def visualize(dir):
    files = os.listdir(dir)
    for file in files:
        print(f'{dir}/'+file)
        f = wave.open(f'{dir}/'+file, 'rb')
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = f.readframes(nframes)
        wavaData = np.frombuffer(strData, dtype=np.int16)
        wavaData = wavaData * 1.0 / max(abs(wavaData))
        wavaData = np.reshape(wavaData, [nframes, nchannels]).T
        f.close()
        time = np.arange(0, nframes) * (1.0 / framerate)
        time = np.reshape(time, [nframes, 1]).T
        plt.plot(time[0, :nframes], wavaData[0, :nframes], c="b")
        plt.xlabel("time(seconds)")
        plt.ylabel("amplitude")
        plt.title("Original wave_{}".format(file[:-4]))
        plt.savefig('waves/{}.jpg'.format(file))
        # plt.show()
        break

def play(dir):
    files = os.listdir(dir)
    # files = [files[i*20+4] for i in range(20)]
    idx = 20
    files = files[(idx-1)*20 :(idx-1)*20+5]
    for file in files:
        playsound(os.path.join(dir, file))
        # break
    # playsound(os.path.join(dir, files[3]))


import librosa

if __name__ == "__main__":
    dirs = os.listdir("dataset")
    # play("dataset/23300240026")
