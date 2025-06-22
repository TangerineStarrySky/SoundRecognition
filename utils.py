import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import librosa
import wave
plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置字体为黑体
plt.rcParams["axes.unicode_minus"] = False # 正常显示负号

RATE = 8192

## 端点检测
# 计算能量
def energy(samples):
    samples = samples.astype(np.float64)
    return np.sum(samples ** 2)

# 计算过零率
def zero_crossing_rate(samples):
    return np.sum(np.diff(np.sign(samples)) != 0) / len(samples)

# 端点检测
def endpoint_detection(samples, zcr_threshold = 0.1, plot=False):
    frame_length = int(0.02*RATE)   # 20ms 帧
    hop_length = frame_length // 2   # 10ms 步长
    frame_energy = []
    frame_zcr = []
    for i in range(0, len(samples) - frame_length, hop_length):
        frame = samples[i:i + frame_length]
        frame_energy.append(energy(frame))
        frame_zcr.append(zero_crossing_rate(frame))

    frame_energy = np.array(frame_energy)
    frame_zcr = np.array(frame_zcr)

    # 绘制能量和过零率分布
    if plot:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        # 绘制能量分布，使用实线和蓝色
        ax1.plot(frame_energy, label='Energy', color='blue')
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Energy', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        # 创建一个共享x轴但具有不同y轴的轴对象
        ax2 = ax1.twinx()
        # 绘制过零率分布，使用虚线和红色
        ax2.plot(frame_zcr, label='Zero-Crossing Rate', linestyle='--', color='red')
        ax2.set_ylabel('Zero-Crossing Rate', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        # 添加基于frame_energy最大值的水平线
        energy_threshold = 0.01 * np.max(frame_energy)
        ax1.axhline(y=energy_threshold, color='green', linestyle='-', label=f'Energy Threshold: {energy_threshold:.2f}')
        ax2.axhline(y=zcr_threshold, color='orange', linestyle='-', label=f'ZCR Threshold: {zcr_threshold:.2f}')
        # 添加图例
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

    num_frames = len(frame_energy)
    in_speech = frame_energy > 0.01 * np.max(frame_energy)
    n1 = np.argmax(in_speech)
    n2 = num_frames - np.argmax(np.flip(in_speech))
    # print("帧数：", num_frames)
    
    # 精确查找起点和终点
    n1_prime = n1
    while n1_prime > 0 and frame_zcr[n1_prime] > zcr_threshold:
        n1_prime -= 1
    n1_prime = max(n1_prime-5, 0)
    
    n2_prime = n2
    while n2_prime < num_frames and frame_zcr[n2_prime] > zcr_threshold:
        n2_prime += 1
    n2_prime = min(n2_prime+10, num_frames - 1)
    start_sample, end_sample = n1_prime * hop_length, n2_prime * hop_length
    
    # 在图上添加竖线
    if plot:
        ax1.axvline(x=n1_prime, color='black', linestyle='--', label=f'Start Index: {n1}')
        ax1.axvline(x=n2_prime, color='black', linestyle='--', label=f'End Index: {n2}')
        plt.title('Energy and Zero-Crossing Rate Distribution with Thresholds and Vertical Lines')

        # 绘制波形
        plt.figure(figsize=(12, 4))
        plt.plot(samples)
        plt.axvline(x=start_sample, color='r', linestyle='--', label='Start')
        plt.axvline(x=end_sample, color='g', linestyle='--', label='End')
        plt.title("Speech Waveform with Endpoint Detection")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()
    return start_sample, end_sample


def extract_mfcc(file_path):
    """
    提取MFCC特征
    """
    with wave.open(file_path, 'rb') as wf:
        nframes = wf.getnframes()
        rate = wf.getframerate()
        strData = wf.readframes(nframes)
        waveData = np.frombuffer(strData, dtype=np.int16)
        # print("nchannels:", wf.getnchannels())
        # print("getsampwidth: ", wf.getsampwidth())

    # print("采样率:", rate)
    # print(len(waveData))
    # if rate != 8192:
    #     y, sr = librosa.load(file_path, sr=None, mono=True)
    #     y_resampled = librosa.resample(y, orig_sr=sr, target_sr=8192)
    #     waveData = y_resampled

    # 端点检测
    s, e = endpoint_detection(waveData)
    signal = waveData[s: e]*1.0
    mfccs = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=40, n_mels=128, n_fft=256, hop_length=80, win_length=256, lifter=12)
    # mfccs = compute_mfcc(signal).T
    mean = np.mean(mfccs, axis=1, keepdims=True)
    std = np.std(mfccs, axis=1, keepdims=True)
    fea = (mfccs-mean)/std
    # 添加一阶差分和二阶差分
    fea_d = librosa.feature.delta(fea, order=1)
    fea_dd = librosa.feature.delta(fea, order=2)
    fea = np.concatenate((fea.T, fea_d.T, fea_dd.T), axis=1)
    # fea = np.concatenate((fea.T, fea_d.T), axis=1)
    return fea


# 长度填充到最接近的2的幂，从而便于fft
def pad_to_power_of_two(x):
    N = len(x)
    next_power_of_two = 2 ** np.ceil(np.log2(N)).astype(int)
    padded_x = np.pad(x, (0, next_power_of_two - N), mode='constant')
    return padded_x

# 分治法实现fft
def MyFFT(x):
    N = len(x)
    if N <= 1:
        return x
    else:
        X_even = MyFFT(x[0::2])  # 分治
        X_odd = MyFFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N//2) / N)
        X = np.concatenate([X_even + factor * X_odd, X_even - factor * X_odd]) # 合并
        return X

# 预加重滤波器
def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

# 汉明窗
def hamming_window(N):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))


# 参数设置
N = 256             # 窗口长度
NFFT = 256          # 最接近N的2的幂次
D = 40              # MFCC系数维度
hop_length = 80     # 帧移长度
num_filters = 128    # 滤波器个数
lifter = 12         # 升倒 liftering 参数

def lifter_ceps(mfcc, lifter=12):
    """对 MFCC 倒谱系数应用升倒 liftering"""
    if lifter > 0:
        n = np.arange(mfcc.shape[1])
        lift = 1 + (lifter / 2) * np.sin(np.pi * n / lifter)
        return mfcc * lift
    else:
        return mfcc


# 计算Mel滤波器组
def mel_filterbank(num_filters):
    # Mel频率范围
    low_mel = 0
    high_mel = 2595 * np.log10(1 + (RATE / 2) / 700)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    # Mel -> Hz
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    # Hz -> FFT bin
    bin_points = np.floor((NFFT + 1) * hz_points / RATE).astype(int)
    # 创建滤波器组
    filters = np.zeros((num_filters, NFFT // 2 + 1))
    for i in range(1, num_filters + 1):
        left = bin_points[i-1]
        center = bin_points[i]
        right = bin_points[i+1]
        # 上升斜坡
        if center != left:
            filters[i-1, left:center] = np.linspace(0, 1, center - left)
        # 下降斜坡
        if right != center:
            filters[i-1, center:right] = np.linspace(1, 0, right - center)
    
    # Slaney风格归一化：每个滤波器能量和为1
    enorm = 2.0 / (hz_points[2:num_filters + 2] - hz_points[:num_filters])
    filters *= enorm[:, np.newaxis]
    return filters


# 计算MFCC系数
def compute_mfcc(signal, D=D, N=N, NFFT=NFFT, hop_length=hop_length, num_filters=num_filters, lifter=lifter):
    mfccs = []

    # 预加重
    signal = pre_emphasis(signal)
    signal = np.pad(signal, (NFFT // 2, NFFT // 2), mode='reflect')
    
    # 创建Mel滤波器组
    filters = mel_filterbank(num_filters)
    
    # 分帧处理
    for start in range(0, len(signal) - N + 1, hop_length):
        # 1. 取窗口
        frame = signal[start:start+N]
        
        # 2. 加窗
        frame = frame * hamming_window(N)
        
        # 3. FFT
        # mag_spectrum = np.abs(np.fft.rfft(frame, n=NFFT))
        mag_spectrum = np.abs(MyFFT(np.pad(frame, (0, NFFT - len(frame)), mode='constant'))[:NFFT//2+1])
        power_spectrum = (mag_spectrum ** 2) / NFFT
        
        # 4. Mel滤波
        mel_energy = np.dot(power_spectrum, filters.T)
        mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)  # 数值稳定性保护
        
        # 5. 取对数
        log_mel_energy = np.log(mel_energy)
        
        # 6. DCT变换得到MFCC
        mfcc = dct(log_mel_energy, type=2, norm='ortho')[:D]
        mfccs.append(mfcc)

    mfccs = np.array(mfccs)
    mfccs = lifter_ceps(mfccs, lifter) 

    return mfccs