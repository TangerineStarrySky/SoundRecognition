import librosa
import utils
import numpy as np

def test_MyFFT():
    x = np.random.rand(256)
    custom_fft = utils.MyFFT(x)
    numpy_fft = np.fft.fft(x)
    
    error = np.max(np.abs(custom_fft - numpy_fft))
    if error < 1e-10:
        print("MyFFT 通过测试!")


def test_pre_emphasis():
    x = np.random.rand(100)
    alpha = 0.97
    emphasized = utils.pre_emphasis(x, alpha)
    expected = np.append(x[0], x[1:] - alpha * x[:-1])
    assert np.allclose(emphasized, expected), "预加重不一致"
    print("pre_emphasis 通过测试!")


def test_hamming_window():
    N = 256
    custom_win = utils.hamming_window(N)
    numpy_win = np.hamming(N)
    
    assert np.allclose(custom_win, numpy_win, atol=1e-10), "Hamming窗差异过大"
    print("hamming_window 通过测试!")


def test_mel_filterbank():
    RATE = 8192
    NFFT = 256
    num_filters = 40
    custom_filters = utils.mel_filterbank(num_filters)
    librosa_filters = librosa.filters.mel(sr=RATE, n_fft=NFFT, n_mels=num_filters)
    assert custom_filters.shape == librosa_filters.shape, \
        f"形状不一致: {custom_filters.shape} vs {librosa_filters.shape}"
    # 允许较小误差
    max_diff = np.max(np.abs(custom_filters - librosa_filters))
    if max_diff < 0.05:
        print("mel_filterbank 通过测试!")
    else:
        print("mel_filterbank 差异较大，最大误差:", max_diff)


def test_compute_mfcc():
    y, sr = librosa.load("dataset/22307110035/22307110035_04_04.dat", sr=8192)
    custom_mfcc = utils.compute_mfcc(y, D=4, num_filters=40, NFFT=256, hop_length=80, lifter=12)
    librosa_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=4, n_mels=40, n_fft=256, hop_length=80, win_length=256, lifter=12).T
    mean = np.mean(librosa_mfcc, axis=1, keepdims=True)
    std = np.std(librosa_mfcc, axis=1, keepdims=True)
    librosa_mfcc = (librosa_mfcc-mean)/std

    mean = np.mean(custom_mfcc, axis=1, keepdims=True)
    std = np.std(custom_mfcc, axis=1, keepdims=True)
    custom_mfcc = (custom_mfcc-mean)/std

    assert custom_mfcc.shape == librosa_mfcc.shape, \
        f"形状不一致: {custom_mfcc.shape} vs {librosa_mfcc.shape}"
    error = np.mean((custom_mfcc - librosa_mfcc)**2)

    if error < 0.05:
        print("compute_mfcc 通过测试!")
    else:
        print("compute_mfcc 测试失败，均方误差:", error)


if __name__ == "__main__":
    test_MyFFT()
    test_pre_emphasis()
    test_hamming_window()
    test_mel_filterbank()
    test_compute_mfcc()