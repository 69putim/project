import os
from typing import Any, Union

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

#ディレクトリを定義
base_dir = "./"
esc_dir = os.path.join(base_dir, "ESC-50-master")
meta_file = os.path.join(base_dir, "ESC-50/meta/esc50.csv")
audio_dir = os.path.join(base_dir, "ESC-50/audio/")

#metadataの読み込み
meta_data = pd.read_csv(meta_file)

#dataサイズの取得
data_size = meta_data.shape
print(data_size)

#ターゲットラベルと名前の変更
class_dict = {}

for i in range(data_size[0]):
    if meta_data.loc[i, "target"] not in class_dict.keys():
        class_dict[meta_data.loc[i, "target"]] = meta_data.loc[i, "category"]

#wavデータの読み込み
def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_path, sr=44100)
    return x, fs

#wavデータをメルケプストラムへ変更
def calculate_melsp(x, n_fft = 1024, hop_length = 128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128)
    return melsp

#wavを表示
def show_wav(x):
    plt.plot(x)
    plt.show()

#メルケプストラムを表示
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs, x_axis="time", y_axis="mel", hop_length=128)
    plt.colorbar(format='%+2.0f db')
    plt.show()

#サンプル
x, fs = load_wave_data(audio_dir, meta_data.loc[100, "filename"])
melsp = calculate_melsp(x)
print("wave size:{0}\nmelsp size:{1}\nsamping rate:{2}".format(x.shape, melsp.shape, fs))
show_wav(x)
show_melsp(melsp, fs)
