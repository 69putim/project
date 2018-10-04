import os 
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn import model_selection


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
#x, fs = load_wave_data(audio_dir, meta_data.loc[100, "filename"])
#melsp = calculate_melsp(x)
#print("wave size:{0}\nmelsp size:{1}\nsamping rate:{2}".format(x.shape, melsp.shape, fs))
#show_wav(x)
#show_melsp(melsp, fs)

#Augmentation(ホワイトノイズ)
def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))

#Augmentation(ストレッチ)
def stretch_sound(x, rate=1.1):
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)
    if len(x)>input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length -len(x))), "constant")

#Augmentation(シフト)
def shift_sound(x, rate=2):
    return np.roll(x, int(len(x)//rate))

#学習データとテストデータの作成
x = list(meta_data.loc[:,"filename"])
y = list(meta_data.loc[:,"target"])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, stratify=y)
print("x_train:{0}\ny train:{1}\nx test:{2}\ny test:{3}".format(len(x_train), len(y_train), len(x_test), len(y_test)))

#各クラスが均等に分割されているかを確認
a = np.zeros(50)
for c in y_test:
    a[c] += 1
print(a)

#作成した学習データの保存

freq = 128 #周波数
time = 1723

#augmentationと一緒にnpz形式で学習用データを保存
def save_np_data(filename, x, y, aug=None, rates=None):
    np_data = np.zeros(freq*time*len(x)).reshape(len(x), freq, time)
    np_targets = np.zeros(len(y))
    for i in range(len(y)):
        _x, fs = load_wave_data(audio_dir, x[i])
        if aug is not None:
            _x = aug(x = _x, rate=rates[i])
        _x = calculate_melsp(_x)
        np_data[i] = _x
        np_targets[i] = y[i]
    np.savez(filename, x=np_data, y=np_targets)

#テスト用データセットを保存
if not os.path.exists("esc_melsp_test.npz"):
    save_np_data("esc_melsp_test.npz", x_test, y_test)

#学習用データセットを保存
if not os.path.exists("esc_melsp_train_raw.npz"):
    save_np_data("esc_melsp_train_raw.npz", x_train, y_train)

#ホワイトノイズが入った学習用データセット
if not os.path.exists("esc_melsp_train_wn.npz"):
    rates = np.random.randint(1, 50, len(x_train))/10000
    save_np_data("esc_melsp_train_wn.npz", x_train, y_train, aug = add_white_noise, rates = rates)

#ストレッチされた学習用データセット
if not os.path.exists("esc_melsp_tarin_st.npz"):
    rates = np.random.choice(np.arange(80, 120), len(y_train))/100
    save_np_data("esc_melsp_train_st.npz", x_train, y_train, aug = stretch_sound, rates = rates)

#シフトされた学習用データセット
if not os.path.exists("esc_melsp_train_ss.npz"):
    rates = np.random.choice(np.arange(2, 6), len(y_train))
    save_np_data("esc_melsp_train_ss.npz", x_train, y_train, aug = shift_sound, rates = rates)

#ホワイトノイズ、ストレッチ、シフトがランダムに組み合わされた学習用データセット
if not os.path.exists("esc_melsp_train_comb.npz"):
    np_data = np.zeros(freq*time*len(x_train)).reshape(len(x_train), freq, time)
    np_targets = np.zeros(len(y_train))
    for i in range(len(y_train)):
        x, fs = load_wave_data(audio_dir, x_train[i])
        x = add_white_noise(x = x, rate = np.random.randint(1, 50)/1000)
        if np.random.choice((True, False)):
            x = stretch_sound(x = x, rate = np.random.choice(np.arange(80, 120))/100)
        else:
            x = shift_sound(x = x, rate = np.random.choice(np.arange(2, 6)))
        x = calculate_melsp(x)
        np_data[i] = x
        np_targets[i] = y_train[i]
    np.savez("esc_melsp_train_comb.npz", x = np_data, y = np_targets)




