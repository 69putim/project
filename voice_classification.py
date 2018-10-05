#x そのままのデータ
#y ラベル

import os 
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn import model_selection

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Conv2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

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
print("x_train:{0}\ny_train:{1}\nx_test:{2}\ny_test:{3}".format(len(x_train), len(y_train), len(x_test), len(y_test)))


#各クラスが均等に分割されているかを確認
a = np.zeros(50) #50個のクラスそれぞれの中に入っているテストデータの数を数える
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



#CNNのそれら

#学習データセット
train_files = ["esc_melsp_train_raw.npz",
               "esc_melsp_train_ss.npz",
               "esc_melsp_train_st.npz",
               "esc_melsp_train_wn.npz",
               "esc_melsp_train_comb.npz"]
test_file = "esc_melsp_test.npz"

train_num = 1500
test_num = 500

#データセット中のパーセプトロンを設定
x_train = np.zeros(freq*time*train_num*len(train_files)).reshape(train_num*len(train_files), freq, time)
y_train = np.zeros(train_num*len(train_files))

#学習データの読み込み
for i in range(len(train_files)):
    data = np.load(train_files[i])
    x_train[i*train_num:(i+1)*train_num] = data["x"]
    y_train[i*train_num:(i+1)*train_num] = data["y"]

#テストデータの読み込み
test_data = np.load(test_file)
x_test = test_data["x"]
y_test = test_data["y"]

#正解を一つのベクトルに再定義
classes = 50
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)

#学習データのリシェイプ
x_train = x_train.reshape(train_num*5, freq, time, 1)
x_test = x_test.reshape(test_num, freq, time, 1)

#リシェイプされた学習データとテストデータのサイズを表示
print("x_train:{0}\ny_train:{1}\nx_test:{2}\ny_test:{3}".format(x_train.shape,
                                                                y_train.shape,
                                                                x_test.shape,
                                                                y_test.shape))

def cba(inputs, filters, kernel_size, strides):#stridesは見る所の動く範囲の大きさ
    x = Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = 'same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

#CNNの詳細
inputs = Input(shape=(x_train.shape[1:]))

x_1 = cba(inputs, filters = 32, kernel_size = (1, 8), strides = (1, 2))
x_1 = cba(x_1, filters = 32, kernel_size = (8, 1), strides = (2, 1))
x_1 = cba(x_1, filters = 64, kernel_size = (1, 8), strides = (1, 2))
x_1 = cba(x_1, filters = 64, kernel_size = (8, 1), strides = (2, 1))

x_2 = cba(inputs, filters = 32, kernel_size = (1, 16), strides = (1, 2))
x_2 = cba(x_2, filters = 32, kernel_size = (16, 1), strides = (2, 1))
x_2 = cba(x_2, filters = 64, kernel_size = (1, 16), strides = (1, 2))
x_2 = cba(x_2, filters = 64, kernel_size = (16, 1), strides = (2,1))

x_3 = cba(inputs, filters = 32, kernel_size = (1, 32), strides = (1, 2))
x_3 = cba(x_3, filters = 32, kernel_size = (32, 1), strides = (2, 1))
x_3 = cba(x_3, filters = 64, kernel_size = (1, 32), strides = (1, 2))
x_3 = cba(x_3, filters = 64, kernel_size = (32, 1), strides = (2, 1))

x_4 = cba(inputs, filters = 32, kernel_size = (1, 64), strides = (1, 2))
x_4 = cba(x_4, filters = 32, kernel_size = (64, 1), strides = (2, 1))
x_4 = cba(x_4, filters = 64, kernel_size = (1, 64), strides = (1, 2))
x_4 = cba(x_4, filters = 64, kernel_size = (64, 1), strides = (2, 1))

x = Add()([x_1, x_2, x_3, x_4])

x = cba(x, filters=128, kernel_size=(1,16), strides=(1,2))
x = cba(x, filters=128, kernel_size=(16,1), strides=(2,1))

x = GlobalAveragePooling2D()(x)
x = Dense(classes)(x)
x = Activation("softmax")(x)

model = Model(inputs, x)

#adam optimizerを起動
opt = keras.optimizers.adam(lr=0.00001, decay=1e-6, amsgrad=True)

#adam amsgradで学習する
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#モデルチェックポイントのディレクトリ
model_dir = "./models"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

#早期停止とチェックポイント
es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
chkpt = os.path.join(model_dir, 'esc50_.{epoch:02d}_{val_loss:.4f}_{val_acc:.4f}.hdf5')
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')



#学習データにmixupを用いてCNNを学習する
class MixupGenerator():
    def __init__(self, x_train, y_train, batch_size=16, alpha=0.2, shuffle=True):
        self.x_train=x_train
        self.y_train=y_train
        self.batch_size=batch_size
        self.alpha=alpha
        self.shuffle=shuffle
        self.sample_num=len(x_train)

    def __call__(self):
        while True:
            indexes=self.__get_exploration_order()
            itr_num=int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                x, y = self.__data_generation(batch_ids)

                yield x, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.x_train.shape
        _, class_num = self.y_train.shape
        x1 = self.x_train[batch_ids[:self.batch_size]]
        x2 = self.x_train[batch_ids[self.batch_size:]]
        y1 = self.y_train[batch_ids[:self.batch_size]]
        y2 = self.y_train[batch_ids[self.batch_size:]]
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        x_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        x = x1 * x_l + x2 * (1 - x_l)
        y = y1 * y_l + y2 * (1 - y_l)

        return x, y


#モデルの訓練
batch_size = 16
epochs = 1000

training_generator = MixupGenerator(x_train, y_train)()
model.fit_generator(generator = training_generator,
                    steps_per_epoch = x_train.shape[0] // batch_size,
                    validation_data = (x_test, y_test),
                    epochs = epochs,
                    verbose = 1,
                    shuffle = True,
                    callbacks = [es_cb, cp_cb])

#モデルの評価
model = load_model("./models/esc50_.105_0.8096_0.8200.hdf5")

evaluation = model.evaluate(x_test, y_test)
print(evaluation)
