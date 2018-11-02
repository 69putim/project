import os
import librosa.display
import matplotlib.pyplot as plt
import glob
import numpy as np

# 音声ファイルの読み込み
def load_audio_file(file_path):
    data, sr = librosa.load(file_path, sr=22050) # sr=22050
    # if len(data)>input_length:
    #     data = data[:input_length]
    # else:
    #     data = np.pad(data, (0, max(0, input_length - len(data))), "constant")# 足りない部分を0で埋める
    return data, sr

# グラフを出力
def plot_graph(data, sr):
    plt.xlabel('time[s]')
    plt.ylabel('Amplitude')
    librosa.display.waveplot(data, sr=sr)
    plt.show()

# stretchしたデータの作成
def stretch_wav_data(data, rate=1):
    input_length = len(data)
    data = librosa.effects.time_stretch(data, rate)
    if len(data) > input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

def make_save_folder(base, target):
    path = os.path.join(base, target)
    if not os.path.isdir(path):
        os.makedirs(path)
        print('make folder')
        print(path)


# main

base_dir = os.path.abspath(os.path.dirname(__file__)) # 現在のディレクトリを絶対パスで表示
my_audio_list = sorted(glob.glob(os.path.join(base_dir, 'input_1st_my_wav/'+'my_raw_wav/'+'*_v_*.wav'))) # .wavデータだけをリスト化
akane_audio_list = sorted(glob.glob(os.path.join(base_dir, 'input_1st_akane_wav/'+'akane_raw_wav/'+'*_v_*.wav'))) # .wavデータだけをリスト化
# print(audio_list)

# 保存先フォルダを作成　作ったデータは同じ階層に***_raw_.wavをそれぞれ作っておけ
make_save_folder(base_dir, 'input_1st_my_wav/' + 'my_noise_wav')
make_save_folder(base_dir, 'input_1st_my_wav/' + 'my_shift_wav')
make_save_folder(base_dir, 'input_1st_my_wav/' + 'my_stretch_wav')

make_save_folder(base_dir, 'input_1st_akane_wav/' + 'akane_noise_wav')
make_save_folder(base_dir, 'input_1st_akane_wav/' + 'akane_shift_wav')
make_save_folder(base_dir, 'input_1st_akane_wav/' + 'akane_stretch_wav')

# input_1st_my_wavでの処理
for i in range(0, 1000): # スタートする番号を合わせる、len(audio_list)でリスト内のデータ総数を取得
    target_file = "input_1st_my_wav/"+'my_raw_wav/'+'tori_v_'+'{0:03d}'.format(i)+'.wav'
    file_path = os.path.join(base_dir, target_file)
    print(file_path)

    if os.path.exists(file_path):
        # 波形情報をwav_dataへ,フレーム周波数をsampling_rateへ格納
        wav_data, sampling_rate = load_audio_file(file_path)
        # print(len(wav_data))

        # 波形情報を表示（デバック）
        # plot_graph(wav_data, sampling_rate)

        # ホワイトノイズ を作成、追加
        white_noise = np.random.randn(len(wav_data))
        wav_data_wn = wav_data + 0.005 * white_noise
        # ホワイトノイズ を加えた波形情報を表示（デバック）
        # plot_graph(wav_data_wn, sampling_rate)

        # Shiftしたデータを作成
        wav_data_shi = np.roll(wav_data, sampling_rate // 2)
        # Shiftしたデータの波形情報を表示（デバック）
        # plot_graph(wav_data_shi, sampling_rate)

        # stretchしたデータを作成
        wav_data_str = stretch_wav_data(wav_data, 1.1)
        # stretchしたデータの波形情報を表示（デバック）
        # plot_graph(wav_data_str, sampling_rate)

        # my_noise_wavにホワイトノイズ を加えたデータを保存
        out_path = os.path.join(base_dir,
                                "input_1st_my_wav/" + 'my_noise_wav/' + 'tori_v_' + '{0:03d}'.format(i) + '.wav')
        librosa.output.write_wav(out_path, wav_data_wn, 22050)

        # my_shift_wavにShiftしたデータを保存
        out_path = os.path.join(base_dir,
                                "input_1st_my_wav/" + 'my_shift_wav/' + 'tori_v_' + '{0:03d}'.format(i) + '.wav')
        librosa.output.write_wav(out_path, wav_data_shi, 22050)

        # my_stretch_wavにShiftしたデータを保存
        out_path = os.path.join(base_dir,
                                "input_1st_my_wav/" + 'my_stretch_wav/' + 'tori_v_' + '{0:03d}'.format(i) + '.wav')
        librosa.output.write_wav(out_path, wav_data_str, 22050)

# input_1st_akane_wavでの処理
for i in range(1, 1000): # スタートする番号を合わせる、len(audio_list)でリスト内のデータ総数を取得
    target_file = "input_1st_akane_wav/"+'akane_raw_wav/'+'tori_v_'+'{0:03d}'.format(i)+'.wav'
    file_path = os.path.join(base_dir, target_file)
    print(file_path)

    if os.path.exists(file_path):
        # 波形情報をwav_dataへ,フレーム周波数をsampling_rateへ格納
        wav_data, sampling_rate = load_audio_file(file_path)
        # print(len(wav_data))

        # 波形情報を表示（デバック）
        # plot_graph(wav_data, sampling_rate)

        # ホワイトノイズ を作成、追加
        white_noise = np.random.randn(len(wav_data))
        wav_data_wn = wav_data + 0.005 * white_noise
        # ホワイトノイズ を加えた波形情報を表示（デバック）
        # plot_graph(wav_data_wn, sampling_rate)

        # Shiftしたデータを作成
        wav_data_shi = np.roll(wav_data, sampling_rate // 2)
        # Shiftしたデータの波形情報を表示（デバック）
        # plot_graph(wav_data_shi, sampling_rate)

        # stretchしたデータを作成
        wav_data_str = stretch_wav_data(wav_data, 1.1)
        # stretchしたデータの波形情報を表示（デバック）
        # plot_graph(wav_data_str, sampling_rate)

        # my_noise_wavにホワイトノイズ を加えたデータを保存
        out_path = os.path.join(base_dir,
                                "input_1st_akane_wav/" + 'akane_noise_wav/' + 'tori_v_' + '{0:03d}'.format(i) + '.wav')
        librosa.output.write_wav(out_path, wav_data_wn, 22050)

        # my_shift_wavにShiftしたデータを保存
        out_path = os.path.join(base_dir,
                                "input_1st_akane_wav/" + 'akane_shift_wav/' + 'tori_v_' + '{0:03d}'.format(i) + '.wav')
        librosa.output.write_wav(out_path, wav_data_shi, 22050)

        # my_stretch_wavにShiftしたデータを保存
        out_path = os.path.join(base_dir, "input_1st_akane_wav/" + 'akane_stretch_wav/' + 'tori_v_' + '{0:03d}'.format(
            i) + '.wav')
        librosa.output.write_wav(out_path, wav_data_str, 22050)