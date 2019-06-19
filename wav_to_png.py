# This script is used to create raw+mel spectograms
# Also, retain.py uses the function "get_mel_spect" from this module for data augmentation

import csv
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def get_mel_spect(y, sr, label, wav_name): # Display a mel-scaled spectrogram

    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    set_fig()
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    image_dir = r'C:\Users\USER1\Desktop\urban_sound\spectrograms_mel'
    plt.savefig(image_dir + '/' + label  + '/' + wav_name + '.jpg')
    plt.close()
    return()


def get_raw_spect(y,label, wav_name):  # Display a raw spectrogram

    log_S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    set_fig()
    librosa.display.specshow(log_S, y_axis='linear')
    image_dir = r'C:\Users\USER1\Desktop\urban_sound\spectrograms_raw'
    plt.savefig(image_dir + '/' + label + '/' + wav_name + '.jpg')
    plt.close()
    return ()

def set_fig():
    # Set figure properties
    fig = plt.figure(figsize=(12, 4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    return


def my_main():
    base_dir = r'C:\Users\USER1\Desktop\urban_sound'
    matadate_file = os.path.join(base_dir,'UrbanSound8K.tar', 'UrbanSound8K', 'metadata', 'UrbanSound8K.csv')
    count = 0
    with open(matadate_file) as csvfile:
        meta_reader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(meta_reader):
            if idx > count:
                count += 1
            else:
                continue
            print(count)
            if not os.path.exists(os.path.join(base_dir,'spectrograms_mel',row[7])):
               os.makedirs('spectrograms_mel/' + row[7])
            if not os.path.exists(os.path.join(base_dir,'spectrograms_raw',row[7])):
               os.makedirs('spectrograms_raw/' + row[7])

            wav_file = os.path.join(base_dir, 'UrbanSound8K.tar', 'UrbanSound8K', 'audio', 'fold'+ str(row[5]),
                                    str(row[0]))
            y, sr = librosa.load(wav_file)
            get_mel_spect(y, sr, row[7], row[0])
            get_raw_spect(y, row[7], row[0])
            print(row[7] + '/' + row[0] + '.jpg')

if __name__ == '__main__':
    my_main()


