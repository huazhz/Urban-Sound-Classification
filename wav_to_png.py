# Using this script I creates raw+mel spectograms
# Also, retain.py use this function "get_mel_spect" from this module for data augmentation

import csv
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


def create_mel_spect(y, sr, label, wav_name): # Create a mel-scaled spectrogram

    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    image_dir = r'C:\Users\USER1\Desktop\urban_sound\spectrograms_mel'
    plt.savefig(image_dir + '/' + label  + '/' + wav_name + '.jpg')
    plt.close()


def create_raw_spect(y,label, wav_name):  # Create a raw spectrogram

    log_S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(log_S, y_axis='linear')
    image_dir = r'C:\Users\USER1\Desktop\urban_sound\spectrograms_raw'
    plt.savefig(image_dir + '/' + label + '/' + wav_name + '.jpg')
    plt.close()


def set_fig():    # Set figure properties

    fig = plt.figure(figsize=(12, 4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)


def create_spect_by_file_type(wav_file, spec_type, label, file_name):

    y, sr = librosa.load(wav_file)
    set_fig()
    if spec_type == 'mel':
        create_mel_spect(y, sr, label, file_name)
    elif spec_type == 'raw':
        create_raw_spect(y, label, file_name)
    print(spec_type + '\n' + label + '/' + file_name + '.jpg')


def createSpectrograms(spec_types):

    base_dir = r'C:\Users\USER1\Desktop\urban_sound'
    matadate_file = os.path.join(base_dir,'UrbanSound8K.tar', 'UrbanSound8K', 'metadata', 'UrbanSound8K.csv')
    with open(matadate_file) as csvfile:
        meta_reader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(meta_reader):
            if idx == 0: # Ignore headers
                continue
            print(idx)

            fold = row[5]
            file_name = row[0]
            label = row[7]
            wav_file = os.path.join(base_dir, 'UrbanSound8K.tar', 'UrbanSound8K', 'audio', 'fold'+ fold, file_name)

            for spec_type in spec_types:
                dir_name = os.path.join(base_dir,'spectrograms_' + spec_type, label)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                create_spect_by_file_type(wav_file, spec_type, label, file_name)


if __name__ == '__main__':
    createSpectrograms(['mel', 'raw'])
