import os
import sys
import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from glob import glob
from tqdm import tqdm
import argparse

num_cores = -1

def compute_melspec(filename, outdir, audio_segment_length):
    try:
        sr = 44100
        wav = librosa.load(filename, sr=sr)[0]
        if(audio_segment_length!=-1 and audio_segment_length!=0):
            wav = wav [:sr*audio_segment_length]
        melspec = librosa.feature.melspectrogram(
            wav,
            sr=sr,
            n_fft=2560,
            hop_length=694,
            n_mels=128,
            fmin=20,
            fmax=22050)
        logmel = librosa.core.power_to_db(melspec)
#        new_name_ = filename.split('/')[-1].split('-')
#        new_name = '{}/{}-{}.wav.npy'.format('/'.join(filename.split('/')[:-1]), new_name_[0], new_name_[1].split('_')[0])
        np.save(outdir + os.path.basename(filename) + '.npy', logmel)
    except ValueError:
        print('ERROR IN:', filename)

# def compute_melspec(filename, outdir, sample_duration=-1):
#     try:
#         sr = 44100
#         wav = librosa.load(filename, sr=sr)[0]
#         audio_len_s = librosa.get_duration(wav, sr=sr)
#         if sample_duration == -1:
#             sample_duration = audio_len_s

#         samples = int(sample_duration*sr)
#         for i in range(int(audio_len_s//sample_duration)):
#             melspec = librosa.feature.melspectrogram(
#                 wav[i*samples: (i+1)*samples],
#                 sr=sr,
#                 n_fft=2560,
#                 hop_length=694,
#                 n_mels=128,
#                 fmin=20,
#                 fmax=22050)
#             logmel = librosa.core.power_to_db(melspec)
#             # new_name_ = filename.split('/')[-1].split('-')
#             # new_name = '{}/{}-{}.wav.npy'.format('/'.join(filename.split('/')[:-1]), new_name_[0], new_name_[1].split('_')[0])
#             np.save(outdir + os.path.basename(filename) + f'.{i+1}.npy', logmel)
#     except ValueError:
#         print('ERROR IN:', filename)


def main(input_path, output_path, audio_segment_length):
    #, sample_duration):
    file_list = glob(input_path + '/*.wav')
    os.makedirs(output_path, exist_ok=True)
    _ = Parallel(n_jobs=num_cores)(
            delayed(lambda x: compute_melspec(x, output_path + '/', audio_segment_length))(x)
            for x in tqdm(file_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input and Output Paths')
    parser.add_argument('input_path', type=str, help="Specifies directory of audio files.")
    parser.add_argument('output_path', type=str, help="Specifies directory for generated spectrograms.")
    parser.add_argument('-a', '--audio_segment_length', type=int, help="Specifies length of audio segment to extract from each audio file. Default -1(Consider full length audio).", default=-1)
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.audio_segment_length)
            #, args.__getattr__('sample_duration'))
