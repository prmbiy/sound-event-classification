import os
import sys
import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from glob import glob
from tqdm import tqdm
import argparse

num_cores = 4

def compute_melspec(filename, outdir):
    try:
        wav = librosa.load(filename, sr=44100)[0]
        melspec = librosa.feature.melspectrogram(
            wav,
            sr=44100,
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
    

def main(input_path, output_path):
    file_list = glob(input_path + '/*.wav')
    os.makedirs(output_path, exist_ok=True)
    _ = Parallel(n_jobs=num_cores)(
            delayed(lambda x: compute_melspec(x, output_path + '/'))(x)
            for x in tqdm(file_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input and Output Paths')
    parser.add_argument('input_path', type=str, help="Specifies directory of audio files")
    parser.add_argument('output_path', type=str, help="Specifies directory for generated spectrograms")
    args = parser.parse_args()
    
    main(args.input_path, args.output_path)
