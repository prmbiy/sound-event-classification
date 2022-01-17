import os
import sys
import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from glob import glob
from tqdm import tqdm
import argparse
from config_global import n_fft, hop_length, n_mels, fmin, fmax, sample_rate, num_cores

def compute_melspec(filename, outdir, audio_segment_length):
    try:
        wav = librosa.load(filename, sr=sample_rate)[0]
        if(audio_segment_length!=-1 and audio_segment_length!=0):
            wav = wav [:sample_rate*audio_segment_length]
        melspec = librosa.feature.melspectrogram(
            wav,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax)
        logmel = librosa.core.power_to_db(melspec)
        np.save(outdir + os.path.basename(filename) + '.npy', logmel)
    except ValueError:
        print('ERROR IN:', filename)

def main(input_path, output_path, audio_segment_length):
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
