'''
For an input audio of any length, the function performs a prediction for every 3 second frame
with a stride of 1 second

Output: Final prediction based on majority voting
'''


import os
import sys
import librosa
import torch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from glob import glob
from tqdm import tqdm
import argparse
from utils import Task5Model
from augmentation.SpecTransforms import ResizeSpectrogram

num_frames = 636
feature_type = "logmelspec"

channel_means = np.load('./data/statistics/channel_means_{}.npy'.format(feature_type)).reshape(1,-1,1)
channel_stds = np.load('./data/statistics/channel_stds_{}.npy'.format(feature_type)).reshape(1,-1,1)


cuda = True
device = torch.device('cuda:0' if cuda else 'cpu')

model = Task5Model(8).to(device)
model.load_state_dict(torch.load('./models/model_{}'.format(feature_type)))
model = model.eval()


num_cores = -1

labels = ['1_engine', '2_machinery-impact', '3_non-machinery-impact',
            '4_powered-saw', '5_alert-signal', '6_music', '7_human-voice', '8_dog']

def compute_melspectr(filename, outdir, audio_segment_length):

    sr = 44100
    wav = librosa.load(filename, sr=sr)[0]
    preds = []
    scores = []
    frame_number = []
    
    if(audio_segment_length == -1):
        audio_segment_length = wav
    
    if(audio_segment_length*sr > len(wav) ):
        audio_segment_length = np.floor(wav/sr)
    
    try:
        start =0
        n = 1
        while( (start + audio_segment_length*sr) < len(wav)):
            segment = wav[start:start+ audio_segment_length*sr]

            melspec = librosa.feature.melspectrogram(
                segment,
                sr=sr,
                n_fft=2560,
                hop_length=694,
                n_mels=128,
                fmin=20,
                fmax=22050)
            logmel = librosa.core.power_to_db(melspec)
            logmel = (logmel - channel_means)/channel_stds
            logmel = torch.Tensor(logmel)
            if len(logmel.shape)<=3:
                logmel = torch.unsqueeze(logmel, 0)
            inputs = logmel.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)
                for j in range(outputs.shape[0]):
                    scores.append(outputs[j,:].detach().cpu().numpy())
                outputs = outputs[0].detach().cpu().numpy()
                frame_number.append(n)
                pred = labels[np.argmax(outputs)]
                preds.append(pred)

            start = start + sr
            n = n+1
        

        scores = np.array(scores)
        output_df = pd.DataFrame(
        scores, columns=[
            '1_engine', '2_machinery-impact', '3_non-machinery-impact',
            '4_powered-saw', '5_alert-signal', '6_music', '7_human-voice', '8_dog'])

        output_df['frame_number'] = pd.Series(
        frame_number,
        index=output_df.index)
    
    
        output_df = output_df.sort_values('frame_number', ignore_index=True)

        #output_df.to_csv('./models/pred_eval_testing_3s.csv', index=False)
        os.makedirs(f"./models/{outdir}", exist_ok = True)
        output_df.to_csv(f"./models/{outdir}pred.csv", index=False)
        print(f"Predictions saved in './models/{outdir}pred.csv'.")

        final_prediction = max(set(preds), key = preds.count)
        print(filename + " " + final_prediction)

        
    except ValueError:
        print('ERROR IN:', filename)   
        
    
        
        

def main(input_path, output_path, audio_segment_length):
    #, sample_duration):
    if not(os.path.exists(output_path)):
            os.mkdir(output_path)
    
    file_list = glob(input_path + '/*.wav')
    n = len(input_path)
    for i in range (len(file_list)):
        pathset = output_path  + '/' +  file_list[i][n+1:-4] + '/'
        if not(os.path.exists(pathset)):
            os.mkdir(pathset)
    
    os.makedirs(output_path, exist_ok=True)
    _ = Parallel(n_jobs=num_cores)(
            delayed(lambda x: compute_melspectr(x, output_path  + '/' +  x[n+1:-4] + '/', audio_segment_length))(x)
            for x in tqdm(file_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input and Output Paths')
    parser.add_argument('input_path', type=str, help="Specifies directory of audio files.")
    parser.add_argument('output_path', type=str, help="Specifies directory to save prediction scores.")
    parser.add_argument('-a', '--audio_segment_length', type=int, help="Specifies length of audio segment to predict for each audio file. Default -1(Consider full length audio).", default=-1)
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.audio_segment_length)
            #, args.__getattr__('sample_duration'))

        
#  Predictions seem to match for: 
#"00_010346_ogg1600.wav"
# "00_010426_ogg450.wav"
# "00_010457_ogg550.wav"
#"00_010471_ogg950.wav"
#"00_010477_ogg3200.wav"
    